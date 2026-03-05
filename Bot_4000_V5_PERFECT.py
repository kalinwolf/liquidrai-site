import os
import json
import time
import math
import hashlib
import sqlite3
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
import numpy as np

try:
    import MetaTrader5 as mt5
except Exception:  # pragma: no cover
    mt5 = None

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

DB_PATH = os.getenv("BOT_DB_PATH", os.path.join(os.getcwd(), "bot_4000.db"))
SYMBOL = os.getenv("BOT_SYMBOL", "EURUSD")
DQN_WEIGHTS_PATH = os.getenv("DQN_WEIGHTS_PATH", os.path.join(os.getcwd(), "dqn_weights.npz"))
SCALER_PATH = os.getenv("SCALER_PATH", os.path.join(os.getcwd(), "scaler.pkl"))
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:11434/api/generate")
TRADE_LOT_SIZE = float(os.getenv("TRADE_LOT_SIZE", "0.01"))
LOOP_SLEEP_SECONDS = int(os.getenv("LOOP_SLEEP_SECONDS", "30"))
COOLDOWN_SECONDS = int(os.getenv("COOLDOWN_SECONDS", "120"))

logger = logging.getLogger("bot4000")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

LAST_TRADE_TS = 0.0
LAST_POS_TICKET = None
LAST_POS_PROFIT = 0.0


# =========================
# V4-compatible core helpers
# =========================
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions_log (
            ts_run TEXT,
            headline_ts TEXT,
            headline_hash TEXT,
            headline TEXT,
            current_price REAL,
            current_vol REAL,
            atr_h2 REAL,
            llm_action TEXT,
            llm_confidence REAL,
            llm_reason TEXT,
            dqn_action TEXT,
            dqn_confidence REAL,
            final_action TEXT,
            trade_executed INTEGER,
            PRIMARY KEY (ts_run, headline_hash)
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experience_log (
            ts TEXT,
            state_features TEXT,
            action TEXT,
            reward REAL,
            next_state_features TEXT,
            done INTEGER
        )
        """
    )

    # V5 tables
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS macro_cache(
            ts TEXT PRIMARY KEY,
            us2y REAL,
            de2y REAL,
            yield_spread REAL,
            dxy REAL,
            dxy_chg REAL,
            eur_pos REAL,
            basis_eurusd REAL,
            basis_chg REAL,
            source TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS options_levels(
            ts TEXT,
            strike REAL,
            notional REAL,
            PRIMARY KEY (ts, strike)
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS headlines(
            headline_hash TEXT PRIMARY KEY,
            first_seen_ts TEXT,
            text TEXT
        )
        """
    )

    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions_log(ts_run)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_hash ON decisions_log(headline_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_macro_ts ON macro_cache(ts)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_options_ts ON options_levels(ts)")

    _ensure_decisions_log_columns(cur)

    conn.commit()
    conn.close()


def _ensure_decisions_log_columns(cur: sqlite3.Cursor) -> None:
    cur.execute("PRAGMA table_info(decisions_log)")
    existing = {r[1] for r in cur.fetchall()}
    wanted = {
        "v5_env_score": "REAL",
        "v5_macro_score": "REAL",
        "v5_session_score": "REAL",
        "v5_gamma_score": "REAL",
        "v5_final_score": "REAL",
        "v5_bias_action": "TEXT",
        "v5_bias_strength": "REAL",
        "v5_reason": "TEXT",
    }
    for col, ctype in wanted.items():
        if col not in existing:
            cur.execute(f"ALTER TABLE decisions_log ADD COLUMN {col} {ctype}")


def make_features() -> np.ndarray:
    # Stable 16-feature vector intended for M5 aggregation.
    # If MT5 unavailable, return neutral vector.
    if mt5 is None:
        return np.zeros(16, dtype=np.float32)

    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 200)
    if rates is None or len(rates) < 50:
        return np.zeros(16, dtype=np.float32)

    close = np.array([r[4] for r in rates], dtype=np.float64)
    high = np.array([r[2] for r in rates], dtype=np.float64)
    low = np.array([r[3] for r in rates], dtype=np.float64)
    vol = np.array([r[5] for r in rates], dtype=np.float64)

    ret = np.diff(close) / np.maximum(close[:-1], 1e-9)
    ma_fast = close[-10:].mean()
    ma_slow = close[-40:].mean()
    tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
    atr = tr[-24:].mean() if len(tr) >= 24 else 0.0

    feats = np.array(
        [
            ret[-1] if len(ret) else 0.0,
            ret[-5:].mean() if len(ret) >= 5 else 0.0,
            ret[-20:].mean() if len(ret) >= 20 else 0.0,
            np.std(ret[-20:]) if len(ret) >= 20 else 0.0,
            ma_fast - ma_slow,
            (close[-1] - ma_fast),
            atr,
            vol[-1] if len(vol) else 0.0,
            vol[-10:].mean() if len(vol) >= 10 else 0.0,
            np.max(close[-20:]) - close[-1],
            close[-1] - np.min(close[-20:]),
            np.percentile(ret[-50:], 90) if len(ret) >= 50 else 0.0,
            np.percentile(ret[-50:], 10) if len(ret) >= 50 else 0.0,
            math.sin(datetime.now(timezone.utc).hour / 24.0 * 2 * math.pi),
            math.cos(datetime.now(timezone.utc).hour / 24.0 * 2 * math.pi),
            1.0,
        ],
        dtype=np.float32,
    )
    return feats


def enter_trade(action: str, lot: float, atr_h2: float) -> bool:
    global LAST_TRADE_TS
    if mt5 is None or lot <= 0:
        return False
    if action not in ("BUY", "SELL"):
        return False
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return False

    price = tick.ask if action == "BUY" else tick.bid
    point = mt5.symbol_info(SYMBOL).point
    sl_dist = max(atr_h2, point * 80)
    tp_dist = sl_dist * 1.5
    sl = price - sl_dist if action == "BUY" else price + sl_dist
    tp = price + tp_dist if action == "BUY" else price - tp_dist
    req = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lot,
        "type": mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 4000,
        "comment": "Bot_4000_V5_PERFECT",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(req)
    ok = result is not None and result.retcode == mt5.TRADE_RETCODE_DONE
    if ok:
        LAST_TRADE_TS = time.time()
    return ok


# =========================
# V5 layer
# =========================
@dataclass
class MacroSnapshot:
    us2y: float = 0.0
    de2y: float = 0.0
    yield_spread: float = 0.0
    dxy: float = 0.0
    dxy_chg: float = 0.0
    eur_pos: float = 0.0
    basis_eurusd: float = 0.0
    basis_chg: float = 0.0
    source: str = "none"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_whitespace(text: str) -> str:
    return " ".join((text or "").split())


def _headline_hash(text: str) -> str:
    return hashlib.sha256(_normalize_whitespace(text).encode("utf-8")).hexdigest()


def _load_macro_inputs() -> MacroSnapshot:
    p = os.path.join(os.getcwd(), "macro_inputs.json")
    if not os.path.exists(p):
        return MacroSnapshot(source="none")
    try:
        with open(p, "r", encoding="utf-8") as f:
            d = json.load(f)
        eur_pos = float(d.get("eur_pos", 0.0))
        eur_pos = max(-1.0, min(1.0, eur_pos))
        return MacroSnapshot(
            us2y=float(d.get("us2y", 0.0)),
            de2y=float(d.get("de2y", 0.0)),
            yield_spread=float(d.get("yield_spread", 0.0)),
            dxy=float(d.get("dxy", 0.0)),
            dxy_chg=float(d.get("dxy_chg", 0.0)),
            eur_pos=eur_pos,
            basis_eurusd=float(d.get("basis_eurusd", 0.0)),
            basis_chg=float(d.get("basis_chg", 0.0)),
            source="file",
        )
    except Exception:
        return MacroSnapshot(source="none")


def _write_macro_cache(conn: sqlite3.Connection, ts: str, snap: MacroSnapshot) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO macro_cache
        (ts, us2y, de2y, yield_spread, dxy, dxy_chg, eur_pos, basis_eurusd, basis_chg, source)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (ts, snap.us2y, snap.de2y, snap.yield_spread, snap.dxy, snap.dxy_chg, snap.eur_pos, snap.basis_eurusd, snap.basis_chg, snap.source),
    )


def _load_options_levels(conn: sqlite3.Connection) -> List[Dict[str, float]]:
    p = os.path.join(os.getcwd(), "options_levels.json")
    levels: List[Dict[str, float]] = []
    if not os.path.exists(p):
        return levels
    try:
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ts = payload.get("ts") or _now_iso()[:13] + ":00:00Z"
        for lv in payload.get("levels", []):
            strike = float(lv.get("strike", 0.0))
            notional = float(lv.get("notional", 0.0))
            levels.append({"strike": strike, "notional": notional, "ts": ts})
            conn.execute(
                "INSERT OR REPLACE INTO options_levels (ts, strike, notional) VALUES (?,?,?)",
                (ts, strike, notional),
            )
    except Exception:
        return []
    return levels


def _compute_session_score(ts: datetime) -> float:
    h = ts.hour
    if 7 <= h < 16:
        return 0.7
    if 12 <= h < 20:
        return 0.9
    return 0.3


def _compute_gamma_score(price: float, levels: List[Dict[str, float]], ts: datetime) -> Tuple[float, str]:
    if not levels:
        near_cut = 1.0 if ts.hour in (14, 15) else 0.0
        return max(0.0, 0.4 - near_cut * 0.2), "no_levels"
    distances = []
    for lv in levels:
        if lv["strike"] > 0:
            distances.append(abs(price - lv["strike"]) / lv["strike"])
    nearest = min(distances) if distances else 1.0
    # Lower score near strike to reduce trend chasing
    score = max(0.0, min(1.0, nearest * 300))
    return score, f"nearest_strike_dist={nearest:.5f}"


def _compute_macro_bias(snap: MacroSnapshot) -> Tuple[float, str, float, str]:
    # SELL EURUSD bias when USD funding tightens / DXY rises
    score = 0.0
    score -= snap.dxy_chg * 40
    score += snap.eur_pos * 0.3
    score += (-snap.basis_chg) * -2.0
    score += (snap.de2y - snap.us2y) * 0.2
    score = max(-1.0, min(1.0, score))
    if score > 0.2:
        return score, "BUY", abs(score), "macro_buy_bias"
    if score < -0.2:
        return score, "SELL", abs(score), "macro_sell_bias"
    return score, "HOLD", abs(score), "macro_neutral"


def _compute_env_score(vol_score: float, session_score: float, gamma_score: float, macro_score: float) -> float:
    return max(0.0, min(1.0, 0.30 * vol_score + 0.25 * session_score + 0.20 * gamma_score + 0.25 * (1 - abs(macro_score))))


def _compute_vol_score(features: np.ndarray) -> float:
    # prefer moderate volatility
    vol = abs(float(features[3])) if len(features) > 3 else 0.0
    return max(0.0, min(1.0, 1.0 - min(vol * 3000, 1.0)))


def _v5_no_trade_gate(candidate_action: str, env_score: float, macro_bias_action: str, macro_strength: float) -> Tuple[bool, str]:
    if time.time() - LAST_TRADE_TS < COOLDOWN_SECONDS:
        return False, "cooldown"
    if env_score < 0.35:
        return False, "env_below_min"
    if candidate_action in ("BUY", "SELL") and macro_bias_action in ("BUY", "SELL"):
        if candidate_action != macro_bias_action and macro_strength >= 0.7:
            return False, "macro_conflict_veto"
    return True, "allowed"


def _insert_decision_base(
    conn: sqlite3.Connection,
    ts_run: str,
    headline_ts: str,
    headline_hash: str,
    headline: str,
    current_price: float,
    current_vol: float,
    atr_h2: float,
    llm_action: str,
    llm_confidence: float,
    llm_reason: str,
    dqn_action: str,
    dqn_confidence: float,
    final_action: str,
    trade_executed: int,
) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO decisions_log
        (ts_run, headline_ts, headline_hash, headline, current_price, current_vol, atr_h2,
         llm_action, llm_confidence, llm_reason, dqn_action, dqn_confidence, final_action, trade_executed)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            ts_run,
            headline_ts,
            headline_hash,
            headline,
            current_price,
            current_vol,
            atr_h2,
            llm_action,
            llm_confidence,
            llm_reason,
            dqn_action,
            dqn_confidence,
            final_action,
            trade_executed,
        ),
    )


def _update_decision_v5(
    conn: sqlite3.Connection,
    ts_run: str,
    headline_hash: str,
    env_score: float,
    macro_score: float,
    session_score: float,
    gamma_score: float,
    final_score: float,
    bias_action: str,
    bias_strength: float,
    reason: str,
) -> None:
    conn.execute(
        """
        UPDATE decisions_log
        SET v5_env_score=?, v5_macro_score=?, v5_session_score=?, v5_gamma_score=?,
            v5_final_score=?, v5_bias_action=?, v5_bias_strength=?, v5_reason=?
        WHERE ts_run=? AND headline_hash=?
        """,
        (env_score, macro_score, session_score, gamma_score, final_score, bias_action, bias_strength, reason, ts_run, headline_hash),
    )


def _llm_decide(headline: str) -> Tuple[str, float, str]:
    try:
        prompt = f"Classify EURUSD action BUY/SELL/HOLD for headline: {headline}. Return JSON keys action, confidence, reason."
        payload = {"model": "llama3", "prompt": prompt, "stream": False}
        r = requests.post(LLM_ENDPOINT, json=payload, timeout=3)
        txt = r.json().get("response", "") if r.ok else ""
        action = "HOLD"
        if "BUY" in txt.upper():
            action = "BUY"
        elif "SELL" in txt.upper():
            action = "SELL"
        return action, 0.55, txt[:240] or "llm_fallback"
    except Exception:
        return "HOLD", 0.0, "llm_unreachable"


def _dqn_decide(features: np.ndarray) -> Tuple[str, float]:
    # placeholder deterministic policy, keeps interface stable
    momentum = float(features[1]) if len(features) > 1 else 0.0
    if momentum > 0.0001:
        return "BUY", 0.55
    if momentum < -0.0001:
        return "SELL", 0.55
    return "HOLD", 0.5


def _get_current_price_vol() -> Tuple[float, float]:
    if mt5 is None:
        return 0.0, 0.0
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        return 0.0, 0.0
    price = (tick.bid + tick.ask) / 2
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M5, 0, 24)
    vol = float(np.std(np.diff(np.array([r[4] for r in rates], dtype=np.float64)))) if rates is not None and len(rates) > 2 else 0.0
    return float(price), vol


def _get_atr_h2() -> float:
    if mt5 is None:
        return 0.0
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H2, 0, 30)
    if rates is None or len(rates) < 5:
        return 0.0
    h = np.array([r[2] for r in rates], dtype=np.float64)
    l = np.array([r[3] for r in rates], dtype=np.float64)
    c = np.array([r[4] for r in rates], dtype=np.float64)
    tr = np.maximum(h[1:] - l[1:], np.maximum(abs(h[1:] - c[:-1]), abs(l[1:] - c[:-1])))
    return float(tr[-14:].mean()) if len(tr) >= 14 else 0.0


def _get_open_position():
    if mt5 is None:
        return None
    pos = mt5.positions_get(symbol=SYMBOL)
    if not pos:
        return None
    return pos[0]


def _log_experience(conn: sqlite3.Connection, state: np.ndarray, action: str, next_state: np.ndarray) -> None:
    global LAST_POS_TICKET, LAST_POS_PROFIT
    position = _get_open_position()

    # Legacy reward (gated: zero when no position)
    legacy_reward = 0.0
    if position is not None:
        legacy_reward = float(next_state[0] - state[0]) if len(state) and len(next_state) else 0.0

    conn.execute(
        "INSERT INTO experience_log (ts, state_features, action, reward, next_state_features, done) VALUES (?,?,?,?,?,?)",
        (_now_iso(), json.dumps(state.tolist()), action, legacy_reward, json.dumps(next_state.tolist()), 0),
    )

    # V5 PnL-aligned reward row
    if position is not None:
        ticket = getattr(position, "ticket", None)
        cur_profit = float(getattr(position, "profit", 0.0))
        pnl_reward = 0.0
        if LAST_POS_TICKET == ticket:
            pnl_reward = cur_profit - LAST_POS_PROFIT
        LAST_POS_TICKET = ticket
        LAST_POS_PROFIT = cur_profit

        conn.execute(
            "INSERT INTO experience_log (ts, state_features, action, reward, next_state_features, done) VALUES (?,?,?,?,?,?)",
            (_now_iso(), json.dumps(state.tolist()), action, pnl_reward, json.dumps(next_state.tolist()), 0),
        )


def startup_diagnostics() -> None:
    logger.info("DB path: %s", DB_PATH)
    logger.info("DB exists: %s", os.path.exists(DB_PATH))
    logger.info("DQN weights exists: %s", os.path.exists(DQN_WEIGHTS_PATH))
    logger.info("Scaler exists: %s", os.path.exists(SCALER_PATH))

    mt5_ok = False
    if mt5 is not None:
        mt5_ok = mt5.initialize()
        logger.info("MT5 init: %s", mt5_ok)
        if mt5_ok:
            info = mt5.symbol_info(SYMBOL)
            if info is not None:
                mt5.symbol_select(SYMBOL, True)
                logger.info("MT5 symbol selected: %s", SYMBOL)
    else:
        logger.warning("MT5 module unavailable")

    try:
        r = requests.post(LLM_ENDPOINT, json={"model": "llama3", "prompt": "ping", "stream": False}, timeout=2)
        logger.info("LLM reachable: %s", r.ok)
    except Exception:
        logger.warning("LLM endpoint unreachable")


def run_cycle() -> None:
    conn = get_db()
    ts_run = _now_iso()
    now_dt = datetime.now(timezone.utc)

    macro = _load_macro_inputs()
    _write_macro_cache(conn, ts_run, macro)
    option_levels = _load_options_levels(conn)

    features = make_features()
    state = features.copy()
    dqn_action, dqn_conf = _dqn_decide(features)
    headline = "N/A"
    headline_ts = ts_run
    headline_hash = "NO_HEADLINE"
    llm_action, llm_conf, llm_reason = _llm_decide(headline)

    # V4 base candidate action (DQN lead, LLM as filter)
    candidate_action = dqn_action
    if llm_action in ("BUY", "SELL") and llm_action != dqn_action:
        candidate_action = "HOLD"

    current_price, current_vol = _get_current_price_vol()
    atr_h2 = _get_atr_h2()

    vol_score = _compute_vol_score(features)
    session_score = _compute_session_score(now_dt)
    gamma_score, gamma_reason = _compute_gamma_score(current_price, option_levels, now_dt)
    macro_score, bias_action, bias_strength, macro_reason = _compute_macro_bias(macro)
    env_score = _compute_env_score(vol_score, session_score, gamma_score, macro_score)

    # Signed v5 final score, direction-sensitive
    direction = 1.0 if candidate_action == "BUY" else -1.0 if candidate_action == "SELL" else 0.0
    v5_final_score = direction * (0.35 * env_score + 0.25 * session_score + 0.20 * gamma_score + 0.20 * (1 - abs(macro_score)))

    allow_trade, gate_reason = _v5_no_trade_gate(candidate_action, env_score, bias_action, bias_strength)
    if atr_h2 <= 0:
        allow_trade = False
        gate_reason = "atr_h2_zero"

    final_action = candidate_action
    if not allow_trade:
        final_action = "HOLD"

    trade_executed = 0
    if final_action in ("BUY", "SELL") and _get_open_position() is None:
        if enter_trade(final_action, TRADE_LOT_SIZE, atr_h2):
            trade_executed = 1

    _insert_decision_base(
        conn,
        ts_run,
        headline_ts,
        headline_hash,
        headline,
        current_price,
        current_vol,
        atr_h2,
        llm_action,
        llm_conf,
        llm_reason,
        dqn_action,
        dqn_conf,
        final_action,
        trade_executed,
    )

    v5_reason = f"gate={gate_reason}; macro={macro_reason}; gamma={gamma_reason}"
    _update_decision_v5(
        conn,
        ts_run,
        headline_hash,
        env_score,
        macro_score,
        session_score,
        gamma_score,
        v5_final_score,
        bias_action,
        bias_strength,
        v5_reason,
    )

    conn.execute(
        "INSERT OR IGNORE INTO headlines (headline_hash, first_seen_ts, text) VALUES (?,?,?)",
        (headline_hash, ts_run, headline),
    )

    next_state = make_features()
    _log_experience(conn, state, final_action, next_state)

    conn.commit()
    conn.close()


# Compatibility wrapper

def main() -> None:
    init_db()
    startup_diagnostics()
    while True:
        try:
            run_cycle()
        except Exception as exc:
            logger.exception("run_cycle failed: %s", exc)
        time.sleep(LOOP_SLEEP_SECONDS)


if __name__ == "__main__":
    main()
