(function(){
  const toast = document.getElementById("toast");
  const toastMsg = document.getElementById("toastMsg");
  const toastBtn = document.getElementById("toastBtn");

  function showToast(msg){
    if(!toast) return;
    toastMsg.textContent = msg;
    toast.classList.add("show");
    if(toastBtn){
      toastBtn.onclick = () => toast.classList.remove("show");
    }
    setTimeout(()=>toast.classList.remove("show"), 4500);
  }

  // smooth scroll for same-page anchors
  document.addEventListener("click", (e)=>{
    const a = e.target.closest("a");
    if(!a) return;
    const href = a.getAttribute("href") || "";
    if(href.startsWith("#")){
      const el = document.querySelector(href);
      if(el){
        e.preventDefault();
        el.scrollIntoView({behavior:"smooth", block:"start"});
      }
    }
  });

  // copy-to-clipboard helpers
  document.querySelectorAll("[data-copy]").forEach(btn=>{
    btn.addEventListener("click", async ()=>{
      const txt = btn.getAttribute("data-copy") || "";
      try{
        await navigator.clipboard.writeText(txt);
        showToast("Copied to clipboard: " + txt);
      }catch{
        showToast("Copy failed. Try manual copy.");
      }
    });
  });

  // contact form (no backend) – mailto fallback
  const form = document.getElementById("contactForm");
  if(form){
    form.addEventListener("submit", (e)=>{
      e.preventDefault();
      const name = (document.getElementById("name")||{}).value || "";
      const email = (document.getElementById("email")||{}).value || "";
      const msg = (document.getElementById("message")||{}).value || "";
      const subject = encodeURIComponent("LiquidRai inquiry — " + (name || "Website visitor"));
      const body = encodeURIComponent(
        "Name: " + name + "\n" +
        "Email: " + email + "\n\n" +
        msg + "\n\n" +
        "Sent from: https://kalinwolf.github.io/"
      );
      const to = "hello@liquidrai.com"; // placeholder; replace if you have a real inbox
      window.location.href = `mailto:${to}?subject=${subject}&body=${body}`;
      showToast("Opening your email client…");
    });
  }

  // mark active nav
  const path = (location.pathname.split("/").pop() || "index.html").toLowerCase();
  document.querySelectorAll(".nav-links a").forEach(a=>{
    const href = (a.getAttribute("href")||"").toLowerCase();
    if(href === path || (path==="" && href==="index.html")){
      a.classList.add("active");
    }
  });
})();
