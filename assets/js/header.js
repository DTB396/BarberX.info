(() => {
  const header = document.getElementById("siteHeader");
  if (!header) return;

  const toggle = header.querySelector("[data-nav-toggle]");
  const nav = header.querySelector("[data-nav]");
  if (!toggle || !nav) return;

  const setOpen = (open) => {
    header.classList.toggle("nav-open", open);
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    toggle.setAttribute("aria-label", open ? "Close menu" : "Open menu");
  };

  toggle.addEventListener("click", () => {
    const open = !header.classList.contains("nav-open");
    setOpen(open);
  });

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && header.classList.contains("nav-open")) setOpen(false);
  });

  nav.addEventListener("click", (e) => {
    const a = e.target.closest("a");
    if (!a) return;
    if (window.matchMedia("(max-width: 768px)").matches) setOpen(false);
  });

  let lastY = window.scrollY;
  const onScroll = () => {
    const y = window.scrollY;
    header.classList.toggle("is-scrolled", y > 8);

    if (header.classList.contains("nav-open")) {
      header.classList.remove("is-hidden");
      lastY = y;
      return;
    }

    const goingDown = y > lastY;
    const past = y > 140;
    header.classList.toggle("is-hidden", goingDown && past);
    lastY = y;
  };

  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();
