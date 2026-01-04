// Barber Cam — Nav + small utilities (additive, preserves visuals)
(function () {
  // Footer year
  const yearEl = document.getElementById("year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  // Mobile nav toggle
  const navToggle = document.querySelector(".nav-toggle");
  const mainNav = document.querySelector(".main-nav");

  if (!navToggle || !mainNav) return;

  function closeNav() {
    navToggle.setAttribute("aria-expanded", "false");
    mainNav.classList.remove("active");
    document.body.style.overflow = ""; // only matters if you lock scroll in CSS/JS
  }

  function openNav() {
    navToggle.setAttribute("aria-expanded", "true");
    mainNav.classList.add("active");
    // Optional: lock background scroll if your drawer overlays content
    // document.body.style.overflow = "hidden";
  }

  navToggle.addEventListener("click", () => {
    const isExpanded = navToggle.getAttribute("aria-expanded") === "true";
    isExpanded ? closeNav() : openNav();
  });

  // Close when internal anchor links are clicked (same-page navigation)
  mainNav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", (e) => {
      // Only close for internal anchor links (hash navigation on same page)
      const href = link.getAttribute("href");
      if (href && href.startsWith("#")) {
        closeNav();
        // Optional: return focus to toggle for accessibility
        requestAnimationFrame(() => {
          if (navToggle && typeof navToggle.focus === "function") {
            navToggle.focus();
          }
        });
      }
    });
  });

  // Close if hash changes (backup for anchor navigation)
  window.addEventListener("hashchange", closeNav);
  window.addEventListener("popstate", closeNav);
})();
