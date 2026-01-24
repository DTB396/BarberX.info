/**
 * Premium Navigation Enhancement
 * Mobile menu, dropdowns, search, and accessibility
 */

(function () {
  "use strict";

  // Wait for DOM
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }

  function init() {
    initMobileNav();
    initDesktopDropdowns();
    initSearch();
    initScrollBehavior();
  }

  /**
   * Mobile Navigation
   */
  function initMobileNav() {
    const toggle = document.querySelector(".premium-nav-toggle");
    const mobileNav = document.querySelector(".premium-nav--mobile");
    const closeBtn = document.querySelector(".premium-nav__close");
    const overlay =
      document.querySelector(".premium-nav-overlay") || createOverlay();
    const submenuToggles = document.querySelectorAll(
      ".premium-nav__submenu-toggle",
    );

    if (!toggle || !mobileNav) return;

    // Open menu
    toggle.addEventListener("click", () => {
      const isOpen = toggle.getAttribute("aria-expanded") === "true";

      if (isOpen) {
        closeMenu();
      } else {
        openMenu();
      }
    });

    // Close button
    if (closeBtn) {
      closeBtn.addEventListener("click", closeMenu);
    }

    // Overlay click
    overlay.addEventListener("click", closeMenu);

    // Submenu toggles
    submenuToggles.forEach((toggleBtn) => {
      toggleBtn.addEventListener("click", (e) => {
        e.preventDefault();
        const isExpanded = toggleBtn.getAttribute("aria-expanded") === "true";
        toggleBtn.setAttribute("aria-expanded", !isExpanded);
      });
    });

    // Close on escape
    document.addEventListener("keydown", (e) => {
      if (
        e.key === "Escape" &&
        mobileNav.getAttribute("aria-hidden") === "false"
      ) {
        closeMenu();
        toggle.focus();
      }
    });

    // Trap focus in mobile menu
    mobileNav.addEventListener("keydown", trapFocus);

    function openMenu() {
      toggle.setAttribute("aria-expanded", "true");
      toggle.setAttribute("aria-label", "Close navigation menu");
      mobileNav.setAttribute("aria-hidden", "false");
      overlay.classList.add("is-active");
      document.body.style.overflow = "hidden";

      // Focus first link
      const firstLink = mobileNav.querySelector("a, button");
      if (firstLink) {
        setTimeout(() => firstLink.focus(), 100);
      }
    }

    function closeMenu() {
      toggle.setAttribute("aria-expanded", "false");
      toggle.setAttribute("aria-label", "Open navigation menu");
      mobileNav.setAttribute("aria-hidden", "true");
      overlay.classList.remove("is-active");
      document.body.style.overflow = "";

      // Close all submenus
      submenuToggles.forEach((btn) => {
        btn.setAttribute("aria-expanded", "false");
      });
    }

    function createOverlay() {
      const div = document.createElement("div");
      div.className = "premium-nav-overlay";
      document.body.appendChild(div);
      return div;
    }

    function trapFocus(e) {
      if (e.key !== "Tab") return;

      const focusableElements = mobileNav.querySelectorAll(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
      );

      const firstElement = focusableElements[0];
      const lastElement = focusableElements[focusableElements.length - 1];

      if (e.shiftKey && document.activeElement === firstElement) {
        e.preventDefault();
        lastElement.focus();
      } else if (!e.shiftKey && document.activeElement === lastElement) {
        e.preventDefault();
        firstElement.focus();
      }
    }
  }

  /**
   * Desktop Dropdowns
   */
  function initDesktopDropdowns() {
    const dropdownToggles = document.querySelectorAll(
      ".premium-nav__dropdown-toggle",
    );

    dropdownToggles.forEach((toggle) => {
      const dropdown = toggle.nextElementSibling;
      if (!dropdown || !dropdown.classList.contains("premium-nav__dropdown"))
        return;

      let timeout;

      toggle.addEventListener("click", (e) => {
        e.preventDefault();
        const isExpanded = toggle.getAttribute("aria-expanded") === "true";
        closeAllDropdowns();

        if (!isExpanded) {
          toggle.setAttribute("aria-expanded", "true");
          dropdown.style.display = "block";
          setTimeout(() => (dropdown.style.opacity = "1"), 10);
        }
      });

      // Hover for desktop
      if (window.innerWidth >= 1024) {
        toggle.parentElement.addEventListener("mouseenter", () => {
          clearTimeout(timeout);
          closeAllDropdowns();
          toggle.setAttribute("aria-expanded", "true");
          dropdown.style.display = "block";
          setTimeout(() => (dropdown.style.opacity = "1"), 10);
        });

        toggle.parentElement.addEventListener("mouseleave", () => {
          timeout = setTimeout(() => {
            toggle.setAttribute("aria-expanded", "false");
            dropdown.style.opacity = "0";
            setTimeout(() => (dropdown.style.display = "none"), 200);
          }, 150);
        });
      }
    });

    // Close dropdowns when clicking outside
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".premium-nav__item--has-dropdown")) {
        closeAllDropdowns();
      }
    });

    // Close on escape
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeAllDropdowns();
      }
    });

    function closeAllDropdowns() {
      dropdownToggles.forEach((toggle) => {
        const dropdown = toggle.nextElementSibling;
        toggle.setAttribute("aria-expanded", "false");
        if (dropdown) {
          dropdown.style.opacity = "0";
          setTimeout(() => (dropdown.style.display = "none"), 200);
        }
      });
    }
  }

  /**
   * Search Toggle
   */
  function initSearch() {
    const searchToggle = document.getElementById("header-search-toggle");
    const searchModal = document.getElementById("search-modal");

    if (!searchToggle) return;

    searchToggle.addEventListener("click", () => {
      if (searchModal) {
        searchModal.classList.add("is-active");
        const searchInput = searchModal.querySelector('input[type="search"]');
        if (searchInput) {
          setTimeout(() => searchInput.focus(), 100);
        }
      }
    });

    // Keyboard shortcut: Ctrl+K or Cmd+K
    document.addEventListener("keydown", (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "k") {
        e.preventDefault();
        searchToggle.click();
      }
    });
  }

  /**
   * Scroll Behavior
   */
  function initScrollBehavior() {
    const header = document.getElementById("siteHeader");
    if (!header) return;

    let lastScroll = 0;
    let ticking = false;

    window.addEventListener("scroll", () => {
      if (!ticking) {
        window.requestAnimationFrame(() => {
          handleScroll();
          ticking = false;
        });
        ticking = true;
      }
    });

    function handleScroll() {
      const currentScroll = window.pageYOffset;

      // Add shadow on scroll
      if (currentScroll > 10) {
        header.classList.add("scrolled");
      } else {
        header.classList.remove("scrolled");
      }

      // Hide on scroll down, show on scroll up (optional)
      // if (currentScroll > lastScroll && currentScroll > 100) {
      //   header.style.transform = 'translateY(-100%)';
      // } else {
      //   header.style.transform = 'translateY(0)';
      // }

      lastScroll = currentScroll;
    }
  }
})();
