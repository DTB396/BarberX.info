// Respect prefers‑reduced‑motion – already handled in CSS,
// but we keep a small helper for any future interactive bits.
(() => {
    const header = document.querySelector('.site-header');
    if (!header) return;

    let lastY = window.scrollY;
    let ticking = false;

    const update = () => {
        const y = window.scrollY;

        header.classList.toggle('is-scrolled', y > 8);

        // hide-on-scroll-down, show-on-scroll-up
        const goingDown = y > lastY;
        header.classList.toggle('is-hidden', goingDown && y > 96);

        lastY = y;
        ticking = false;
    };

    window.addEventListener('scroll', () => {
        if (!ticking) {
            window.requestAnimationFrame(update);
            ticking = true;
        }
    }, { passive: true });

    update();
})();

    () => {
    const header = document.querySelector('.site-header');
    if (!header) return;

    let lastY = window.scrollY;
    let ticking = false;

    const update = () => {
        const y = window.scrollY;
        header.classList.toggle('is-scrolled', y > 8);

        const goingDown = y > lastY;
        header.classList.toggle('is-hidden', goingDown && y > 96);

        lastY = y;
        ticking = false;
    };

    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(update);
            ticking = true;
        }
    }, { passive: true });

    update();
})();

document.addEventListener('DOMContentLoaded', () => {
    const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)');
    if (prefersReduced.matches) {
        console.log('User prefers reduced motion – additional JS animations disabled.');
        // Insert any extra cleanup here if you later add JS‑driven animations.
    }

    // Example: log CTA clicks (replace with real analytics if desired)
    const cta = document.querySelector('.hero-cta');
    if (cta) {
        cta.addEventListener('click', () => console.log('CTA clicked'));
    }
});
(() => {
    const header = document.querySelector('.site-header');
    const navToggle = document.querySelector('.nav-toggle');
    const mainNav = document.querySelector('.main-nav');

    // Mobile nav
    if (navToggle && mainNav) {
        navToggle.addEventListener('click', () => {
            const expanded = navToggle.getAttribute('aria-expanded') === 'true';
            navToggle.setAttribute('aria-expanded', String(!expanded));
            mainNav.classList.toggle('active');
        });
    }

    // Sticky header behavior
    if (!header) return;

    let lastY = window.scrollY;
    let ticking = false;

    const update = () => {
        const y = window.scrollY;
        header.classList.toggle('is-scrolled', y > 8);

        const goingDown = y > lastY;
        header.classList.toggle('is-hidden', goingDown && y > 120);

        lastY = y;
        ticking = false;
    };

    window.addEventListener('scroll', () => {
        if (!ticking) {
            requestAnimationFrame(update);
            ticking = true;
        }
    }, { passive: true });

    update();
})();
