---
layout: default
title: "Interactive US Medication & Gardening Guide"
description: "Discover which medications are overprescribed in your state, their natural origins, and what you can garden as alternatives. Interactive map for all 50 states with verified CDC data."
audience: "Public"
status: "Living document"
permalink: /health/holistic-independence/
show_breadcrumbs: true
hide_hero: true

page_includes:
  - holistic-data.html

page_css:
  - /assets/css/pages/holistic-map.css
  - /assets/css/pages/holistic-map-enhanced.css

page_js:
  - /assets/js/holistic-interactive-map.js
  - /assets/js/symptom-plant-tool.js

<section class="section-block">
  <div class="container">
    <p class="section-eyebrow">Stewardship • Food-first • Informed consent</p>
    <h1 class="section-heading">Holistic Independence & Stewardship Guide</h1>
    <p class="section-lead">
      A practical reference for families and communities who want to lower baseline inflammation, stress, and preventable pain—
      while collaborating safely with licensed clinicians.
    </p>

    <blockquote>
      <p><em>“Jesus entered the temple and overturned the tables of the money changers.”</em></p>
      <p class="muted">
        This guide challenges dependency-by-default and encourages responsibility, discernment, and stewardship.
        It does not reject legitimate medical care.
      </p>
    </blockquote>
  </div>
</section>

<section class="section-block">
  <div class="container">

    <h2 class="h3">Read first</h2>

    <div class="callout">
      <ul>
        <li><strong>No medical advice.</strong> Educational and collaborative only.</li>
        <li><strong>No “stop your meds” messaging.</strong> Medication changes must be made with a licensed clinician.</li>
        <li><strong>Safety first.</strong> If symptoms are severe, sudden, or worsening, seek urgent medical care.</li>
        <li><strong>Verify locally.</strong> Zones and frost windows are planning aids; microclimates vary by neighborhood and elevation.</li>
      </ul>
    </div>

    <p class="muted">
      Faith Frontier does not diagnose, treat, or cure disease. Individuals assume responsibility for personal decisions and should
      coordinate care with qualified professionals.
    </p>

  </div>
</section>

<section class="holistic-hero">
  <div class="container">
    <h1>🇺🇸 Interactive US Medication & Gardening Guide</h1>
    <p>
      Click your state to discover which medications are overprescribed, their natural origins, 
      and what you can garden as healthy alternatives.
    </p>
    
    <div class="hero-steps">
      <div class="hero-step">
        <span class="hero-step-number">1</span>
        <span class="hero-step-text">Click Your State</span>
      </div>
      <div class="hero-step">
        <span class="hero-step-number">2</span>
        <span class="hero-step-text">See Medication Data</span>
      </div>
      <div class="hero-step">
        <span class="hero-step-number">3</span>
        <span class="hero-step-text">Learn What to Garden</span>
      </div>
    </div>
  </div>
</section>

<section class="section-block">
  <div class="container">

    <div class="step-container">
      <div class="step-card">
        <div class="step-header">
          <div class="step-number-badge">1</div>
          <div>
            <h2 class="step-title">Select Your State</h2>
            <p class="step-subtitle">Click any state on the map below to get started</p>
          </div>
        </div>

        <div class="map-section">
          <div class="map-instructions">
            <h3>👆 Click Any State on the Map</h3>
            <p>See prescription rates, top medications, and gardening recommendations for your area</p>
          </div>

          <div class="map-shell" id="usMapShell">
            <object id="usMapObj" type="image/svg+xml" data="{{ '/assets/svg/us-states.svg' | relative_url }}"></object>
          </div>

          <div id="mapTooltip" class="map-tooltip" hidden></div>

          <div class="map-legend-box">
            <h4>Prescription Rate Levels</h4>
            <div class="map-legend-grid">
              <div class="legend-item">
                <div class="legend-color-box" style="background: #10b981;"></div>
                <span class="legend-label">Lower (&lt;60%)</span>
              </div>
              <div class="legend-item">
                <div class="legend-color-box" style="background: #fbbf24;"></div>
                <span class="legend-label">Moderate (60-65%)</span>
              </div>
              <div class="legend-item">
                <div class="legend-color-box" style="background: #f59e0b;"></div>
                <span class="legend-label">High (66-67%)</span>
              </div>
              <div class="legend-item">
                <div class="legend-color-box" style="background: #dc2626;"></div>
                <span class="legend-label">Crisis (68%+)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="step-card">
        <div class="step-header">
          <div class="step-number-badge">2</div>
          <div>
            <h2 class="step-title">Your State's Data</h2>
            <p class="step-subtitle">Medication statistics, natural origins, and alternatives</p>
          </div>
        </div>

        <div id="resultsMeta" class="results-meta"></div>
      </div>

      <div class="step-card">
        <div class="step-header">
          <div class="step-number-badge">3</div>
          <div>
            <h2 class="step-title">Plants You Can Garden</h2>
            <p class="step-subtitle">Grow these in your hardiness zone</p>
          </div>
        </div>

        <div id="resultsGrid" class="results-grid"></div>
      </div>
    </div>

  </div>
</section>

<section class="section-block">
  <div class="container">
    <h2 class="h3">Practical foundations</h2>

    <h3 class="h4">Core principle</h3>
    <p>
      Inflammation, pain, and nervous-system load are influenced throughout the day by food, movement, posture, sleep, stress,
      and environment. Medication can be appropriate and sometimes essential—this guide focuses on strengthening daily inputs.
    </p>

    <h3 class="h4">Florida-friendly starter set (adaptable by zone)</h3>
    <ul>
      <li><strong>Turmeric</strong> + black pepper + dietary fat (food/tea)</li>
      <li><strong>Ginger</strong> (tea + cooking)</li>
      <li><strong>Tulsi</strong> (gentle tea)</li>
      <li><strong>Rosemary</strong> (food + infused oil for massage)</li>
      <li><strong>Leafy greens</strong>, <strong>sweet potatoes</strong>, <strong>okra</strong>, <strong>peppers</strong>, <strong>cooked tomatoes</strong></li>
    </ul>

    <h3 class="h4">Oil infusion (beginner-safe)</h3>
    <ol>
      <li>Chop rosemary and/or ginger.</li>
      <li>Cover in olive or coconut oil.</li>
      <li>Infuse 2–4 weeks (or warm gently on very low heat).</li>
      <li>Strain and label with date.</li>
    </ol>
    <p class="muted">
      Essential oils are concentrated and can irritate skin—avoid undiluted use. Food + tea + gentle topical oils are a safer first step.
    </p>

    <h3 class="h4">Caution list</h3>
    <ul>
      <li><strong>St. John’s Wort</strong>: broad medication interactions.</li>
      <li><strong>Kava</strong>: potential liver risk.</li>
      <li>High-dose extracts: avoid early; start low and coordinate with a clinician.</li>
    </ul>

  </div>
</section>

<section class="section-block">
  <div class="container">
    <h2 class="h3">Expansion roadmap</h2>

    <ul>
      <li>State tooltips: zone hint + approximate frost window + featured plants</li>
      <li>Condition subguides: pain/nerve, sleep, stress, inflammation, digestion</li>
      <li>Printable regional garden plans</li>
      <li>Clinician appendix: collaboration workflow, contraindication checklist, documentation templates</li>
    </ul>

    <p>
      This guide is a starting point —
      <strong>a table overturned against exploitation, while honoring the sanctity of the temple itself.</strong>
    </p>

    <p class="muted">Living document. Contributions welcome, with safety and sourcing standards.</p>
  </div>
</section>
