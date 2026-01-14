---
layout: default
title: "Case Records & Legal Archive"
permalink: /cases/
description: "Complete archive of Faith Frontier's legal proceedings—documenting competence, transparency, and constitutional literacy through public court records."
stylesheet: /assets/css/cases-index.css
---

{%- comment -%}
==============================================================================
FAITH FRONTIER CASE RECORDS INDEX
==============================================================================

This page displays the complete legal archive with:
- Hero section with case statistics
- Filter controls for status, court, and search
- Responsive card grid of all published cases
- Smart resource display (notes/evidence only when present)

Styling: /assets/css/cases-index.css
Case Layout: _layouts/case-enhanced.html
Resources Include: _includes/case-resources.html
{%- endcomment -%}

<!-- ==================== HERO SECTION ==================== -->
<div class="cases-hero">
  <div class="container">
    <h1>Case Records & Legal Archive</h1>
    <p class="cases-hero-lead">
      Every filing. Every order. Every motion. Faith Frontier's complete legal record demonstrates <strong>procedural competence</strong>, 
      <strong>constitutional literacy</strong>, and <strong>transparent accountability</strong>—the foundation of trust for government partnerships and community stewardship.
    </p>
    
    <!-- Case Statistics -->
    <div class="cases-stats">
      {% assign total_cases = site.cases | where_exp: "case", "case.published != false" | size %}
      {% assign active_cases = site.cases | where: "status", "active" | where_exp: "case", "case.published != false" | size %}
      {% assign pending_cases = site.cases | where: "status", "pending" | where_exp: "case", "case.published != false" | size %}
      {% assign closed_cases = site.cases | where: "status", "closed" | where_exp: "case", "case.published != false" | size %}
      
      <div class="case-stat">
        <span class="case-stat-number">{{ total_cases }}</span>
        <span class="case-stat-label">Total Cases</span>
      </div>
      <div class="case-stat">
        <span class="case-stat-number">{{ active_cases }}</span>
        <span class="case-stat-label">Active Litigation</span>
      </div>
      <div class="case-stat">
        <span class="case-stat-number">{{ pending_cases }}</span>
        <span class="case-stat-label">Pending Decisions</span>
      </div>
      <div class="case-stat">
        <span class="case-stat-number">100%</span>
        <span class="case-stat-label">Transparency</span>
      </div>
    </div>
    
    <!-- Call-to-Action Buttons -->
    <div class="cases-actions">
      <a href="/stewardship/" class="btn-cases btn-primary">Join the Stewardship Journey</a>
      <a href="/government-partnerships/" class="btn-cases btn-secondary">Municipal Partnerships</a>
      <a href="#active-cases" class="btn-cases btn-secondary">Active Case Digest</a>
    </div>
  </div>
</div>

<!-- ==================== MAIN CONTENT ==================== -->
<div class="container" style="max-width: 1100px; margin: 0 auto; padding: 0 1.5rem;">
  
  <!-- ==================== WHY THIS MATTERS ==================== -->
  <section class="why-matters-section">
    <h2 class="why-matters-title">Why This Archive Matters</h2>
    <div class="why-matters-grid">
      
      <!-- For Government Partners -->
      <div class="why-matters-card">
        <h3>For Government Partners</h3>
        <p>
          Documented track record of procedural competence, constitutional knowledge, and ability to navigate complex legal systems—essential for emergency housing contracts and reentry partnerships.
        </p>
      </div>
      
      <!-- For Community Members -->
      <div class="why-matters-card">
        <h3>For Community Members</h3>
        <p>
          Real-world proof that Faith Frontier operates with transparency, humility, and respect for due process—building trust through verifiable actions, not empty promises.
        </p>
      </div>
      
      <!-- For Legal Researchers -->
      <div class="why-matters-card">
        <h3>For Legal Researchers</h3>
        <p>
          Comprehensive docket entries, filings, and procedural history demonstrating constitutional issues, procedural challenges, and the evolution of legal strategy over time.
        </p>
      </div>
    </div>
  </section>
  
  <!-- ==================== FILTER CONTROLS ==================== -->
  <div class="case-filters" id="case-filters">
    <div class="filter-row">
      <div class="filter-group">
        <label for="filter-status">Status</label>
        <select id="filter-status" class="filter-select">
          <option value="">All Statuses</option>
          <option value="active">Active</option>
          <option value="pending">Pending</option>
          <option value="closed">Closed</option>
        </select>
      </div>
      
      <div class="filter-group">
        <label for="filter-court">Court</label>
        <select id="filter-court" class="filter-select">
          <option value="">All Courts</option>
          {% assign all_courts = site.cases | where_exp: "case", "case.published != false" | map: "court" | uniq %}
          {% for court in all_courts %}
            {% if court and court != "" %}
            <option value="{{ court }}">{{ court }}</option>
            {% endif %}
          {% endfor %}
        </select>
      </div>
      
      <div class="filter-group" style="flex: 2;">
        <label for="filter-search">Search</label>
        <input type="text" id="filter-search" class="filter-input" placeholder="Search by title, docket, or description...">
      </div>
    </div>
  </div>
  
  <!-- ==================== ACTIVE CASES ==================== -->
  <section id="active-cases" style="margin-bottom: 3rem;">
    <h2 class="section-heading">Active Cases</h2>
    <div class="cases-grid" id="active-cases-grid">
      {% assign active_cases = site.cases | where: "status", "active" | where_exp: "case", "case.published != false" %}
      {% if active_cases.size > 0 %}
        {% for case in active_cases %}
          {% include case-card.html case=case %}
        {% endfor %}
      {% else %}
        <div class="no-results">
          <div class="no-results-icon">⚖️</div>
          <h3 class="no-results-title">No Active Cases</h3>
          <p class="no-results-text">There are currently no active cases in the system.</p>
        </div>
      {% endif %}
    </div>
  </section>
  
  <!-- ==================== PENDING CASES ==================== -->
  {% assign pending_cases = site.cases | where: "status", "pending" | where_exp: "case", "case.published != false" %}
  {% if pending_cases.size > 0 %}
  <section style="margin-bottom: 3rem;">
    <h2 class="section-heading">Pending Decisions</h2>
    <div class="cases-grid">
      {% for case in pending_cases %}
        {% include case-card.html case=case %}
      {% endfor %}
    </div>
  </section>
  {% endif %}
  
  <!-- ==================== CLOSED CASES ==================== -->
  {% assign closed_cases = site.cases | where: "status", "closed" | where_exp: "case", "case.published != false" %}
  {% if closed_cases.size > 0 %}
  <section style="margin-bottom: 3rem;">
    <h2 class="section-heading">Closed Cases</h2>
    <div class="cases-grid">
      {% for case in closed_cases %}
        {% include case-card.html case=case %}
      {% endfor %}
    </div>
  </section>
  {% endif %}
  
  <!-- ==================== TRANSPARENCY STATEMENT ==================== -->
  <section class="transparency-section">
    <h2 class="transparency-title">Our Commitment to Transparency</h2>
    <p class="transparency-text">
      Every case listed here is backed by public court records. We publish filings, dockets, and procedural history to demonstrate:
    </p>
    <ul class="transparency-list">
      <li><strong>Constitutional literacy:</strong> Understanding of due process, rights, and legal procedure</li>
      <li><strong>Procedural competence:</strong> Ability to navigate court systems and administrative processes</li>
      <li><strong>Accountability:</strong> Nothing hidden, everything verifiable through official records</li>
      <li><strong>Educational value:</strong> Real-world examples of civic engagement and legal advocacy</li>
    </ul>
    <p class="transparency-text-final">
      This archive exists to build trust through transparency and to demonstrate that Faith Frontier operates with integrity, humility, and respect for lawful processes.
    </p>
  </section>

  <!-- ==================== OPRA RECORDS INTEGRATION ==================== -->
  <section class="opra-section">
    <h2 class="opra-title">
      📋 Supporting OPRA Administrative Records
    </h2>
    <p class="opra-intro">
      Many cases are supported by Open Public Records Act (OPRA) requests that document vendor relationships, 
      policy frameworks, and operational oversight. These administrative records complement litigation by establishing 
      factual foundations through official government documents.
    </p>
    
    {% assign opra_count = site.opra | size %}
    {% assign active_opra = site.opra | where: "status", "Active" | size %}
    {% assign pending_opra = site.opra | where: "status", "Awaiting Response" | size %}
    
    <div class="opra-stats">
      <div class="opra-stat-card">
        <div class="opra-stat-number">{{ opra_count }}</div>
        <div class="opra-stat-label">Total OPRA Records</div>
      </div>
      <div class="opra-stat-card">
        <div class="opra-stat-number">{{ active_opra }}</div>
        <div class="opra-stat-label">Active Requests</div>
      </div>
      <div class="opra-stat-card">
        <div class="opra-stat-number">{{ pending_opra }}</div>
        <div class="opra-stat-label">Pending Responses</div>
      </div>
    </div>
    
    <div class="opra-cta">
      <a href="/opra/" class="opra-btn">
        View All OPRA Records →
      </a>
    </div>
    
    <div class="opra-footer">
      <strong>What are OPRA records?</strong> OPRA (Open Public Records Act) allows citizens to request government documents. 
      Faith Frontier uses OPRA to gather factual evidence about vendor contracts, policy decisions, and operational oversight—
      creating transparent documentation that supports legal strategy and public accountability.
    </div>
  </section>
  
</div>

<!-- ==================== CLIENT-SIDE FILTERING ==================== -->
<script>
(function() {
  'use strict';
  
  const filterStatus = document.getElementById('filter-status');
  const filterCourt = document.getElementById('filter-court');
  const filterSearch = document.getElementById('filter-search');
  const casesGrids = document.querySelectorAll('.cases-grid');
  
  if (!filterStatus || !filterCourt || !filterSearch) return;
  
  function applyFilters() {
    const statusValue = filterStatus.value.toLowerCase();
    const courtValue = filterCourt.value.toLowerCase();
    const searchValue = filterSearch.value.toLowerCase();
    
    casesGrids.forEach(function(grid) {
      const cards = grid.querySelectorAll('.case-card');
      let visibleCount = 0;
      
      cards.forEach(function(card) {
        const cardStatus = (card.dataset.status || '').toLowerCase();
        const cardCourt = (card.dataset.court || '').toLowerCase();
        const cardText = card.textContent.toLowerCase();
        
        const matchesStatus = !statusValue || cardStatus === statusValue;
        const matchesCourt = !courtValue || cardCourt.includes(courtValue);
        const matchesSearch = !searchValue || cardText.includes(searchValue);
        
        if (matchesStatus && matchesCourt && matchesSearch) {
          card.style.display = '';
          visibleCount++;
        } else {
          card.style.display = 'none';
        }
      });
      
      // Show/hide no results message
      let noResults = grid.querySelector('.no-results');
      if (visibleCount === 0 && !noResults) {
        noResults = document.createElement('div');
        noResults.className = 'no-results';
        
        const icon = document.createElement('div');
        icon.className = 'no-results-icon';
        icon.textContent = '🔍';
        
        const title = document.createElement('h3');
        title.className = 'no-results-title';
        title.textContent = 'No Cases Found';
        
        const text = document.createElement('p');
        text.className = 'no-results-text';
        text.textContent = 'Try adjusting your filters or search terms.';
        
        noResults.appendChild(icon);
        noResults.appendChild(title);
        noResults.appendChild(text);
        
        grid.appendChild(noResults);
      } else if (visibleCount > 0 && noResults) {
        noResults.remove();
      }
    });
  }
  
  filterStatus.addEventListener('change', applyFilters);
  filterCourt.addEventListener('change', applyFilters);
  filterSearch.addEventListener('input', applyFilters);
})();
</script>
