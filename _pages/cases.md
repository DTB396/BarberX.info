---
layout: default
title: "Case Records (Legacy)"
permalink: /cases-legacy/
description: "Documentation of lived experience navigating institutions — a public archive of the intersection between law, conscience, and personal truth."
published: false
---

# Case Records

Documentation of lived experience navigating institutions — a public archive of the intersection between law, conscience, and personal truth. Born from a journey of childhood duality and spiritual awakening, these records bear witness to the procedural gaps that ordinary New Jersey residents face, turning hard lessons into pathways toward clarity, due process, and restored trust.

Everything is always changing, and we do our best to stay on top of it.

Need a rapid snapshot? Visit the [Active Case Digest](/cases/active/) for the latest filings and prayer priorities.

<div class="case-index-filters" style="margin: 2rem 0;">
  <input type="search" id="case-search" placeholder="Search cases by title, docket, or court..." style="width: 100%; max-width: 500px; padding: 0.75rem; font-size: 1rem; border-radius: 8px; border: 2px solid var(--accent-brass-muted); background: var(--color-surface-alt); color: var(--color-text);">
</div>

<div class="case-list" style="display: grid; gap: 1.5rem; margin-top: 2rem;">
{% assign cases = site.cases %}
{% for case in cases %}
  <article class="case-card" style="background: var(--color-surface-alt); padding: 1.5rem; border-radius: 12px; border: 2px solid rgba(212, 165, 116, 0.2); transition: border-color 0.2s;">
    <h2 style="margin-top: 0; font-size: 1.5rem;">
      <a href="{{ case.url | relative_url }}" style="color: var(--color-highlight); text-decoration: none;">
        {{ case.short_title | default: case.title }}
      </a>
    </h2>
    
    <div class="case-meta" style="display: grid; gap: 0.5rem; margin: 1rem 0; font-size: 0.95rem; color: var(--color-muted);">
      <div><strong>Court:</strong> {{ case.court }}</div>
      <div><strong>Docket:</strong> {{ case.primary_docket | default: case.docket | default: case.dockets | join: ", " }}</div>
      <div><strong>Status:</strong> <span style="color: {% if case.status == 'active' %}var(--accent-emerald-light){% elsif case.status == 'pending' %}var(--accent-brass){% else %}var(--color-muted){% endif %};">{{ case.status | capitalize }}</span></div>
      {% if case.filed_date %}
      <div><strong>Filed:</strong> {{ case.filed_date | date: "%B %-d, %Y" }}</div>
      {% endif %}
    </div>
    
    {% if case.overview %}
    <p style="margin: 1rem 0; line-height: 1.6;">{{ case.overview }}</p>
    {% endif %}
    
    <a href="{{ case.url | relative_url }}" style="display: inline-block; margin-top: 1rem; padding: 0.5rem 1.25rem; background: var(--accent-brass); color: #0f172a; text-decoration: none; border-radius: 6px; font-weight: 600; transition: background 0.2s;">
      View Case Details →
    </a>
  </article>
{% endfor %}
</div>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const searchInput = document.getElementById('case-search');
  const caseCards = document.querySelectorAll('.case-card');
  
  if (searchInput) {
    searchInput.addEventListener('input', function(e) {
      const searchTerm = e.target.value.toLowerCase();
      
      caseCards.forEach(card => {
        const text = card.textContent.toLowerCase();
        if (text.includes(searchTerm)) {
          card.style.display = 'block';
        } else {
          card.style.display = 'none';
        }
      });
    });
  }
});
</script>
