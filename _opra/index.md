---
layout: default
title: "OPRA Administrative Records"
permalink: /opra/
description: "Open Public Records Act (OPRA) administrative records preserved for transparency, documentation, and public accountability."
stylesheet: /assets/css/pages/opra.css
hero_panel: false
hide_hero: true
show_breadcrumbs: true
---

<!-- OPRA Hero Section with Live Stats -->
<div class="opra-hero">
  <h1>📋 OPRA Administrative Records</h1>
  <p style="max-width: 700px; margin: 0 auto 1.5rem; opacity: 0.9; font-size: 1.1rem;">
    Open Public Records Act requests documenting vendor relationships, policy frameworks,
    and operational oversight across New Jersey government agencies.
  </p>

{% assign opra_all = site.opra | where_exp: "item", "item.slug != 'template'" %}
{% assign opra_active = site.opra | where: "status", "Active" | where_exp: "item", "item.slug != 'template'" %}
{% assign opra_awaiting = site.opra | where: "status", "Awaiting Response" | where_exp: "item", "item.slug != 'template'" %}
{% assign opra_review = site.opra | where: "status", "Internal Review Pending" | where_exp: "item", "item.slug != 'template'" %}

  <div class="opra-hero-stats">
    <div class="opra-hero-stat">
      <span class="opra-hero-stat-number">{{ opra_all.size }}</span>
      <span class="opra-hero-stat-label">Total Records</span>
    </div>
    <div class="opra-hero-stat">
      <span class="opra-hero-stat-number">{{ opra_active.size }}</span>
      <span class="opra-hero-stat-label">Active</span>
    </div>
    <div class="opra-hero-stat">
      <span class="opra-hero-stat-number">{{ opra_awaiting.size }}</span>
      <span class="opra-hero-stat-label">Awaiting Response</span>
    </div>
    <div class="opra-hero-stat">
      <span class="opra-hero-stat-number">{{ opra_review.size }}</span>
      <span class="opra-hero-stat-label">Under Review</span>
    </div>
  </div>
</div>

<div class="container" style="max-width: 1000px; margin: 0 auto; padding: 0 1.5rem;">

<!-- Search Bar -->
<section class="search-filter-section" style="margin-bottom: 2rem; margin-top: 2rem;">
  <div class="search-container">
    <input type="text" class="search-bar" placeholder="Search OPRA records by agency, topic, or request..." aria-label="Search OPRA records">
    <span class="search-icon">🔍</span>
    <button class="search-clear" aria-label="Clear search">✕</button>
  </div>
</section>

<section class="opra-intro-section" style="margin-bottom: 2rem;">
  <p style="font-size: 1.1rem; line-height: 1.7; color: var(--ff-text, #e5e7eb);">
    This page serves as a centralized index of <strong style="color: #d4a574;">Open Public Records Act (OPRA)</strong> administrative records maintained for transparency, documentation, and public accountability.
  </p>
  <p style="font-size: 1rem; line-height: 1.7; color: var(--ff-text-muted, #9ca3af); margin-top: 1rem;">
    OPRA records listed here may relate to pending or concluded litigation, but they are preserved and organized as independent administrative threads, separate from pleadings, motions, or legal argument. This structure ensures clarity, auditability, and proper recordkeeping.
  </p>
</section>

<hr style="border: none; border-top: 1px solid rgb(255 255 255 / 10%); margin: 2rem 0;">

<section style="margin-bottom: 2rem;">
  <h2 style="color: #d4a574; font-size: 1.5rem; margin-bottom: 1rem;">Purpose and Scope</h2>
  <p style="margin-bottom: 1rem; color: var(--ff-text-muted, #9ca3af);">These records are maintained to:</p>
  <ul style="list-style: none; padding: 0; margin: 0 0 1rem;">
    <li style="padding: 0.5rem 0; padding-left: 1.5rem; position: relative; color: var(--ff-text, #e5e7eb);">
      <span style="position: absolute; left: 0; color: #10b981;">✓</span>
      Document public-records requests and agency responses
    </li>
    <li style="padding: 0.5rem 0; padding-left: 1.5rem; position: relative; color: var(--ff-text, #e5e7eb);">
      <span style="position: absolute; left: 0; color: #10b981;">✓</span>
      Preserve timelines, correspondence, and productions
    </li>
    <li style="padding: 0.5rem 0; padding-left: 1.5rem; position: relative; color: var(--ff-text, #e5e7eb);">
      <span style="position: absolute; left: 0; color: #10b981;">✓</span>
      Support orderly review by courts, counsel, custodians, and the public
    </li>
    <li style="padding: 0.5rem 0; padding-left: 1.5rem; position: relative; color: var(--ff-text, #e5e7eb);">
      <span style="position: absolute; left: 0; color: #10b981;">✓</span>
      Maintain clear separation between records administration and legal advocacy
    </li>
  </ul>
  <p style="font-size: 0.9rem; color: var(--ff-text-muted, #6b7280); font-style: italic;">
    Nothing on this page asserts findings of fact, legal conclusions, or allegations. All materials are indexed as records only.
  </p>
</section>

<hr style="border: none; border-top: 1px solid rgb(255 255 255 / 10%); margin: 2rem 0;">

<section style="margin-bottom: 2rem;">
  <h2 style="color: #d4a574; font-size: 1.5rem; margin-bottom: 1rem;">Live Public Tracking (OPRAMachine)</h2>
  <p style="margin-bottom: 1rem; color: var(--ff-text, #e5e7eb);">
    All OPRA requests indexed here are publicly tracked and viewable on <strong style="color: #34d399;">OPRAMachine</strong>, New Jersey's public OPRA transparency platform.
  </p>
  
  <div style="background: rgb(16 185 129 / 10%); border: 1px solid rgb(16 185 129 / 30%); border-radius: 12px; padding: 1.25rem; margin: 1rem 0;">
    <p style="margin: 0 0 0.75rem; font-weight: 600; color: #34d399;">View all requests by filing account:</p>
    <ul style="list-style: none; padding: 0; margin: 0;">
      <li style="margin-bottom: 0.5rem;">
        <a href="https://opramachine.com/user/x_devon_tyler_of_the_barber_mate" target="_blank" rel="noopener" style="color: #10b981; text-decoration: none;">
          X (Devon Tyler of the Barber-Materio Family) ↗
        </a>
      </li>
      <li>
        <a href="https://opramachine.com/user/mr_barber" target="_blank" rel="noopener" style="color: #10b981; text-decoration: none;">
          Mr. Barber ↗
        </a>
      </li>
    </ul>
    <p style="font-size: 0.85rem; color: var(--ff-text-muted, #6b7280); margin: 0.75rem 0 0;">
      All OPRA requests filed under these accounts are publicly accessible on OPRAMachine.
    </p>
  </div>
  
  <h3 style="color: #e5e7eb; font-size: 1.125rem; margin: 1.5rem 0 0.75rem;">What is OPRAMachine?</h3>
  <p style="color: var(--ff-text-muted, #9ca3af);">
    OPRAMachine is a public platform that allows New Jersey residents to file, track, and browse OPRA requests. Requests and government responses published there are publicly accessible to promote transparency and accountability.
  </p>
</section>

<hr style="border: none; border-top: 1px solid rgb(255 255 255 / 10%); margin: 2rem 0;">

<section style="margin-bottom: 2rem;">
  <h2 style="color: #d4a574; font-size: 1.5rem; margin-bottom: 1rem;">How to Use This Page</h2>
  <p style="margin-bottom: 1rem; color: var(--ff-text, #e5e7eb);">Each OPRA record below links to:</p>
  <ul style="list-style: none; padding: 0; margin: 0;">
    <li style="padding: 0.5rem 0; padding-left: 1.5rem; position: relative; color: var(--ff-text, #e5e7eb);">
      <span style="position: absolute; left: 0; color: #d4a574;">→</span>
      A dedicated record index page on this site (documentation only)
    </li>
    <li style="padding: 0.5rem 0; padding-left: 1.5rem; position: relative; color: var(--ff-text, #e5e7eb);">
      <span style="position: absolute; left: 0; color: #d4a574;">→</span>
      When available, the corresponding live OPRAMachine thread
    </li>
  </ul>
  <p style="margin-top: 1rem; color: var(--ff-text-muted, #9ca3af);">
    Records are grouped by current administrative status for ease of navigation by courts, custodians, and reviewers.
  </p>
</section>

<hr style="border: none; border-top: 1px solid rgb(255 255 255 / 10%); margin: 2rem 0;">

<section style="margin-bottom: 2rem;">
  <h2 style="color: #d4a574; font-size: 1.5rem; margin-bottom: 1rem;">Active</h2>
  <p style="color: var(--ff-text-muted, #9ca3af); margin-bottom: 1.5rem;">Requests currently open or producing records.</p>

{% include opra-list.html status="Active" %}

</section>

<hr style="border: none; border-top: 1px solid rgb(255 255 255 / 10%); margin: 2rem 0;">

<section style="margin-bottom: 2rem;">
  <h2 style="color: #d4a574; font-size: 1.5rem; margin-bottom: 1rem;">Awaiting Response</h2>
  <p style="color: var(--ff-text-muted, #9ca3af); margin-bottom: 1.5rem;">Requests properly filed and pending an initial response or statutory extension.</p>

{% include opra-list.html status="Awaiting Response" %}

</section>

<hr style="border: none; border-top: 1px solid rgb(255 255 255 / 10%); margin: 2rem 0;">

<section style="margin-bottom: 2rem;">
  <h2 style="color: #d4a574; font-size: 1.5rem; margin-bottom: 1rem;">Internal Review Pending</h2>
  <p style="color: var(--ff-text-muted, #9ca3af); margin-bottom: 1.5rem;">Requests subject to internal review, clarification, or custodian reconsideration.</p>

{% include opra-list.html status="Internal Review Pending" %}

</section>

<hr style="border: none; border-top: 1px solid rgb(255 255 255 / 10%); margin: 2rem 0;">

<section style="margin-bottom: 2rem;">
  <h2 style="color: #d4a574; font-size: 1.5rem; margin-bottom: 1rem;">Referred or No Records Maintained</h2>
  <p style="color: var(--ff-text-muted, #9ca3af); margin-bottom: 1.5rem;">Requests referred to another agency or responded to with a "no records maintained" determination.</p>

{% include opra-list.html status="Referred — No Records Maintained" %}

</section>

<hr style="border: none; border-top: 1px solid rgb(255 255 255 / 10%); margin: 2rem 0;">

<section style="margin-bottom: 2rem;">
  <h2 style="color: #d4a574; font-size: 1.5rem; margin-bottom: 1rem;">All OPRA Records</h2>
  <p style="color: var(--ff-text-muted, #9ca3af); margin-bottom: 1.5rem;">Complete index of all OPRA administrative records maintained on this site.</p>

{% include opra-list.html status="ALL" %}

</section>

<hr style="border: none; border-top: 1px solid rgb(255 255 255 / 10%); margin: 2rem 0;">

<section style="margin-bottom: 2rem;">
  <h2 style="color: #d4a574; font-size: 1.5rem; margin-bottom: 1rem;">Administrative Discipline</h2>
  <ul style="list-style: none; padding: 0; margin: 0;">
    <li style="padding: 0.5rem 0; padding-left: 1.5rem; position: relative; color: var(--ff-text, #e5e7eb);">
      <span style="position: absolute; left: 0; color: #10b981;">•</span>
      Each OPRA record is maintained in its own directory with a clear timeline and document structure.
    </li>
    <li style="padding: 0.5rem 0; padding-left: 1.5rem; position: relative; color: var(--ff-text, #e5e7eb);">
      <span style="position: absolute; left: 0; color: #10b981;">•</span>
      Interpretive notes, if any, are kept separate from record indexes.
    </li>
    <li style="padding: 0.5rem 0; padding-left: 1.5rem; position: relative; color: var(--ff-text, #e5e7eb);">
      <span style="position: absolute; left: 0; color: #10b981;">•</span>
      Litigation filings and arguments are maintained elsewhere and are not mixed into OPRA administration.
    </li>
  </ul>
  <p style="margin-top: 1.5rem; padding: 1rem; background: rgb(212 165 116 / 10%); border-left: 3px solid #d4a574; color: var(--ff-text, #e5e7eb); font-style: italic;">
    This page exists solely to support lawful transparency, procedural clarity, and durable public recordkeeping.
  </p>
</section>

</div><!-- End container -->
