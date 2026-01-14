---
layout: default
title: "Active Case Digest"
permalink: /cases/active/
description: "Live litigation tracker for the Faith Frontier Ecclesiastical Trust."
hero_panel: false
hide_hero: true
show_breadcrumbs: true
---

<section class="section-block cases-active-intro">
  <div class="container">
    <p class="section-eyebrow">Live litigation</p>
    <h1 class="section-heading">Active Case Digest</h1>
    <p class="section-lead">
      A public index of active matters and records for quick navigation. Built for transparency and orderly
      recordkeeping.
    </p>

    <div class="digest-jumplinks">
      <a class="btn btn-ghost btn--sm" href="#capacity-notice">Capacity notice</a>
      <a class="btn btn-ghost btn--sm" href="#active-cases">Active cases</a>
      <a class="btn btn-ghost btn--sm" href="#opra">OPRA / public records</a>
      <a class="btn btn-ghost btn--sm" href="#counsel-screening">Counsel screening summary</a>
    </div>
  </div>
</section>

<section id="capacity-notice" class="section-block capacity-notice-acd-page">
  <div class="container">
    <div class="capacity-notice-arr card">
      <p class="section-eyebrow">Capacity Notice</p>
      <h2 class="section-heading h3">In God (Yeshua Christos) We TrVst</h2>
      <p class="section-lead">
        The matters referenced on this page are pursued by <strong>Devon T. Barber</strong> in his
        <strong>individual capacity</strong>, <strong>pro se</strong>. <strong>Faith Frontier Ecclesiastical Trust is not a
        party</strong> and does not provide legal representation. This digest is for transparency, record navigation, and
        public accountability only.
      </p>

      <div class="digest-mini-grid">
        <div class="digest-mini">
          <p class="section-eyebrow">For court staff</p>
          <p class="small">
            Use this page to locate the case record page, docket timeline, and linked exhibits.
          </p>
        </div>
        <div class="digest-mini">
          <p class="section-eyebrow">For counsel / reviewers</p>
          <p class="small">
            The “Counsel screening summary” below is descriptive and non-argument.
          </p>
        </div>
        <div class="digest-mini">
          <p class="section-eyebrow">For any party</p>
          <p class="small">
            Nothing here is legal advice, a finding of fact, or a substitute for filed pleadings.
          </p>
        </div>
      </div>
    </div>
  </div>
</section>

<section id="active-cases" class="section-block cases-active-list">
  <div class="container">
    <div class="section-head">
      <p class="section-eyebrow">Current matters</p>
      <h2 class="section-heading h3">Active case index</h2>
      <p class="section-lead">
        Click a case to open the record page (caption, docket, filings, exhibits, and timelines).
      </p>
    </div>

    {% assign active_cases = site.cases | where: "status", "active" %}
    {% if active_cases.size == 0 %}
      <p>No cases are flagged as active right now. Check back soon.</p>
    {% else %}
      <div class="active-cases-grid">
        {% for case in active_cases %}
          <article class="active-case-card">
            <header>
              <p class="case-docket">
                {{ case.primary_docket | default: case.short_title | escape }}
              </p>
              <h3 class="h4">
                <a href="{{ case.url | relative_url }}">{{ case.title | escape }}</a>
              </h3>
              <p class="case-meta">
                {{ case.court | escape }}{% if case.status %} • {{ case.status | capitalize | escape }}{% endif %}
                {% if case.venue %} • {{ case.venue | escape }}{% endif %}
              </p>
            </header>

            {% if case.overview %}
              <p class="case-overview">{{ case.overview | strip_html | truncatewords: 30 }}</p>
            {% endif %}

            <div class="case-links">
              <a class="btn btn-ghost btn--sm" href="{{ case.url | relative_url }}">Open record →</a>

              {% if case.opra_anchor %}
                <a class="btn btn-ghost btn--sm" href="{{ case.url | relative_url }}#opra">OPRA →</a>
              {% endif %}
            </div>
          </article>
        {% endfor %}
      </div>
    {% endif %}
  </div>
</section>

{%- comment -%}
  OPRA DATA SOURCES (your repo):

- _data/opra.yml               (records)
- _data/opra_case_map.yml      (maps OPRA records to cases)

  Assumptions (safe defaults):

- Each OPRA record has a stable key: `id` (preferred) or `slug`.
- Case pages have `slug` in front matter.
- opra_case_map.yml maps case_slug -> list of opra ids (or slugs).
{%- endcomment -%}

<section id="opra" class="section-block cases-active-opra">
  <div class="container">
    <div class="case-digest-opra card">
      <p class="section-eyebrow">OPRA / public records</p>
      <h2 class="section-heading h3">Public records requests</h2>
      <p class="section-lead">
        Consolidated public-records requests and responses. When a request is tracked on OPRAMachine, this page links to
        the canonical public thread.
      </p>

      {% assign opra_items = site.data.opra %}
      {% assign opra_case_map = site.data.opra_case_map %}

      {% if opra_items and opra_items.size > 0 %}
        <div class="opra-toolbar">
          <p class="small">
            Portal:
            <a href="https://opramachine.com/" target="_blank" rel="noopener">opramachine.com</a>
          </p>
          <p class="small">
            Tip: Each OPRA record should include <code>opramachine_url</code> for direct linking.
          </p>
        </div>

        {% assign opra_sorted = opra_items | sort: "date_submitted" | reverse %}

        <div class="opra-grid">
          {% for r in opra_sorted %}
            <article class="opra-card">
              <header>
                <p class="opra-status">
                  {% if r.status %}
                    {{ r.status | upcase | escape }}
                  {% else %}
                    OPEN
                  {% endif %}
                </p>

                <h3 class="h5 opra-title">
                  {% if r.opramachine_url %}
                    <a href="{{ r.opramachine_url | escape }}" target="_blank" rel="noopener">
                      {{ r.title | default: "OPRA Request" | escape }} ↗
                    </a>
                  {% else %}
                    {{ r.title | default: "OPRA Request" | escape }}
                  {% endif %}
                </h3>

                <p class="opra-meta small">
                  {% if r.agency %}
                    <span>{{ r.agency | escape }}</span>
                  {% endif %}
                  {% if r.county %}
                    <span> • {{ r.county | escape }}</span>
                  {% endif %}
                  {% if r.date_submitted %}
                    <span> • {{ r.date_submitted | date: "%b %-d, %Y" }}</span>
                  {% endif %}
                </p>
              </header>

              {% if r.notes %}
                <p class="small">{{ r.notes | strip_html | truncatewords: 28 }}</p>
              {% endif %}

              <div class="opra-actions">
                {% if r.opramachine_url %}
                  <a
                    class="btn btn-primary btn--sm"
                    href="{{ r.opramachine_url | escape }}"
                    target="_blank"
                    rel="noopener"
                  >
                    Open on OPRAMachine ↗
                  </a>
                {% endif %}

                {%- comment -%}
                  Optional: show mapped related cases for this OPRA record using opra_case_map.
                  Supports map formats:
                  A) opra_case_map:
                       <case_slug>:
                         - <opra_id>
                         - <opra_id>
                  B) opra_case_map:
                       <opra_id>:
                         - <case_slug>
                         - <case_slug>
                  We detect B by checking whether opra_case_map[r.id] exists.
                {%- endcomment -%}

                {% assign rid = r.id | default: r.slug %}
                {% assign mapped_cases = nil %}

                {% if rid and opra_case_map and opra_case_map[rid] %}
                  {% assign mapped_cases = opra_case_map[rid] %}
                {% endif %}

                {% if mapped_cases %}
                  <div class="opra-related">
                    <p class="small"><strong>Related cases:</strong></p>
                    <ul class="small">
                      {% for case_slug in mapped_cases %}
                        {% assign rc = site.cases | where: "slug", case_slug | first %}
                        {% if rc %}
                          <li>
                            <a href="{{ rc.url | relative_url }}">
                              {{ rc.short_title | default: rc.title | escape }}
                            </a>
                          </li>
                        {% else %}
                          <li>{{ case_slug | escape }}</li>
                        {% endif %}
                      {% endfor %}
                    </ul>
                  </div>
                {% endif %}
              </div>
            </article>
          {% endfor %}
        </div>

        <hr />

        <div class="card">
          <p class="section-eyebrow">OPRA by active case</p>
          <h3 class="section-heading h4">Mapped requests (active matters)</h3>
          <p class="small">
            This view uses <code>_data/opra_case_map.yml</code> to list OPRA items per active case for fast review.
          </p>

          {% if opra_case_map %}
            <div class="digest-grid">
              {% for case in active_cases %}
                <div class="digest-block">
                  <h4 class="h5">
                    <a href="{{ case.url | relative_url }}">{{ case.short_title | default: case.title | escape }}</a>
                  </h4>

                  {% assign case_slug = case.slug %}
                  {% assign case_opra_ids = opra_case_map[case_slug] %}

                  {% if case_opra_ids and case_opra_ids.size > 0 %}
                    <ul class="digest-list">
                      {% for oid in case_opra_ids %}
                        {% assign record = nil %}

                        {%- comment -%}
                          Find OPRA record by matching:
                          - id == oid OR slug == oid
                        {%- endcomment -%}
                        {% for rr in opra_items %}
                          {% assign rrid = rr.id | default: rr.slug %}
                          {% if rrid == oid %}
                            {% assign record = rr %}
                            {% break %}
                          {% endif %}
                        {% endfor %}

                        {% if record %}
                          <li>
                            {% if record.opramachine_url %}
                              <a href="{{ record.opramachine_url | escape }}" target="_blank" rel="noopener">
                                {{ record.title | default: "OPRA Request" | escape }} ↗
                              </a>
                            {% else %}
                              {{ record.title | default: "OPRA Request" | escape }}
                            {% endif %}
                            {% if record.status %}
                              <span class="small"> ({{ record.status | escape }})</span>
                            {% endif %}
                          </li>
                        {% else %}
                          <li>{{ oid | escape }}</li>
                        {% endif %}
                      {% endfor %}
                    </ul>
                  {% else %}
                    <p class="small">No mapped OPRA items for this case yet.</p>
                  {% endif %}
                </div>
              {% endfor %}
            </div>
          {% else %}
            <p class="small">
              No OPRA case map found. Add <code>_data/opra_case_map.yml</code> to enable case-level linking.
            </p>
          {% endif %}
        </div>
      {% else %}
        <p class="small">
          No OPRA records found in <code>_data/opra.yml</code>.
        </p>
      {% endif %}
    </div>
  </div>
</section>

<section id="counsel-screening" class="section-block cases-active-summary">
  <div class="container">
    <div class="case-digest-summary card">
      <p class="section-eyebrow">Counsel screening summary</p>
      <h2 class="section-heading h3">Consolidated Case Analysis (2022–2025)</h2>
      <p class="section-lead">
        A condensed, non-argument summary of interconnected matters for contingency-fee review. Provided for navigation
        and independent assessment—not as a finding of fault.
      </p>

      <div class="digest-grid">
        <div class="digest-block">
          <h3 class="h5">Executive pattern</h3>
          <p>
            Across three matters (2022, 2024, 2025), civil or low-level encounters escalated into criminal or high-risk
            enforcement actions following stigmatizing misclassification and procedural breakdowns. The recurring theme
            is escalation that appears to rely on unsupported inferences rather than contemporaneous objective conduct.
          </p>
        </div>

        <div class="digest-block">
          <h3 class="h5">2022: Wage dispute → criminalization</h3>
          <ul class="digest-list">
            <li>
              Civil wage dispute; alleged disposal of property and self-help eviction without lawful process.
            </li>
            <li>
              Complainants later asserted fear; dispute escalated into criminal posture.
            </li>
            <li>
              Detention extension granted; plaintiff reports muted participation and missing wage/tax evidence at critical
              stages.
            </li>
            <li>
              Representation transition: public defender shortcomings → retained counsel (~$5,000) based on “Certified
              Supreme Court Trial Attorney” credentials; expectation of litigation and evidentiary development.
            </li>
          </ul>
        </div>

        <div class="digest-block">
          <h3 class="h5">2024: Pedestrian stop escalation (Boyd)</h3>
          <ul class="digest-list">
            <li>
              Jaywalking alleged; no citation stated; ID demand followed by escalation into warrant checks/enforcement.
            </li>
            <li>
              Plaintiff proceeded pro se; matter escalated into Superior Court; currently on appeal.
            </li>
          </ul>
        </div>

        <div class="digest-block">
          <h3 class="h5">2025: Traffic stop + CAD labeling (Ruiz)</h3>
          <ul class="digest-list">
            <li>
              Stop escalated with reliance on CAD “sovereign citizen” labeling rather than observed conduct.
            </li>
            <li>
              Plaintiff contemporaneously engaged in state assistance/reintegration programs (e.g., GA/SNAP/EA/WorkFirst
              NJ), indicating ordinary civic compliance.
            </li>
          </ul>
        </div>

        <div class="digest-block">
          <h3 class="h5">Damages indicators</h3>
          <ul class="digest-list">
            <li>Liberty impacts: detention, escalations, court involvement.</li>
            <li>Property impacts: alleged 2022 disposal/eviction-related losses.</li>
            <li>Financial impacts: retained counsel fees; ongoing legal burden.</li>
            <li>Reputational impacts and chilling effect on lawful assertion of rights.</li>
          </ul>
        </div>

        <div class="digest-block">
          <h3 class="h5">Neutral framing for review</h3>
          <p>
            This is presented as a pattern-of-misclassification and escalation matter. It is not a request for the reader to
            adopt conclusions, but an organized pointer to records and issues for independent assessment by counsel.
          </p>
        </div>
      </div>

      <div class="digest-actions">
        <a class="btn btn-primary" href="{{ '/cases/' | relative_url }}">Browse all case records →</a>
        <a class="btn btn-ghost" href="{{ '/contact/' | relative_url }}">Contact / record access →</a>
      </div>
    </div>
  </div>
</section>
