---
title: "Ecclesiastical Trust Inner Room"
permalink: /sanctuary/trust/
hide_hero: true
---

<section class="section-content trust-hero">
  <p class="trust-eyebrow">Sanctuary Inner Room</p>
  <h1>Ecclesiastical Trust Inner Room</h1>
  <p class="lead">
    A quiet archive space within the Sanctuary where Ecclesiastical Trust instruments, banking resolutions, and governance minutes are preserved for prayerful reference and public traceability.
  </p>
</section>

<section class="section-content">
  <div class="trust-note trust-note--muted">
    <h2>Ecclesiastical &amp; Spiritual Context</h2>
    <p>
    These records are maintained as part of Faith Frontier’s ecclesiastical and spiritual stewardship. They are not legal advice or court filings. Court dockets and documents filed under penalty of perjury (such as sworn statements and certifications) remain under the jurisdiction of the appropriate court and are referenced here in good faith to honor transparency and integrity. Nothing authored on this site constitutes sworn testimony unless explicitly identified as a court document. For clarity or pastoral context, email <a href="mailto:info@Tillerstead.com">info@Tillerstead.com</a>.
  </p>
</div>
</section>

<section class="section-content trust-room">
  <h2>Ecclesiastical Trust Records</h2>
  <p class="inner-room-subtitle">
    Inner-room documents related to the Faith Frontier Ecclesiastical Trust: formation instruments, banking resolutions, governance minutes, and related ecclesiastical records. Each item remains paired with its civic counterpart so auditors, regulators, and community partners can reconcile what is held spiritually with what is filed legally.
  </p>

  <div class="trust-doc-grid">
    {% assign trust_docs = site.trust | sort: "date" | reverse %}
    {% for doc in trust_docs %}
      <article class="trust-doc-card">
        <h3><a href="{{ doc.url | relative_url }}">{{ doc.title }}</a></h3>
        <p class="trust-doc-meta">
          {% if doc.date %}{{ doc.date | date: "%B %d, %Y" }}{% endif %}
          {% if doc.doc_type %} · {{ doc.doc_type | replace: '_', ' ' | capitalize }}{% endif %}
          {% if doc.status %} · {{ doc.status | replace: '_', ' ' | capitalize }}{% endif %}
        </p>
        {% if doc.excerpt %}
          <p>{{ doc.excerpt }}</p>
        {% endif %}
        <p class="trust-doc-links">
          <a href="{{ doc.url | relative_url }}" class="btn btn-ghost">Read document</a>
          {% if doc.pdf_path %}
            <a href="{{ doc.pdf_path | relative_url }}" class="trust-download-inline">Signed PDF</a>
          {% endif %}
        </p>
      </article>
    {% endfor %}
  </div>
</section>
