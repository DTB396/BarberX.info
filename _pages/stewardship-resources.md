---
layout: default
title: Stewardship Resources
permalink: /stewardship-resources/
description: Products and resources that have been helpful in our journey of faithful stewardship and neighbor-care. Your support helps sustain our mission.
stylesheet: /assets/css/stewardship-resources.css
---

{% assign content = site.data.stewardship-content %}
{% assign products = site.data.stewardship-resources %}

<article class="stewardship-resources">

  {%- comment -%} HERO SECTION {%- endcomment -%}
  <header class="resources-hero">
    <h1>{{ content.hero.title }}</h1>
    <p class="hero-tagline">{{ content.hero.tagline }}</p>
  </header>

  <section class="resources-intro container">

    {%- comment -%} INTRODUCTION {%- endcomment -%}
    <h2>{{ content.intro.heading }}</h2>
    {{ content.intro.body | markdownify }}

    {%- comment -%} DISCLAIMER BOX {%- endcomment -%}
    <div class="disclaimer-box" role="note">
      <h4>{{ content.disclaimer.title }}</h4>
      <p>{{ content.disclaimer.intro }}</p>
      
      <ul class="disclaimer-list">
        {% for point in content.disclaimer.points %}
        <li>
          <strong>{{ point.heading }}:</strong> {{ point.text }}
        </li>
        {% endfor %}
      </ul>
      
      <p>{{ content.disclaimer.closing }}</p>
    </div>

  </section>

  {%- comment -%} PRODUCT FILTERS & GRID {%- endcomment -%}
  <section class="resources-products container">
    
    <div class="filter-controls">
      {% for category in content.categories %}
      <button class="filter-btn {% if category.active %}active{% endif %}" 
              data-category="{{ category.id }}">
        {{ category.label }}
      </button>
      {% endfor %}
    </div>

    <div class="resources-grid" id="resourcesGrid">
      {%- comment -%} Products loaded dynamically via JavaScript {%- endcomment -%}
      <div class="resources-loading">
        <p>Loading products...</p>
      </div>
    </div>

  </section>

  {%- comment -%} MISSION SUPPORT {%- endcomment -%}
  <section class="mission-support container">
    <h3>{{ content.mission_support.title }}</h3>
    <p>{{ content.mission_support.intro }}</p>
    <ul>
      {% for area in content.mission_support.areas %}
      <li>{{ area | markdownify | remove: '<p>' | remove: '</p>' }}</li>
      {% endfor %}
    </ul>
    <p>{{ content.mission_support.statement }}</p>
  </section>

  {%- comment -%} QUESTIONS SECTION {%- endcomment -%}
  <section class="resources-questions container">
    <h3>{{ content.questions.title }}</h3>
    <ul>
      {% for item in content.questions.items %}
      <li>
        <strong>{{ item.heading }}:</strong> {{ item.text | markdownify | remove: '<p>' | remove: '</p>' }}
      </li>
      {% endfor %}
    </ul>
  </section>

  {%- comment -%} FOOTER DISCLOSURES {%- endcomment -%}
  <footer class="resources-footer container">
    <hr class="footer-divider" />
    
    {{ content.footer.amazon_disclosure | markdownify }}
    {{ content.footer.legal_status | markdownify }}
    <p><small><strong>Last Updated:</strong> {{ content.footer.last_updated }}</small></p>
  </footer>

</article>

{%- comment -%} Embed product data for JavaScript {%- endcomment -%}
<script>
window.stewardshipProducts = {{ site.data.stewardship-resources | jsonify }};
</script>

<script src="{{ '/assets/js/stewardship-resources.js' | relative_url }}" defer></script>
