---
layout: listing
title: Essays & Reflections
permalink: /essays/
body_class: page-essays
description: "Reflections on faith, covenant, sovereignty, and the tension between personal identity and global systems — memoir fused with spiritual and civic manifesto."
---

<section aria-labelledby="essays-heading" class="essays-index shell">
  <header class="page-header">
    <h1 id="essays-heading">Essays & Reflections</h1>
    <p class="essays-lead">
      Reflections on faith, covenant, sovereignty, and the tension between personal identity and 
      global systems. These essays weave memoir with spiritual and civic manifesto — honest, 
      reflective, grounded, and searching.
    </p>
  </header>

  <div class="essays-list" style="max-width: 60rem; margin: 0 auto;">
    {% assign sorted_essays = site.essays | sort: "date" | reverse %}
    {% for essay in sorted_essays %}
      <article class="essay-card">
        <h2 class="essay-card-title">
          <a href="{{ essay.url | relative_url }}">
            {{ essay.title }}
          </a>
        </h2>
        {% if essay.date %}
        <p class="essay-card-date">{{ essay.date | date: "%B %-d, %Y" }}</p>
        {% endif %}
        {% if essay.description %}
        <p class="essay-card-excerpt">{{ essay.description }}</p>
        {% endif %}
      </article>
    {% endfor %}
  </div>
</section>
