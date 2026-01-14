---
layout: default
title: Docs
permalink: /docs/
---

# Docs

<ul>
  {% assign sorted_docs = site.docs | sort: 'title' %}
  {% for doc in sorted_docs %}
    <li><a href="{{ doc.url | relative_url }}">{{ doc.title | default: doc.slug | default: doc.name }}</a></li>
  {% endfor %}
</ul>
