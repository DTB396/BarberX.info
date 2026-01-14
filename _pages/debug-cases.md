---
layout: default
title: "Debug Cases Collection"
permalink: /debug-cases/
---

# Debug: site.cases

<ul>
{% for c in site.cases %}
  <li>{{ c.title }} ({{ c.path }})</li>
{% else %}
  <li>No cases found in site.cases.</li>
{% endfor %}
</ul>
