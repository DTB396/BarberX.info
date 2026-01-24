---
layout: default
title: "FAQ"
permalink: /faq/
---

{% capture faq_content %}
{% include_relative ../faq.md %}
{% endcapture %}

{{ faq_content | markdownify }}
