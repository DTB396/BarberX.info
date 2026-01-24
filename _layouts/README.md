# Layout Documentation

This document explains all available layout options and front matter variables for pages on this site.

## 📄 Available Layouts

### 1. `default.html`

Base layout for all pages. Provides the HTML structure, header, footer, and navigation.

**Front Matter Options:**

```yaml
layout: default
title: "Page Title"
body_class: "custom-body-class" # Add custom CSS class to <body>
data_theme: "dark" # Set data-theme attribute
main_class: "custom-main-class" # Add custom CSS class to <main>
main_role: "document" # Set ARIA role for main content
hide_header: true # Hide site header
hide_footer: true # Hide site footer
hide_barber_pole: true # Hide decorative barber pole
use_barber_spinner: true # Show barber pole spinner
barber_pole_position: "bottom-right" # Spinner position
barber_pole_size: "medium" # Spinner size
no_main_js: true # Don't load main.js
extra_css: # Additional CSS files
  - /assets/css/custom.css
extra_js: # Additional JS files
  - /assets/js/custom.js
custom_head: "<meta...>" # Raw HTML for <head>
custom_js: "console.log('test');" # Inline JavaScript
track_analytics: false # Disable analytics
maintenance_banner: true # Show maintenance notice
```

---

### 2. `page.html`

Enhanced page layout with rich metadata, TOC, CTAs, and related content.

**Front Matter Options:**

```yaml
layout: page

# Header Configuration
title: "Page Title"
subtitle: "Optional subtitle"
description: "Page description shown below title"
category: "Category Name" # Or use 'type'
type: "Documentation"

# Metadata
author: "Author Name"
date: 2026-01-19
last_updated: 2026-01-19
reading_time: "5 min" # Auto-calculated if omitted

# Alert Banner
alert: "Important message here"
alert_title: "Attention"
alert_type: "warning" # info, warning, error, success

# Navigation
show_breadcrumbs: true # Show breadcrumb trail
toc: true # Enable table of contents
show_toc: true # Alternative to 'toc'

# Tags
tags:
  - Documentation
  - Guide
  - Tutorial

# Call to Action
cta_title: "Ready to get started?"
cta_description: "Take the next step"
cta_text: "Get Started"
cta_link: /get-started/

# Related Content
related_pages:
  - title: "Related Page 1"
    url: /related-1/
    description: "Brief description"
  - title: "Related Page 2"
    url: /related-2/

# External Resources
resources:
  - title: "Resource Name"
    url: https://example.com
    description: "What this resource covers"

# Footer Metadata
source: "Original source or attribution"
attribution: "Alternative to 'source'"
license: "CC BY-SA 4.0"
version: "1.0.0"
changelog_url: /changelog/

# Styling
page_class: "custom-page-class"
header_background: "linear-gradient(...)"
```

---

### 3. `article.html`

Layout for long-form articles and narratives.

**Front Matter Options:**

```yaml
layout: article
title: "Article Title"
description: "Article summary"
author: "Author Name"
date: 2026-01-19
category: "Biography"
```

---

### 4. `doc.html`

Simple documentation layout without decorative elements.

**Front Matter Options:**

```yaml
layout: doc
title: "Documentation Title"
```

---

### 5. `case.html` / `case-enhanced.html`

Legal case layouts with docket management and specialized fields.

**Front Matter Options:**

```yaml
layout: case-enhanced

# Case Information
title: "Case Title"
short_title: "Short Title"
court: "Court Name"
venue: "Venue"
primary_docket: "ATL-DC-007956-25"
case_type: "Civil Rights"
status: "Active"
filed_date: 2025-01-01
forum_level: "Superior Court"
judge: "Judge Name"
role: "Pro Se Plaintiff"

# Case Details
procedural_posture: "Motion pending..."
factual_background: "The facts are..."
my_involvement: "My role in this case..."
current_status: "Currently awaiting..."
next_steps:
  - "File motion by X date"
  - "Attend hearing on Y date"

# Timeline
timeline:
  - date: 2025-01-01
    event: "Complaint filed"
  - date: 2025-02-01
    event: "Answer received"

# Documents
documents:
  - path: "complaint.pdf"
    label: "Original Complaint"
    note: "Filed 1/1/2025"

# Related Cases
related_cases:
  - title: "Related Case"
    url: /cases/related/
    relationship: "Consolidated with"
related_dockets:
  - "Related Docket Number"

# Provenance
source_url: https://example.com
received_via: "OPRA Request"
provenance_note: "Received via..."

# Assets
assets_dir: /cases/case-slug/filings/
```

---

## 🎨 Global Site Variables

These are set in `_config.yml` and affect all pages:

```yaml
# Site Info
title: "Site Title"
description: "Site description"
lang: en

# Features
maintenance_mode: true # Show maintenance banner
analytics_id: "GA-XXXXX" # Google Analytics
```

---

## 📋 Common Patterns

### Minimal Page

```yaml
---
layout: page
title: "Simple Page"
---
Content goes here...
```

### Rich Documentation Page

```yaml
---
layout: page
title: "Complete Guide"
description: "Everything you need to know"
category: "Documentation"
author: "Team"
date: 2026-01-19
show_breadcrumbs: true
toc: true
tags:
  - Guide
  - Documentation
cta_text: "Start Learning"
cta_link: /start/
related_pages:
  - title: "Next Steps"
    url: /next-steps/
---
Content with headers for TOC...
```

### Alert Page

```yaml
---
layout: page
title: "Important Notice"
alert: "This page contains time-sensitive information"
alert_title: "Notice"
alert_type: "warning"
---
Content...
```

---

## 🔧 Tips

1. **Use `page.html` for content pages** - It provides the most flexibility
2. **Use `case.html` for legal cases** - Structured with proper metadata
3. **Use `default.html` for custom layouts** - When you need full control
4. **Use `doc.html` for simple docs** - Clean and minimal
5. **CSS loads automatically** - Based on layout and collection type
6. **Enable TOC for long pages** - Improves navigation
7. **Add related pages** - Helps users discover content
8. **Use alerts sparingly** - Only for important messages
9. **Set reading time** - Or let it auto-calculate
10. **Always set title and description** - Important for SEO

---

## 🎨 Automatic CSS Loading

The default layout automatically loads page-specific CSS based on:

### By Layout Type

```yaml
layout: case          # Loads cases.css, case-enhanced.css, case-analysis.css
layout: case-enhanced # Loads cases.css, case-enhanced.css, case-analysis.css
layout: essay         # Loads essays.css
layout: article       # Loads articles.css
```

### By Collection

```yaml
collection: cases     # Loads cases.css, case-enhanced.css
collection: essays    # Loads essays.css
collection: opra      # Loads opra.css
```

### By Body Class

```yaml
body_class: "page-case" # Loads cases.css (if contains 'case')
```

You don't need to manually add extra_css for these - it's automatic!

---

## 🐛 Troubleshooting

**Page not rendering?**

- Check that layout name is spelled correctly
- Verify front matter is valid YAML
- Ensure layout file exists in `_layouts/`

**Conditionals not working?**

- Check variable names (case-sensitive)
- Verify indentation in YAML
- Use `true` not `"true"` for booleans

**Styles not applying?**

- Add custom CSS via `extra_css`
- Use `body_class` or `page_class` for targeting
- Check browser console for errors

---

## 📚 See Also

- `_includes/README.md` - Component documentation
- `_config.yml` - Site configuration
- Jekyll Documentation: https://jekyllrb.com/docs/
