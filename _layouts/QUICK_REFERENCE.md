# Layout Quick Reference

Quick reference for the most commonly used front matter options.

## 🚀 Quick Start

### Basic Page

```yaml
---
layout: page
title: "Page Title"
---
```

### Documentation Page

```yaml
---
layout: page
title: "Guide Title"
description: "What this guide covers"
toc: true
show_breadcrumbs: true
---
```

### Alert Page

```yaml
---
layout: page
title: "Important Notice"
alert: "Time-sensitive information"
alert_type: "warning" # info, warning, error, success
---
```

## 📋 Common Options

| Option             | Type    | Example             | Description            |
| ------------------ | ------- | ------------------- | ---------------------- |
| `layout`           | string  | `page`              | Layout template to use |
| `title`            | string  | `"My Page"`         | Page title (required)  |
| `description`      | string  | `"Page summary"`    | Shown below title      |
| `category`         | string  | `"Documentation"`   | Page category          |
| `author`           | string  | `"Devon Tyler"`     | Content author         |
| `date`             | date    | `2026-01-19`        | Publication date       |
| `toc`              | boolean | `true`              | Show table of contents |
| `show_breadcrumbs` | boolean | `true`              | Show breadcrumb trail  |
| `tags`             | array   | `[Guide, Tutorial]` | Page tags              |

## 🎨 Styling Options

| Option       | Type   | Example         |
| ------------ | ------ | --------------- |
| `body_class` | string | `"dark-theme"`  |
| `page_class` | string | `"wide-layout"` |
| `main_class` | string | `"container"`   |

## 🔔 Alert Types

```yaml
alert_type: "info"     # Blue - informational
alert_type: "warning"  # Yellow - caution
alert_type: "error"    # Red - critical
alert_type: "success"  # Green - confirmation
```

## 🔗 CTAs & Navigation

```yaml
# Call to action
cta_text: "Get Started"
cta_link: /start/
cta_title: "Ready to begin?"
cta_description: "Take the first step"

# Related pages
related_pages:
  - title: "Next Guide"
    url: /next/
    description: "Continue learning"
```

## 🎯 Metadata

```yaml
author: "Devon Tyler"
date: 2026-01-19
last_updated: 2026-01-19
reading_time: "5 min" # Or auto-calculated
license: "CC BY-SA 4.0"
version: "1.0.0"
```

## 🙈 Hide Elements

```yaml
hide_header: true # No site header
hide_footer: true # No site footer
hide_barber_pole: true # No decorative element
no_main_js: true # No main.js
```

## 📦 Custom Assets

```yaml
extra_css:
  - /assets/css/custom.css
extra_js:
  - /assets/js/custom.js
```

## 💡 Pro Tips

1. **Always set title** - Required for good UX
2. **Use descriptions** - Improves SEO and clarity
3. **Enable TOC for long content** - Helps navigation
4. **Add related pages** - Keeps users engaged
5. **Use alerts sparingly** - Only for important info
6. **Test with different browsers** - Verify compatibility

## 🐛 Common Mistakes

❌ **Don't do this:**

```yaml
toc: "true" # String instead of boolean
hide_header: yes # Use true/false
```

✅ **Do this:**

```yaml
toc: true # Boolean
hide_header: true # Boolean
```

---

For complete documentation, see `_layouts/README.md`
