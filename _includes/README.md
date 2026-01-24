# Organized \_includes Directory Structure

This directory contains all Jekyll include files organized by function following best practices.

## 📁 Directory Structure

```
_includes/
├── components/          # Reusable UI components
│   ├── heroes/         # Hero sections (barber-hero, hero, page-hero, premium-hero, case-hero)
│   ├── navigation/     # Navigation elements (header, breadcrumbs, nav)
│   ├── cards/          # Card components (case-card)
│   ├── banners/        # Banner components (maintenance-banner, faith-conscience-banner)
│   └── forms/          # Form components (connect)
│
├── sections/           # Page-specific sections
│   ├── home/          # Homepage sections (preview, principles, status, compliance, faq, featured-essays)
│   ├── cases/         # Case page sections (case-analysis, case-docket, case-resources, opra-*)
│   └── faith/         # Faith sections (daily-verse-enhanced, spotify-player)
│
├── layout/            # Site-wide layout elements
│   ├── head/         # Head elements (head.html)
│   └── footer/       # Footer elements (footer.html, footer-links.html)
│
├── seo/              # SEO and metadata
│   ├── seo.html      # SEO meta tags
│   ├── schema.html   # Schema.org structured data
│   └── structured-data.html
│
├── data/             # Data-related includes
│   ├── scripts.html  # JavaScript includes
│   └── holistic-data.html
│
└── assets/           # Asset-related includes
    └── logos/        # Logo files and related documentation
```

## 🔧 Usage Examples

### Heroes

```liquid
{% include components/heroes/barber-hero.html %}
{% include components/heroes/page-hero.html title="Page Title" %}
```

### Navigation

```liquid
{% include components/navigation/header.html %}
{% include components/navigation/breadcrumbs.html %}
```

### Home Sections

```liquid
{% include sections/home/preview.html %}
{% include sections/home/principles.html %}
{% include sections/home/status.html %}
```

### Case Sections

```liquid
{% include sections/cases/case-docket.html %}
{% include sections/cases/case-analysis.html %}
{% include sections/cases/opra-records.html %}
```

### Layout

```liquid
{% include layout/head/head.html %}
{% include layout/footer/footer.html %}
```

### SEO

```liquid
{% include seo/seo.html %}
{% include seo/schema.html %}
```

## 📋 Migration Reference

| Old Path             | New Path                                 |
| -------------------- | ---------------------------------------- |
| `barber-hero.html`   | `components/heroes/barber-hero.html`     |
| `header.html`        | `components/navigation/header.html`      |
| `breadcrumbs.html`   | `components/navigation/breadcrumbs.html` |
| `preview.html`       | `sections/home/preview.html`             |
| `principles.html`    | `sections/home/principles.html`          |
| `status.html`        | `sections/home/status.html`              |
| `case-docket.html`   | `sections/cases/case-docket.html`        |
| `case-analysis.html` | `sections/cases/case-analysis.html`      |
| `head.html`          | `layout/head/head.html`                  |
| `footer.html`        | `layout/footer/footer.html`              |
| `seo.html`           | `seo/seo.html`                           |
| `schema.html`        | `seo/schema.html`                        |

## ✅ Updated Files

The following files have been updated to use the new paths:

- `index.html`
- `_layouts/default.html`
- `_layouts/case.html`
- `_layouts/case-enhanced.html`

## 🎯 Benefits

1. **Clear Organization**: Files grouped by function, not alphabetically
2. **Easy to Find**: Logical hierarchy makes finding components intuitive
3. **Scalable**: New components can be added to appropriate categories
4. **Maintainable**: Related files are grouped together
5. **Best Practice**: Follows Jekyll and component-based architecture standards
