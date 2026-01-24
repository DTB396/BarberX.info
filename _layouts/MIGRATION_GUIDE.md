# Migration Guide: Upgrading Pages to Enhanced Layouts

This guide helps you migrate existing pages to use the new enhanced layout system.

## 🎯 Quick Migration Path

### Step 1: Identify Pages to Migrate

Pages currently using `layout: default` that would benefit from enhanced features:

- Documentation pages
- About/info pages
- Guide pages
- Tutorial pages

### Step 2: Change Layout

**Before:**

```yaml
---
layout: default
title: "My Page"
---
```

**After:**

```yaml
---
layout: page
title: "My Page"
---
```

### Step 3: Add Metadata

Enhance with recommended metadata:

```yaml
---
layout: page
title: "My Page"
description: "What this page is about" # NEW
category: "Documentation" # NEW
author: "Devon Tyler" # NEW
date: 2026-01-19 # NEW
---
```

### Step 4: Enable Useful Features

```yaml
---
layout: page
title: "My Page"
description: "What this page is about"
category: "Documentation"
show_breadcrumbs: true # NEW - Navigation trail
toc: true # NEW - Table of contents
tags: # NEW - Categorization
  - Guide
  - Important
---
```

## 📋 Migration Checklist

For each page you migrate:

- [ ] Change `layout: default` to `layout: page`
- [ ] Add `description` field
- [ ] Add `category` or `type` if applicable
- [ ] Add `author` if not already set
- [ ] Add `date` for publication date
- [ ] Consider adding `show_breadcrumbs: true`
- [ ] Enable `toc: true` if page is long
- [ ] Add relevant `tags`
- [ ] Test the page locally
- [ ] Check mobile responsiveness
- [ ] Verify print layout

## 🔄 Migration by Page Type

### Documentation Pages

```yaml
---
layout: page
title: "Documentation Title"
description: "Brief overview"
category: "Documentation"
show_breadcrumbs: true
toc: true
tags:
  - Docs
  - Reference
---
```

### About/Info Pages

```yaml
---
layout: page
title: "About Us"
description: "Learn about our mission"
category: "About"
author: "Devon Tyler"
date: 2026-01-19
---
```

### Guide/Tutorial Pages

```yaml
---
layout: page
title: "How to Guide"
description: "Step-by-step instructions"
category: "Tutorial"
toc: true
reading_time: "10 min"
cta_text: "Next Steps"
cta_link: /next-guide/
related_pages:
  - title: "Related Guide"
    url: /related/
---
```

### Landing Pages

```yaml
---
layout: page
title: "Get Started"
description: "Begin your journey"
cta_title: "Ready to start?"
cta_description: "Take the first step today"
cta_text: "Sign Up"
cta_link: /signup/
---
```

## ⚠️ Pages That Should NOT Migrate

Keep `layout: default` for:

- Homepage (`index.html`)
- Special layouts (case pages, essays)
- Custom designed pages
- Pages with unique structure needs

## 🎨 Optional Enhancements

### Add Alerts for Important Info

```yaml
alert: "This information was updated recently"
alert_type: "info"
```

### Link Related Content

```yaml
related_pages:
  - title: "Part 2 of this series"
    url: /part-2/
    description: "Continue learning"
```

### Add External Resources

```yaml
resources:
  - title: "Official Docs"
    url: https://example.com/docs
    description: "Complete reference"
```

### Include Metadata

```yaml
last_updated: 2026-01-19
license: "CC BY-SA 4.0"
version: "2.0"
```

## 🔍 Testing Your Migration

### Local Testing

1. Build the site: `bundle exec jekyll build`
2. Serve locally: `bundle exec jekyll serve`
3. Visit: `http://127.0.0.1:4000/your-page/`

### What to Check

- [ ] Page renders correctly
- [ ] All content appears
- [ ] Links work
- [ ] Images load
- [ ] TOC generates (if enabled)
- [ ] Breadcrumbs show correct path
- [ ] Alert displays properly
- [ ] CTA appears and links work
- [ ] Related pages render
- [ ] Mobile view looks good
- [ ] Print preview is clean

## 📊 Batch Migration Script

To update multiple files at once:

```powershell
# PowerShell script to update layout in multiple files
$files = Get-ChildItem "_pages" -Filter "*.md"

foreach ($file in $files) {
    $content = Get-Content $file.FullName -Raw

    # Replace layout: default with layout: page
    $updated = $content -replace 'layout: default', 'layout: page'

    # Only update if changed
    if ($updated -ne $content) {
        Set-Content $file.FullName $updated
        Write-Host "Updated: $($file.Name)" -ForegroundColor Green
    }
}
```

**⚠️ Warning:** Review each file manually after batch updates!

## 🎯 Priority Migration List

Migrate in this order:

1. **High Priority** - Frequently visited pages
   - About page
   - Contact page
   - Main documentation

2. **Medium Priority** - Important but less visited
   - Guide pages
   - Tutorial pages
   - Resource pages

3. **Low Priority** - Rarely updated
   - Archive pages
   - Old blog posts
   - Historical documents

## 💡 Best Practices

1. **Migrate gradually** - Don't try to update everything at once
2. **Test each page** - Verify it works before moving to the next
3. **Keep backups** - Use git to track changes
4. **Document decisions** - Note why certain pages weren't migrated
5. **Get feedback** - Have others review migrated pages
6. **Monitor analytics** - Check if engagement improves

## 🐛 Common Issues

### Page Looks Broken

**Problem:** Layout appears wrong after migration

**Solution:**

- Check front matter syntax (YAML must be valid)
- Verify all quotes match
- Ensure boolean values are `true` not `"true"`
- Check for special characters in text

### TOC Not Showing

**Problem:** Table of contents doesn't appear

**Solution:**

- Verify `toc: true` is set
- Check that content has headings (`##`, `###`)
- Ensure Jekyll TOC plugin is installed

### Breadcrumbs Wrong

**Problem:** Breadcrumb trail shows incorrect path

**Solution:**

- Check `permalink` setting
- Verify page is in correct directory
- Review site.navigation configuration

## 📚 Additional Resources

- `_layouts/README.md` - Complete layout documentation
- `_layouts/QUICK_REFERENCE.md` - Quick reference guide
- `/layout-example/` - Full-featured example page
- Jekyll Docs: https://jekyllrb.com/docs/

## ✅ Migration Complete!

Once you've migrated your pages:

1. Commit changes to git
2. Test thoroughly
3. Deploy to production
4. Monitor for issues
5. Gather user feedback
6. Iterate and improve

---

**Questions?** Check the documentation or create an issue!
