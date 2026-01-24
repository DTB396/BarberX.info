# BarberX UX Quick Reference

## 🎯 Component Usage Guide

### Usage Meter

Shows visual progress with tier limits:

```jinja2
{% include 'components/usage-meter.html' with
   title='BWC Videos',
   resource_name='bwc_videos',
   current=usage.bwc_videos_processed,
   limit=limits.bwc_videos_per_month,
   show_upgrade=true,
   icon_svg='<svg>...</svg>'
%}
```

### Tier Upgrade Card

Contextual upgrade prompts:

```jinja2
{% include 'components/tier-upgrade-card.html' with
   title='Unlock Professional Features',
   description='Get 25x more capacity',
   benefits=['25 BWC videos', 'No watermarks', 'Priority support']
%}
```

### Onboarding Tour

First-time user walkthrough:

```html
<!-- Add once per page, usually in dashboard -->
{% include 'components/onboarding-tour.html' %}
```

## 🔧 Python Helper Functions

### Number Formatting

```python
from ux_helpers import format_number, format_file_size, format_duration

format_number(1500)  # "1,500"
format_file_size(1048576)  # "1.00 MB"
format_duration(90)  # "1m 30s"
```

### Tier Management

```python
from ux_helpers import tier_features, tier_pricing, tier_upgrade_suggestion

features = tier_features('PROFESSIONAL')
price = tier_pricing('PREMIUM')  # 149
next_tier = tier_upgrade_suggestion('FREE')  # 'PROFESSIONAL'
```

### Usage Tracking

```python
from ux_helpers import usage_percentage, usage_status

percent = usage_percentage(current=8, limit=10)  # 80
status = usage_status(current=8, limit=10)  # 'warning'
# Returns: 'healthy', 'warning', 'critical', or 'unlimited'
```

### Route Protection

```python
from ux_helpers import requires_feature

@app.route('/advanced-tools')
@login_required
@requires_feature('forensic_analysis')
def advanced_tools():
    return render_template('tools/advanced.html')
```

## 🎨 Jinja2 Filters

In templates, use these filters:

```jinja2
{{ 1500 | format_number }}  <!-- 1,500 -->
{{ 1048576 | format_file_size }}  <!-- 1.00 MB -->
{{ 90 | format_duration }}  <!-- 1m 30s -->
{{ current_user.tier_name | tier_color }}  <!-- #c41e3a -->
{{ usage.bwc_videos_processed | usage_percentage(limits.bwc_videos_per_month) }}
```

## 🔐 Tier Access Control

### In Routes

```python
# Method 1: Decorator
from auth_routes import tier_required
from models_auth import TierLevel

@app.route('/premium-feature')
@tier_required(TierLevel.PREMIUM)
def premium_feature():
    return render_template('premium.html')

# Method 2: Manual check
@app.route('/some-route')
@login_required
def some_route():
    if not current_user.can_access_feature('api_access'):
        flash('Upgrade to access API', 'warning')
        return redirect(url_for('pricing'))
    return render_template('api.html')
```

### In Templates

```jinja2
{% if current_user.tier_name == 'PREMIUM' %}
  <div class="premium-features">...</div>
{% endif %}

{% if current_user.can_access_feature('api_access') %}
  <a href="/api/docs">API Documentation</a>
{% endif %}
```

## 💬 Flash Messages

### Python

```python
# Simple message
flash('Operation successful!', 'success')

# With HTML (e.g., links)
flash('Limit reached. <a href="/pricing">Upgrade now</a>', 'warning')

# Categories: success, danger, error, warning, info
flash('Invalid input', 'danger')
```

### Template Display

```jinja2
{% with messages = get_flashed_messages(with_categories=true) %}
  {% if messages %}
    {% for category, message in messages %}
      <div class="flash-message {{ category }}">{{ message | safe }}</div>
    {% endfor %}
  {% endif %}
{% endwith %}
```

## ♿ Accessibility Checklist

- [ ] All interactive elements have `aria-label` or visible text
- [ ] Forms have associated `<label>` elements
- [ ] Buttons use `<button>` not `<div onclick>`
- [ ] Images have `alt` text
- [ ] Modals have `role="dialog"` and `aria-modal="true"`
- [ ] Color is not the only indicator (use icons too)
- [ ] Keyboard navigation works (test with Tab)
- [ ] Focus indicators are visible
- [ ] Text has sufficient contrast (4.5:1 minimum)

## 📱 Responsive Design

### Breakpoints

```css
/* Mobile first */
.element {
  /* base styles */
}

@media (min-width: 640px) {
  /* sm */
}
@media (min-width: 768px) {
  /* md */
}
@media (min-width: 1024px) {
  /* lg */
}
@media (min-width: 1280px) {
  /* xl */
}
```

### Grid Classes

```html
<!-- Auto-fit grid -->
<div
  style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;"
>
  <!-- Cards -->
</div>
```

## 🎯 Common Patterns

### Loading States

```html
<button id="submitBtn" onclick="showLoading(this)">Submit</button>

<script>
  function showLoading(btn) {
    btn.disabled = true;
    btn.innerHTML = "<span>Processing...</span>";
  }
</script>
```

### Empty States

```html
{% if items %} {% for item in items %}
<!-- Display items -->
{% endfor %} {% else %}
<div class="empty-state">
  <p>No items found. <a href="/create">Create one</a></p>
</div>
{% endif %}
```

### Error Handling

```python
try:
    # Operation
    flash('Success!', 'success')
except Exception as e:
    app.logger.error(f'Error: {e}')
    flash('Something went wrong. Please try again.', 'danger')
```

## 🔍 Debugging Tips

### Check User Tier

```python
# In Python
print(f"User tier: {current_user.tier_name}")
print(f"Limits: {current_user.get_tier_limits()}")

# In Template
{{ current_user.tier_name }}
{{ current_user.get_tier_limits() }}
```

### View Usage Data

```python
from models_auth import UsageTracking

usage = UsageTracking.get_or_create_current(user_id)
print(f"BWC videos: {usage.bwc_videos_processed}")
print(f"Storage: {usage.storage_used_mb} MB")
```

## 🚀 Performance Tips

1. **Lazy Load Components**

   ```html
   <script defer src="/assets/js/heavy-component.js"></script>
   ```

2. **Cache Static Assets**

   ```python
   @app.route('/assets/<path:filename>')
   def serve_assets(filename):
       response = send_file(os.path.join('assets', filename))
       response.cache_control.max_age = 31536000  # 1 year
       return response
   ```

3. **Minimize Database Queries**
   ```python
   # Use eager loading
   users = User.query.options(
       db.joinedload(User.usage)
   ).all()
   ```

## 📋 Testing Checklist

- [ ] Test with keyboard only (no mouse)
- [ ] Test with screen reader (NVDA/JAWS)
- [ ] Test on mobile device
- [ ] Test in high contrast mode
- [ ] Test with slow network
- [ ] Test all tier levels
- [ ] Test error states
- [ ] Test edge cases (0, max, overflow)

## 🎨 Brand Colors

```css
/* Primary */
--barberx-red: #c41e3a;
--barberx-blue: #1e40af;
--barberx-gold: #d4a574;

/* Status */
--success: #10b981;
--warning: #f59e0b;
--error: #ef4444;
--info: #3b82f6;

/* Neutrals */
--text-primary: #0a0a0f;
--text-secondary: #6b7280;
--border-color: #e0e0e8;
--surface-color: #ffffff;
```

## 📞 Support

For questions or issues:

1. Check `UX-IMPROVEMENTS-COMPLETE.md` for detailed docs
2. Review component examples in `templates/components/`
3. Test with `ux_helpers.py` functions
4. Ensure accessibility with `assets/css/accessibility.css`

---

**Pro Tip:** Always test new features with a FREE tier account first to ensure upgrade prompts work correctly!
