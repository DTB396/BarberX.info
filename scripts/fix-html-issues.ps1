# Fix HTML Structure Issues - Comprehensive Solution
# This script fixes all validation issues found across the repository

Write-Host "🔧 FIXING HTML STRUCTURE ISSUES" -ForegroundColor Cyan
Write-Host ("=" * 60) -ForegroundColor Gray

# Function to safely replace content
function Fix-File {
    param(
        [string]$Path,
        [hashtable[]]$Replacements
    )
    
    if (!(Test-Path $Path)) {
        Write-Host "⚠️  File not found: $Path" -ForegroundColor Yellow
        return
    }
    
    $content = Get-Content $Path -Raw -Encoding UTF8
    $modified = $false
    
    foreach ($replacement in $Replacements) {
        if ($content -match [regex]::Escape($replacement.Old)) {
            $content = $content -replace [regex]::Escape($replacement.Old), $replacement.New
            $modified = $true
            Write-Host "  ✓ $($replacement.Description)" -ForegroundColor Green
        }
    }
    
    if ($modified) {
        Set-Content $Path -Value $content -Encoding UTF8 -NoNewline
        Write-Host "✅ Fixed: $Path`n" -ForegroundColor Green
    }
}

# Function to fix placeholder tags
function Fix-PlaceholderTags {
    param([string]$Path)
    
    if (!(Test-Path $Path)) { return }
    
    $content = Get-Content $Path -Raw -Encoding UTF8
    $modified = $false
    
    # Escape placeholder tags - order matters, do closing tags first
    $tags = @(
        @{ Pattern = '</slug>'; Replace = '' },
        @{ Pattern = '<slug>'; Replace = '`&lt;slug&gt;`' },
        @{ Pattern = '</name>'; Replace = '' },
        @{ Pattern = '<name>'; Replace = '`&lt;name&gt;`' },
        @{ Pattern = '</path>'; Replace = '' },
        @{ Pattern = '<path>'; Replace = '`&lt;path&gt;`' },
        @{ Pattern = '</DATE>'; Replace = '' },
        @{ Pattern = '<DATE>'; Replace = '`&lt;DATE&gt;`' },
        @{ Pattern = '</date>'; Replace = '' },
        @{ Pattern = '<date>'; Replace = '`&lt;date&gt;`' },
        @{ Pattern = '</n>'; Replace = '' },
        @{ Pattern = '<n>'; Replace = '`&lt;n&gt;`' }
    )
    
    foreach ($tag in $tags) {
        $escapedPattern = [regex]::Escape($tag.Pattern)
        if ($content -match $escapedPattern) {
            $content = $content -replace $escapedPattern, $tag.Replace
            $modified = $true
        }
    }
    
    if ($modified) {
        Set-Content $Path -Value $content -Encoding UTF8 -NoNewline
        Write-Host "  ✅ Fixed placeholder tags in $Path" -ForegroundColor Green
    }
}

# Function to replace hardcoded colors
function Fix-HardcodedColors {
    param([string]$Path)
    
    if (!(Test-Path $Path)) { return }
    
    $content = Get-Content $Path -Raw -Encoding UTF8
    $modified = $false
    
    # Color mappings
    $colorMappings = @{
        'rgba(16,92,74,1)' = 'var(--emerald-700)'
        'rgba(58,56,52,1)' = 'var(--color-text-muted)'
        'rgba(28,27,25,1)' = 'var(--color-bg)'
        '#d4af37' = 'var(--accent-brass)'
        'rgba(212, 165, 116, 0.2)' = 'var(--brass-alpha-20)'
        'rgba(0, 0, 0, 0.2)' = 'var(--shadow-alpha-20)'
        '#4CAF50' = 'var(--success-green)'
        'rgba(76, 175, 80, 0.2)' = 'var(--success-alpha-20)'
        '#FFC107' = 'var(--warning-yellow)'
        'rgba(255, 193, 7, 0.2)' = 'var(--warning-alpha-20)'
        '#9E9E9E' = 'var(--muted-gray)'
        'rgba(158, 158, 158, 0.2)' = 'var(--muted-alpha-20)'
        'rgba(15, 23, 42, 0.6)' = 'var(--navy-alpha-60)'
        'rgba(212, 165, 116, 0.3)' = 'var(--brass-alpha-30)'
        'rgba(184,138,57,0.18)' = 'var(--brass-alpha-18)'
        'rgba(184,138,57,0.35)' = 'var(--brass-alpha-35)'
    }
    
    foreach ($color in $colorMappings.Keys) {
        $escapedColor = [regex]::Escape($color)
        if ($content -match $escapedColor) {
            $content = $content -replace $escapedColor, $colorMappings[$color]
            $modified = $true
        }
    }
    
    if ($modified) {
        Set-Content $Path -Value $content -Encoding UTF8 -NoNewline
        Write-Host "  ✅ Replaced hardcoded colors in $Path" -ForegroundColor Green
    }
}

# =============================================================================
# FIX ESSAYS
# =============================================================================
Write-Host "`n📚 Fixing _essays directory..." -ForegroundColor Cyan

# Fix thomas-becket-proclamation.md
Fix-File -Path "_essays\2020-12-28-thomas-becket-proclamation.md" -Replacements @(
    @{
        Old = '<meta charset="UTF-8"><meta charset="UTF-8"> <br>'
        New = '<meta charset="UTF-8"><br>'
        Description = "Remove duplicate meta charset"
    },
    @{
        Old = '<span class="meta__label">ISSUED ON:</span><span> DECEMBER</span><time> 28, 2020,</time>'
        New = '<span class="meta__label">ISSUED ON:</span> DECEMBER 28, 2020'
        Description = "Fix nested span/time tags"
    }
)

# Fix faith-frontier-ministry-charter.md - add missing closing section tag
if (Test-Path "_essays\2025-06-13-faith-frontier-ministry-charter.md") {
    $content = Get-Content "_essays\2025-06-13-faith-frontier-ministry-charter.md" -Raw -Encoding UTF8
    # Find line with unclosed section and add closing tag before next section
    if ($content -match '<section class="charter-section">[\s\S]*?<h2>') {
        # This requires manual inspection - flag for review
        Write-Host "  ⚠️  faith-frontier-ministry-charter.md requires manual review (section tag)" -ForegroundColor Yellow
    }
}

# Fix tiller-earth.md
Fix-File -Path "_essays\2025-10-11-tiller-earth.md" -Replacements @(
    @{
        Old = '— <strong>Devon Tyler Barber</strong> • <em>Tillerstead LLC · Faith Frontier</em><br>'
        New = '— <strong>Devon Tyler Barber • <em>Tillerstead LLC · Faith Frontier</em></strong><br>'
        Description = "Fix strong/em tag nesting"
    }
)

# Fix hardcoded colors in tiller-earth.md
Fix-HardcodedColors -Path "_essays\2025-10-11-tiller-earth.md"

# =============================================================================
# FIX PLACEHOLDER TAGS
# =============================================================================
Write-Host "`n📋 Fixing placeholder tags..." -ForegroundColor Cyan

$placeholderFiles = @(
    ".github\agents\my-agent.agent.md",
    ".github\copilot-instructions.md",
    "_internal\BROKEN-PDFS-TODO.md",
    "_internal\FIXES-SUMMARY.md",
    "_internal\GOVERNANCE-REFACTOR-QUICKSTART.md",
    "_internal\GOVERNANCE-REFACTOR-TOOL.md",
    "_internal\IMPLEMENTATION-CHECKLIST.md",
    "worker\README.md"
)

foreach ($file in $placeholderFiles) {
    Fix-PlaceholderTags -Path $file
}

# =============================================================================
# FIX SVG ISSUES
# =============================================================================
Write-Host "`n🎨 Fixing SVG issues..." -ForegroundColor Cyan
if (Test-Path "assets\img\README.md") {
    $content = Get-Content "assets\img\README.md" -Raw -Encoding UTF8
    if ($content -match '<svg>(?!\s*</svg>)') {
        $content = $content -replace '<svg>', '```html`n&lt;svg&gt;`n```'
        Set-Content "assets\img\README.md" -Value $content -Encoding UTF8 -NoNewline
        Write-Host "  ✅ Fixed SVG tags" -ForegroundColor Green
    }
}

# =============================================================================
# FIX STYLE TAGS
# =============================================================================
Write-Host "`n💅 Fixing <style> tags in documentation..." -ForegroundColor Cyan
$styleFiles = @(
    "_internal\CASE-RESOURCES-QUICKREF.md",
    "_internal\CASE-RESOURCES-SYSTEM.md",
    "_internal\STYLE-RULES.md"
)

foreach ($file in $styleFiles) {
    if (Test-Path $file) {
        $content = Get-Content $file -Raw -Encoding UTF8
        if ($content -match '<style>(?![\s\S]*?</style>)') {
            $content = $content -replace '<style>', '```html`n&lt;style&gt;`n```'
            Set-Content $file -Value $content -Encoding UTF8 -NoNewline
            Write-Host "  ✅ Fixed <style> tags in $file" -ForegroundColor Green
        }
    }
}

# =============================================================================
# FIX HARDCODED COLORS IN INCLUDES
# =============================================================================
Write-Host "`n🎨 Fixing hardcoded colors in _includes..." -ForegroundColor Cyan
Fix-HardcodedColors -Path "_includes\nav.component.html"
Fix-HardcodedColors -Path "_includes\opra-quick-links.html"

# =============================================================================
# FIX HARDCODED COLORS IN LAYOUTS
# =============================================================================
Write-Host "`n🎨 Fixing hardcoded colors in _layouts..." -ForegroundColor Cyan
Fix-HardcodedColors -Path "_layouts\article.html"

# =============================================================================
# FIX HARDCODED COLORS IN PAGES
# =============================================================================
Write-Host "`n🎨 Fixing hardcoded colors in _pages..." -ForegroundColor Cyan
$pageFiles = @(
    "_pages\about.md",
    "_pages\the-story.md"
)

foreach ($file in $pageFiles) {
    Fix-HardcodedColors -Path $file
}

# =============================================================================
# FIX UNCLOSED LIQUID BLOCKS
# =============================================================================
Write-Host "`n🔧 Fixing Liquid template issues..." -ForegroundColor Cyan
if (Test-Path "_internal\CASE-FILES-CLEANUP-SUMMARY.md") {
    $content = Get-Content "_internal\CASE-FILES-CLEANUP-SUMMARY.md" -Raw -Encoding UTF8
    # Escape Liquid syntax in code examples
    if ($content -match '{% if %}') {
        $content = $content -replace '{% if %}', '`{% if %}`'
        $content = $content -replace '{% endif %}', '`{% endif %}`'
        Set-Content "_internal\CASE-FILES-CLEANUP-SUMMARY.md" -Value $content -Encoding UTF8 -NoNewline
        Write-Host "  ✅ Fixed Liquid blocks in CASE-FILES-CLEANUP-SUMMARY.md" -ForegroundColor Green
    }
}

# =============================================================================
# SUMMARY
# =============================================================================
Write-Host "`n" + ("=" * 60) -ForegroundColor Gray
Write-Host "✅ Automated fixes complete!" -ForegroundColor Green
Write-Host "`nNote: Some issues require manual review:" -ForegroundColor Yellow
Write-Host "  - _essays/2025-06-13-faith-frontier-ministry-charter.md (section tags)" -ForegroundColor Yellow
Write-Host "  - _essays/2025-10-11-tiller-earth.md (complex nested structure)" -ForegroundColor Yellow
Write-Host "  - _includes/*.html (partial templates - may have intentional incomplete HTML)" -ForegroundColor Yellow
Write-Host "  - _layouts/*.html (template structure)" -ForegroundColor Yellow
Write-Host "  - _pages/*.md (complex inline styles with Liquid)" -ForegroundColor Yellow
Write-Host "`nRun validation to check progress:" -ForegroundColor Cyan
Write-Host "  node scripts/validate-html-structure.js --scan <directory>`n" -ForegroundColor Gray
