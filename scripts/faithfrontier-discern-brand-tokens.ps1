<# FaithFrontier Brand Token Discernment (PowerShell)
   Interactive by default. Use -Auto for no prompts.
#>
[CmdletBinding()]
param(
  [switch]$Auto,
  [ValidateSet("rock","covenant","luminous")]
  [string]$Preset = "rock",
  [switch]$WriteOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}

function Ask([string]$Prompt, [string]$Default = "") {
  if ($Auto) { return $Default }
  $msg = if ([string]::IsNullOrWhiteSpace($Default)) { $Prompt } else { "{0} [{1}]" -f $Prompt, $Default }
  $in = Read-Host $msg
  if ([string]::IsNullOrWhiteSpace($in)) { return $Default }
  return $in.Trim()
}

function Ensure-HexColor([string]$v, [string]$fallback) {
  if ([string]::IsNullOrWhiteSpace($v)) { return $fallback }
  $vv = $v.Trim()
  if ($vv -match "^#[0-9A-Fa-f]{6}$") { return $vv.ToUpperInvariant() }
  return $fallback
}

function Repo-Root { (Resolve-Path (Join-Path $PSScriptRoot "..")).Path }
$ROOT = Repo-Root

$presets = @{
  rock = @{
    ink="#0B1220"; navy="#0E1B2A"; emerald="#0F5D4D"; gold="#B08D2F"; parchment="#F5F0E6"; stone="#E6E2DA"
    danger="#8A1F2E"; success="#1E6B4E"; info="#1B4F72"
    radius=16; maxw=1120; shadow="0 14px 40px rgba(11,18,32,.16)"; motionStyle="gentle-purposeful"; motionSpeed="medium"; easing="ease-out"
    fontSans="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"
    fontSerif="ui-serif, Georgia, ''Times New Roman'', serif"
    feel="steadfast, calm, luminous, disciplined, protective"
  }
  covenant = @{
    ink="#0B1220"; navy="#0C1A24"; emerald="#0D5A45"; gold="#B28A2E"; parchment="#F7F2E9"; stone="#E8E4DD"
    danger="#8A1F2E"; success="#1E6B4E"; info="#1B4F72"
    radius=14; maxw=1160; shadow="0 16px 44px rgba(11,18,32,.14)"; motionStyle="disciplined"; motionSpeed="slow"; easing="ease-in-out"
    fontSans="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"
    fontSerif="ui-serif, Georgia, ''Times New Roman'', serif"
    feel="reverent, disciplined, stable, clear, protective"
  }
  luminous = @{
    ink="#09101E"; navy="#0B1A2D"; emerald="#0E6A56"; gold="#C49A33"; parchment="#FBF6EC"; stone="#EEE9E1"
    danger="#8A1F2E"; success="#1E6B4E"; info="#1B4F72"
    radius=18; maxw=1120; shadow="0 18px 48px rgba(9,16,30,.14)"; motionStyle="luminous"; motionSpeed="medium"; easing="ease-out"
    fontSans="ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"
    fontSerif="ui-serif, Georgia, ''Times New Roman'', serif"
    feel="luminous, calm, confident, warm, ordered"
  }
}
$p = $presets[$Preset]

if (-not $WriteOnly) {
  Write-Host "============================================================"
  Write-Host " FaithFrontier Brand Discernment (VS Code / PowerShell)"
  Write-Host "============================================================"
  Write-Host ""
  Write-Host "Note: This script cannot contact God on your behalf."
  Write-Host "It translates prayerful discernment into usable design tokens."
  Write-Host ""
}

$virtue1 = Ask "Virtue #1" "Integrity"
$virtue2 = Ask "Virtue #2" "Faith"
$virtue3 = Ask "Virtue #3" "Perseverance"

$feel = Ask "UX feel (3-6 words, comma-separated)" $p.feel
$motionStyle = Ask "Motion style" $p.motionStyle
$motionSpeed = Ask "Motion speed (fast/medium/slow)" $p.motionSpeed
$easing      = Ask "Easing (standard/ease-out/ease-in-out)" $p.easing

$ink       = Ensure-HexColor (Ask "Color ink hex" $p.ink) $p.ink
$navy      = Ensure-HexColor (Ask "Color navy hex" $p.navy) $p.navy
$emerald   = Ensure-HexColor (Ask "Color emerald hex" $p.emerald) $p.emerald
$gold      = Ensure-HexColor (Ask "Color gold hex" $p.gold) $p.gold
$parchment = Ensure-HexColor (Ask "Color parchment hex" $p.parchment) $p.parchment
$stone     = Ensure-HexColor (Ask "Color stone hex" $p.stone) $p.stone
$danger    = Ensure-HexColor (Ask "Color danger hex" $p.danger) $p.danger
$success   = Ensure-HexColor (Ask "Color success hex" $p.success) $p.success
$info      = Ensure-HexColor (Ask "Color info hex" $p.info) $p.info

$radius = [int](Ask "Border radius px" ($p.radius.ToString()))
$maxw   = [int](Ask "Max content width px" ($p.maxw.ToString()))
$shadow = Ask "Shadow CSS" $p.shadow

$fontSans  = Ask "Primary sans stack" $p.fontSans
$fontSerif = Ask "Accent serif stack (optional)" $p.fontSerif

$sem = [ordered]@{
  bg=$parchment; surface=$stone; text=$ink; muted=$navy; brand=$emerald; accent=$gold
  info=$info; success=$success; danger=$danger; border="rgba(11,18,32,.18)"; focus=$gold
}

$tokensDir = Join-Path $ROOT "assets\css\tokens"
$cssOut    = Join-Path $tokensDir "faithfrontier.tokens.css"
$docsDir   = Join-Path $ROOT "docs"
$mdOut     = Join-Path $docsDir "BRAND_TOKENS.md"
New-Item -ItemType Directory -Force -Path $tokensDir | Out-Null
New-Item -ItemType Directory -Force -Path $docsDir | Out-Null

$css = ":root {`r`n"
$css += "  /* FaithFrontier tokens */`r`n"
$css += "  /* Virtues: $virtue1 | $virtue2 | $virtue3 */`r`n"
$css += "  --ff-ink: $ink;`r`n"
$css += "  --ff-navy: $navy;`r`n"
$css += "  --ff-emerald: $emerald;`r`n"
$css += "  --ff-gold: $gold;`r`n"
$css += "  --ff-parchment: $parchment;`r`n"
$css += "  --ff-stone: $stone;`r`n"
$css += "  --ff-danger: $danger;`r`n"
$css += "  --ff-success: $success;`r`n"
$css += "  --ff-info: $info;`r`n"
$css += "  --ff-radius: ${radius}px;`r`n"
$css += "  --ff-maxw: ${maxw}px;`r`n"
$css += "  --ff-shadow: $shadow;`r`n"
$css += "  --ff-font-sans: $fontSans;`r`n"
$css += "  --ff-font-serif: $fontSerif;`r`n"
$css += "  --ff-motion-style: $motionStyle;`r`n"
$css += "  --ff-motion-speed: $motionSpeed;`r`n"
$css += "  --ff-easing: $easing;`r`n"

foreach ($k in $sem.Keys) {
  # IMPORTANT: ${k} prevents the --ff-$k: parsing issue
  $css += "  --ff-${k}: $($sem[$k]);`r`n"
}
$css += "}`r`n"

$semLines = ($sem.Keys | Sort-Object | ForEach-Object { "- **$($_)**: ``$($sem[$_])``" }) -join "`n"

$md = @"
---
title: Brand Tokens
description: FaithFrontier design tokens generated by the discernment script.
---

# FaithFrontier Brand Tokens

**Virtues:** $virtue1 | $virtue2 | $virtue3  
**UX feel:** $feel  
**Preset:** $Preset

## Palette

- Ink: ``$ink``
- Navy: ``$navy``
- Emerald: ``$emerald``
- Gold: ``$gold``
- Parchment: ``$parchment``
- Stone: ``$stone``

## Semantic tokens

$semLines

## Motion

- Style: **$motionStyle**
- Speed: **$motionSpeed**
- Easing: **$easing**

## Typography

- Sans: ``$fontSans``
- Serif: ``$fontSerif``

## Governance note

This file records an intentional token decision to prevent silent design drift.
"@

[System.IO.File]::WriteAllText($cssOut, $css, [System.Text.UTF8Encoding]::new($false))
[System.IO.File]::WriteAllText($mdOut,  $md,  [System.Text.UTF8Encoding]::new($false))

if (-not $WriteOnly) {
  Write-Host ""
  Write-Host "Wrote tokens CSS:"
  Write-Host "  $cssOut"
  Write-Host "Wrote docs:"
  Write-Host "  $mdOut"
}
