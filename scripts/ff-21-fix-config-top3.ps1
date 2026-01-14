# scripts/ff-21-fix-config-top3.ps1
[CmdletBinding()]
param(
  [string]$RepoRoot = (Get-Location).Path,
  [string]$ExpectedUrl = "https://faithfrontier.org",
  [string]$LogoAssetPath = "/assets/img/logo/logo-faithfrontier-header.svg",
  [switch]$WhatIf
)

$ErrorActionPreference = "Stop"

function Assert-GitClean([string]$root) {
  if (-not (Test-Path (Join-Path $root ".git"))) { throw "Not a git repo (no .git). Run at repo root." }
  $porcelain = git -C $root status --porcelain
  if (-not [string]::IsNullOrWhiteSpace($porcelain)) {
    throw "Safety stop: working tree not clean. Commit/stash first."
  }
}

function Backup-File([string]$path, [string]$root) {
  $dir = Join-Path $root "docs\reports"
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $bak = Join-Path $dir ("_config.yml.backup-{0}" -f $stamp)
  Copy-Item -Force $path $bak
  return $bak
}

Assert-GitClean $RepoRoot

$configPath = Join-Path $RepoRoot "_config.yml"
if (-not (Test-Path $configPath)) { throw "Missing _config.yml" }

$cfg = Get-Content -Raw $configPath

# Hard stop if config looks corrupted (protect healthy repo)
if ($cfg -match '"scripts"\s*:' -or $cfg -match '^\s*\{' ) {
  throw "Safety stop: _config.yml contains JSON-like content."
}

$backup = Backup-File $configPath $RepoRoot

# --- Fix 1: ensure url is correct (idempotent) ---
$cfg2 = [regex]::Replace($cfg, '(?m)^\s*url\s*:\s*.*$', "url: $ExpectedUrl")

# --- Fix 2: manifesto permalink collision -> /manifesto/:path/ ---
# Replace the line inside the manifesto collection block
$cfg2 = [regex]::Replace(
  $cfg2,
  '(?ms)(manifesto:\s*\n\s*output:\s*true\s*\n\s*permalink:\s*)([^\n]+)',
  '${1}/manifesto/:path/'
)

# --- Fix 3: docs/trust permalinks :name -> :path (future-safe) ---
$cfg2 = [regex]::Replace(
  $cfg2,
  '(?ms)(docs:\s*\n\s*output:\s*true\s*\n\s*permalink:\s*)([^\n]+)',
  '${1}/docs/:path/'
)
$cfg2 = [regex]::Replace(
  $cfg2,
  '(?ms)(trust:\s*\n\s*output:\s*true\s*\n\s*permalink:\s*)([^\n]+)',
  '${1}/sanctuary/trust/:path/'
)

# --- Logo fix: move references away from /_includes/... to /assets/... ---
$cfg2 = [regex]::Replace($cfg2, '(?m)^(logo_main_dark:\s*).*$' , "`$1`"$LogoAssetPath`"")
$cfg2 = [regex]::Replace($cfg2, '(?m)^(logo_main_light:\s*).*$', "`$1`"$LogoAssetPath`"")
$cfg2 = [regex]::Replace($cfg2, '(?m)^(logo_main:\s*).*$'      , "`$1`"$LogoAssetPath`"")

# Add standard `logo:` key if missing (helps SEO/includes)
if ($cfg2 -notmatch '(?m)^\s*logo\s*:') {
  $cfg2 = $cfg2.TrimEnd() + "`n`n# Standard logo key (used by SEO/includes)`nlogo: `"$LogoAssetPath`"`n"
}

if ($cfg2 -eq $cfg) {
  Write-Host "No changes needed."
  Write-Host "Backup created anyway: $backup"
  exit 0
}

if ($WhatIf) {
  Write-Host "WhatIf: would update _config.yml"
  Write-Host "Backup: $backup"
  exit 0
}

Set-Content -Encoding UTF8 $configPath $cfg2
Write-Host "Applied top-3 config fixes safely."
Write-Host "Backup: $backup"
Write-Host "IMPORTANT: ensure the logo SVG exists at: $LogoAssetPath"
