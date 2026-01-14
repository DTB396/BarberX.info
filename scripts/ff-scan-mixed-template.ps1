[CmdletBinding()]
param([string]$RepoRoot = (Get-Location).Path)

$ErrorActionPreference = "Stop"

if (-not (Test-Path (Join-Path $RepoRoot ".git"))) { throw "Run from repo root (no .git found)." }

$hits = @()
$patterns = @(
  '</html>\s*<header',
  '<meta\s+charset=',
  '/\*\*\s*\r?\n\s*\*\s*PREMIUM HEADER INTERACTIONS',
  'premium-header__inner',
  '{%\s*include\s+head\.html\s*%}',
  '{%\s*include\s+header\.html\s*%}'
)

Get-ChildItem $RepoRoot -Recurse -File -Include *.html,*.liquid,*.md,*.js -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch "\\_site\\|\\vendor\\|\\node_modules\\" } |
  ForEach-Object {
    $raw = Get-Content -Raw $_.FullName
    $matched = @()
    foreach ($p in $patterns) {
      if ($raw -match $p) { $matched += $p }
    }
    if ($matched.Count -ge 2) {
      $hits += [PSCustomObject]@{ File = $_.FullName; Matches = ($matched -join " | ") }
    }
  }

$reportDir = Join-Path $RepoRoot "docs\reports"
if (-not (Test-Path $reportDir)) { New-Item -ItemType Directory -Path $reportDir | Out-Null }
$report = Join-Path $reportDir ("ff-mixed-template-scan-{0}.txt" -f (Get-Date -Format "yyyyMMdd-HHmmss"))

$hits | Sort-Object File | Format-Table -AutoSize | Out-String | Set-Content -Encoding UTF8 $report
Write-Host "Scan report: $report"
Write-Host "If you paste the report here, I’ll tell you the exact file to repair."
