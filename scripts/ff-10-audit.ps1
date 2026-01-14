# scripts/ff-10-audit.ps1
[CmdletBinding()]
param(
  [string]$RepoRoot = (Get-Location).Path
)

. (Join-Path $PSScriptRoot "ff-00-lib-safe.ps1")

$ErrorActionPreference = "Stop"
Assert-GitRepo $RepoRoot

$reportDir = Join-Path $RepoRoot "docs\reports"
if (-not (Test-Path $reportDir)) { New-Item -ItemType Directory -Path $reportDir | Out-Null }

function ReadIf([string]$p) { if (Test-Path $p) { Get-Content -Raw $p } else { $null } }

$cfgPath = Join-Path $RepoRoot "_config.yml"
$cfg = ReadIf $cfgPath
$gem = Test-Path (Join-Path $RepoRoot "Gemfile")
$pkg = Test-Path (Join-Path $RepoRoot "package.json")

$stack =
  if ($gem) { "jekyll" }
  elseif ($pkg) { "node" }
  else { "unknown" }

$workflows = Get-ChildItem (Join-Path $RepoRoot ".github\workflows") -Filter *.yml -ErrorAction SilentlyContinue

# Quick YAML sanity check: look for obvious corruption
$cfgFlags = @()
if ($cfg) {
  if ($cfg -match '^\s*\{\s*"$' -or $cfg -match '"scripts"\s*:') { $cfgFlags += "config_contains_json_like_blob" }
  if ([regex]::Matches($cfg, '(?m)^\s*url\s*:').Count -gt 1) { $cfgFlags += "duplicate_url_keys" }
  if ([regex]::Matches($cfg, '(?m)^\s*baseurl\s*:').Count -gt 1) { $cfgFlags += "duplicate_baseurl_keys" }
}

# CSS references scan (helps diagnose style issues without editing)
$cssRefs = @()
Get-ChildItem $RepoRoot -Recurse -Include *.html,*.md,*.liquid -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch "\\_site\\" } |
  ForEach-Object {
    $raw = Get-Content -Raw $_.FullName
    $matches = [regex]::Matches($raw, 'href\s*=\s*["'']([^"'']+\.css[^"'']*)["'']')
    foreach ($m in $matches) {
      $cssRefs += [PSCustomObject]@{ file=$_.FullName; href=$m.Groups[1].Value }
    }
  }

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$reportPath = Join-Path $reportDir "faithfrontier-audit-$stamp.txt"

@"
FaithFrontier Repo Audit
========================
RepoRoot: $RepoRoot
Stack: $stack
Has _config.yml: $(Test-Path $cfgPath)
Config Flags: $($cfgFlags -join ", ")
Workflow files:
$($workflows.FullName -join "`n")

CSS refs found: $($cssRefs.Count)
CSS refs sample:
$($cssRefs | Select-Object -First 20 | Format-Table -AutoSize | Out-String)
"@ | Set-Content -Encoding UTF8 $reportPath

Write-Host "Audit written: $reportPath"
