[CmdletBinding()]
param(
  [switch]$Fix,              # rewrite broken links where we can safely infer the target
  [switch]$RunNodeAudits,     # run existing node scripts after scanning
  [switch]$VerboseReport      # more detail in report
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Repo-Root {
  $here = Resolve-Path (Join-Path $PSScriptRoot "..")
  return $here.Path
}

function Is-ExternalLink([string]$url) {
  if ([string]::IsNullOrWhiteSpace($url)) { return $false }
  return ($url -match '^(https?:)?//') -or ($url -match '^mailto:') -or ($url -match '^tel:')
}

function Normalize-LinkPath([string]$url) {
  # Strip query/hash for filesystem check.
  $u = $url
  $u = $u -replace '\?.*$', ''
  $u = $u -replace '\#.*$', ''
  return $u
}

function Get-MarkdownLinks([string]$text) {
  # Captures: [label](url) and ![alt](url)
  $rx = [regex]::new('(?m)(?<!\!)\[[^\]]*\]\(([^)]+)\)|\!\[[^\]]*\]\(([^)]+)\)')
  $matches = $rx.Matches($text)
  $urls = New-Object System.Collections.Generic.List[string]
  foreach ($m in $matches) {
    # group1 or group2 depending on which alternative matched
    $u = $m.Groups[1].Value
    if ([string]::IsNullOrWhiteSpace($u)) { $u = $m.Groups[2].Value }
    if (-not [string]::IsNullOrWhiteSpace($u)) {
      $urls.Add($u.Trim())
    }
  }
  return $urls
}

function Resolve-LocalTarget {
  param(
    [string]$RepoRoot,
    [string]$FromFile,
    [string]$Url
  )

  $raw = $Url.Trim().Trim('"').Trim("'")
  $u = Normalize-LinkPath $raw

  if (Is-ExternalLink $u) { return $null }

  # Skip anchors only
  if ($u -match '^\#') { return $null }

  # Absolute site-root paths (Jekyll-style): /assets/...
  if ($u.StartsWith("/")) {
    $p = Join-Path $RepoRoot ($u.TrimStart("/") -replace '/', '\')
    return $p
  }

  # Relative paths: resolve from the markdown file directory
  $fromDir = Split-Path -Parent $FromFile
  $p2 = Join-Path $fromDir ($u -replace '/', '\')
  return $p2
}

function Find-FileByName {
  param(
    [string]$RepoRoot,
    [string]$FileName
  )
  if ([string]::IsNullOrWhiteSpace($FileName)) { return @() }

  # Prefer typical publishable locations
  $preferredRoots = @(
    (Join-Path $RepoRoot "assets"),
    (Join-Path $RepoRoot "files"),
    (Join-Path $RepoRoot "uploads"),
    (Join-Path $RepoRoot "docs"),
    (Join-Path $RepoRoot "_cases"),
    (Join-Path $RepoRoot "_filings"),
    $RepoRoot
  ) | Where-Object { Test-Path $_ }

  $hits = New-Object System.Collections.Generic.List[string]

  foreach ($r in $preferredRoots) {
    $found = Get-ChildItem -Path $r -Recurse -File -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -ieq $FileName } |
      Select-Object -ExpandProperty FullName
    foreach ($f in $found) { $hits.Add($f) }
    if ($hits.Count -gt 0) { break }
  }

  # Fallback: whole repo
  if ($hits.Count -eq 0) {
    $found2 = Get-ChildItem -Path $RepoRoot -Recurse -File -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -ieq $FileName } |
      Select-Object -ExpandProperty FullName
    foreach ($f in $found2) { $hits.Add($f) }
  }

  return $hits.ToArray()
}

function To-SitePath {
  param(
    [string]$RepoRoot,
    [string]$FullPath
  )
  # Convert absolute file path -> site path best effort.
  $rel = Resolve-Path $FullPath
  $relStr = $rel.Path.Substring($RepoRoot.Length).TrimStart('\')
  $site = "/" + ($relStr -replace '\\','/')
  return $site
}

$ROOT = Repo-Root

Write-Host ""
Write-Host "================================================================================"
Write-Host "FaithFrontier: Harmonize links + case/docket integrity scan"
Write-Host "Repo root: $ROOT"
Write-Host "Fix mode: $Fix"
Write-Host "================================================================================"
Write-Host ""

# Basic structure checks
$must = @("_cases","_data","assets")
foreach ($m in $must) {
  $p = Join-Path $ROOT $m
  if (-not (Test-Path $p)) {
    Write-Warning "Missing expected folder: $m  (path: $p)"
  }
}

# Collect markdown files (cases first)
$mdFiles = New-Object System.Collections.Generic.List[string]
$casesDir = Join-Path $ROOT "_cases"
if (Test-Path $casesDir) {
  Get-ChildItem -Path $casesDir -Recurse -Filter *.md -File | ForEach-Object { $mdFiles.Add($_.FullName) }
}

# Also scan these common docs locations
$extraScan = @(
  (Join-Path $ROOT "docs"),
  (Join-Path $ROOT "_pages"),
  (Join-Path $ROOT "_posts")
) | Where-Object { Test-Path $_ }

foreach ($d in $extraScan) {
  Get-ChildItem -Path $d -Recurse -Filter *.md -File | ForEach-Object { $mdFiles.Add($_.FullName) }
}

if ($mdFiles.Count -eq 0) {
  throw "No markdown files found in _cases/docs/_pages/_posts. Check your repo structure."
}

# Report objects
$broken = New-Object System.Collections.Generic.List[object]
$fixed  = New-Object System.Collections.Generic.List[object]

foreach ($f in $mdFiles | Sort-Object) {
  $txt = Get-Content -LiteralPath $f -Raw -ErrorAction Stop
  $urls = Get-MarkdownLinks $txt
  if ($urls.Count -eq 0) { continue }

  foreach ($url in $urls) {
    $u = $url.Trim()
    if (Is-ExternalLink $u) { continue }

    # Only focus on file-like links (pdf + common attachments)
    $uNorm = Normalize-LinkPath $u
    if ($uNorm -notmatch '\.(pdf|png|jpg|jpeg|webp|svg|zip)$') { continue }

    $target = Resolve-LocalTarget -RepoRoot $ROOT -FromFile $f -Url $u
    if ($null -eq $target) { continue }

    $exists = Test-Path -LiteralPath $target
    if ($exists) { continue }

    # If missing, try to locate by filename and optionally rewrite link
    $fileName = [System.IO.Path]::GetFileName(($uNorm -replace '/', '\'))
    $candidates = Find-FileByName -RepoRoot $ROOT -FileName $fileName

    $item = [pscustomobject]@{
      FromFile   = $f
      Link       = $u
      LinkNorm   = $uNorm
      ResolvedFS = $target
      FoundByNameCount = $candidates.Count
      FoundByName = ($candidates -join "; ")
    }
    $broken.Add($item)

    if ($Fix -and $candidates.Count -ge 1) {
      # Choose first candidate, but prefer publishable paths
      $best = $candidates | Select-Object -First 1
      $newSitePath = To-SitePath -RepoRoot $ROOT -FullPath $best

      # Rewrite exactly the url substring inside parentheses
      $escapedOld = [regex]::Escape($u)
      $newTxt = [regex]::Replace($txt, "\(($escapedOld)\)", "($newSitePath)")

      if ($newTxt -ne $txt) {
        # Backup once per file
        $bak = "$f.bak"
        if (-not (Test-Path $bak)) {
          Copy-Item -LiteralPath $f -Destination $bak -Force
        }
        Set-Content -LiteralPath $f -Value $newTxt -Encoding utf8
        $fixed.Add([pscustomobject]@{
          FromFile = $f
          OldLink  = $u
          NewLink  = $newSitePath
          Target   = $best
        })
        # Update in-memory text for subsequent replacements
        $txt = $newTxt
      }
    }
  }
}

# Write reports
$reportsDir = Join-Path $ROOT "reports"
New-Item -ItemType Directory -Force -Path $reportsDir | Out-Null

$brokenPath = Join-Path $reportsDir "broken-links.report.json"
$fixedPath  = Join-Path $reportsDir "fixed-links.report.json"
$summaryPath = Join-Path $reportsDir "broken-links.summary.txt"

($broken | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $brokenPath -Encoding utf8
($fixed  | ConvertTo-Json -Depth 6) | Set-Content -LiteralPath $fixedPath  -Encoding utf8

$summary = New-Object System.Collections.Generic.List[string]
$summary.Add("Broken local attachment links found: $($broken.Count)")
$summary.Add("Auto-fixed links written: $($fixed.Count)")
$summary.Add("")
if ($broken.Count -gt 0) {
  $grouped = $broken | Group-Object FromFile | Sort-Object Count -Descending
  $summary.Add("Top files with broken links:")
  foreach ($g in $grouped | Select-Object -First 20) {
    $summary.Add(("  {0}  ({1})" -f $g.Name, $g.Count))
  }
}
$summary | Set-Content -LiteralPath $summaryPath -Encoding utf8

Write-Host "Broken links: $($broken.Count)"
Write-Host "Auto-fixed:   $($fixed.Count)"
Write-Host "Reports:"
Write-Host "  $summaryPath"
Write-Host "  $brokenPath"
Write-Host "  $fixedPath"
Write-Host ""

# Optional: run node audits you already have
if ($RunNodeAudits) {
  $node = Get-Command node -ErrorAction SilentlyContinue
  if (-not $node) {
    Write-Warning "Node not found. Skipping node audits."
  } else {
    $scripts = @(
      "check-site-links.js",
      "check-pdf-links.js",
      "check-pdf-health.js",
      "validate-everything.js"
    )

    foreach ($s in $scripts) {
      $p = Join-Path $ROOT ("scripts\" + $s)
      if (Test-Path $p) {
        Write-Host "Running: node scripts/$s"
        & node $p
        Write-Host ""
      } else {
        if ($VerboseReport) { Write-Warning "Missing node script: $p" }
      }
    }
  }
}

Write-Host "Done."
