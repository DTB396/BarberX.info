[CmdletBinding()]
param(
  [string]$RepoRoot = (Resolve-Path ".").Path
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

Write-Host "`n== Front matter guard =="

$targets = Get-ChildItem -Recurse -File -Include *.html,*.md,*.markdown,*.xml,*.json,*.yml,*.yaml |
  Where-Object { $_.FullName -notmatch "\\(_site|\.jekyll-cache|vendor|node_modules)\\" }

$bad = @()

foreach ($f in $targets) {
  $firstLine = (Get-Content -LiteralPath $f.FullName -TotalCount 1 -ErrorAction SilentlyContinue)
  if ($firstLine -match '^\s*[—–]\s*$' -or $firstLine -match '^\s*[—–]\s*layout\s*:') {
    $bad += $f.FullName
  }
}

if ($bad.Count -gt 0) {
  Write-Host "`n❌ Invalid front matter delimiter detected (—/–). These files will be copied as static:" -ForegroundColor Red
  $bad | ForEach-Object { Write-Host " - $_" }
  throw "Front matter guard failed. Replace smart dashes with '---'."
}

Write-Host "✔ Front matter guard OK"
