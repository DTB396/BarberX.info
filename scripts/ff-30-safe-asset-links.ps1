# scripts/ff-30-safe-asset-links.ps1
[CmdletBinding()]
param(
  [string]$RepoRoot = (Get-Location).Path,
  [int]$MaxFilesToChange = 10,
  [switch]$WhatIf
)

. (Join-Path $PSScriptRoot "ff-00-lib-safe.ps1")

$ErrorActionPreference = "Stop"
Assert-GitRepo $RepoRoot
Assert-CleanWorkingTree $RepoRoot

$backup = New-FFBackup $RepoRoot "assetlinks"

$targets = Get-ChildItem $RepoRoot -Recurse -Include *.html,*.md,*.liquid -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch "\\_site\\" }

$repls = @(
  @{ pattern = 'href\s*=\s*"\/(assets\/[^"]+)"'; replace = 'href="{{ ''/$1'' | relative_url }}"' },
  @{ pattern = 'src\s*=\s*"\/(assets\/[^"]+)"';  replace = 'src="{{ ''/$1'' | relative_url }}"' }
)

$changedFiles = @()

foreach ($f in $targets) {
  $raw = Get-Content -Raw $f.FullName
  $new = $raw
  foreach ($r in $repls) { $new = [regex]::Replace($new, $r.pattern, $r.replace) }

  if ($new -ne $raw) {
    $changedFiles += $f.FullName
    if (-not $WhatIf) {
      # backup only the files we touch
      $rel = $f.FullName.Substring($RepoRoot.Length).TrimStart("\","/")
      $destDir = Join-Path $backup (Split-Path $rel -Parent)
      if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
      Copy-Item -Force $f.FullName (Join-Path $destDir (Split-Path $rel -Leaf))
      Set-Content -Encoding UTF8 $f.FullName $new
    }
  }
}

Assert-ChangeLimit $changedFiles.Count $MaxFilesToChange "Increase -MaxFilesToChange only if you intend a broad refactor."

Write-Host "Would change: $($changedFiles.Count) file(s)."
if ($WhatIf) {
  $changedFiles | Select-Object -First 50 | ForEach-Object { Write-Host " - $_" }
  Write-Host "WhatIf: no changes written."
} else {
  Write-Host "Changes written. Backup created at: $backup"
}
