# scripts/ff-00-lib-safe.ps1
Set-StrictMode -Version Latest

function Assert-GitRepo([string]$RepoRoot) {
  if (-not (Test-Path (Join-Path $RepoRoot ".git"))) {
    throw "Not a git repo (no .git). Run from repo root."
  }
}

function Get-GitStatusSummary([string]$RepoRoot) {
  Push-Location $RepoRoot
  try {
    $porcelain = git status --porcelain 2>$null
    $branch = (git rev-parse --abbrev-ref HEAD 2>$null)
    [PSCustomObject]@{
      Branch = $branch
      Dirty  = ([string]::IsNullOrWhiteSpace($porcelain) -eq $false)
      Count  = ($porcelain -split "`n" | Where-Object { $_.Trim() -ne "" }).Count
      Lines  = $porcelain
    }
  } finally { Pop-Location }
}

function Assert-CleanWorkingTree([string]$RepoRoot) {
  $st = Get-GitStatusSummary $RepoRoot
  if ($st.Dirty) {
    throw "Working tree is not clean (changes: $($st.Count)). Commit/stash first for safety."
  }
}

function New-FFBackup([string]$RepoRoot, [string]$Label) {
  $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
  $dir = Join-Path $RepoRoot "docs\reports"
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $backup = Join-Path $dir ("backup-$Label-$stamp")
  New-Item -ItemType Directory -Path $backup | Out-Null
  return $backup
}

function Copy-IfExists([string]$Path, [string]$DestDir) {
  if (Test-Path $Path) {
    Copy-Item -Force -Recurse -Path $Path -Destination $DestDir
  }
}

function Assert-ChangeLimit([int]$ChangedFiles, [int]$Limit, [string]$Hint) {
  if ($ChangedFiles -gt $Limit) {
    throw "Safety stop: would change $ChangedFiles files (limit $Limit). $Hint"
  }
}
