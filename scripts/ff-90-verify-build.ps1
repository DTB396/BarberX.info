# scripts/ff-90-verify-build.ps1
[CmdletBinding()]
param(
  [string]$RepoRoot = (Get-Location).Path
)

. (Join-Path $PSScriptRoot "ff-00-lib-safe.ps1")

$ErrorActionPreference = "Stop"
Assert-GitRepo $RepoRoot

Push-Location $RepoRoot
try {
  $hasGem = Test-Path "Gemfile"
  $hasPkg = Test-Path "package.json"

  if ($hasPkg) {
    Write-Host "npm ci..."
    npm ci
  }

  if ($hasGem) {
    Write-Host "bundle install..."
    bundle install
    Write-Host "jekyll build..."
    bundle exec jekyll build --trace
  } else {
    Write-Host "No Gemfile found; skipping Jekyll build."
  }

  Write-Host "OK: verification complete."
} finally {
  Pop-Location
}
