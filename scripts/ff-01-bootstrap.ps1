# scripts/ff-01-bootstrap.ps1
[CmdletBinding()]
param(
  [string]$RepoRoot = (Get-Location).Path
)

. (Join-Path $PSScriptRoot "ff-00-lib-safe.ps1")

$ErrorActionPreference = "Stop"
Assert-GitRepo $RepoRoot
Assert-CleanWorkingTree $RepoRoot

$paths = @(
  "scripts",
  "docs",
  "docs\reports",
  ".github\workflows"
)

foreach ($p in $paths) {
  $full = Join-Path $RepoRoot $p
  if (-not (Test-Path $full)) { New-Item -ItemType Directory -Path $full | Out-Null }
}

# Add .gitattributes only if missing (avoid disrupting working repo)
$gitattributes = Join-Path $RepoRoot ".gitattributes"
if (-not (Test-Path $gitattributes)) {
@"
* text=auto
*.ps1 text eol=crlf
*.sh  text eol=lf
*.yml text eol=lf
*.md  text eol=lf
*.html text eol=lf
*.css text eol=lf
*.scss text eol=lf
"@ | Set-Content -Encoding UTF8 $gitattributes
}

Write-Host "Bootstrap complete (safe). No existing files overwritten except optional new .gitattributes."
