[CmdletBinding(SupportsShouldProcess=$true)]
param(
  [string]$RepoRoot = (Resolve-Path ".").Path
)

$ErrorActionPreference = "Stop"
Set-Location $RepoRoot

function Ensure-Line([string]$Path, [string]$Line) {
  if (-not (Test-Path $Path)) { New-Item -ItemType File -Path $Path | Out-Null }
  $raw = Get-Content $Path -Raw
  if ($raw -notmatch [regex]::Escape($Line)) {
    Add-Content -Path $Path -Value $Line
  }
}

Write-Host "`n== Fix Pipeline =="

# A) .gitattributes (LF policy)
if (-not (Test-Path ".gitattributes")) {
@"
* text=auto
.gitattributes text eol=lf
*.yml  text eol=lf
*.yaml text eol=lf
*.md   text eol=lf
*.html text eol=lf
*.css  text eol=lf
*.scss text eol=lf
*.js   text eol=lf
*.json text eol=lf
"@ | Set-Content -Path ".gitattributes" -Encoding ascii
  Write-Host "✔ Created .gitattributes"
} else {
  Ensure-Line ".gitattributes" "* text=auto"
  Ensure-Line ".gitattributes" ".gitattributes text eol=lf"
  Ensure-Line ".gitattributes" "*.yml  text eol=lf"
  Ensure-Line ".gitattributes" "*.yaml text eol=lf"
  Ensure-Line ".gitattributes" "*.md   text eol=lf"
  Ensure-Line ".gitattributes" "*.html text eol=lf"
  Ensure-Line ".gitattributes" "*.css  text eol=lf"
  Ensure-Line ".gitattributes" "*.scss text eol=lf"
  Ensure-Line ".gitattributes" "*.js   text eol=lf"
  Ensure-Line ".gitattributes" "*.json text eol=lf"
  Write-Host "✔ Updated .gitattributes"
}

# B) .gitignore hygiene (don’t track build artifacts / backups)
Ensure-Line ".gitignore" "_site/"
Ensure-Line ".gitignore" ".jekyll-cache/"
Ensure-Line ".gitignore" ".sass-cache/"
Ensure-Line ".gitignore" "*.bak"
Ensure-Line ".gitignore" "_backup/"

Write-Host "✔ Ensured .gitignore rules"

# C) Jekyll config: normalize keep_files and exclude backups
if (Test-Path "_config.yml") {
  $cfg = Get-Content "_config.yml" -Raw

  # normalize keep_files block to only CNAME + assets
  $cfg = [regex]::Replace($cfg, '(?ms)^keep_files:\s*\n(?:[ \t]*-\s*.*\n)*', @"
keep_files:
  - "CNAME"
  - "assets/"
"@)

  if ($cfg -notmatch '(?m)^exclude:\s*$') { $cfg += "`nexclude:`n" }

  # remove stray bak/backup lines anywhere
  $cfg = [regex]::Replace($cfg, '(?m)^[ \t]*-\s*"_backup\/"\s*$\r?\n?', '')
  $cfg = [regex]::Replace($cfg, '(?m)^[ \t]*-\s*_backup\/\s*$\r?\n?', '')
  $cfg = [regex]::Replace($cfg, '(?m)^[ \t]*-\s*"\*\.bak"\s*$\r?\n?', '')
  $cfg = [regex]::Replace($cfg, '(?m)^[ \t]*-\s*"\*\*\/\*\.bak"\s*$\r?\n?', '')

  # insert clean excludes right under exclude:
  $insert = @(
    '  - "_backup/"',
    '  - "**/*.bak"',
    '  - "*.bak"'
  ) -join "`n"

  $cfg = [regex]::Replace($cfg, '(?m)^exclude:\s*$',
    { param($m) $m.Value + "`n" + $insert },
    1
  )

  Set-Content "_config.yml" $cfg -Encoding utf8
  Write-Host "✔ Normalized _config.yml (keep_files/exclude)"
} else {
  Write-Host "⚠ No _config.yml found"
}

# D) Remove stray .bak files from disk (optional but recommended)
Get-ChildItem -Recurse -File -Filter "*.bak" -ErrorAction SilentlyContinue |
  Where-Object { $_.FullName -notmatch '\\\.git\\' } |
  ForEach-Object {
    Remove-Item $_.FullName -Force
  }
Write-Host "✔ Removed *.bak files from working tree (if any)"

# E) Renormalize per .gitattributes
git add --renormalize . | Out-Null
Write-Host "✔ Renormalized line endings (git add --renormalize .)"

Write-Host "`nDone. Review changes with: git status"
