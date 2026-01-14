# Quick Commit Script for Docket PDF Fix

Write-Host "`n=== Docket PDF Link Fix - Commit Changes ===" -ForegroundColor Cyan
Write-Host ""

cd "C:\Users\Devon Tyler\faithfrontier"

# Check if git repo
if (-not (Test-Path ".git")) {
    Write-Host "ERROR: Not a git repository!" -ForegroundColor Red
    exit 1
}

Write-Host "Files to commit:" -ForegroundColor Yellow
Write-Host "  - cases/atl-l-003252-25/README.md" -ForegroundColor White
Write-Host "  - _data/docket/atl-l-003252-25.yml" -ForegroundColor White
Write-Host "  - DOCKET-PDF-LINK-FIX.md (documentation)" -ForegroundColor White
Write-Host ""

# Show git status
Write-Host "Current git status:" -ForegroundColor Yellow
git status --short

Write-Host ""
$confirm = Read-Host "Commit and push these changes? (y/n)"

if ($confirm -eq 'y' -or $confirm -eq 'Y') {
    Write-Host "`nStaging files..." -ForegroundColor Cyan
    git add cases/atl-l-003252-25/README.md
    git add _data/docket/atl-l-003252-25.yml
    git add DOCKET-PDF-LINK-FIX.md
    
    Write-Host "Creating commit..." -ForegroundColor Cyan
    git commit -m "Fix PDF 404 errors: Update docket paths and merge with README

- Changed docket YAML paths from /assets/cases/ to /cases/ structure
- Merged docket metadata into README with chronological organization
- Added document types and improved formatting
- Included December 20, 2025 latest filings
- Added URL encoding for filenames with spaces
- Created documentation in DOCKET-PDF-LINK-FIX.md"
    
    Write-Host "`nPushing to remote..." -ForegroundColor Cyan
    git push
    
    Write-Host "`n✓ Changes pushed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Wait 2-3 minutes for GitHub Pages to rebuild"
    Write-Host "  2. Visit: https://faithfrontier.org/cases/atl-l-003252-25/"
    Write-Host "  3. Verify PDF links work"
    Write-Host ""
} else {
    Write-Host "`nCommit cancelled." -ForegroundColor Yellow
}
