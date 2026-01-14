# Fix All Case Docket PDF Links - Complete Update Script

Write-Host "`n╔═══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     FIX ALL CASE PDF LINKS - COMMIT & PUSH            ║" -ForegroundColor Cyan
Write-Host "╚═══════════════════════════════════════════════════════════╝`n" -ForegroundColor Cyan

cd "C:\Users\Devon Tyler\faithfrontier"

# Check git repo
if (-not (Test-Path ".git")) {
    Write-Host "ERROR: Not a git repository!" -ForegroundColor Red
    exit 1
}

Write-Host "Summary of Changes:" -ForegroundColor Yellow
Write-Host ""
Write-Host "Fixed 11 docket YAML files:" -ForegroundColor White
Write-Host "  • a-000313-25.yml" -ForegroundColor Gray
Write-Host "  • atl-22-002292.yml" -ForegroundColor Gray
Write-Host "  • atl-24-001934.yml" -ForegroundColor Gray
Write-Host "  • atl-dc-007956-25.yml" -ForegroundColor Gray
Write-Host "  • atl-l-002794-25.yml" -ForegroundColor Gray
Write-Host "  • atl-l-002869-25.yml" -ForegroundColor Gray
Write-Host "  • atl-l-003252-25.yml" -ForegroundColor Gray
Write-Host "  • barber-nj-pcr-2022.yml" -ForegroundColor Gray
Write-Host "  • street-crossing-pcr-appeal.yml" -ForegroundColor Gray
Write-Host "  • usdj-1-22-cv-06206.yml" -ForegroundColor Gray
Write-Host "  • usdj-1-25-cv-15641.yml" -ForegroundColor Gray
Write-Host ""
Write-Host "Enhanced README.md for ATL-L-003252-25" -ForegroundColor White
Write-Host ""
Write-Host "Change: /assets/cases/ → /cases/" -ForegroundColor Cyan
Write-Host ""

# Show git status
Write-Host "Git status:" -ForegroundColor Yellow
git status --short

Write-Host ""
$confirm = Read-Host "Commit and push all changes? (y/n)"

if ($confirm -eq 'y' -or $confirm -eq 'Y') {
    Write-Host "`nStaging all docket files..." -ForegroundColor Cyan
    git add _data/docket/*.yml
    git add cases/atl-l-003252-25/README.md
    git add fix-all-docket-paths.js
    git add DOCKET-PDF-LINK-FIX.md
    git add FAITHFRONTIER_DOCKET_FIX_COMPLETE.md
    
    Write-Host "Creating commit..." -ForegroundColor Cyan
    git commit -m "Fix PDF 404 errors across all cases: Update docket paths

Changed all docket YAML files from /assets/cases/ to /cases/ structure
to match actual built site paths in Jekyll.

Files fixed:
- a-000313-25.yml
- atl-22-002292.yml
- atl-24-001934.yml
- atl-dc-007956-25.yml
- atl-l-002794-25.yml
- atl-l-002869-25.yml
- atl-l-003252-25.yml
- barber-nj-pcr-2022.yml
- street-crossing-pcr-appeal.yml
- usdj-1-22-cv-06206.yml
- usdj-1-25-cv-15641.yml

Also enhanced ATL-L-003252-25 README with chronological docket
and document metadata.

This fixes 404 errors on all case pages at faithfrontier.org/cases/"
    
    Write-Host "`nPushing to remote..." -ForegroundColor Cyan
    git push
    
    Write-Host "`n✓ Changes pushed successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host "NEXT STEPS" -ForegroundColor Yellow
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "1. Wait 2-3 minutes for GitHub Pages to rebuild" -ForegroundColor White
    Write-Host ""
    Write-Host "2. Test these case pages:" -ForegroundColor White
    Write-Host "   • https://faithfrontier.org/cases/" -ForegroundColor Gray
    Write-Host "   • https://faithfrontier.org/cases/atl-l-003252-25/" -ForegroundColor Gray
    Write-Host "   • https://faithfrontier.org/cases/atl-l-002869-25/" -ForegroundColor Gray
    Write-Host "   • https://faithfrontier.org/cases/usdj-1-25-cv-15641/" -ForegroundColor Gray
    Write-Host ""
    Write-Host "3. Click on any PDF links - should work now!" -ForegroundColor White
    Write-Host ""
    Write-Host "═══════════════════════════════════════════════════════════" -ForegroundColor Cyan
    Write-Host ""
} else {
    Write-Host "`nCommit cancelled." -ForegroundColor Yellow
    Write-Host "Run this script again when ready to push." -ForegroundColor Gray
}
