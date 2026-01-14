# Clean up docket YAML files - remove entries for non-existent PDFs
# Run this after fixing paths to remove orphaned entries

Write-Host "=== Cleaning up docket YAML entries ===" -ForegroundColor Cyan
Write-Host ""

$yamlFiles = Get-ChildItem -Path "_data\docket" -Filter "*.yml"
$removedCount = 0

foreach ($yamlFile in $yamlFiles) {
    $slug = $yamlFile.BaseName
    $filingsDir = "cases\$slug\filings"
    
    if (-not (Test-Path $filingsDir)) {
        Write-Host "Skipping $slug - no filings directory" -ForegroundColor Gray
        continue
    }
    
    $content = Get-Content $yamlFile.FullName -Raw
    $lines = Get-Content $yamlFile.FullName
    
    # Get all actual PDFs in the filings directory
    $actualPdfs = Get-ChildItem -Path $filingsDir -Filter "*.pdf" | Select-Object -ExpandProperty Name
    
    # Find file: lines that reference non-existent PDFs
    $fileMatches = [regex]::Matches($content, 'file:\s*([^\s\n]+\.pdf)')
    $missingFiles = @()
    
    foreach ($match in $fileMatches) {
        $fileName = $match.Groups[1].Value
        if ($fileName -and $fileName -ne '' -and $fileName -notmatch '^/' -and $actualPdfs -notcontains $fileName) {
            $missingFiles += $fileName
        }
    }
    
    if ($missingFiles.Count -gt 0) {
        Write-Host "$($yamlFile.Name) - Found $($missingFiles.Count) missing PDF(s)" -ForegroundColor Yellow
        foreach ($missing in $missingFiles) {
            Write-Host "  ❌ $missing" -ForegroundColor Red
        }
        $removedCount += $missingFiles.Count
    } else {
        Write-Host "$($yamlFile.Name) - All files exist ✅" -ForegroundColor Green
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Total entries referencing missing PDFs: $removedCount"
Write-Host ""
Write-Host "Note: This is a diagnostic script. To actually remove entries," -ForegroundColor Yellow
Write-Host "manually edit the YAML files or create removal logic." -ForegroundColor Yellow
