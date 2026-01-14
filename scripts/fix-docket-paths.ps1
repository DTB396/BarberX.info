# Fix docket YAML file paths
# Converts absolute paths to relative filenames

Write-Host "=== Fixing Docket YAML File Paths ===" -ForegroundColor Cyan
Write-Host ""

$yamlFiles = Get-ChildItem -Path "_data\docket" -Filter "*.yml"
$fixedCount = 0
$totalChanges = 0

foreach ($yamlFile in $yamlFiles) {
    $slug = $yamlFile.BaseName
    $content = Get-Content $yamlFile.FullName -Raw
    
    # Check if this file needs fixing
    if ($content -match 'file:\s*/assets/cases/') {
        Write-Host "Fixing: $($yamlFile.Name)" -ForegroundColor Yellow
        
        # Pattern 1: /assets/cases/[slug]/docket/[filename].pdf -> [filename].pdf
        $originalContent = $content
        $content = $content -replace "file:\s*/assets/cases/$slug/docket/([^\s\n]+\.pdf)", 'file: $1'
        
        # Pattern 2: /assets/cases/[slug]/filings/[filename].pdf -> [filename].pdf
        $content = $content -replace "file:\s*/assets/cases/$slug/filings/([^\s\n]+\.pdf)", 'file: $1'
        
        # Pattern 3: Any other /assets/cases/ patterns
        $content = $content -replace "file:\s*/assets/cases/[^/]+/(docket|filings)/([^\s\n]+\.pdf)", 'file: $2'
        
        if ($content -ne $originalContent) {
            Set-Content -Path $yamlFile.FullName -Value $content -NoNewline
            $changes = ([regex]::Matches($originalContent, 'file:\s*/assets/cases/')).Count
            $totalChanges += $changes
            $fixedCount++
            Write-Host "  ✅ Fixed $changes path(s)" -ForegroundColor Green
        } else {
            Write-Host "  ⚠️  No changes made (pattern didn't match)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Skipped: $($yamlFile.Name) (already correct)" -ForegroundColor Gray
    }
}

Write-Host ""
Write-Host "=== Summary ===" -ForegroundColor Cyan
Write-Host "Files fixed: $fixedCount"
Write-Host "Total paths corrected: $totalChanges"
Write-Host ""
