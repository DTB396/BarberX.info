# Manual Fixes for Remaining HTML Issues
# Targets specific problematic lines that automated script couldn't fix

Write-Host "🔧 APPLYING MANUAL HTML FIXES" -ForegroundColor Cyan

# Fix thomas-becket line 32
$file = "_essays\2020-12-28-thomas-becket-proclamation.md"
if (Test-Path $file) {
    $lines = Get-Content $file
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match '<span class="meta__label">ISSUED ON:</span><span> DECEMBER</span><time>') {
            $lines[$i] = '<p><span class="meta__label">ISSUED ON:</span> DECEMBER 28, 2020</p>'
            Write-Host "✅ Fixed thomas-becket-proclamation.md line $($i+1)" -ForegroundColor Green
            break
        }
    }
    $lines | Set-Content $file -Encoding UTF8
}

# Fix faith-frontier-ministry-charter.md - close section tag
$file = "_essays\2025-06-13-faith-frontier-ministry-charter.md"
if (Test-Path $file) {
    $content = Get-Content $file -Raw
    # Find the unclosed section around line 125-145
    $lines = $content -split "`n"
    for ($i = 120; $i -lt 150; $i++) {
        if ($i -lt $lines.Count -and $lines[$i] -match '<h2>' -and $i -gt 120) {
            # Insert </section> before this h2
            $lines[$i] = "</section>`n`n" + $lines[$i]
            Write-Host "✅ Added closing section tag in faith-frontier-ministry-charter.md at line $($i+1)" -ForegroundColor Green
            break
        }
    }
    $lines -join "`n" | Set-Content $file -Encoding UTF8 -NoNewline
}

# Fix tiller-earth.md unclosed tags
$file = "_essays\2025-10-11-tiller-earth.md"
if (Test-Path $file) {
    $lines = Get-Content $file
    for ($i = 0; $i -lt $lines.Count; $i++) {
        # Find the footer closing around line 260
        if ($lines[$i] -match '</footer>' -and $i -gt 255 -and $i -lt 265) {
            # Add missing closing tags
            $lines[$i] = '</footer>'
            $lines[$i+1] = '</div>'
            $lines[$i+2] = '</div>'
            $lines[$i+3] = '</details>'
            Write-Host "✅ Added closing tags in tiller-earth.md at line $($i+1)" -ForegroundColor Green
            break
        }
    }
    $lines | Set-Content $file -Encoding UTF8
}

Write-Host "`n✅ Manual fixes complete!" -ForegroundColor Green
Write-Host "Run validation to verify: node scripts/validate-html-structure.js --scan _essays" -ForegroundColor Cyan
