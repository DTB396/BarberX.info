<#
Patches an existing US states SVG to:
- ensure each state path has data-state="XX"
- add CSS classes
- inject subtle overlay groups for relief + landmarks (non-interfering)
This does NOT create state geometry. You must supply a base SVG with state paths.
Recommended base: Wikimedia "Blank US Map (states only).svg" which uses state IDs (AL, AK, ...).
#>

param(
    [Parameter(Mandatory = $true)]
    [string]$SvgPath = ".\assets\svg\us-states.svg"
)

if (!(Test-Path $SvgPath)) { throw "SVG not found: $SvgPath" }

$svg = Get-Content $SvgPath -Raw

# Ensure basic stylesheet for hover and selection
if ($svg -notmatch "<style") {
    $style = @"
<style>
  .state { cursor: pointer; transition: opacity .12s ease; }
  .state:hover { opacity: .85; }
  .state--selected { opacity: .75; }
  #relief, #landmarks { pointer-events: none; opacity: .20; }
</style>
"@
    $svg = $svg -replace "<svg([^>]+)>", "<svg`$1>`n$style"
}

# Add data-state to paths whose id is a 2-letter code (common in Wikimedia file)
$svg = [regex]::Replace($svg, '<path([^>]*?)\sid="([A-Z]{2})"([^>]*?)>', {
        param($m)
        $pre = $m.Groups[1].Value
        $id = $m.Groups[2].Value
        $post = $m.Groups[3].Value

        # If already has data-state, keep it
        if ($m.Value -match 'data-state=') { return $m.Value }

        # Add class + data-state
        $updated = "<path$pre id=`"$id`" class=`"state`" data-state=`"$id`"$post>"
        return $updated
    })

# Inject subtle relief + landmark overlay groups near end (if not present)
if ($svg -notmatch 'id="relief"') {
    $overlay = @"
<g id="relief">
  <!-- subtle mountain bands (stylized, not geographic-precise) -->
  <path d="M120 210 C 240 140, 340 150, 460 210" fill="none" stroke="currentColor" stroke-width="10" opacity="0.10"/>
  <path d="M520 260 C 660 190, 820 200, 980 275" fill="none" stroke="currentColor" stroke-width="8" opacity="0.08"/>
</g>
<g id="landmarks">
  <!-- tiny, low-opacity icons: star markers (placeholder points) -->
  <circle cx="220" cy="280" r="6" fill="currentColor" opacity="0.10"/>
  <circle cx="700" cy="340" r="6" fill="currentColor" opacity="0.10"/>
  <circle cx="980" cy="390" r="6" fill="currentColor" opacity="0.10"/>
</g>
"@
    $svg = $svg -replace "</svg>\s*$", "$overlay`n</svg>"
}

Set-Content $SvgPath $svg -Encoding UTF8
Write-Host "Patched SVG: $SvgPath"
