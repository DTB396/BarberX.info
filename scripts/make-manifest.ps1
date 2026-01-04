param(
  [Parameter(Mandatory=$true)][string]$Root,
  [Parameter(Mandatory=$true)][string]$OutDir,
  [string]$BaseUrl = "https://faithfrontier.org/cases"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path -LiteralPath $Root)) {
  throw "Root path not found: $Root"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$items = Get-ChildItem -LiteralPath $Root -Directory | ForEach-Object {
  $caseDir = $_.FullName
  $caseId  = $_.Name

  Get-ChildItem -LiteralPath $caseDir -Filter *.pdf -Recurse -File | ForEach-Object {
    $fileName = $_.Name

    [PSCustomObject]@{
      caseId    = $caseId
      caseTitle = $caseId
      docId     = $fileName
      docTitle  = $fileName
      docType   = "document"
      date      = ""
      url       = ("{0}/{1}/{2}" -f $BaseUrl.TrimEnd('/'), $caseId, $fileName)
      tags      = @("Faith Frontier","Cases")
      text      = ""
      snippet   = ""
    }
  }
}

$manifestPath = Join-Path $OutDir "manifest.json"
$items | ConvertTo-Json -Depth 7 | Set-Content -Encoding UTF8 -LiteralPath $manifestPath

Write-Host "Wrote $manifestPath with $($items.Count) PDFs"
