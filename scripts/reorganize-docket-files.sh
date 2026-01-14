#!/bin/bash
# Reorganize docket files from cases/<slug>/filings/ to assets/cases/<slug>/docket/

set -e

CASES_DIR="cases"
ASSETS_CASES_DIR="assets/cases"

echo "=== Creating assets/cases directory structure ==="
mkdir -p "$ASSETS_CASES_DIR"

echo ""
echo "=== Copying PDF files to new locations ==="

for case_dir in "$CASES_DIR"/*/; do
    if [ ! -d "$case_dir" ]; then
        continue
    fi
    
    slug=$(basename "$case_dir")
    filings_dir="$case_dir/filings"
    
    if [ ! -d "$filings_dir" ]; then
        echo "⊘ No filings directory for $slug"
        continue
    fi
    
    echo ""
    echo "📁 Processing $slug..."
    
    # Create target docket directory
    docket_dir="$ASSETS_CASES_DIR/$slug/docket"
    mkdir -p "$docket_dir"
    
    # Count PDFs
    pdf_count=$(find "$filings_dir" -maxdepth 1 -name "*.pdf" | wc -l)
    
    if [ "$pdf_count" -eq 0 ]; then
        echo "  ⊘ No PDF files found"
        continue
    fi
    
    # Copy PDFs
    cp -v "$filings_dir"/*.pdf "$docket_dir/" 2>/dev/null || true
    
    echo "  ✓ Copied $pdf_count PDF files"
done

echo ""
echo "=== Summary ==="
echo "Files have been copied to assets/cases/<slug>/docket/"
echo ""
echo "Next: Run the YAML update script to fix file paths"
