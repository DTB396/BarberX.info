#!/bin/bash
# Simple bash script to reorganize docket files and update YAML paths
# No dependencies required - uses only standard Unix tools

set -e

echo "=== FaithFrontier Docket System Repair ==="
echo ""

# Step 1: Copy PDFs to new location
echo "Step 1: Copying PDF files..."
echo ""

declare -a cases=("a-000313-25" "atl-l-002794-25" "atl-l-002869-25" "atl-l-003252-25" "atl-dc-007956-25" "usdj-1-22-cv-06206" "usdj-1-25-cv-15641")

total_copied=0
total_skipped=0

for slug in "${cases[@]}"; do
    source_dir="cases/$slug/filings"
    dest_dir="assets/cases/$slug/docket"
    
    if [ ! -d "$source_dir" ]; then
        echo "⊘ No filings directory for $slug"
        continue
    fi
    
    # Count PDFs
    pdf_count=$(find "$source_dir" -maxdepth 1 -name "*.pdf" 2>/dev/null | wc -l)
    
    if [ "$pdf_count" -eq 0 ]; then
        echo "⊘ No PDF files in $slug"
        continue
    fi
    
    echo "📁 Processing $slug... ($pdf_count files)"
    
    # Ensure destination exists
    mkdir -p "$dest_dir"
    
    # Copy files
    for pdf in "$source_dir"/*.pdf; do
        if [ -f "$pdf" ]; then
            filename=$(basename "$pdf")
            dest_file="$dest_dir/$filename"
            
            if [ -f "$dest_file" ]; then
                echo "  ⊘ Skipped (exists): $filename"
                ((total_skipped++))
            else
                cp "$pdf" "$dest_file"
                echo "  ✓ Copied: $filename"
                ((total_copied++))
            fi
        fi
    done
done

echo ""
echo "📊 Copy Summary:"
echo "  • Copied: $total_copied"
echo "  • Skipped: $total_skipped"
echo ""

# Step 2: Update YAML files
echo "=================================================="
echo "Step 2: Updating YAML file paths..."
echo ""

docket_data_dir="_data/docket"
yaml_files_updated=0
paths_updated=0

for yaml_file in "$docket_data_dir"/*.yml; do
    if [ ! -f "$yaml_file" ]; then
        continue
    fi
    
    slug=$(basename "$yaml_file" .yml)
    echo "📝 Updating YAML for $slug..."
    
    # Create backup
    cp "$yaml_file" "$yaml_file.bak"
    
    # Count replacements
    count=0
    
    # Replace various path patterns
    # Pattern 1: /cases/<slug>/filings/ → /assets/cases/<slug>/docket/
    if grep -q "file: /cases/$slug/filings/" "$yaml_file" 2>/dev/null; then
        sed -i.tmp "s|file: /cases/$slug/filings/|file: /assets/cases/$slug/docket/|g" "$yaml_file"
        c=$(grep -o "file: /assets/cases/$slug/docket/" "$yaml_file" | wc -l)
        count=$((count + c))
        echo "  ✓ Updated $c paths: /cases/$slug/filings/ → /assets/cases/$slug/docket/"
    fi
    
    # Pattern 2: /cases/<slug>/ (no subdirectory) → /assets/cases/<slug>/docket/
    if grep -q "file: /cases/$slug/[^/]*\.pdf" "$yaml_file" 2>/dev/null; then
        sed -i.tmp "s|file: /cases/$slug/\([^/]*\.pdf\)|file: /assets/cases/$slug/docket/\1|g" "$yaml_file"
        c=$(grep -c "file: /assets/cases/$slug/docket/" "$yaml_file" || echo 0)
        echo "  ✓ Updated paths: /cases/$slug/ → /assets/cases/$slug>/docket/"
    fi
    
    # Pattern 3: /cases/<slug>/pcr/ → /assets/cases/<slug>/docket/
    if grep -q "file: /cases/$slug/pcr/" "$yaml_file" 2>/dev/null; then
        sed -i.tmp "s|file: /cases/$slug/pcr/|file: /assets/cases/$slug/docket/|g" "$yaml_file"
        echo "  ✓ Updated paths: /cases/$slug/pcr/ → /assets/cases/$slug/docket/"
    fi
    
    # Pattern 4: Special case for old ATL-24-001934 paths in a-000313-25
    if grep -q "file: /cases/atl-24-001934/pcr/" "$yaml_file" 2>/dev/null; then
        sed -i.tmp "s|file: /cases/atl-24-001934/pcr/|file: /assets/cases/$slug/docket/|g" "$yaml_file"
        echo "  ✓ Updated legacy paths: /cases/atl-24-001934/pcr/ → /assets/cases/$slug/docket/"
    fi
    
    # Clean up temp file
    rm -f "$yaml_file.tmp"
    
    # Check if file was actually modified
    if ! cmp -s "$yaml_file" "$yaml_file.bak"; then
        ((yaml_files_updated++))
        echo "  ✓ Saved $yaml_file with updates"
    else
        echo "  → No changes needed for $yaml_file"
    fi
    
    # Remove backup
    rm -f "$yaml_file.bak"
done

echo ""
echo "📊 YAML Update Summary:"
echo "  • Files updated: $yaml_files_updated"
echo ""

# Final summary
echo "=================================================="
echo "✅ Reorganization complete!"
echo ""
echo "Next steps:"
echo "1. Review changes: git status"
echo "2. Test Jekyll build: bundle exec jekyll build"
echo "3. Verify case pages display correctly"
echo "4. Commit changes if everything looks good"
echo ""
echo "Files copied to: assets/cases/<slug>/docket/"
echo "Original files remain in: cases/<slug>/filings/"
