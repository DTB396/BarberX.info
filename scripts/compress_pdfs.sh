#!/usr/bin/env bash
# Usage:
#   Default: ./scripts/compress_pdfs.sh
#   Custom:  ./scripts/compress_pdfs.sh assets/pdfs assets/pdfs/compressed
#   Quality: GS_QUALITY=/screen ./scripts/compress_pdfs.sh

set -euo pipefail

if ! command -v gs >/dev/null 2>&1; then
  echo "Error: 'gs' (Ghostscript) is not installed. Install it with: sudo apt-get update && sudo apt-get install -y ghostscript" >&2
  exit 1
fi

input_dir=${1:-assets/pdfs}
output_dir=${2:-assets/pdfs/compressed}
quality=${GS_QUALITY:-/ebook}

mkdir -p "$output_dir"

shopt -s nullglob
pdfs=("${input_dir}"/*.pdf "${input_dir}"/*.PDF)
shopt -u nullglob

if [ ${#pdfs[@]} -eq 0 ]; then
  echo "No PDF files found in '$input_dir'."
  echo "Compressed files (if any) will be in: $output_dir"
  exit 0
fi

for pdf in "${pdfs[@]}"; do
  filename=$(basename "$pdf")
  base_name=${filename%.*}
  output_file="$output_dir/${base_name}-compressed.pdf"

  gs -sDEVICE=pdfwrite \
    -dCompatibilityLevel=1.4 \
    -dPDFSETTINGS="$quality" \
    -dNOPAUSE -dQUIET -dBATCH \
    -sOutputFile="$output_file" \
    "$pdf"

  original_size=$(stat -c%s "$pdf")
  compressed_size=$(stat -c%s "$output_file")

  if [ "$compressed_size" -ge "$original_size" ]; then
    echo "Warning: $filename not compressed (output not smaller). Original retained."
    rm -f "$output_file"
  else
    reduction=$((original_size - compressed_size))
    percent=$((reduction * 100 / original_size))
    echo "Compressed $filename: saved ${reduction} bytes (${percent}% reduction)."
  fi
done

echo "Compression complete. Check compressed PDFs in: $output_dir"
