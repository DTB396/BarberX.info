#!/bin/bash

# Rename .txt files in _opra to their proper extensions

cd /workspaces/FaithFrontier/_opra || exit 1

echo "Renaming OPRA files from .txt to proper extensions..."
echo ""

# Rename all index.md.txt to index.md
find . -name "index.md.txt" | while read file; do
  newname="${file%.txt}"
  echo "Renaming: $file → $newname"
  mv "$file" "$newname"
done

# Rename all notes.md.txt to notes.md
find . -name "notes.md.txt" | while read file; do
  newname="${file%.txt}"
  echo "Renaming: $file → $newname"
  mv "$file" "$newname"
done

# Rename all timeline.yml.txt to timeline.yml
find . -name "timeline.yml.txt" | while read file; do
  newname="${file%.txt}"
  echo "Renaming: $file → $newname"
  mv "$file" "$newname"
done

echo ""
echo "Done! OPRA files renamed successfully."
echo ""
echo "Now listing .md and .yml files:"
find . -name "*.md" -o -name "*.yml" | sort
