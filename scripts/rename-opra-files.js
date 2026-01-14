#!/usr/bin/env node

/**
 * Rename .txt files in _opra to their proper extensions
 * - *.md.txt → *.md
 * - *.yml.txt → *.yml
 * - correspondence.txt stays as .txt (not a structured file)
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const opraDir = path.join(__dirname, '../_opra');

function renameFilesRecursive(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    
    if (entry.isDirectory()) {
      renameFilesRecursive(fullPath);
    } else if (entry.isFile()) {
      // Rename .md.txt to .md
      if (entry.name.endsWith('.md.txt')) {
        const newPath = fullPath.replace(/\.md\.txt$/, '.md');
        console.log(`Renaming: ${path.relative(opraDir, fullPath)} → ${path.relative(opraDir, newPath)}`);
        fs.renameSync(fullPath, newPath);
      }
      // Rename .yml.txt to .yml
      else if (entry.name.endsWith('.yml.txt')) {
        const newPath = fullPath.replace(/\.yml\.txt$/, '.yml');
        console.log(`Renaming: ${path.relative(opraDir, fullPath)} → ${path.relative(opraDir, newPath)}`);
        fs.renameSync(fullPath, newPath);
      }
      // Keep correspondence.txt as is (it's just a placeholder)
    }
  }
}

console.log('Renaming OPRA files from .txt to proper extensions...\n');
renameFilesRecursive(opraDir);
console.log('\nDone! OPRA files renamed successfully.');
