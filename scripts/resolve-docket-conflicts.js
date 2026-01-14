#!/usr/bin/env node

/**
 * Resolve Git merge conflicts in docket YAML files
 * Chooses the newer format (from repair script) and removes duplicate /assets/ prefix
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

const docketDir = path.join(rootDir, '_data', 'docket');

console.log('🔧 Resolving merge conflicts in docket YAML files...\n');

const files = fs.readdirSync(docketDir).filter(f => f.endsWith('.yml'));

let totalConflicts = 0;
let totalFixed = 0;

for (const file of files) {
  const filePath = path.join(docketDir, file);
  let content = fs.readFileSync(filePath, 'utf8');
  
  if (!content.includes('<<<<<<< HEAD')) {
    continue;
  }
  
  console.log(`📝 Processing ${file}...`);
  
  let conflictCount = 0;
  
  // Resolve conflicts by taking the newer version (after =======)
  // and removing duplicate /assets/ prefix
  content = content.replace(
    /<<<<<<< HEAD\n  file: \/assets\/assets\/(.*?)\n=======\n  file: \/assets\/assets\/(.*?)\n>>>>>>> [^\n]+\n/g,
    (match, oldPath, newPath) => {
      conflictCount++;
      // Use the newer path (second one) and fix double /assets/
      return `  file: /assets/${newPath}\n`;
    }
  );
  
  // Handle nested conflicts (some files have double conflicts)
  content = content.replace(
    /<<<<<<< HEAD\n<<<<<<< HEAD\n  file: \/assets\/assets\/(.*?)\n=======\n  file: \/assets\/assets\/(.*?)\n>>>>>>> [^\n]+\n=======\n  file: \/assets\/assets\/(.*?)\n>>>>>>> [^\n]+\n/g,
    (match, old1, old2, newPath) => {
      conflictCount++;
      return `  file: /assets/${newPath}\n`;
    }
  );
  
  if (conflictCount > 0) {
    fs.writeFileSync(filePath, content, 'utf8');
    console.log(`  ✅ Resolved ${conflictCount} conflicts`);
    totalConflicts += conflictCount;
    totalFixed++;
  }
}

console.log(`\n✨ Complete!`);
console.log(`   Fixed ${totalConflicts} conflicts in ${totalFixed} files\n`);
