/**
 * Consolidate Hardcoded Colors to CSS Variables
 * Replaces common hex colors with their CSS variable equivalents
 */

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Color mappings: hex value -> CSS variable
const colorMappings = {
  // Pure colors
  '#ffffff': 'var(--pure-white)',
  '#fff': 'var(--pure-white)',
  '#000000': 'var(--pure-black)',
  '#000': 'var(--pure-black)',
  
  // Grays
  '#666666': 'var(--gray-dark)',
  '#666': 'var(--gray-dark)',
  '#999999': 'var(--gray-mid)',
  '#999': 'var(--gray-mid)',
  '#cccccc': 'var(--border-gray)',
  '#ccc': 'var(--border-gray)',
  '#eeeeee': 'var(--border-gray-light)',
  '#eee': 'var(--border-gray-light)',
  '#f0f0f0': 'var(--gray-light)',
  
  // Text/Cream colors
  '#f9fafb': 'var(--cream-text)',
  
  // Success colors
  '#10b981': 'var(--success-green-bright)',
  '#4caf50': 'var(--success-green)',
  '#4CAF50': 'var(--success-green)',
  
  // Warning colors
  '#fbbf24': 'var(--warning-yellow-bright)',
  '#ffc107': 'var(--warning-yellow)',
  '#FFC107': 'var(--warning-yellow)',
  
  // Muted
  '#9e9e9e': 'var(--muted-gray)',
  '#9E9E9E': 'var(--muted-gray)',
  
  // Brand colors (already have variables but catch stragglers)
  '#d4a574': 'var(--accent-brass)',
  '#d4af37': 'var(--accent-brass)',
};

async function getAllCssFiles(dir) {
  const files = [];
  
  async function walk(directory) {
    const entries = await fs.readdir(directory, { withFileTypes: true });
    
    for (const entry of entries) {
      const fullPath = path.join(directory, entry.name);
      
      if (entry.isDirectory() && entry.name !== 'node_modules' && entry.name !== '_site') {
        await walk(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.css')) {
        files.push(fullPath);
      }
    }
  }
  
  await walk(dir);
  return files;
}

async function consolidateColorsInFile(filePath) {
  try {
    let content = await fs.readFile(filePath, 'utf8');
    let modified = false;
    
    // Replace each color mapping
    for (const [hex, variable] of Object.entries(colorMappings)) {
      // Skip if already using variable
      if (content.includes(`var(${hex})`)) continue;
      
      // Create regex that matches the hex but not when already in var()
      const regex = new RegExp(`(?<!var\\([^)]*)(${hex.replace('#', '#')})(?!\\))`, 'gi');
      
      if (regex.test(content)) {
        content = content.replace(regex, variable);
        modified = true;
      }
    }
    
    if (modified) {
      await fs.writeFile(filePath, content, 'utf8');
      return { file: filePath, modified: true };
    }
    
    return { file: filePath, modified: false };
  } catch (error) {
    return { file: filePath, error: error.message };
  }
}

async function main() {
  console.log('🎨 Consolidating hardcoded colors to CSS variables...\n');
  
  const projectRoot = path.resolve(__dirname, '..');
  const cssDir = path.join(projectRoot, 'assets', 'css');
  
  // Find all CSS files
  const cssFiles = await getAllCssFiles(cssDir);
  
  console.log(`Found ${cssFiles.length} CSS files to process\n`);
  
  const results = await Promise.all(
    cssFiles.map(file => consolidateColorsInFile(file))
  );
  
  const modified = results.filter(r => r.modified);
  const errors = results.filter(r => r.error);
  
  console.log('\n📊 Results:');
  console.log(`✅ Modified: ${modified.length} files`);
  console.log(`⚠️  Errors: ${errors.length} files`);
  console.log(`📝 Unchanged: ${results.length - modified.length - errors.length} files`);
  
  if (modified.length > 0) {
    console.log('\n✨ Modified files:');
    modified.forEach(r => {
      const relativePath = path.relative(projectRoot, r.file);
      console.log(`  - ${relativePath}`);
    });
  }
  
  if (errors.length > 0) {
    console.log('\n❌ Errors:');
    errors.forEach(r => {
      const relativePath = path.relative(projectRoot, r.file);
      console.log(`  - ${relativePath}: ${r.error}`);
    });
  }
  
  console.log('\n✅ Color consolidation complete!');
}

main().catch(console.error);
