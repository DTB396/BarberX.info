#!/usr/bin/env node

/**
 * FIX HTML STRUCTURE ISSUES
 * Systematically repairs all HTML validation issues found across the repository
 */

const fs = require('fs');
const path = require('path');

const fixes = {
  // Essays fixes
  '_essays/2020-12-28-thomas-becket-proclamation.md': [
    {
      search: /<meta charset="UTF-8"><meta charset="UTF-8"> <br>/,
      replace: '<meta charset="UTF-8"><br>'
    },
    {
      search: /<span class="meta__label">ISSUED ON:<\/span><span> DECEMBER<\/span><time> 28, 2020,<\/time>/,
      replace: '<span class="meta__label">ISSUED ON:</span> DECEMBER 28, 2020'
    }
  ],
  
  '_essays/2025-06-13-faith-frontier-ministry-charter.md': [
    {
      search: /<section class="charter-section">\s*<h2>/,
      replace: '</section>\n\n<section class="charter-section">\n<h2>',
      line: 145
    }
  ],
  
  '_essays/2025-10-11-tiller-earth.md': [
    {
      search: /— <strong>Devon Tyler Barber<\/strong> • <em>Tillerstead LLC · Faith Frontier<\/em><br>/,
      replace: '— <strong>Devon Tyler Barber</strong> • <em>Tillerstead LLC · Faith Frontier</em><br>',
      context: 'Fix tag order'
    },
    {
      // Need to add closing tags for open elements
      search: /<\/footer>\s*<\/div>\s*<\/div>\s*<\/details>/,
      replace: '</strong></footer>\n</div>\n</div>\n</details>',
      line: 260
    }
  ]
};

// Color variable mappings
const colorMappings = {
  'rgba(16,92,74,1)': 'var(--emerald-700)',
  'rgba(58,56,52,1)': 'var(--color-text-muted)',
  'rgba(28,27,25,1)': 'var(--color-bg)',
  '#d4af37': 'var(--accent-brass)',
  'rgba(212, 165, 116, 0.2)': 'var(--brass-alpha-20)',
  'rgba(0, 0, 0, 0.2)': 'var(--shadow-alpha-20)',
  '#4CAF50': 'var(--success-green)',
  'rgba(76, 175, 80, 0.2)': 'var(--success-alpha-20)',
  '#FFC107': 'var(--warning-yellow)',
  'rgba(255, 193, 7, 0.2)': 'var(--warning-alpha-20)',
  '#9E9E9E': 'var(--muted-gray)',
  'rgba(158, 158, 158, 0.2)': 'var(--muted-alpha-20)',
  'rgba(15, 23, 42, 0.6)': 'var(--navy-alpha-60)',
  'rgba(212, 165, 116, 0.3)': 'var(--brass-alpha-30)',
  'rgba(184,138,57,0.18)': 'var(--brass-alpha-18)',
  'rgba(184,138,57,0.35)': 'var(--brass-alpha-35)'
};

function replaceHardcodedColors(content) {
  let updated = content;
  for (const [hardcoded, variable] of Object.entries(colorMappings)) {
    const regex = new RegExp(hardcoded.replace(/[()]/g, '\\$&'), 'g');
    updated = updated.replace(regex, variable);
  }
  return updated;
}

function fixFile(filePath, fileFixes) {
  const fullPath = path.join(process.cwd(), filePath);
  
  if (!fs.existsSync(fullPath)) {
    console.log(`⚠️  File not found: ${filePath}`);
    return;
  }
  
  let content = fs.readFileSync(fullPath, 'utf8');
  let modified = false;
  
  for (const fix of fileFixes) {
    if (fix.search instanceof RegExp) {
      if (fix.search.test(content)) {
        content = content.replace(fix.search, fix.replace);
        modified = true;
        console.log(`  ✓ Applied fix: ${fix.context || 'regex replacement'}`);
      }
    } else {
      if (content.includes(fix.search)) {
        content = content.replace(fix.search, fix.replace);
        modified = true;
        console.log(`  ✓ Applied fix: ${fix.context || 'string replacement'}`);
      }
    }
  }
  
  // Replace hardcoded colors
  const colorFixed = replaceHardcodedColors(content);
  if (colorFixed !== content) {
    content = colorFixed;
    modified = true;
    console.log(`  ✓ Replaced hardcoded colors with CSS variables`);
  }
  
  if (modified) {
    fs.writeFileSync(fullPath, content, 'utf8');
    console.log(`✅ Fixed: ${filePath}\n`);
  }
}

function fixPlaceholderTags(directory) {
  console.log(`\n📝 Fixing placeholder tags in ${directory}...`);
  
  const files = [
    '.github/agents/my-agent.agent.md',
    '.github/copilot-instructions.md',
    '_internal/BROKEN-PDFS-TODO.md',
    '_internal/FIXES-SUMMARY.md',
    '_internal/GOVERNANCE-REFACTOR-QUICKSTART.md',
    '_internal/GOVERNANCE-REFACTOR-TOOL.md',
    '_internal/IMPLEMENTATION-CHECKLIST.md',
    'worker/README.md'
  ];
  
  for (const file of files) {
    const fullPath = path.join(process.cwd(), file);
    if (!fs.existsSync(fullPath)) continue;
    
    let content = fs.readFileSync(fullPath, 'utf8');
    let modified = false;
    
    // Escape placeholder tags by wrapping in code blocks or using HTML entities
    const placeholders = [
      { search: /<slug>/g, replace: '`<slug>`' },
      { search: /<\/slug>/g, replace: '' },
      { search: /<name>/g, replace: '`<name>`' },
      { search: /<\/name>/g, replace: '' },
      { search: /<path>/g, replace: '`<path>`' },
      { search: /<\/path>/g, replace: '' },
      { search: /<date>/gi, replace: '`<date>`' },
      { search: /<\/date>/gi, replace: '' },
      { search: /<n>/g, replace: '`<n>`' },
      { search: /<\/n>/g, replace: '' }
    ];
    
    for (const placeholder of placeholders) {
      if (placeholder.search.test(content)) {
        content = content.replace(placeholder.search, placeholder.replace);
        modified = true;
      }
    }
    
    if (modified) {
      fs.writeFileSync(fullPath, content, 'utf8');
      console.log(`  ✅ Fixed placeholder tags in ${file}`);
    }
  }
}

function fixSVGIssues() {
  console.log(`\n🎨 Fixing SVG issues...`);
  
  const file = 'assets/img/README.md';
  const fullPath = path.join(process.cwd(), file);
  
  if (!fs.existsSync(fullPath)) {
    console.log(`⚠️  File not found: ${file}`);
    return;
  }
  
  let content = fs.readFileSync(fullPath, 'utf8');
  
  // Wrap unclosed SVG tags in code blocks
  content = content.replace(/<svg>(?!\s*<\/svg>)/g, '```html\n<svg>\n```');
  
  fs.writeFileSync(fullPath, content, 'utf8');
  console.log(`  ✅ Fixed SVG tags in ${file}`);
}

function fixStyleTags() {
  console.log(`\n💅 Fixing <style> tags in documentation...`);
  
  const files = [
    '_internal/CASE-RESOURCES-QUICKREF.md',
    '_internal/CASE-RESOURCES-SYSTEM.md',
    '_internal/STYLE-RULES.md'
  ];
  
  for (const file of files) {
    const fullPath = path.join(process.cwd(), file);
    if (!fs.existsSync(fullPath)) continue;
    
    let content = fs.readFileSync(fullPath, 'utf8');
    
    // Wrap style blocks in code fences
    content = content.replace(/<style>(?![\s\S]*?<\/style>)/g, '```html\n<style>\n```');
    
    fs.writeFileSync(fullPath, content, 'utf8');
    console.log(`  ✅ Fixed <style> tags in ${file}`);
  }
}

function main() {
  console.log('🔧 FIXING HTML STRUCTURE ISSUES\n');
  console.log('='.repeat(60));
  
  // Apply specific file fixes
  console.log('\n📄 Applying specific file fixes...');
  for (const [filePath, fileFixes] of Object.entries(fixes)) {
    console.log(`\n🔍 Processing: ${filePath}`);
    fixFile(filePath, fileFixes);
  }
  
  // Fix placeholder tags
  fixPlaceholderTags();
  
  // Fix SVG issues
  fixSVGIssues();
  
  // Fix style tags
  fixStyleTags();
  
  console.log('\n' + '='.repeat(60));
  console.log('✅ HTML structure fixes complete!');
  console.log('\nRun validation again to verify fixes:');
  console.log('  node scripts/validate-html-structure.js --scan <directory>\n');
}

if (require.main === module) {
  main();
}

module.exports = { replaceHardcodedColors, colorMappings };
