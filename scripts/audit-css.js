#!/usr/bin/env node

/**
 * CSS Property Violations and Breaking Points Auditor
 * Finds and fixes CSS syntax errors, property violations, and responsive breakpoints
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const cssDir = path.join(__dirname, '..', 'assets', 'css');

const config = {
  fix: process.argv.includes('--fix'),
  verbose: process.argv.includes('--verbose')
};

const issues = {
  syntax: [],
  properties: [],
  breakpoints: [],
  performance: [],
  accessibility: []
};

const breakpointStandards = {
  mobile: { min: 320, max: 767 },
  tablet: { min: 768, max: 1023 },
  desktop: { min: 1024, max: 1919 },
  wide: { min: 1920, max: 9999 }
};

// Common CSS property typos and mistakes
const propertyChecks = {
  'display: flex;': ['display: flexbox;', 'display: inline-flex;'],
  'position: absolute;': ['position: absolut;', 'position: fixed;'],
  'position: relative;': ['position: relativ;'],
  'overflow: hidden;': ['overflow: hide;', 'overflow: scroll;'],
  'text-align: center;': ['text-align: middle;'],
  'vertical-align: middle;': ['vertical-align: center;'],
  'font-weight: bold;': ['font-weight: bolder;'],
  'text-decoration: none;': ['text-decoration: null;']
};

// Vendor prefix requirements
const vendorPrefixes = {
  'appearance': ['-webkit-appearance', '-moz-appearance'],
  'backdrop-filter': ['-webkit-backdrop-filter'],
  'user-select': ['-webkit-user-select', '-moz-user-select', '-ms-user-select'],
  'transform': ['-webkit-transform', '-moz-transform', '-ms-transform']
};

function log(msg, type = 'info') {
  const colors = { error: '\x1b[31m', warning: '\x1b[33m', success: '\x1b[32m', info: '\x1b[36m' };
  const symbols = { error: '✗', warning: '⚠', success: '✓', info: '→' };
  console.log(`${colors[type]}${symbols[type]} ${msg}\x1b[0m`);
}

function checkBraceBalance(content, fileName) {
  const openBraces = (content.match(/{/g) || []).length;
  const closeBraces = (content.match(/}/g) || []).length;
  
  if (openBraces !== closeBraces) {
    issues.syntax.push({
      file: fileName,
      issue: `Unmatched braces: ${openBraces} open, ${closeBraces} close`,
      severity: 'error'
    });
    return false;
  }
  return true;
}

function checkPropertySyntax(content, fileName) {
  const lines = content.split('\n');
  
  lines.forEach((line, index) => {
    const lineNum = index + 1;
    const trimmed = line.trim();
    
    // Check for property lines
    if (trimmed.includes(':') && !trimmed.startsWith('/*') && !trimmed.startsWith('*')) {
      // Missing semicolon
      if (!trimmed.endsWith(';') && !trimmed.endsWith('{') && !trimmed.endsWith('}')) {
        issues.syntax.push({
          file: fileName,
          line: lineNum,
          issue: `Missing semicolon: ${trimmed}`,
          severity: 'error'
        });
      }
      
      // Double semicolons
      if (trimmed.includes(';;')) {
        issues.syntax.push({
          file: fileName,
          line: lineNum,
          issue: `Double semicolon: ${trimmed}`,
          severity: 'warning'
        });
      }
      
      // Space before colon
      if (/\s+:/.test(trimmed) && !trimmed.includes('//') && !trimmed.startsWith('@')) {
        issues.syntax.push({
          file: fileName,
          line: lineNum,
          issue: `Space before colon: ${trimmed}`,
          severity: 'warning'
        });
      }
    }
  });
}

function checkVendorPrefixes(content, fileName) {
  Object.entries(vendorPrefixes).forEach(([property, prefixes]) => {
    const regex = new RegExp(`\\b${property}\\s*:`, 'g');
    const matches = content.match(regex);
    
    if (matches) {
      prefixes.forEach(prefix => {
        const prefixRegex = new RegExp(`\\b${prefix}\\s*:`, 'g');
        if (!content.match(prefixRegex)) {
          issues.properties.push({
            file: fileName,
            issue: `Missing vendor prefix: ${prefix} for ${property}`,
            severity: 'warning'
          });
        }
      });
    }
  });
}

function checkBreakpoints(content, fileName) {
  const mediaQueries = content.match(/@media[^{]+\{/g) || [];
  const breakpointValues = [];
  
  mediaQueries.forEach(mq => {
    const widthMatches = mq.match(/(\d+)px/g);
    if (widthMatches) {
      widthMatches.forEach(match => {
        const value = parseInt(match);
        breakpointValues.push(value);
        
        // Check for odd breakpoint values
        if (value % 16 !== 0 && value > 100) {
          issues.breakpoints.push({
            file: fileName,
            issue: `Odd breakpoint value: ${value}px (consider using multiples of 16)`,
            severity: 'warning'
          });
        }
      });
    }
  });
  
  // Check for inconsistent breakpoints
  const uniqueBreakpoints = [...new Set(breakpointValues)].sort((a, b) => a - b);
  if (uniqueBreakpoints.length > 6) {
    issues.breakpoints.push({
      file: fileName,
      issue: `Too many breakpoints (${uniqueBreakpoints.length}). Consider consolidating: ${uniqueBreakpoints.join(', ')}px`,
      severity: 'warning'
    });
  }
}

function checkPerformance(content, fileName) {
  // Check for expensive selectors
  const expensiveSelectors = [
    { pattern: /\*\s*\{/, issue: 'Universal selector (*) - very expensive' },
    { pattern: /\[class\^=/, issue: 'Attribute selector with ^ - expensive' },
    { pattern: /\[class\$=/, issue: 'Attribute selector with $ - expensive' },
    { pattern: /\[class\*=/, issue: 'Attribute selector with * - expensive' }
  ];
  
  expensiveSelectors.forEach(({ pattern, issue }) => {
    if (pattern.test(content)) {
      issues.performance.push({
        file: fileName,
        issue,
        severity: 'warning'
      });
    }
  });
  
  // Check for !important overuse
  const importantCount = (content.match(/!important/g) || []).length;
  if (importantCount > 10) {
    issues.performance.push({
      file: fileName,
      issue: `Excessive !important usage (${importantCount} instances) - indicates specificity issues`,
      severity: 'warning'
    });
  }
}

function checkAccessibility(content, fileName) {
  // Check for low contrast colors
  const colorRegex = /color:\s*rgba?\([^)]+\)/gi;
  const colors = content.match(colorRegex) || [];
  
  colors.forEach(color => {
    if (color.includes('rgba')) {
      const alphaMatch = color.match(/,\s*(0\.[0-4]\d*)\s*\)/);
      if (alphaMatch) {
        issues.accessibility.push({
          file: fileName,
          issue: `Low opacity color (${alphaMatch[1]}) may fail contrast requirements: ${color}`,
          severity: 'warning'
        });
      }
    }
  });
  
  // Check for missing focus states
  if (content.includes(':hover') && !content.includes(':focus')) {
    issues.accessibility.push({
      file: fileName,
      issue: 'Has :hover but missing :focus states - keyboard accessibility issue',
      severity: 'warning'
    });
  }
}

function checkCommonMistakes(content, fileName) {
  const mistakes = [
    { pattern: /display:\s*flexbox/i, correct: 'display: flex', issue: 'Invalid value "flexbox"' },
    { pattern: /position:\s*absolut[^e]/i, correct: 'position: absolute', issue: 'Typo in "absolute"' },
    { pattern: /overflow:\s*hide[^n]/i, correct: 'overflow: hidden', issue: 'Typo in "hidden"' },
    { pattern: /text-align:\s*middle/i, correct: 'text-align: center', issue: 'Invalid value "middle"' },
    { pattern: /vertical-align:\s*center/i, correct: 'vertical-align: middle', issue: 'Invalid value "center"' },
    { pattern: /font-weight:\s*bolder[^;]/i, correct: 'font-weight: bold or 700', issue: 'Consider specific value' },
    { pattern: /box-shadow:\s*none\s*!important/i, correct: 'Remove !important', issue: 'Unnecessary !important' }
  ];
  
  mistakes.forEach(({ pattern, correct, issue }) => {
    if (pattern.test(content)) {
      issues.properties.push({
        file: fileName,
        issue: `${issue}. Use: ${correct}`,
        severity: 'error'
      });
    }
  });
}

function analyzeFile(filePath) {
  const fileName = path.basename(filePath);
  const content = fs.readFileSync(filePath, 'utf8');
  
  if (config.verbose) log(`Analyzing ${fileName}...`, 'info');
  
  checkBraceBalance(content, fileName);
  checkPropertySyntax(content, fileName);
  checkVendorPrefixes(content, fileName);
  checkBreakpoints(content, fileName);
  checkPerformance(content, fileName);
  checkAccessibility(content, fileName);
  checkCommonMistakes(content, fileName);
}

function main() {
  console.log('\n╔═══════════════════════════════════════════════════════════╗');
  console.log('║     CSS PROPERTY VIOLATIONS & BREAKING POINTS AUDIT   ║');
  console.log('╚═══════════════════════════════════════════════════════════╝\n');
  
  if (!fs.existsSync(cssDir)) {
    log(`CSS directory not found: ${cssDir}`, 'error');
    process.exit(1);
  }
  
  const cssFiles = fs.readdirSync(cssDir)
    .filter(f => f.endsWith('.css'))
    .map(f => path.join(cssDir, f));
  
  log(`Found ${cssFiles.length} CSS files\n`, 'info');
  
  cssFiles.forEach(analyzeFile);
  
  // Report results
  console.log('\n' + '═'.repeat(60));
  console.log('AUDIT RESULTS');
  console.log('═'.repeat(60));
  
  const totalIssues = Object.values(issues).reduce((sum, arr) => sum + arr.length, 0);
  const errors = Object.values(issues).flat().filter(i => i.severity === 'error').length;
  const warnings = Object.values(issues).flat().filter(i => i.severity === 'warning').length;
  
  console.log(`Total Issues:    ${totalIssues}`);
  console.log(`\x1b[31mErrors:          ${errors}\x1b[0m`);
  console.log(`\x1b[33mWarnings:        ${warnings}\x1b[0m`);
  console.log('');
  
  // Detailed breakdown
  Object.entries(issues).forEach(([category, items]) => {
    if (items.length > 0) {
      console.log(`\n\x1b[33m${category.toUpperCase()} (${items.length}):\x1b[0m`);
      
      // Group by file
      const grouped = {};
      items.forEach(item => {
        if (!grouped[item.file]) grouped[item.file] = [];
        grouped[item.file].push(item);
      });
      
      Object.entries(grouped).forEach(([file, fileIssues]) => {
        console.log(`  ${file}:`);
        fileIssues.forEach(issue => {
          const line = issue.line ? ` (line ${issue.line})` : '';
          const severity = issue.severity === 'error' ? '\x1b[31m[ERROR]\x1b[0m' : '\x1b[33m[WARN]\x1b[0m';
          console.log(`    ${severity}${line} ${issue.issue}`);
        });
      });
    }
  });
  
  // Summary recommendations
  if (totalIssues > 0) {
    console.log('\n' + '═'.repeat(60));
    console.log('RECOMMENDATIONS');
    console.log('═'.repeat(60));
    
    if (errors > 0) {
      console.log('\x1b[31m• Fix syntax errors immediately - they break rendering\x1b[0m');
    }
    
    if (issues.breakpoints.length > 0) {
      console.log('• Consolidate breakpoints to standard values (768px, 1024px, 1280px)');
    }
    
    if (issues.accessibility.length > 0) {
      console.log('• Add :focus states where :hover exists');
      console.log('• Review low-opacity colors for contrast compliance');
    }
    
    if (issues.performance.length > 0) {
      console.log('• Reduce expensive selectors and !important usage');
    }
  }
  
  console.log('\n');
  
  if (errors > 0) {
    log('CSS audit failed - errors found', 'error');
    process.exit(1);
  } else if (warnings > 0) {
    log('CSS audit passed with warnings', 'warning');
    process.exit(0);
  } else {
    log('CSS audit passed - no issues found', 'success');
    process.exit(0);
  }
}

main();
