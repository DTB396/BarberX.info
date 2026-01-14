/**
 * Comprehensive Responsive Audit for Faith Frontier
 * Scans ALL files for responsive issues and generates fix recommendations
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const config = {
  rootDir: path.join(__dirname, '..'),
  extensions: ['.html', '.md', '.css'],
  excludeDirs: ['node_modules', '.git', '_site'],
  
  // Responsive standards
  maxWidths: {
    reading: '65ch',        // Optimal reading width
    content: '1200px',      // Max content container
    narrow: '900px',        // Narrow content sections
    wide: '1400px'          // Wide sections (rarely used)
  },
  
  breakpoints: {
    mobile: '640px',
    tablet: '768px',
    desktop: '1024px',
    wide: '1280px'
  }
};

// Issues tracker
const issues = {
  hardcodedWidths: [],
  missingMediaQueries: [],
  inlineMaxWidths: [],
  containerIssues: [],
  viewportIssues: []
};

/**
 * Recursively scan directory
 */
function scanDirectory(dir) {
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    
    if (entry.isDirectory()) {
      if (!config.excludeDirs.includes(entry.name)) {
        scanDirectory(fullPath);
      }
    } else if (entry.isFile()) {
      const ext = path.extname(entry.name);
      if (config.extensions.includes(ext)) {
        analyzeFile(fullPath);
      }
    }
  }
}

/**
 * Analyze individual file
 */
function analyzeFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const relativePath = path.relative(config.rootDir, filePath);
  
  // Check for hardcoded pixel widths in HTML/MD
  if (filePath.endsWith('.html') || filePath.endsWith('.md')) {
    // Inline max-width without clamp or responsive units
    const inlineMaxWidth = /max-width:\s*(\d{3,4})px(?![^<]*clamp)/g;
    let match;
    while ((match = inlineMaxWidth.exec(content)) !== null) {
      issues.inlineMaxWidths.push({
        file: relativePath,
        line: getLineNumber(content, match.index),
        value: match[1] + 'px',
        context: getContext(content, match.index)
      });
    }
    
    // Container without max-width
    const containerPattern = /<div[^>]*class="container"[^>]*>/g;
    while ((match = containerPattern.exec(content)) !== null) {
      if (!match[0].includes('max-width')) {
        issues.containerIssues.push({
          file: relativePath,
          line: getLineNumber(content, match.index),
          issue: 'Container without max-width',
          context: getContext(content, match.index)
        });
      }
    }
  }
  
  // Check CSS files
  if (filePath.endsWith('.css')) {
    // Fixed widths without media queries nearby
    const fixedWidthPattern = /max-width:\s*(\d{3,4})px;/g;
    let cssMatch;
    while ((cssMatch = fixedWidthPattern.exec(content)) !== null) {
      const nearbyContent = content.substring(Math.max(0, cssMatch.index - 500), cssMatch.index + 500);
      if (!nearbyContent.includes('@media')) {
        issues.hardcodedWidths.push({
          file: relativePath,
          line: getLineNumber(content, cssMatch.index),
          value: cssMatch[1] + 'px',
          context: getContext(content, cssMatch.index)
        });
      }
    }
    
    // Check for missing mobile breakpoints
    if (content.includes('max-width') && !content.includes('@media (max-width: 768px)')) {
      issues.missingMediaQueries.push({
        file: relativePath,
        issue: 'Has max-width but no tablet breakpoint (@media max-width: 768px)'
      });
    }
  }
}

/**
 * Get line number from character index
 */
function getLineNumber(content, index) {
  return content.substring(0, index).split('\n').length;
}

/**
 * Get context around match
 */
function getContext(content, index, chars = 100) {
  const start = Math.max(0, index - chars);
  const end = Math.min(content.length, index + chars);
  return content.substring(start, end).trim();
}

/**
 * Generate report
 */
function generateReport() {
  let report = '# COMPREHENSIVE RESPONSIVE AUDIT REPORT\n\n';
  report += `Generated: ${new Date().toISOString()}\n\n`;
  
  report += '## SUMMARY\n\n';
  report += `- Inline max-widths: ${issues.inlineMaxWidths.length}\n`;
  report += `- Hardcoded widths in CSS: ${issues.hardcodedWidths.length}\n`;
  report += `- Container issues: ${issues.containerIssues.length}\n`;
  report += `- Missing media queries: ${issues.missingMediaQueries.length}\n\n`;
  
  if (issues.inlineMaxWidths.length > 0) {
    report += '## INLINE MAX-WIDTH ISSUES\n\n';
    issues.inlineMaxWidths.forEach(issue => {
      report += `### ${issue.file}:${issue.line}\n`;
      report += `**Current:** \`${issue.value}\`\n`;
      report += `**Recommendation:** Use \`max-width: min(${issue.value}, 95vw)\` or clamp\n`;
      report += `**Context:** \`${issue.context.substring(0, 80)}...\`\n\n`;
    });
  }
  
  if (issues.hardcodedWidths.length > 0) {
    report += '## HARDCODED CSS WIDTHS\n\n';
    issues.hardcodedWidths.forEach(issue => {
      report += `### ${issue.file}:${issue.line}\n`;
      report += `**Value:** \`${issue.value}\`\n`;
      report += `**Recommendation:** Add responsive breakpoints\n\n`;
    });
  }
  
  if (issues.containerIssues.length > 0) {
    report += '## CONTAINER ISSUES\n\n';
    issues.containerIssues.forEach(issue => {
      report += `### ${issue.file}:${issue.line}\n`;
      report += `**Issue:** ${issue.issue}\n`;
      report += `**Context:** \`${issue.context.substring(0, 80)}...\`\n\n`;
    });
  }
  
  if (issues.missingMediaQueries.length > 0) {
    report += '## MISSING MEDIA QUERIES\n\n';
    issues.missingMediaQueries.forEach(issue => {
      report += `### ${issue.file}\n`;
      report += `**Issue:** ${issue.issue}\n\n`;
    });
  }
  
  report += '## RECOMMENDED RESPONSIVE STANDARDS\n\n';
  report += '```css\n';
  report += '/* Content Containers */\n';
  report += `.container { max-width: min(${config.maxWidths.content}, 95vw); }\n`;
  report += `.container-narrow { max-width: min(${config.maxWidths.narrow}, 92vw); }\n`;
  report += `.reading-width { max-width: min(${config.maxWidths.reading}, 90vw); }\n\n`;
  report += '/* Breakpoints */\n';
  report += `@media (max-width: ${config.breakpoints.tablet}) { /* Tablet */ }\n`;
  report += `@media (max-width: ${config.breakpoints.mobile}) { /* Mobile */ }\n`;
  report += '```\n\n';
  
  return report;
}

// Run audit
console.log('Starting comprehensive responsive audit...');
scanDirectory(config.rootDir);
const report = generateReport();

// Write report
const reportPath = path.join(config.rootDir, 'RESPONSIVE_AUDIT_REPORT.md');
fs.writeFileSync(reportPath, report);

console.log(`\nAudit complete! Found:`);
console.log(`- ${issues.inlineMaxWidths.length} inline max-width issues`);
console.log(`- ${issues.hardcodedWidths.length} hardcoded width issues`);
console.log(`- ${issues.containerIssues.length} container issues`);
console.log(`- ${issues.missingMediaQueries.length} missing media queries`);
console.log(`\nReport saved to: ${reportPath}`);
