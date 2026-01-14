#!/usr/bin/env node
/**
 * HTML/CSS Structure Validator for Faith Frontier
 * Finds blocks breaking our design intentions
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

class StructureValidator {
  constructor() {
    this.issues = [];
    this.warnings = [];
  }

  // Validate HTML tag nesting and balance
  validateHtmlStructure(content, filePath) {
    const lines = content.split('\n');
    const tagStack = [];
    const selfClosingTags = ['img', 'br', 'hr', 'input', 'meta', 'link', 'area', 'base', 'col', 'command', 'embed', 'keygen', 'param', 'source', 'track', 'wbr'];
    
    lines.forEach((line, idx) => {
      const lineNum = idx + 1;
      
      // Find opening tags
      const openMatches = line.matchAll(/<(\w+)(?:\s[^>]*)?>/g);
      for (const match of openMatches) {
        const tag = match[1].toLowerCase();
        if (!selfClosingTags.includes(tag)) {
          tagStack.push({ tag, line: lineNum, content: match[0] });
        }
      }
      
      // Find closing tags
      const closeMatches = line.matchAll(/<\/(\w+)>/g);
      for (const match of closeMatches) {
        const tag = match[1].toLowerCase();
        if (tagStack.length === 0) {
          this.issues.push({
            file: filePath,
            line: lineNum,
            type: 'UNMATCHED_CLOSE',
            message: `Closing tag </${tag}> has no matching opening tag`,
            context: line.trim()
          });
        } else {
          const last = tagStack[tagStack.length - 1];
          if (last.tag === tag) {
            tagStack.pop();
          } else {
            this.issues.push({
              file: filePath,
              line: lineNum,
              type: 'TAG_MISMATCH',
              message: `Expected </${last.tag}> but found </${tag}> (opened at line ${last.line})`,
              context: line.trim(),
              suggestion: `Close <${last.tag}> before closing <${tag}>`
            });
          }
        }
      }
    });
    
    // Check for unclosed tags
    tagStack.forEach(item => {
      this.issues.push({
        file: filePath,
        line: item.line,
        type: 'UNCLOSED_TAG',
        message: `Tag <${item.tag}> is never closed`,
        context: item.content
      });
    });
  }

  // Validate inline styles vs CSS classes
  validateStyleUsage(content, filePath) {
    const lines = content.split('\n');
    const inlineStylePattern = /<[^>]+style="[^"]*"/g;
    
    lines.forEach((line, idx) => {
      const lineNum = idx + 1;
      const matches = line.match(inlineStylePattern);
      
      if (matches && matches.length > 0) {
        // Check for repeated inline styles (should be classes)
        const styleAttr = line.match(/style="([^"]*)"/);
        if (styleAttr) {
          const styles = styleAttr[1];
          
          // Flag complex inline styles that should be classes
          if (styles.length > 100) {
            this.warnings.push({
              file: filePath,
              line: lineNum,
              type: 'COMPLEX_INLINE_STYLE',
              message: 'Complex inline style should be moved to CSS class',
              context: line.trim().substring(0, 80) + '...'
            });
          }
        }
      }
    });
  }

  // Validate CSS variable usage
  validateCssVariables(content, filePath) {
    const lines = content.split('\n');
    const hardcodedColors = /#[0-9a-fA-F]{3,6}|rgba?\([^)]+\)/g;
    
    lines.forEach((line, idx) => {
      const lineNum = idx + 1;
      
      // Skip if line already uses var()
      if (line.includes('var(--')) return;
      
      // Check for hardcoded colors in style attributes
      if (line.includes('style=')) {
        const matches = line.match(hardcodedColors);
        if (matches) {
          matches.forEach(color => {
            this.warnings.push({
              file: filePath,
              line: lineNum,
              type: 'HARDCODED_COLOR',
              message: `Hardcoded color "${color}" should use CSS variable`,
              context: line.trim().substring(0, 80) + '...',
              suggestion: 'Use var(--color-name) instead'
            });
          });
        }
      }
    });
  }

  // Validate accessibility
  validateAccessibility(content, filePath) {
    const lines = content.split('\n');
    
    lines.forEach((line, idx) => {
      const lineNum = idx + 1;
      
      // Check for images without alt text
      if (/<img[^>]*>/i.test(line) && !line.includes('alt=')) {
        this.issues.push({
          file: filePath,
          line: lineNum,
          type: 'MISSING_ALT',
          message: 'Image missing alt attribute',
          context: line.trim()
        });
      }
      
      // Check for buttons/links without accessible text
      if (/<button[^>]*>\s*<\/button>/i.test(line)) {
        this.warnings.push({
          file: filePath,
          line: lineNum,
          type: 'EMPTY_BUTTON',
          message: 'Button has no text content',
          context: line.trim()
        });
      }
    });
  }

  // Validate liquid template syntax
  validateLiquidSyntax(content, filePath) {
    const lines = content.split('\n');
    const liquidStack = [];
    
    lines.forEach((line, idx) => {
      const lineNum = idx + 1;
      
      // Check for opening liquid blocks
      const openMatches = line.matchAll(/{%\s*(if|for|unless|case)\s/g);
      for (const match of openMatches) {
        liquidStack.push({ tag: match[1], line: lineNum });
      }
      
      // Check for closing liquid blocks
      const closeMatches = line.matchAll(/{%\s*end(if|for|unless|case)\s*%}/g);
      for (const match of closeMatches) {
        const tag = match[1];
        if (liquidStack.length === 0) {
          this.issues.push({
            file: filePath,
            line: lineNum,
            type: 'UNMATCHED_LIQUID_END',
            message: `Closing {% end${tag} %} has no matching opening`,
            context: line.trim()
          });
        } else {
          const last = liquidStack[liquidStack.length - 1];
          if (last.tag === tag) {
            liquidStack.pop();
          } else {
            this.issues.push({
              file: filePath,
              line: lineNum,
              type: 'LIQUID_MISMATCH',
              message: `Expected {% end${last.tag} %} but found {% end${tag} %} (opened at line ${last.line})`,
              context: line.trim()
            });
          }
        }
      }
    });
    
    // Check for unclosed liquid blocks
    liquidStack.forEach(item => {
      this.issues.push({
        file: filePath,
        line: item.line,
        type: 'UNCLOSED_LIQUID',
        message: `Liquid block {% ${item.tag} %} is never closed`,
        context: `Line ${item.line}`
      });
    });
  }

  // Validate semantic HTML
  validateSemanticHtml(content, filePath) {
    const lines = content.split('\n');
    
    lines.forEach((line, idx) => {
      const lineNum = idx + 1;
      
      // Check for div soup where semantic tags should be used
      if (/<div class="(header|footer|nav|article|aside|main)"/i.test(line)) {
        const match = line.match(/class="(header|footer|nav|article|aside|main)"/i);
        if (match) {
          this.warnings.push({
            file: filePath,
            line: lineNum,
            type: 'NON_SEMANTIC_HTML',
            message: `Consider using <${match[1]}> instead of <div class="${match[1]}">`,
            context: line.trim().substring(0, 80) + '...'
          });
        }
      }
    });
  }

  // Validate file
  validateFile(filePath) {
    try {
      const content = fs.readFileSync(filePath, 'utf-8');
      const ext = path.extname(filePath);
      
      console.log(`\n🔍 Validating: ${filePath}`);
      
      // HTML structure validation
      this.validateHtmlStructure(content, filePath);
      
      // Liquid template validation
      if (ext === '.md' || ext === '.html') {
        this.validateLiquidSyntax(content, filePath);
      }
      
      // Style validation
      this.validateStyleUsage(content, filePath);
      this.validateCssVariables(content, filePath);
      
      // Accessibility validation
      this.validateAccessibility(content, filePath);
      
      // Semantic HTML validation
      this.validateSemanticHtml(content, filePath);
      
    } catch (error) {
      console.error(`❌ Error reading ${filePath}:`, error.message);
    }
  }

  // Print report
  printReport() {
    console.log('\n' + '='.repeat(80));
    console.log('📊 VALIDATION REPORT');
    console.log('='.repeat(80));
    
    if (this.issues.length === 0 && this.warnings.length === 0) {
      console.log('\n✅ No issues found! Structure looks good.');
      return 0;
    }
    
    if (this.issues.length > 0) {
      console.log(`\n🚨 CRITICAL ISSUES (${this.issues.length}):\n`);
      this.issues.forEach((issue, idx) => {
        console.log(`${idx + 1}. [${issue.type}] ${issue.file}:${issue.line}`);
        console.log(`   ${issue.message}`);
        console.log(`   Context: ${issue.context}`);
        if (issue.suggestion) {
          console.log(`   💡 Suggestion: ${issue.suggestion}`);
        }
        console.log('');
      });
    }
    
    if (this.warnings.length > 0) {
      console.log(`\n⚠️  WARNINGS (${this.warnings.length}):\n`);
      this.warnings.forEach((warning, idx) => {
        console.log(`${idx + 1}. [${warning.type}] ${warning.file}:${warning.line}`);
        console.log(`   ${warning.message}`);
        if (warning.suggestion) {
          console.log(`   💡 Suggestion: ${warning.suggestion}`);
        }
        console.log('');
      });
    }
    
    console.log('='.repeat(80));
    console.log(`\n📈 Summary: ${this.issues.length} critical issues, ${this.warnings.length} warnings`);
    
    return this.issues.length > 0 ? 1 : 0;
  }
}

// Main execution
function main() {
  const args = process.argv.slice(2);
  
  if (args.length === 0) {
    console.log('Usage: node validate-html-structure.js <file1> [file2] ...');
    console.log('   or: node validate-html-structure.js --scan [directory]');
    process.exit(1);
  }
  
  const validator = new StructureValidator();
  
  if (args[0] === '--scan') {
    const scanDir = args[1] || '.';
    console.log(`📁 Scanning directory: ${scanDir}`);
    
    // Simple directory traversal without glob
    const walkDir = (dir) => {
      const files = [];
      try {
        const items = fs.readdirSync(dir);
        for (const item of items) {
          const fullPath = path.join(dir, item);
          const stat = fs.statSync(fullPath);
          if (stat.isDirectory()) {
            if (!['node_modules', '_site', 'vendor', '.git'].includes(item)) {
              files.push(...walkDir(fullPath));
            }
          } else if (stat.isFile() && /\.(md|html)$/i.test(item)) {
            files.push(fullPath);
          }
        }
      } catch (e) {
        console.error(`Error reading ${dir}: ${e.message}`);
      }
      return files;
    };
    
    const files = walkDir(scanDir);
    
    console.log(`Found ${files.length} files to validate`);
    files.forEach(file => validator.validateFile(file));
  } else {
    // Validate specific files
    args.forEach(file => validator.validateFile(file));
  }
  
  process.exit(validator.printReport());
}

main();
