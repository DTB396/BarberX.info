#!/usr/bin/env node
// Pre-commit validation hook for docket intake
// Checks PDFs and YAML files before commit
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const RED = '\x1b[31m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const RESET = '\x1b[0m';

let hasErrors = false;
let hasWarnings = false;

const log = {
  error: (msg) => { hasErrors = true; console.error(`${RED}✗${RESET} ${msg}`); },
  success: (msg) => console.log(`${GREEN}✓${RESET} ${msg}`),
  warn: (msg) => { hasWarnings = true; console.log(`${YELLOW}⚠${RESET} ${msg}`); },
  info: (msg) => console.log(`  ${msg}`),
};

// Validate PDF file
const validatePdf = (filePath) => {
  if (!fs.existsSync(filePath)) {
    log.error(`File not found: ${filePath}`);
    return false;
  }
  
  const stats = fs.statSync(filePath);
  
  // Check minimum size
  if (stats.size < 1024) {
    log.error(`PDF too small (${stats.size} bytes): ${filePath}`);
    return false;
  }
  
  // Check PDF magic number
  try {
    const fd = fs.openSync(filePath, 'r');
    const buffer = Buffer.alloc(5);
    fs.readSync(fd, buffer, 0, 5, 0);
    fs.closeSync(fd);
    
    if (!buffer.toString('ascii').startsWith('%PDF-')) {
      log.error(`Invalid PDF format: ${filePath}`);
      return false;
    }
  } catch (e) {
    log.error(`Cannot read PDF: ${filePath} - ${e.message}`);
    return false;
  }
  
  return true;
};

// Validate YAML file
const validateYaml = (filePath) => {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const data = yaml.load(content, { schema: yaml.JSON_SCHEMA });
    
    // Check if it's an array
    if (!Array.isArray(data)) {
      log.error(`Docket file must be an array: ${filePath}`);
      return false;
    }
    
    // Validate each entry
    let entryValid = true;
    data.forEach((entry, idx) => {
      const entryNum = idx + 1;
      
      // Check required fields
      if (!entry.id) {
        log.error(`Entry ${entryNum} missing 'id' field in ${filePath}`);
        entryValid = false;
      }
      if (!entry.date) {
        log.error(`Entry ${entryNum} missing 'date' field in ${filePath}`);
        entryValid = false;
      }
      if (!entry.type) {
        log.error(`Entry ${entryNum} missing 'type' field in ${filePath}`);
        entryValid = false;
      }
      if (!entry.title) {
        log.error(`Entry ${entryNum} missing 'title' field in ${filePath}`);
        entryValid = false;
      }
      if (!entry.file) {
        log.error(`Entry ${entryNum} missing 'file' field in ${filePath}`);
        entryValid = false;
      }
      
      // Validate date format
      if (entry.date && !/^\d{4}-\d{2}-\d{2}$/.test(entry.date)) {
        log.error(`Entry ${entryNum} has invalid date format: ${entry.date} (should be YYYY-MM-DD)`);
        entryValid = false;
      }
      
      // Validate type
      const validTypes = [
        'Filing', 'Order', 'Notice', 'Brief', 'Exhibit', 'Motion', 'Other',
        'Complaint', 'Proposed Order', 'Proof of Service', 'Request', 'Certification'
      ];
      if (entry.type && !validTypes.includes(entry.type)) {
        log.warn(`Entry ${entryNum} has non-standard type: ${entry.type}`);
        log.info(`Valid types: ${validTypes.join(', ')}`);
      }
      
      // Check if referenced file exists
      if (entry.file) {
        const fileUrl = entry.file.startsWith('/') ? entry.file.slice(1) : entry.file;
        const filePath = path.normalize(fileUrl);
        if (!fs.existsSync(filePath)) {
          log.warn(`Entry ${entryNum} references missing file: ${entry.file}`);
        }
      }
    });
    
    return entryValid;
  } catch (e) {
    log.error(`Invalid YAML in ${filePath}: ${e.message}`);
    return false;
  }
};

// Validate cases-map.yml
const validateCasesMap = (filePath) => {
  try {
    const content = fs.readFileSync(filePath, 'utf8');
    const data = yaml.load(content, { schema: yaml.JSON_SCHEMA });
    
    if (!data || typeof data !== 'object') {
      log.error(`Cases map must be an object: ${filePath}`);
      return false;
    }
    
    // Check each mapping
    let valid = true;
    Object.entries(data).forEach(([docket, slug]) => {
      // Check slug format
      if (!/^[a-z0-9-]+$/.test(slug)) {
        log.error(`Invalid slug format for ${docket}: ${slug} (must be lowercase, alphanumeric, hyphens only)`);
        valid = false;
      }
      
      // Check if case exists
      const caseDir = path.join('cases', slug);
      if (!fs.existsSync(caseDir)) {
        log.warn(`Case directory not found for ${docket}: ${caseDir}`);
      }
      
      // Check if docket YAML exists
      const docketFile = path.join('_data', 'docket', `${slug}.yml`);
      if (!fs.existsSync(docketFile)) {
        log.warn(`Docket file not found for ${docket}: ${docketFile}`);
      }
    });
    
    return valid;
  } catch (e) {
    log.error(`Invalid YAML in ${filePath}: ${e.message}`);
    return false;
  }
};

// Main validation
const main = () => {
  console.log('\n🔍 Running pre-commit validation...\n');
  
  // Check for PDFs in inbox
  const inboxDirs = ['_inbox', 'assets/uploads'];
  let pdfCount = 0;
  
  inboxDirs.forEach(dir => {
    if (fs.existsSync(dir)) {
      const files = fs.readdirSync(dir, { recursive: true });
      files.forEach(file => {
        if (file.toLowerCase().endsWith('.pdf')) {
          const fullPath = path.join(dir, file);
          if (validatePdf(fullPath)) {
            log.success(`Valid PDF: ${fullPath}`);
            pdfCount++;
          }
        }
      });
    }
  });
  
  // Check docket YAML files in staged changes
  if (fs.existsSync('_data/docket')) {
    const files = fs.readdirSync('_data/docket');
    files.forEach(file => {
      if (file.endsWith('.yml') || file.endsWith('.yaml')) {
        const fullPath = path.join('_data', 'docket', file);
        if (validateYaml(fullPath)) {
          log.success(`Valid docket: ${fullPath}`);
        }
      }
    });
  }
  
  // Check cases-map.yml
  const casesMapFile = '_data/cases-map.yml';
  if (fs.existsSync(casesMapFile)) {
    if (validateCasesMap(casesMapFile)) {
      log.success(`Valid cases map: ${casesMapFile}`);
    }
  }
  
  // Summary
  console.log('\n' + '─'.repeat(60));
  if (hasErrors) {
    console.log(`${RED}❌ Validation FAILED${RESET} - Fix errors before committing`);
    process.exit(1);
  } else if (hasWarnings) {
    console.log(`${YELLOW}⚠️  Validation passed with warnings${RESET}`);
    console.log('Review warnings above and proceed with commit if acceptable');
  } else {
    console.log(`${GREEN}✅ All validations passed${RESET}`);
    if (pdfCount > 0) {
      console.log(`\n💡 Tip: ${pdfCount} PDF(s) in inbox will be processed by GitHub Actions after push`);
    }
  }
  console.log('─'.repeat(60) + '\n');
};

main();
