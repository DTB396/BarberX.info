#!/usr/bin/env node
// Batch PDF upload helper - validates and prepares PDFs for intake
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';

const INBOX = '_inbox';
const MIN_PDF_SIZE = 1024; // 1KB minimum
const MAX_PDF_SIZE = 50 * 1024 * 1024; // 50MB maximum
const CASES_MAP_FILE = '_data/cases-map.yml';

// Color output helpers
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
};

const log = {
  error: (msg) => console.error(`${colors.red}✗${colors.reset} ${msg}`),
  success: (msg) => console.log(`${colors.green}✓${colors.reset} ${msg}`),
  warn: (msg) => console.log(`${colors.yellow}⚠${colors.reset} ${msg}`),
  info: (msg) => console.log(`${colors.blue}ℹ${colors.reset} ${msg}`),
  section: (msg) => console.log(`\n${colors.cyan}${msg}${colors.reset}`),
};

// Load cases map
const loadCasesMap = () => {
  try {
    if (fs.existsSync(CASES_MAP_FILE)) {
      return yaml.load(fs.readFileSync(CASES_MAP_FILE, 'utf8')) || {};
    }
  } catch (e) {
    log.warn(`Could not load cases map: ${e.message}`);
  }
  return {};
};

// Validate PDF file
const validatePdf = (filePath) => {
  const errors = [];
  const warnings = [];
  
  // Check if file exists
  if (!fs.existsSync(filePath)) {
    errors.push('File does not exist');
    return { valid: false, errors, warnings };
  }
  
  const stats = fs.statSync(filePath);
  
  // Check if it's a file
  if (!stats.isFile()) {
    errors.push('Not a regular file');
    return { valid: false, errors, warnings };
  }
  
  // Check extension
  if (!filePath.toLowerCase().endsWith('.pdf')) {
    errors.push('Not a PDF file (wrong extension)');
    return { valid: false, errors, warnings };
  }
  
  // Check file size
  if (stats.size < MIN_PDF_SIZE) {
    errors.push(`File too small (${stats.size} bytes, minimum ${MIN_PDF_SIZE})`);
  } else if (stats.size > MAX_PDF_SIZE) {
    warnings.push(`File very large (${Math.round(stats.size / 1024 / 1024)}MB)`);
  }
  
  // Check PDF magic number
  try {
    const fd = fs.openSync(filePath, 'r');
    const buffer = Buffer.alloc(5);
    fs.readSync(fd, buffer, 0, 5, 0);
    fs.closeSync(fd);
    
    const magic = buffer.toString('ascii');
    if (!magic.startsWith('%PDF-')) {
      errors.push('Not a valid PDF file (wrong magic number)');
    }
  } catch (e) {
    errors.push(`Could not read file: ${e.message}`);
  }
  
  return {
    valid: errors.length === 0,
    errors,
    warnings,
    size: stats.size,
  };
};

// Extract docket number from filename
const extractDocketNumber = (filename, casesMap) => {
  const patterns = [
    { regex: /\b(\d:\d{2}-cv-\d{5}(?:-[A-Z]{3}(?:-[A-Z]{3})?)?)\b/i, name: 'Federal' },
    { regex: /\b(ATL-[A-Z]{1,2}-\d{6}-\d{2})\b/i, name: 'NJ Superior' },
    { regex: /\b(MER-[A-Z]{1,2}-\d{6}-\d{2})\b/i, name: 'NJ Mercer' },
    { regex: /\b(ATL-\d{2}-\d{6})\b/i, name: 'NJ Legacy' },
    { regex: /\b(A-\d{6}-\d{2})\b/, name: 'NJ Appellate' },
  ];
  
  for (const { regex, name } of patterns) {
    const match = filename.match(regex);
    if (match) {
      const docket = match[1];
      const slug = casesMap[docket];
      return { docket, slug, type: name };
    }
  }
  
  return null;
};

// Suggest better filename
const suggestFilename = (originalName) => {
  // Extract date if present
  const dateMatch = originalName.match(/(20\d{2})[-_\.]?(0[1-9]|1[0-2])[-_\.]?(0[1-9]|[12]\d|3[01])/);
  const date = dateMatch ? `${dateMatch[1]}-${dateMatch[2]}-${dateMatch[3]}` : 'YYYY-MM-DD';
  
  // Detect type
  const lower = originalName.toLowerCase();
  let type = 'Filing';
  if (lower.includes('order')) type = 'Order';
  else if (lower.includes('notice')) type = 'Notice';
  else if (lower.includes('brief')) type = 'Brief';
  else if (lower.includes('exhibit')) type = 'Exhibit';
  else if (lower.includes('motion')) type = 'Motion';
  
  // Clean up name
  const base = originalName
    .replace(/\.pdf$/i, '')
    .replace(/^\d{8}[-_]?/, '') // Remove leading date
    .replace(/^20\d{2}-\d{2}-\d{2}[-_]?/, '') // Remove ISO date
    .replace(/[^a-z0-9]+/gi, '-')
    .replace(/^-+|-+$/g, '')
    .toLowerCase();
  
  return `${date}_${type}_${base || 'document'}.pdf`;
};

// Analyze a single PDF
const analyzePdf = (filePath, casesMap) => {
  const basename = path.basename(filePath);
  const validation = validatePdf(filePath);
  const docketInfo = extractDocketNumber(basename, casesMap);
  const suggestedName = suggestFilename(basename);
  
  return {
    path: filePath,
    basename,
    validation,
    docketInfo,
    suggestedName,
  };
};

// Process batch upload
const processBatch = (sourceDir) => {
  log.section('📦 Batch PDF Upload Helper');
  
  // Validate source directory
  if (!fs.existsSync(sourceDir)) {
    log.error(`Source directory not found: ${sourceDir}`);
    process.exit(1);
  }
  
  // Load cases map
  log.info('Loading cases map...');
  const casesMap = loadCasesMap();
  const knownDockets = Object.keys(casesMap);
  log.info(`Found ${knownDockets.length} known docket mappings`);
  
  // Find all PDFs
  log.info(`Scanning ${sourceDir} for PDFs...`);
  const pdfs = [];
  const scanDir = (dir) => {
    const items = fs.readdirSync(dir);
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);
      if (stat.isDirectory()) {
        scanDir(fullPath);
      } else if (item.toLowerCase().endsWith('.pdf')) {
        pdfs.push(fullPath);
      }
    }
  };
  scanDir(sourceDir);
  
  if (pdfs.length === 0) {
    log.warn('No PDF files found');
    return;
  }
  
  log.success(`Found ${pdfs.length} PDF file(s)`);
  
  // Analyze each PDF
  log.section('\n📋 Analysis Results');
  const results = pdfs.map(pdf => analyzePdf(pdf, casesMap));
  
  let validCount = 0;
  let invalidCount = 0;
  let matchedCount = 0;
  let unmatchedCount = 0;
  
  results.forEach((result, index) => {
    console.log(`\n${colors.cyan}[${index + 1}/${results.length}] ${result.basename}${colors.reset}`);
    
    // Validation results
    if (result.validation.valid) {
      log.success(`Valid PDF (${Math.round(result.validation.size / 1024)}KB)`);
      validCount++;
    } else {
      log.error('Invalid PDF:');
      result.validation.errors.forEach(err => console.log(`    ${err}`));
      invalidCount++;
      return; // Skip further analysis for invalid files
    }
    
    result.validation.warnings.forEach(warn => log.warn(warn));
    
    // Docket matching
    if (result.docketInfo) {
      if (result.docketInfo.slug) {
        log.success(`Matched: ${result.docketInfo.docket} → ${result.docketInfo.slug}`);
        matchedCount++;
      } else {
        log.warn(`Docket found but not mapped: ${result.docketInfo.docket}`);
        log.info(`Add to ${CASES_MAP_FILE}: ${result.docketInfo.docket}: <slug>`);
        unmatchedCount++;
      }
    } else {
      log.warn('No docket number detected in filename');
      log.info('Tip: Include case number in filename (e.g., ATL-L-002794-25)');
      unmatchedCount++;
    }
    
    // Filename suggestion
    if (result.suggestedName !== result.basename) {
      log.info(`Suggested: ${result.suggestedName}`);
    }
  });
  
  // Summary
  log.section('\n📊 Summary');
  console.log(`Total PDFs:       ${pdfs.length}`);
  console.log(`Valid:            ${colors.green}${validCount}${colors.reset}`);
  console.log(`Invalid:          ${colors.red}${invalidCount}${colors.reset}`);
  console.log(`Matched to case:  ${colors.green}${matchedCount}${colors.reset}`);
  console.log(`Unmatched:        ${colors.yellow}${unmatchedCount}${colors.reset}`);
  
  // Ready for intake?
  const invalidFiles = results.filter(r => !r.validation.valid);
  const unmatchedFiles = results.filter(
    r => r.validation.valid && (!r.docketInfo || !r.docketInfo.slug)
  );
  
  if (invalidFiles.length > 0) {
    log.section('\n❌ Cannot Proceed');
    log.error(`${invalidFiles.length} file(s) are invalid and must be fixed`);
    process.exit(1);
  }
  
  if (unmatchedFiles.length > 0) {
    log.section('\n⚠️  Ready with Warnings');
    log.warn(`${unmatchedFiles.length} file(s) will go to "unassigned" folder`);
    log.info('Add docket mappings to avoid this, or proceed anyway');
  } else {
    log.section('\n✅ Ready for Intake');
    log.success('All files are valid and matched to cases');
  }
  
  // Copy to inbox instructions
  log.section('\n🚀 Next Steps');
  console.log('1. Review the analysis above');
  console.log('2. Fix any invalid PDFs or add missing case mappings');
  console.log(`3. Copy files to ${INBOX}/:`);
  console.log(`   ${colors.cyan}cp ${sourceDir}/*.pdf ${INBOX}/${colors.reset}`);
  console.log('4. Commit and push:');
  console.log(`   ${colors.cyan}git add ${INBOX}/ && git commit -m "intake: batch upload" && git push${colors.reset}`);
  console.log('5. GitHub Actions will automatically process the files');
  console.log('6. Review the generated PR and merge');
};

// Interactive copy mode
const copyToInbox = (sourceDir) => {
  log.section('📦 Copying PDFs to Inbox');
  
  if (!fs.existsSync(sourceDir)) {
    log.error(`Source directory not found: ${sourceDir}`);
    process.exit(1);
  }
  
  // Ensure inbox exists
  if (!fs.existsSync(INBOX)) {
    fs.mkdirSync(INBOX, { recursive: true });
    log.info(`Created ${INBOX} directory`);
  }
  
  // Find all PDFs
  const pdfs = [];
  const scanDir = (dir) => {
    const items = fs.readdirSync(dir);
    for (const item of items) {
      const fullPath = path.join(dir, item);
      const stat = fs.statSync(fullPath);
      if (stat.isDirectory()) {
        scanDir(fullPath);
      } else if (item.toLowerCase().endsWith('.pdf')) {
        pdfs.push(fullPath);
      }
    }
  };
  scanDir(sourceDir);
  
  if (pdfs.length === 0) {
    log.warn('No PDF files found');
    return;
  }
  
  let copied = 0;
  let skipped = 0;
  
  for (const pdfPath of pdfs) {
    const basename = path.basename(pdfPath);
    const destPath = path.join(INBOX, basename);
    
    // Check if already exists
    if (fs.existsSync(destPath)) {
      log.warn(`Skipped (already exists): ${basename}`);
      skipped++;
      continue;
    }
    
    try {
      fs.copyFileSync(pdfPath, destPath);
      log.success(`Copied: ${basename}`);
      copied++;
    } catch (e) {
      log.error(`Failed to copy ${basename}: ${e.message}`);
    }
  }
  
  log.section('\n📊 Copy Summary');
  console.log(`Copied:  ${colors.green}${copied}${colors.reset}`);
  console.log(`Skipped: ${colors.yellow}${skipped}${colors.reset}`);
  
  if (copied > 0) {
    log.section('\n🚀 Next Steps');
    console.log('Commit and push to trigger intake:');
    console.log(`   ${colors.cyan}git add ${INBOX}/ && git commit -m "intake: ${copied} files" && git push${colors.reset}`);
  }
};

// Main
const main = () => {
  const args = process.argv.slice(2);
  
  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`
${colors.cyan}Batch PDF Upload Helper${colors.reset}

Validates and prepares PDF files for docket intake.

${colors.yellow}USAGE:${colors.reset}
  node scripts/batch-upload-helper.js [OPTIONS] <source-directory>

${colors.yellow}OPTIONS:${colors.reset}
  --analyze         Analyze PDFs without copying (default)
  --copy            Copy validated PDFs to inbox
  --help, -h        Show this help message

${colors.yellow}EXAMPLES:${colors.reset}
  # Analyze PDFs in Downloads folder
  node scripts/batch-upload-helper.js ~/Downloads

  # Analyze PDFs in specific directory
  node scripts/batch-upload-helper.js /path/to/pdfs

  # Copy PDFs to inbox after validation
  node scripts/batch-upload-helper.js --copy ~/Downloads

${colors.yellow}WORKFLOW:${colors.reset}
  1. Drop PDFs in a folder (with docket numbers in filenames)
  2. Run: node scripts/batch-upload-helper.js <folder>
  3. Review analysis and fix any issues
  4. Run: node scripts/batch-upload-helper.js --copy <folder>
  5. Commit and push to trigger automated intake
`);
    process.exit(0);
  }
  
  const copyMode = args.includes('--copy');
  const sourceDir = args.filter(a => !a.startsWith('--')).pop();
  
  if (!sourceDir) {
    log.error('Source directory is required');
    console.log('Run with --help for usage information');
    process.exit(1);
  }
  
  if (copyMode) {
    copyToInbox(sourceDir);
  } else {
    processBatch(sourceDir);
  }
};

main();
