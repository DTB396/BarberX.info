#!/usr/bin/env node
/**
 * Validate Stewardship Resources Product Links
 * 
 * Ensures all product entries in _data/stewardship-resources.yml:
 * - Have valid Amazon URLs with Associate tag
 * - Include required fields
 * - Follow naming conventions
 * - Comply with governance rules
 * 
 * Usage:
 *   node scripts/validate-product-links.js
 *   node scripts/validate-product-links.js --fix  (auto-fix issues)
 */

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const DATA_FILE = path.join(ROOT, '_data/stewardship-resources.yml');
const ASSOCIATE_TAG_PATTERN = /tag=([a-z0-9-]+)/;
const AMAZON_URL_PATTERN = /^https:\/\/(www\.)?amazon\.com\/(dp|gp\/product)\/[A-Z0-9]+/;

// Prohibited patterns from COMPLIANCE.md
const PROHIBITED_PATTERNS = [
  /guaranteed|guarantee/i,
  /cure|cures|treat|treats/i,
  /diagnose|diagnosis/i,
  /medical advice/i,
  /legal advice/i,
  /financial advice/i,
  /will fix|will solve/i,
  /must buy|need to buy/i,
];

// Required fields
const REQUIRED_FIELDS = ['id', 'title', 'description', 'amazon_url', 'category', 'date_added'];

let errors = [];
let warnings = [];
let fixable = [];

function validateProduct(product, category, index) {
  const ctx = `${category}[${index}]`;
  
  // Check required fields
  REQUIRED_FIELDS.forEach(field => {
    if (!product[field]) {
      errors.push(`${ctx}: Missing required field: ${field}`);
    }
  });
  
  if (!product.amazon_url) return; // Can't validate further
  
  // Validate Amazon URL format
  if (!AMAZON_URL_PATTERN.test(product.amazon_url)) {
    errors.push(`${ctx}: Invalid Amazon URL format: ${product.amazon_url}`);
  }
  
  // Check for Associate tag
  const tagMatch = product.amazon_url.match(ASSOCIATE_TAG_PATTERN);
  if (!tagMatch) {
    errors.push(`${ctx}: Amazon URL missing Associate tag (?tag=YOUR-TAG)`);
  } else {
    const tag = tagMatch[1];
    if (tag === 'YOUR-ASSOCIATE-TAG' || tag === 'XXXXX') {
      warnings.push(`${ctx}: Placeholder Associate tag detected: ${tag}`);
    }
  }
  
  // Check for prohibited compliance language
  const fullText = `${product.title} ${product.description} ${product.personal_note || ''}`;
  PROHIBITED_PATTERNS.forEach(pattern => {
    if (pattern.test(fullText)) {
      errors.push(`${ctx}: Prohibited language pattern detected: ${pattern.source}`);
    }
  });
  
  // Validate ID format (kebab-case)
  if (product.id && !/^[a-z0-9-]+$/.test(product.id)) {
    warnings.push(`${ctx}: ID should be kebab-case: ${product.id}`);
  }
  
  // Check description length
  if (product.description && product.description.length < 50) {
    warnings.push(`${ctx}: Description is too short (< 50 chars)`);
  }
  if (product.description && product.description.length > 300) {
    warnings.push(`${ctx}: Description is very long (> 300 chars) - consider shortening`);
  }
  
  // Validate date format
  if (product.date_added && !/^\d{4}-\d{2}-\d{2}$/.test(product.date_added)) {
    errors.push(`${ctx}: date_added must be YYYY-MM-DD format: ${product.date_added}`);
  }
  
  // Check category matches container
  if (product.category && product.category !== category) {
    fixable.push({
      ctx,
      issue: `Category mismatch: product.category="${product.category}" but in "${category}" section`,
      fix: () => { product.category = category; }
    });
  }
}

function generateJSON(data) {
  // Convert YAML to JSON for web consumption
  const jsonPath = path.join(ROOT, '_data/stewardship-resources.json');
  
  // Filter out empty categories and comments
  const cleanData = {};
  Object.keys(data).forEach(category => {
    if (Array.isArray(data[category]) && data[category].length > 0) {
      cleanData[category] = data[category];
    }
  });
  
  fs.writeFileSync(jsonPath, JSON.stringify(cleanData, null, 2), 'utf8');
  console.log(`✓ Generated ${jsonPath}`);
}

async function main() {
  console.log('🔍 Validating Stewardship Resources...\n');
  
  // Check if file exists
  if (!fs.existsSync(DATA_FILE)) {
    console.error(`✗ Data file not found: ${DATA_FILE}`);
    process.exit(1);
  }
  
  // Load YAML
  let data;
  try {
    data = yaml.load(fs.readFileSync(DATA_FILE, 'utf8'));
  } catch (error) {
    console.error('✗ Failed to parse YAML:', error.message);
    process.exit(1);
  }
  
  // Validate each category
  const categories = ['books', 'tools', 'health', 'homestead', 'faith', 'legal', 'other'];
  let totalProducts = 0;
  
  categories.forEach(category => {
    if (data[category] && Array.isArray(data[category])) {
      data[category].forEach((product, index) => {
        validateProduct(product, category, index);
        totalProducts++;
      });
    }
  });
  
  // Apply fixes if requested
  if (process.argv.includes('--fix') && fixable.length > 0) {
    console.log(`🔧 Applying ${fixable.length} fixable issue(s)...\n`);
    fixable.forEach(item => {
      console.log(`  Fixing: ${item.issue}`);
      item.fix();
    });
    
    // Write fixed data back
    const yamlStr = yaml.dump(data, {
      indent: 2,
      lineWidth: 100,
      noRefs: true
    });
    fs.writeFileSync(DATA_FILE, yamlStr, 'utf8');
    console.log(`✓ Fixed data written to ${DATA_FILE}\n`);
  }
  
  // Generate JSON for web
  generateJSON(data);
  
  // Report results
  console.log('\n' + '='.repeat(60));
  console.log('VALIDATION REPORT');
  console.log('='.repeat(60));
  console.log(`Total products: ${totalProducts}`);
  console.log(`Errors: ${errors.length}`);
  console.log(`Warnings: ${warnings.length}`);
  console.log(`Fixable issues: ${fixable.length}`);
  
  if (errors.length > 0) {
    console.log('\n❌ ERRORS:');
    errors.forEach(err => console.log(`  - ${err}`));
  }
  
  if (warnings.length > 0) {
    console.log('\n⚠️  WARNINGS:');
    warnings.forEach(warn => console.log(`  - ${warn}`));
  }
  
  if (fixable.length > 0 && !process.argv.includes('--fix')) {
    console.log('\n💡 Run with --fix to automatically fix these issues');
  }
  
  if (errors.length === 0 && warnings.length === 0) {
    console.log('\n✅ All products validated successfully!');
  }
  
  console.log('\n' + '='.repeat(60) + '\n');
  
  // Exit with error if validation failed
  if (errors.length > 0) {
    process.exit(1);
  }
}

main().catch(console.error);
