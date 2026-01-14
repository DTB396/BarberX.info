#!/usr/bin/env node
/**
 * Add Product to Stewardship Resources
 * 
 * Interactive CLI tool to add new products with proper validation
 * 
 * Usage:
 *   node scripts/add-product.js
 */

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { fileURLToPath } from 'url';
import readline from 'readline';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT = path.resolve(__dirname, '..');

const DATA_FILE = path.join(ROOT, '_data/stewardship-resources.yml');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

function question(prompt) {
  return new Promise(resolve => {
    rl.question(prompt, resolve);
  });
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '');
}

async function main() {
  console.log('📦 Add Product to Stewardship Resources\n');
  console.log('This tool will guide you through adding a new product.\n');
  
  // Category selection
  const categories = ['books', 'tools', 'health', 'homestead', 'faith', 'legal', 'other'];
  console.log('Available categories:');
  categories.forEach((cat, i) => console.log(`  ${i + 1}. ${cat}`));
  
  const categoryIndex = await question('\nSelect category (1-7): ');
  const category = categories[parseInt(categoryIndex) - 1];
  
  if (!category) {
    console.error('Invalid category selection');
    process.exit(1);
  }
  
  // Product details
  const title = await question('\nProduct title: ');
  const description = await question('Product description (2-3 sentences): ');
  const amazonUrl = await question('Amazon Associate URL (https://amazon.com/dp/...): ');
  const personalNote = await question('Personal note (optional): ');
  
  // Generate ID
  const id = slugify(title);
  
  // Build product object
  const product = {
    id,
    title,
    description,
    amazon_url: amazonUrl,
    category,
    tags: [],
    date_added: new Date().toISOString().split('T')[0]
  };
  
  if (personalNote) {
    product.personal_note = personalNote;
  }
  
  // Confirm
  console.log('\n' + '='.repeat(60));
  console.log('PRODUCT PREVIEW:');
  console.log('='.repeat(60));
  console.log(yaml.dump(product, { indent: 2 }));
  
  const confirm = await question('Add this product? (yes/no): ');
  
  if (confirm.toLowerCase() !== 'yes' && confirm.toLowerCase() !== 'y') {
    console.log('Cancelled.');
    rl.close();
    return;
  }
  
  // Load existing data
  let data = yaml.load(fs.readFileSync(DATA_FILE, 'utf8'));
  
  // Add product to appropriate category
  if (!Array.isArray(data[category])) {
    data[category] = [];
  }
  
  data[category].push(product);
  
  // Write back
  const yamlStr = yaml.dump(data, {
    indent: 2,
    lineWidth: 100,
    noRefs: true
  });
  
  fs.writeFileSync(DATA_FILE, yamlStr, 'utf8');
  
  console.log('\n✅ Product added successfully!');
  console.log('\nNext steps:');
  console.log('  1. Run: node scripts/validate-product-links.js');
  console.log('  2. Test locally: bundle exec jekyll serve');
  console.log('  3. Commit: git add _data/stewardship-resources.yml');
  console.log(`  4. git commit -m "feat(resources): add ${title}"`);
  
  rl.close();
}

main().catch(error => {
  console.error('Error:', error);
  rl.close();
  process.exit(1);
});
