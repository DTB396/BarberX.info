// Script to validate YAML front matter in all _cases/*.md files (CommonJS)
const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

const CASES_DIR = path.join(__dirname, '..', '_cases');

function extractFrontMatter(content) {
  const match = content.match(/^---\s*([\s\S]*?)---/);
  return match ? match[1] : null;
}

function validateYamlFields(file, data) {
  const errors = [];
  if (!('filed_date' in data)) {
    errors.push('Missing filed_date');
  } else if (!data.filed_date || typeof data.filed_date !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(data.filed_date)) {
    errors.push(`Invalid filed_date: ${JSON.stringify(data.filed_date)}`);
  }
  // Check for null, ~, or empty string in any field
  for (const [key, value] of Object.entries(data)) {
    if (value === null || value === '' || value === '~') {
      errors.push(`Field ${key} is null/empty/~`);
    }
  }
  return errors;
}

fs.readdirSync(CASES_DIR).forEach(file => {
  if (file.endsWith('.md')) {
    const content = fs.readFileSync(path.join(CASES_DIR, file), 'utf8');
    const frontMatter = extractFrontMatter(content);
    if (!frontMatter) {
      console.log(`${file}: No front matter found`);
      return;
    }
    let data;
    try {
      data = yaml.load(frontMatter);
    } catch (e) {
      console.log(`${file}: YAML parse error: ${e.message}`);
      return;
    }
    const errors = validateYamlFields(file, data);
    if (errors.length > 0) {
      console.log(`${file}:`);
      errors.forEach(err => console.log('  ' + err));
    } else {
      console.log(`${file}: OK`);
    }
  }
});
