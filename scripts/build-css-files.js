import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const cssDir = path.join(__dirname, '..', 'assets', 'css');
const baseOutputDir = path.join(__dirname, '..', '_site', 'assets', 'css');

// Also output to baseurl path if exists
const baseurlOutputDir = path.join(__dirname, '..', '_site', 'faithfrontier-sandbox', 'assets', 'css');

// Ensure output directories exist
if (!fs.existsSync(baseOutputDir)) {
  fs.mkdirSync(baseOutputDir, { recursive: true });
}
if (!fs.existsSync(baseurlOutputDir)) {
  fs.mkdirSync(baseurlOutputDir, { recursive: true });
}

// Function to resolve and read CSS file with imports
function resolveCSS(filePath, basePath) {
  const fullPath = path.join(basePath, filePath);
  if (!fs.existsSync(fullPath)) {
    console.warn(`Warning: CSS file not found: ${fullPath}`);
    return '';
  }
  
  const content = fs.readFileSync(fullPath, 'utf8');
  
  // Replace @import statements with actual content
  return content.replace(/@import\s+url\(['"](.+?)['"]\);?/g, (match, importPath) => {
    const importBasePath = path.dirname(fullPath);
    return resolveCSS(importPath, importBasePath);
  });
}

// Create individual CSS files
const cssFiles = {
  'tokens.css': ['base/tokens.css'],
  'utilities.css': ['utilities/utilities.css'],
  'responsive-enhancements.css': ['layouts/responsive-enhancements.css'],
  'print.css': ['utilities/print.css'],
  'case-analysis.css': ['pages/case-analysis.css'],
  'case-enhanced.css': ['pages/case-enhanced.css'],
  'cases-index.css': ['pages/cases-index.css'],
  'home.css': ['pages/home.css'],
  'stewardship-resources.css': ['pages/stewardship-resources.css']
};

console.log('Building CSS files...');

for (const [outputFile, inputFiles] of Object.entries(cssFiles)) {
  let combinedCSS = '';
  
  for (const inputFile of inputFiles) {
    const resolved = resolveCSS(inputFile, cssDir);
    if (resolved) {
      combinedCSS += resolved + '\n';
    }
  }
  
  // Write to both output directories
  const outputPath = path.join(baseOutputDir, outputFile);
  const baseurlPath = path.join(baseurlOutputDir, outputFile);
  fs.writeFileSync(outputPath, combinedCSS);
  fs.writeFileSync(baseurlPath, combinedCSS);
  console.log(`✓ Created ${outputFile}`);
}

// Copy main.css to both locations
const mainCSS = fs.readFileSync(path.join(cssDir, 'main.css'), 'utf8');
fs.writeFileSync(path.join(baseOutputDir, 'main.css'), mainCSS);
fs.writeFileSync(path.join(baseurlOutputDir, 'main.css'), mainCSS);
console.log('✓ Copied main.css');

// Copy style.css if exists to both locations
const styleCSSPath = path.join(cssDir, 'style.css');
if (fs.existsSync(styleCSSPath)) {
  const styleCSS = fs.readFileSync(styleCSSPath, 'utf8');
  fs.writeFileSync(path.join(baseOutputDir, 'style.css'), styleCSS);
  fs.writeFileSync(path.join(baseurlOutputDir, 'style.css'), styleCSS);
  console.log('✓ Copied style.css');
}

console.log('\nCSS build complete!');
