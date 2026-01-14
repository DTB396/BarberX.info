#!/usr/bin/env node
/**
 * Faith Frontier Content Refactor Tool
 * 
 * Uses centralized /.ai/ governance framework to validate and refactor
 * site content for compliance with mission, tone, and legal standards.
 * 
 * Loads governance in hierarchical order:
 *   1. SYSTEM.md (foundational rules)
 *   2. STYLE.md (writing standards)
 *   3. DOMAIN.md (project context)
 *   4. COMPLIANCE.md (legal boundaries)
 *   5. OUTPUT_RULES.md (technical specs)
 * 
 * Usage:
 *   node scripts/refactor-with-governance.js --all
 *   node scripts/refactor-with-governance.js --section essays
 *   node scripts/refactor-with-governance.js --file _essays/2025-11-10-revelations.md
 *   node scripts/refactor-with-governance.js --audit-only
 *   node scripts/refactor-with-governance.js --interactive
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { OpenAI } from 'openai';
import 'dotenv/config';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const ROOT_DIR = path.resolve(__dirname, '..');

// Configuration
const CONFIG = {
  // Load governance files from /.ai/ directory in prescribed order
  governanceFiles: [
    path.join(ROOT_DIR, '.ai/SYSTEM.md'),
    path.join(ROOT_DIR, '.ai/STYLE.md'),
    path.join(ROOT_DIR, '.ai/DOMAIN.md'),
    path.join(ROOT_DIR, '.ai/COMPLIANCE.md'),
    path.join(ROOT_DIR, '.ai/OUTPUT_RULES.md'),
  ],
  sectionsToScan: {
    essays: path.join(ROOT_DIR, '_essays'),
    cases: path.join(ROOT_DIR, '_cases'),
    trust: path.join(ROOT_DIR, '_trust'),
    manifesto: path.join(ROOT_DIR, '_manifesto'),
    pages: path.join(ROOT_DIR, '_pages'),
    posts: path.join(ROOT_DIR, '_posts'),
  },
  outputDir: path.join(ROOT_DIR, 'reports/governance-refactor'),
  backupDir: path.join(ROOT_DIR, 'reports/governance-refactor/backups'),
};

// Governance validation rules extracted from copilot instructions
const GOVERNANCE_RULES = {
  prohibited: {
    patterns: [
      /alternative currenc/i,
      /replace.*fiat/i,
      /asset-backed money/i,
      /extralegal/i,
      /financial sovereignty/i,
      /parallel government/i,
      /reject.*civil law/i,
      /divine mandate/i,
      /prophetic certainty/i,
      /weaponize.*scripture/i,
      /religious authority over.*law/i,
    ],
    keywords: [
      'revolutionary movement',
      'currency issuer',
      'overthrow',
      'lawless',
      'above the law',
    ]
  },
  required: {
    tone: ['calm', 'grounded', 'sober', 'modest', 'factual'],
    legalFramework: ['New Jersey', 'U.S. law', 'compliant', 'lawful'],
    boundaries: ['accountability', 'transparency', 'stewardship'],
  },
  encouraged: {
    keywords: [
      'stewardship',
      'local trade',
      'community',
      'accountability',
      'neighbor-care',
      'lawful',
      'dignity',
      'craft',
      'vocation',
    ]
  }
};

class GovernanceRefactorTool {
  constructor(options = {}) {
    this.options = {
      auditOnly: options.auditOnly || false,
      interactive: options.interactive || false,
      dryRun: options.dryRun || false,
      useAI: options.useAI !== false, // Default true
    };
    
    this.openai = null;
    if (this.options.useAI && process.env.OPENAI_API_KEY) {
      this.openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    }
    
    this.governanceInstructions = '';
    this.results = {
      scanned: 0,
      issues: [],
      refactored: [],
      skipped: [],
      errors: [],
    };
  }

  /**
   * Initialize the tool
   */
  async init() {
    console.log('🔧 Faith Frontier Governance Refactor Tool');
    console.log('='.repeat(60));
    
    // Load governance files from /.ai/ directory
    try {
      const governanceParts = [];
      for (const govFile of CONFIG.governanceFiles) {
        if (fs.existsSync(govFile)) {
          const content = fs.readFileSync(govFile, 'utf8');
          const fileName = path.basename(govFile);
          governanceParts.push(`\n---\n# ${fileName}\n---\n${content}`);
          console.log(`✓ Loaded ${fileName}`);
        } else {
          console.warn(`⚠️  Missing governance file: ${path.basename(govFile)}`);
        }
      }
      
      this.governanceInstructions = governanceParts.join('\n\n');
      console.log('✓ Governance framework loaded from /.ai/ directory');
      console.log(`  Files loaded: ${CONFIG.governanceFiles.length}`);
      console.log(`  Total size: ${(this.governanceInstructions.length / 1024).toFixed(1)} KB`);
    } catch (error) {
      console.error('✗ Failed to load governance framework:', error.message);
      process.exit(1);
    }

    // Create output directories
    [CONFIG.outputDir, CONFIG.backupDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });

    console.log('✓ Output directory ready:', CONFIG.outputDir);
    console.log('');
  }

  /**
   * Scan all sections or specific targets
   */
  async scan(target = 'all') {
    console.log(`📂 Scanning target: ${target}`);
    console.log('');

    const files = this.getFilesToScan(target);
    console.log(`📄 Found ${files.length} files to scan\n`);

    for (const file of files) {
      await this.analyzeFile(file);
    }

    return this.results;
  }

  /**
   * Get list of files to scan based on target
   */
  getFilesToScan(target) {
    const files = [];

    if (target === 'all') {
      // Scan all sections
      Object.values(CONFIG.sectionsToScan).forEach(dir => {
        if (fs.existsSync(dir)) {
          files.push(...this.getMarkdownFiles(dir));
        }
      });
    } else if (CONFIG.sectionsToScan[target]) {
      // Scan specific section
      const dir = CONFIG.sectionsToScan[target];
      if (fs.existsSync(dir)) {
        files.push(...this.getMarkdownFiles(dir));
      }
    } else if (fs.existsSync(target)) {
      // Single file
      files.push(target);
    } else {
      console.error(`✗ Target not found: ${target}`);
    }

    return files;
  }

  /**
   * Get all markdown files in directory (recursive)
   */
  getMarkdownFiles(dir) {
    const files = [];
    const entries = fs.readdirSync(dir, { withFileTypes: true });

    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        files.push(...this.getMarkdownFiles(fullPath));
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        files.push(fullPath);
      }
    }

    return files;
  }

  /**
   * Analyze a single file for governance compliance
   */
  async analyzeFile(filePath) {
    this.results.scanned++;
    const relativePath = path.relative(ROOT_DIR, filePath);
    console.log(`\n📝 Analyzing: ${relativePath}`);

    try {
      const content = fs.readFileSync(filePath, 'utf8');
      const issues = this.detectIssues(content);

      if (issues.length === 0) {
        console.log('   ✓ No issues found');
        return;
      }

      console.log(`   ⚠️  Found ${issues.length} issue(s):`);
      issues.forEach((issue, i) => {
        console.log(`      ${i + 1}. ${issue.type}: ${issue.message}`);
      });

      this.results.issues.push({
        file: relativePath,
        issues,
      });

      // Refactor if not audit-only
      if (!this.options.auditOnly) {
        await this.refactorFile(filePath, content, issues);
      }
    } catch (error) {
      console.error(`   ✗ Error analyzing file: ${error.message}`);
      this.results.errors.push({
        file: relativePath,
        error: error.message,
      });
    }
  }

  /**
   * Detect governance issues in content
   */
  detectIssues(content) {
    const issues = [];

    // Check for prohibited patterns
    GOVERNANCE_RULES.prohibited.patterns.forEach(pattern => {
      const matches = content.match(pattern);
      if (matches) {
        issues.push({
          type: 'PROHIBITED_LANGUAGE',
          message: `Contains prohibited pattern: "${matches[0]}"`,
          severity: 'high',
          pattern: pattern.source,
        });
      }
    });

    // Check for prohibited keywords
    const contentLower = content.toLowerCase();
    GOVERNANCE_RULES.prohibited.keywords.forEach(keyword => {
      if (contentLower.includes(keyword.toLowerCase())) {
        issues.push({
          type: 'PROHIBITED_KEYWORD',
          message: `Contains prohibited keyword: "${keyword}"`,
          severity: 'high',
          keyword,
        });
      }
    });

    // Check for tone issues (simple heuristics)
    const toneIssues = this.checkTone(content);
    issues.push(...toneIssues);

    return issues;
  }

  /**
   * Check tone and style
   */
  checkTone(content) {
    const issues = [];
    
    // Check for excessive exclamation marks (indicates excitement over sobriety)
    const exclamationCount = (content.match(/!/g) || []).length;
    if (exclamationCount > 10) {
      issues.push({
        type: 'TONE_ISSUE',
        message: `Excessive exclamation marks (${exclamationCount}) - tone should be sober and calm`,
        severity: 'medium',
      });
    }

    // Check for all-caps words (indicates shouting)
    const allCapsWords = content.match(/\b[A-Z]{4,}\b/g) || [];
    if (allCapsWords.length > 5) {
      issues.push({
        type: 'TONE_ISSUE',
        message: `Excessive all-caps words (${allCapsWords.length}) - avoid emphatic styling`,
        severity: 'medium',
      });
    }

    // Check for fear-based language
    const fearWords = ['crisis', 'collapse', 'emergency', 'urgent', 'critical'];
    const fearCount = fearWords.filter(word => 
      content.toLowerCase().includes(word)
    ).length;
    
    if (fearCount > 3) {
      issues.push({
        type: 'TONE_ISSUE',
        message: `Fear-based language detected - maintain calm, grounded tone`,
        severity: 'medium',
      });
    }

    return issues;
  }

  /**
   * Refactor file content to comply with governance
   */
  async refactorFile(filePath, content, issues) {
    const relativePath = path.relative(ROOT_DIR, filePath);

    if (this.options.interactive) {
      // Ask user for confirmation
      console.log(`\n   🤔 Refactor ${relativePath}?`);
      // In real implementation, would use readline for user input
      // For now, skip interactive mode
      console.log('   ⏭️  Interactive mode not fully implemented - skipping');
      this.results.skipped.push(relativePath);
      return;
    }

    if (this.openai) {
      // Use AI to refactor
      await this.aiRefactor(filePath, content, issues);
    } else {
      // Use rule-based refactoring
      this.ruleBasedRefactor(filePath, content, issues);
    }
  }

  /**
   * AI-powered refactoring using OpenAI
   */
  async aiRefactor(filePath, content, issues) {
    const relativePath = path.relative(ROOT_DIR, filePath);
    console.log('   🤖 Requesting AI refactoring...');

    try {
      const prompt = this.buildRefactorPrompt(content, issues);
      
      const response = await this.openai.chat.completions.create({
        model: 'gpt-4o',
        messages: [
          {
            role: 'system',
            content: this.governanceInstructions,
          },
          {
            role: 'user',
            content: prompt,
          }
        ],
        temperature: 0.3,
      });

      const refactoredContent = response.choices[0].message.content;

      if (this.options.dryRun) {
        console.log('   📋 Dry run - would refactor file');
        this.results.refactored.push({
          file: relativePath,
          preview: refactoredContent.substring(0, 200) + '...',
        });
      } else {
        // Backup original
        this.backupFile(filePath);
        
        // Write refactored content
        fs.writeFileSync(filePath, refactoredContent, 'utf8');
        console.log('   ✓ File refactored successfully');
        
        this.results.refactored.push({
          file: relativePath,
          issuesFixed: issues.length,
        });
      }
    } catch (error) {
      console.error(`   ✗ AI refactoring failed: ${error.message}`);
      this.results.errors.push({
        file: relativePath,
        error: error.message,
      });
    }
  }

  /**
   * Build refactoring prompt for AI
   */
  buildRefactorPrompt(content, issues) {
    return `Please refactor the following content to comply with Faith Frontier governance standards.

ISSUES DETECTED:
${issues.map((issue, i) => `${i + 1}. ${issue.type}: ${issue.message}`).join('\n')}

CONTENT TO REFACTOR:
${content}

REQUIREMENTS:
1. Fix all detected issues
2. Maintain the core message and structure
3. Use calm, grounded, sober tone
4. Ensure legal compliance language (New Jersey, U.S. law)
5. Avoid prohibited language (alternative currencies, financial sovereignty, etc.)
6. Keep faith references humble and non-coercive
7. Focus on local stewardship, community accountability, lawful alternatives
8. Write for diverse audiences (judges, regulators, pastors, neighbors)

Return ONLY the refactored content, maintaining original markdown formatting.`;
  }

  /**
   * Rule-based refactoring (fallback when no AI)
   */
  ruleBasedRefactor(filePath, content, issues) {
    const relativePath = path.relative(ROOT_DIR, filePath);
    console.log('   🔧 Applying rule-based refactoring...');

    let refactored = content;
    let changesApplied = 0;

    // Simple find-and-replace for prohibited terms
    const replacements = {
      'alternative currency': 'local trade network',
      'asset-backed money': 'tangible value exchange',
      'financial sovereignty': 'economic self-reliance',
      'divine mandate': 'faith-inspired purpose',
      'prophetic certainty': 'scriptural guidance',
    };

    Object.entries(replacements).forEach(([bad, good]) => {
      const regex = new RegExp(bad, 'gi');
      if (regex.test(refactored)) {
        refactored = refactored.replace(regex, good);
        changesApplied++;
      }
    });

    if (changesApplied > 0) {
      if (this.options.dryRun) {
        console.log(`   📋 Dry run - would apply ${changesApplied} rule-based changes`);
      } else {
        this.backupFile(filePath);
        fs.writeFileSync(filePath, refactored, 'utf8');
        console.log(`   ✓ Applied ${changesApplied} rule-based changes`);
      }

      this.results.refactored.push({
        file: relativePath,
        changesApplied,
      });
    } else {
      console.log('   ⚠️  No automatic fixes available - manual review needed');
      this.results.skipped.push(relativePath);
    }
  }

  /**
   * Backup file before modification
   */
  backupFile(filePath) {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = path.basename(filePath);
    const backupPath = path.join(CONFIG.backupDir, `${filename}.${timestamp}.bak`);
    fs.copyFileSync(filePath, backupPath);
  }

  /**
   * Generate report
   */
  generateReport() {
    console.log('\n' + '='.repeat(60));
    console.log('📊 GOVERNANCE REFACTOR REPORT');
    console.log('='.repeat(60));
    console.log(`\nFiles scanned: ${this.results.scanned}`);
    console.log(`Files with issues: ${this.results.issues.length}`);
    console.log(`Files refactored: ${this.results.refactored.length}`);
    console.log(`Files skipped: ${this.results.skipped.length}`);
    console.log(`Errors: ${this.results.errors.length}`);

    if (this.results.issues.length > 0) {
      console.log('\n📋 ISSUES BY FILE:');
      this.results.issues.forEach(({ file, issues }) => {
        console.log(`\n  ${file}:`);
        issues.forEach(issue => {
          const severity = issue.severity === 'high' ? '🔴' : '🟡';
          console.log(`    ${severity} ${issue.type}: ${issue.message}`);
        });
      });
    }

    if (this.results.errors.length > 0) {
      console.log('\n❌ ERRORS:');
      this.results.errors.forEach(({ file, error }) => {
        console.log(`  ${file}: ${error}`);
      });
    }

    // Save JSON report
    const reportPath = path.join(
      CONFIG.outputDir,
      `report-${new Date().toISOString().replace(/[:.]/g, '-')}.json`
    );
    fs.writeFileSync(reportPath, JSON.stringify(this.results, null, 2), 'utf8');
    console.log(`\n💾 Full report saved: ${reportPath}`);

    // Save markdown summary
    const summaryPath = path.join(CONFIG.outputDir, 'LATEST-REPORT.md');
    this.generateMarkdownReport(summaryPath);
    console.log(`📄 Markdown summary: ${summaryPath}`);
  }

  /**
   * Generate markdown report
   */
  generateMarkdownReport(outputPath) {
    const lines = [
      '# Governance Refactor Report',
      `\nGenerated: ${new Date().toISOString()}`,
      '\n## Summary',
      `\n- Files scanned: ${this.results.scanned}`,
      `- Files with issues: ${this.results.issues.length}`,
      `- Files refactored: ${this.results.refactored.length}`,
      `- Files skipped: ${this.results.skipped.length}`,
      `- Errors: ${this.results.errors.length}`,
    ];

    if (this.results.issues.length > 0) {
      lines.push('\n## Issues Detected\n');
      this.results.issues.forEach(({ file, issues }) => {
        lines.push(`### ${file}\n`);
        issues.forEach(issue => {
          lines.push(`- **${issue.type}** (${issue.severity}): ${issue.message}`);
        });
        lines.push('');
      });
    }

    if (this.results.refactored.length > 0) {
      lines.push('\n## Files Refactored\n');
      this.results.refactored.forEach(item => {
        lines.push(`- ${item.file}`);
      });
    }

    fs.writeFileSync(outputPath, lines.join('\n'), 'utf8');
  }
}

// CLI handling
async function main() {
  const args = process.argv.slice(2);
  
  const options = {
    auditOnly: args.includes('--audit-only'),
    interactive: args.includes('--interactive'),
    dryRun: args.includes('--dry-run'),
    useAI: !args.includes('--no-ai'),
  };

  let target = 'all';
  
  if (args.includes('--all')) {
    target = 'all';
  } else if (args.includes('--section')) {
    const idx = args.indexOf('--section');
    target = args[idx + 1];
  } else if (args.includes('--file')) {
    const idx = args.indexOf('--file');
    target = args[idx + 1];
  }

  const tool = new GovernanceRefactorTool(options);
  await tool.init();
  await tool.scan(target);
  tool.generateReport();
}

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { GovernanceRefactorTool, GOVERNANCE_RULES };
