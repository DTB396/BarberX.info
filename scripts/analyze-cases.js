#!/usr/bin/env node
// OpenAI Case Analysis Script
// Analyzes case records and dockets from judicial oversight and journalistic perspectives

import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import OpenAI from 'openai';
import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

// Check for API key before initializing OpenAI
if (!process.env.OPENAI_API_KEY) {
  console.log('⏭️  OpenAI API key not found - skipping analysis');
  console.log('');
  console.log('To enable AI-powered case analysis:');
  console.log('  1. Get an API key from https://platform.openai.com/api-keys');
  console.log('  2. Add it to .env file: OPENAI_API_KEY=sk-...');
  console.log('  3. Or set as environment variable');
  console.log('');
  console.log('See _docs/ANALYSIS-SYSTEM.md for complete setup instructions.');
  console.log('');
  process.exit(0); // Exit gracefully - this is not an error condition
}

// Initialize OpenAI client
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const CASES_DIR = '_cases';
const DOCKET_DIR = '_data/docket';
const ANALYSIS_DIR = '_data/analysis';
const CASES_MAP_FILE = '_data/cases-map.yml';

// Helper functions
const ensureDir = (dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
};

const readYaml = (filePath) => {
  if (!fs.existsSync(filePath)) return null;
  try {
    return yaml.load(fs.readFileSync(filePath, 'utf8'), { schema: yaml.JSON_SCHEMA });
  } catch (e) {
    console.error(`Error reading YAML file ${filePath}:`, e.message);
    return null;
  }
};

const writeYaml = (filePath, data) => {
  try {
    fs.writeFileSync(filePath, yaml.dump(data, { lineWidth: 1000, noRefs: true, schema: yaml.JSON_SCHEMA }));
  } catch (e) {
    console.error(`Error writing YAML file ${filePath}:`, e.message);
  }
};

// Extract front matter from markdown file
const extractFrontMatter = (content) => {
  const match = content.match(/^---\n([\s\S]*?)\n---/);
  if (!match) return null;
  try {
    return yaml.load(match[1], { schema: yaml.JSON_SCHEMA });
  } catch (e) {
    console.error('Error parsing front matter:', e.message);
    return null;
  }
};

// Read case file
const readCaseFile = (filename) => {
  const filePath = path.join(CASES_DIR, filename);
  if (!fs.existsSync(filePath)) return null;
  
  const content = fs.readFileSync(filePath, 'utf8');
  const frontMatter = extractFrontMatter(content);
  
  if (!frontMatter) return null;
  
  // Extract the markdown content after front matter
  const markdownContent = content.replace(/^---\n[\s\S]*?\n---\n/, '');
  
  return {
    filename,
    frontMatter,
    content: markdownContent,
  };
};

// Get case slug from filename or permalink
const getCaseSlug = (caseData) => {
  if (caseData.frontMatter.permalink) {
    const match = caseData.frontMatter.permalink.match(/\/cases\/([^\/]+)\/?$/);
    if (match) return match[1];
  }
  return caseData.filename.replace(/\.md$/, '');
};

// Read docket entries for a case
const readDocketEntries = (slug) => {
  const docketFile = path.join(DOCKET_DIR, `${slug}.yml`);
  return readYaml(docketFile) || [];
};

// Format case data for OpenAI analysis
const formatCaseForAnalysis = (caseData, docketEntries) => {
  const fm = caseData.frontMatter;
  const docketSummary = docketEntries.map(entry => 
    `- ${entry.date}: ${entry.type} - ${entry.title}${entry.notes ? ' (' + entry.notes + ')' : ''}`
  ).join('\n');
  
  return `
CASE INFORMATION:
Title: ${fm.title || 'Unknown'}
Docket: ${fm.docket || fm.dockets?.join(', ') || 'N/A'}
Court: ${fm.court || 'N/A'}
Case Type: ${fm.case_type || 'N/A'}
Status: ${fm.status || 'N/A'}
Filed Date: ${fm.filed_date || 'N/A'}
Judge: ${fm.judge || 'N/A'}

CASE CONTENT:
${caseData.content}

DOCKET ENTRIES:
${docketSummary || 'No docket entries available'}
`.trim();
};

// Helper function to retry API calls with exponential backoff
const retryWithBackoff = async (fn, maxRetries = 5) => {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (error) {
      if (error.status === 429 && attempt < maxRetries - 1) {
        // Rate limit hit - extract wait time from error message or use exponential backoff
        const retryMatch = error.message.match(/try again in (\d+)([smh])/);
        let waitTime = 20000; // Default 20 seconds
        
        if (retryMatch) {
          const value = parseInt(retryMatch[1]);
          const unit = retryMatch[2];
          if (unit === 's') waitTime = value * 1000;
          else if (unit === 'm') waitTime = value * 60 * 1000;
          else if (unit === 'h') waitTime = value * 60 * 60 * 1000;
        } else {
          // Exponential backoff: 20s, 40s, 80s, 160s, 320s
          waitTime = Math.min(20000 * Math.pow(2, attempt), 320000);
        }
        
        console.log(`Rate limit hit. Waiting ${Math.round(waitTime / 1000)}s before retry ${attempt + 1}/${maxRetries}...`);
        await new Promise(resolve => setTimeout(resolve, waitTime));
      } else {
        throw error;
      }
    }
  }
  throw new Error('Max retries exceeded');
};

// Generate judicial oversight analysis
const generateJudicialAnalysis = async (caseData, docketEntries) => {
  const caseContent = formatCaseForAnalysis(caseData, docketEntries);
  
  const prompt = `You are a legal analyst with expertise in judicial oversight and constitutional law. 
Analyze the following case from a judicial oversight perspective, focusing on:
- Due process considerations
- Procedural propriety
- Constitutional issues raised
- Judicial conduct and adherence to law
- Administrative justice concerns

Provide a thorough, objective analysis that highlights key judicial oversight issues.

${caseContent}`;

  try {
    return await retryWithBackoff(async () => {
      const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: 'You are an expert legal analyst specializing in judicial oversight, constitutional law, and due process. Provide detailed, thoughtful analysis based on legal principles.',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
        temperature: 0.7,
        max_tokens: 2000,
      });
      
      return response.choices[0].message.content;
    });
  } catch (error) {
    console.error('Error generating judicial analysis:', error.message);
    return null;
  }
};

// Generate journalistic commentary
const generateJournalisticCommentary = async (caseData, docketEntries) => {
  const caseContent = formatCaseForAnalysis(caseData, docketEntries);
  
  const prompt = `You are an American journalist committed to freedom, transparency, and holding power accountable. 
Analyze this case from a journalistic perspective, focusing on:
- Public interest and transparency
- Individual rights and freedoms
- Government accountability
- Access to justice
- Broader societal implications
- The human story behind the legal proceedings

Write in a clear, compelling style that makes the case accessible to the general public while maintaining journalistic integrity.

${caseContent}`;

  try {
    return await retryWithBackoff(async () => {
      const response = await openai.chat.completions.create({
        model: 'gpt-4o-mini',
        messages: [
          {
            role: 'system',
            content: 'You are an investigative journalist focused on civil liberties, government accountability, and constitutional rights. Write with clarity, passion for truth, and respect for individual freedom.',
          },
          {
            role: 'user',
            content: prompt,
          },
        ],
        temperature: 0.8,
        max_tokens: 2000,
      });
      
      return response.choices[0].message.content;
    });
  } catch (error) {
    console.error('Error generating journalistic commentary:', error.message);
    return null;
  }
};

// Save analysis to file
const saveAnalysis = (slug, analysis) => {
  ensureDir(ANALYSIS_DIR);
  const analysisFile = path.join(ANALYSIS_DIR, `${slug}.yml`);
  
  const data = {
    generated_at: new Date().toISOString(),
    slug,
    judicial_oversight: analysis.judicial || null,
    journalistic_commentary: analysis.journalistic || null,
  };
  
  writeYaml(analysisFile, data);
  console.log(`✓ Analysis saved: ${analysisFile}`);
};

// Main function
const main = async () => {
  // Ensure analysis directory exists
  ensureDir(ANALYSIS_DIR);
  
  // Get all case files
  const caseFiles = fs.readdirSync(CASES_DIR)
    .filter(f => f.endsWith('.md') && !f.startsWith('_TEMPLATE'));
  
  if (caseFiles.length === 0) {
    console.log('No case files found to analyze.');
    return;
  }
  
  console.log(`Found ${caseFiles.length} case(s) to analyze.\n`);
  
  // Process each case
  for (const filename of caseFiles) {
    console.log(`\nAnalyzing: ${filename}`);
    console.log('─'.repeat(60));
    
    const caseData = readCaseFile(filename);
    if (!caseData) {
      console.log(`⚠ Skipped: Unable to read case file`);
      continue;
    }
    
    // Skip unpublished cases
    if (caseData.frontMatter.published === false) {
      console.log(`⚠ Skipped: Case is not published`);
      continue;
    }
    
    const slug = getCaseSlug(caseData);
    const docketEntries = readDocketEntries(slug);
    
    console.log(`Case: ${caseData.frontMatter.title || 'Untitled'}`);
    console.log(`Slug: ${slug}`);
    console.log(`Docket entries: ${docketEntries.length}`);
    
    // Check if analysis already exists and is recent
    const analysisFile = path.join(ANALYSIS_DIR, `${slug}.yml`);
    if (fs.existsSync(analysisFile)) {
      const existingAnalysis = readYaml(analysisFile);
      if (existingAnalysis && existingAnalysis.generated_at) {
        const generatedDate = new Date(existingAnalysis.generated_at);
        const daysSinceGenerated = (Date.now() - generatedDate.getTime()) / (1000 * 60 * 60 * 24);
        
        if (daysSinceGenerated < 7) {
          console.log(`⚠ Skipped: Analysis already exists (generated ${daysSinceGenerated.toFixed(1)} days ago)`);
          continue;
        }
      }
    }
    
    console.log('\n🔍 Generating judicial oversight analysis...');
    const judicialAnalysis = await generateJudicialAnalysis(caseData, docketEntries);
    
    // Save partial result if judicial analysis succeeded
    if (judicialAnalysis) {
      saveAnalysis(slug, {
        judicial: judicialAnalysis,
        journalistic: null,
      });
      console.log('✓ Judicial analysis saved');
    }
    
    // Delay to respect rate limits (3 RPM = 20 seconds between requests minimum)
    console.log('⏱️  Waiting 20s to respect rate limits...');
    await new Promise(resolve => setTimeout(resolve, 20000));
    
    console.log('📰 Generating journalistic commentary...');
    const journalisticCommentary = await generateJournalisticCommentary(caseData, docketEntries);
    
    if (journalisticCommentary) {
      // Update with both analyses
      saveAnalysis(slug, {
        judicial: judicialAnalysis,
        journalistic: journalisticCommentary,
      });
      console.log('✓ Journalistic commentary saved');
    } else if (!judicialAnalysis) {
      console.log('⚠ No analysis generated due to errors');
    }
    
    // Add a delay between cases to avoid rate limiting
    await new Promise(resolve => setTimeout(resolve, 20000));
  }
  
  console.log('\n' + '═'.repeat(60));
  console.log('✓ Analysis complete!');
  console.log(`Results saved to: ${ANALYSIS_DIR}/`);
};

// Run the script
main().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});
