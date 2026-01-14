// Minimal, Pages-safe intake: rename/move PDFs, update docket YAML, stage changes
import fs from 'fs';
import path from 'path';
import yaml from 'js-yaml';
import { spawnSync } from 'node:child_process';
import { getCourtStamp } from './extractCourtStamp.js';

const INBOX = ['_inbox', 'assets/uploads'];
// Canonical public filing home: /cases/<slug>/filings/<file>.pdf
const CASES_DIR = 'cases';
const CASES_INDEX_DIR = '_cases_index';
const DOCKET_DIR = '_data/docket_index';
const MAP_FILE = '_data/cases-map.yml';

const MIN_PDF_BYTES = 4096;

const readYml = p => fs.existsSync(p) ? yaml.load(fs.readFileSync(p, 'utf8'), { schema: yaml.JSON_SCHEMA }) : undefined;
const writeYml = (p, obj) => fs.writeFileSync(p, yaml.dump(obj, { lineWidth: 1000, schema: yaml.JSON_SCHEMA }));

const kebab = s => (s||'').toLowerCase().replace(/[^a-z0-9]+/g,'-').replace(/^-+|-+$/g,'');
const ensureDir = p => fs.mkdirSync(p, { recursive: true });
const slugifyFile = name => {
  const base = name.replace(/\.pdf$/i, '');
  const slug = kebab(base);
  return slug || 'document';
};

// Find all case markdown files (handles both direct .md files and subdirectory index.md files)
const findCaseFiles = () => {
  const files = [];
  if (!fs.existsSync(CASES_INDEX_DIR)) return files;
  const entries = fs.readdirSync(CASES_INDEX_DIR);
  for (const entry of entries) {
    const fullPath = path.join(CASES_INDEX_DIR, entry);
    const stat = fs.statSync(fullPath);
    if (stat.isFile() && entry.endsWith('.md') && entry !== '_TEMPLATE.md') {
      files.push(entry);
    } else if (stat.isDirectory()) {
      const indexPath = path.join(fullPath, 'index.md');
      if (fs.existsSync(indexPath)) {
        files.push(path.join(entry, 'index.md'));
      }
    }
  }
  return files;
};

const caseFiles = findCaseFiles();

// Bootstrap cases-map.yml from case front matter if missing/empty
let casesMap = readYml(MAP_FILE) || {};
const bootstrapMapFromCases = () => {
  const map = {};
  
  // Parse all case files and collect docket info
  const caseData = [];
  for (const f of caseFiles) {
    const txt = fs.readFileSync(path.join(CASES_INDEX_DIR, f), 'utf8');
    const fm = (txt.match(/^---\n([\s\S]*?)\n---/)||[])[1]||'';
    
    // Extract slug from permalink or use directory name for subdirectories
    let slug = (fm.match(/permalink:\s*\/cases\/([a-z0-9-]+)\//i)||[])[1];
    if (!slug) {
      // If f is like "dirname/index.md", use dirname as slug
      const parts = f.split(path.sep);
      slug = parts.length > 1 ? parts[0] : f.replace(/\.md$/,'');
    }
    
    // collect tokens from dockets[] and primary_docket
    const arr = [];
    
    // Handle inline array format: dockets: ["A-123", "B-456"]
    const mArr = fm.match(/dockets:\s*\[([^\]]+)\]/);
    if (mArr) {
      mArr[1].split(',').forEach(x => arr.push(x.replace(/["'\s]/g,'')));
    }
    
    // Handle YAML list format: dockets:\n  - "A-123"\n  - "B-456"
    const docketsSection = fm.match(/dockets:\s*\n((?:\s+-\s+[^\n]+\n?)+)/);
    if (docketsSection) {
      const mLine = [...docketsSection[1].matchAll(/^\s*-\s*["']?([A-Za-z0-9:\-]+)["']?\s*$/gm)];
      mLine.forEach(x => arr.push((x[1]||'').trim()));
    }
    
    // Handle single docket field
    const mSingle = fm.match(/^docket:\s*["']?([A-Za-z0-9:\-]+)["']?\s*$/m);
    if (mSingle) arr.push(mSingle[1]);
    
    // Handle primary_docket field
    const mPrimary = fm.match(/primary_docket:\s*["']?([A-Za-z0-9:\-]+)["']?/);
    if (mPrimary) arr.push(mPrimary[1]);
    
    const dockets = arr.filter(Boolean);
    if (dockets.length > 0) {
      caseData.push({ slug, dockets });
    }
  }
  
  // Sort by number of dockets descending - process consolidated cases first
  // This ensures multi-docket cases take priority over single-docket pages
  caseData.sort((a, b) => b.dockets.length - a.dockets.length);
  
  // Build the map
  for (const { slug, dockets } of caseData) {
    dockets.forEach(tok => { 
      if (!map[tok]) map[tok] = slug; 
    });
  }
  
  return map;
};

if (Object.keys(casesMap).length === 0) {
  casesMap = bootstrapMapFromCases();
  ensureDir(path.dirname(MAP_FILE));
  writeYml(MAP_FILE, casesMap);
  console.log('Bootstrapped _data/cases-map.yml from case front matter.');
}

// Build reverse index: token->slug from case front matter + map file
const slugFromDocket = token => {
  if (casesMap[token]) return casesMap[token];
  // Try filename inference (last segment of permalink or directory name)
  for (const f of caseFiles) {
    const txt = fs.readFileSync(path.join(CASES_INDEX_DIR, f), 'utf8');
    const fm = (txt.match(/^---\n([\s\S]*?)\n---/)||[])[1]||'';
    const m = fm.match(/dockets:\s*\[([^\]]+)\]/) || fm.match(/dockets:\s*-\s*(.+)/);
    
    let slug = (fm.match(/permalink:\s*\/cases\/([a-z0-9-]+)\//i)||[])[1];
    if (!slug) {
      // If f is like "dirname/index.md", use dirname as slug
      const parts = f.split(path.sep);
      slug = parts.length > 1 ? parts[0] : f.replace(/\.md$/,'');
    }
    
    if (m && m[0].includes(token)) return slug;
  }
  return null;
};

const guessType = name => {
  const n = name.toLowerCase();
  if (n.includes('order')) return 'Order';
  if (n.includes('notice')) return 'Notice';
  if (n.includes('brief')) return 'Brief';
  if (n.includes('exhibit')) return 'Exhibit';
  if (n.includes('motion')) return 'Motion';
  return 'Filing';
};

const guessDate = (name, statsMtime) => {
  const m = name.match(/(20\d{2})[-_\.]?(0[1-9]|1[0-2])[-_\.]?(0[1-9]|[12]\d|3[01])/);
  if (m) return `${m[1]}-${m[2]}-${m[3]}`;
  const d = new Date(statsMtime);
  return d.toISOString().slice(0,10);
};

const discoverPdfs = () => {
  const out = [];
  INBOX.forEach(dir=>{
    if (!fs.existsSync(dir)) return;
    const walk = d => {
      const items = fs.readdirSync(d);
      items.forEach(it=>{
        const p = path.join(d,it);
        const st = fs.statSync(p);
        if (st.isDirectory()) return walk(p);
        if (p.toLowerCase().endsWith('.pdf')) out.push({p, st});
      });
    };
    walk(dir);
  });
  return out;
};

const loadDocket = slug => {
  const p = path.join(DOCKET_DIR, `${slug}.yml`);
  return { p, list: readYml(p) || [] };
};

const fileUrlToRepoPath = (fileUrl) => {
  if (typeof fileUrl !== 'string' || fileUrl.length === 0) return null;
  const normalized = fileUrl.startsWith('/') ? fileUrl.slice(1) : fileUrl;
  return path.normalize(normalized);
};

const ensurePdfHomesForDocket = (slug) => {
  const { list } = loadDocket(slug);
  const missing = [];

  for (const entry of list) {
    if (!entry || typeof entry.file !== 'string') continue;
    const repoPath = fileUrlToRepoPath(entry.file);
    if (!repoPath) continue;
    if (!fs.existsSync(repoPath)) {
      missing.push(repoPath);
      continue;
    }
    try {
      const st = fs.statSync(repoPath);
      if (st.size === 0) missing.push(repoPath);
    } catch {
      missing.push(repoPath);
    }
  }

  if (missing.length === 0) return;

  const args = ['scripts/generate-placeholder-pdfs.js', '--min-bytes', String(MIN_PDF_BYTES), ...missing];
  const res = spawnSync('node', args, { stdio: 'inherit' });
  if (res.status !== 0) {
    console.warn(`Warning: placeholder generation exited with status ${res.status}`);
  }
};

const main = () => {
  console.log('='.repeat(60));
  console.log('📥 DOCKET INTAKE - Starting automated PDF processing');
  console.log('='.repeat(60));
  
  const items = discoverPdfs();
  if (!items.length) { 
    console.log('✓ No PDFs to intake - inbox is empty'); 
    return; 
  }

  console.log(`\n📋 Found ${items.length} PDF file(s) to process\n`);
  const changes = [];
  for (const {p, st} of items) {
    const base = path.basename(p);
    console.log(`\n${'─'.repeat(60)}`);
    console.log(`📄 Processing: ${base}`);
    // Try to match by known tokens in filename - support multiple patterns:
    // ATL-L-002794-25, ATL-22-002292, A-000313-25, 1:25-cv-15641-RMB-MJS, etc.
    // Patterns are ordered from most specific to least specific
    const patterns = [
      /\b(\d:\d{2}-cv-\d{5}(?:-[A-Z]{3}(?:-[A-Z]{3})?)?)\b/i,  // 1:25-cv-15641-RMB-MJS (federal)
      /\b(ATL-[A-Z]{1,2}-\d{6}-\d{2})\b/i,      // ATL-L-002794-25, ATL-DC-007956-25
      /\b(ATL-\d{2}-\d{6})\b/i,                 // ATL-22-002292, ATL-24-001934
      /\b(A-\d{6}-\d{2})\b/                     // A-000313-25
    ];
    let token = null;
    for (const pattern of patterns) {
      const match = base.match(pattern);
      if (match) {
        token = match[1];
        break;
      }
    }
    let slug = token ? slugFromDocket(token) : null;

    // If path already under cases/<slug>/..., respect it
    const mUnder = p.match(/cases\/([a-z0-9-]+)\//i);
    if (mUnder) slug = mUnder[1];

    // NEW: Check if file is in _inbox/<case-slug>/ subdirectory
    if (!slug) {
      const mInbox = p.match(/_inbox[\\\/]([a-z0-9-]+)[\\\/]/i);
      if (mInbox) {
        const inboxSlug = mInbox[1];
        // Verify this slug exists in cases map
        if (Object.values(casesMap).includes(inboxSlug)) {
          slug = inboxSlug;
          console.log(`   ✓ Matched from inbox folder: _inbox/${inboxSlug}/ → ${slug}`);
        }
      }
    }

    // If still unknown, park under 'unassigned'
    if (!slug) {
      console.log(`   ⚠️  No case match found - filing to "unassigned"`);
      slug = 'unassigned';
    } else if (token) {
      console.log(`   ✓ Matched docket: ${token} → ${slug}`);
    }

    const date = guessDate(base, st.mtimeMs);
    const type = guessType(base);
    
    // Clean title: remove date prefixes and normalize
    let title = base.replace(/\.pdf$/i,'');
    // Remove date prefix (YYYY-MM-DD or YYYY_MM_DD or YYYYMMDD)
    title = title.replace(/^\d{4}[-_]?\d{2}[-_]?\d{2}[-_]?/g, '');
    // Remove type prefix if redundant
    title = title.replace(new RegExp(`^${type}[-_\\s]+`, 'i'), '');
    // Normalize separators and spacing
    title = title.replace(/[_-]+/g,' ').replace(/\s+/g,' ').trim();
    
    const stub = slugifyFile(base);
    let newName = `${date}_${type}_${stub}.pdf`;

    const destDir = path.join(CASES_DIR, slug, 'filings');
    ensureDir(destDir);
    let destPath = path.join(destDir, newName);

    // Avoid collisions when slugified filenames match existing items
    let suffix = 1;
    while (fs.existsSync(destPath)) {
      newName = `${date}_${type}_${stub}-${suffix}.pdf`;
      destPath = path.join(destDir, newName);
      suffix++;
    }

    console.log(`   📁 Destination: ${destPath.replace(/\\/g, '/')}`);
    console.log(`   📝 Metadata: ${date} | ${type} | ${title}`);
    
    fs.renameSync(p, destPath);
    console.log(`   ✓ File moved successfully`);

    // Extract court stamp from PDF (Node.js first, fallback to Python OCR)
    let courtStamp = null;
    try {
      courtStamp = getCourtStamp(destPath);
      if (courtStamp) {
        console.log(`   🏛️ Court stamp extracted: ${courtStamp}`);
      } else {
        console.log('   ⚠️ No court stamp found');
      }
    } catch (err) {
      console.log('   ⚠️ Court stamp extraction error:', err);
    }

    const { p: docketFile, list } = loadDocket(slug);
    const id = `${date}-${stub}`.slice(0,64);
    const fileUrl = `/${destPath.replace(/\\/g,'/')}`;

    // De-dup if file already indexed
    const exists = list.find(e => e.file === fileUrl);
    if (!exists) {
      list.push({
        id, date,
        type, title: title || stub.replace(/-/g,' '),
        file: fileUrl,
        court_stamp: courtStamp,
        notes: p.includes('_inbox') ? 'Intake: moved from _inbox' : 'Intake: assets/uploads'
      });
      ensureDir(DOCKET_DIR);
      writeYml(docketFile, list.sort((a,b)=> (a.date<b.date?1:-1)));
      console.log(`   ✓ Added to docket: ${docketFile}`);
    } else {
      console.log(`   ⚠️  Already in docket - skipped metadata update`);
    }

    // Make sure every docket entry has a real file behind it.
    // This only creates placeholders for missing/0-byte PDFs (it does not overwrite real PDFs).
    ensurePdfHomesForDocket(slug);

    changes.push({ from: p, to: destPath, docketFile });
  }

  // Write a simple report
  ensureDir('reports');
  fs.writeFileSync('reports/docket-intake.json', JSON.stringify(changes, null, 2));
  
  console.log('\n' + '='.repeat(60));
  console.log(`✅ INTAKE COMPLETE: ${changes.length} file(s) processed`);
  console.log('='.repeat(60));
  
  // Summary by case
  const byCaseSlug = {};
  changes.forEach(c => {
    const slug = c.to.match(/cases[\/\\]([^\/\\]+)/)?.[1] || 'unknown';
    byCaseSlug[slug] = (byCaseSlug[slug] || 0) + 1;
  });
  
  console.log('\n📊 Files by case:');
  Object.entries(byCaseSlug).sort((a,b) => b[1] - a[1]).forEach(([slug, count]) => {
    console.log(`   ${slug.padEnd(30)} ${count} file(s)`);
  });
  
  console.log(`\n📄 Full report: reports/docket-intake.json`);
  console.log('\n💡 Next: Review changes and create pull request\n');
};

main();
