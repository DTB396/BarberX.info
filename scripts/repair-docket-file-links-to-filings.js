import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import yaml from 'js-yaml';

const repoRoot = process.cwd();
const docketDir = path.join(repoRoot, '_data', 'docket');

function parseArgs(argv) {
  const args = { dryRun: false, only: null };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--dry-run') args.dryRun = true;
    if (arg === '--only') {
      args.only = argv[i + 1];
      i += 1;
    }
  }
  return args;
}

async function exists(filePath) {
  try {
    await fs.access(filePath);
    return true;
  } catch {
    return false;
  }
}

function normalizeTokens(text) {
  return String(text)
    .toLowerCase()
    .replace(/\.pdf$/i, '')
    .replace(/\d{4}-\d{2}-\d{2}/g, ' ')
    .replace(/\d{8}/g, ' ')
    .replace(/\d{2}-\d{2}-\d{4}/g, ' ')
    .split(/[^a-z0-9]+/g)
    .map((t) => t.trim())
    .filter(Boolean);
}

function intersectionScore(aTokens, bTokens) {
  const a = new Set(aTokens);
  let score = 0;
  for (const token of bTokens) {
    if (a.has(token)) score += 1;
  }
  return score;
}

function dateVariants(dateStr) {
  // dateStr: YYYY-MM-DD
  const m = String(dateStr).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!m) return [];
  const yyyy = m[1];
  const mm = m[2];
  const dd = m[3];
  return [`${yyyy}${mm}${dd}`, `${yyyy}-${mm}-${dd}`, `${mm}-${dd}-${yyyy}`];
}

async function listPdfCandidates(slug) {
  const dirs = [
    path.join(repoRoot, 'cases', slug, 'filings'),
    path.join(repoRoot, 'cases', slug),
  ];

  const candidates = [];
  for (const dir of dirs) {
    if (!(await exists(dir))) continue;
    const entries = await fs.readdir(dir, { withFileTypes: true });
    for (const entry of entries) {
      if (!entry.isFile()) continue;
      if (!entry.name.toLowerCase().endsWith('.pdf')) continue;
      candidates.push({
        filename: entry.name,
        absPath: path.join(dir, entry.name),
        // we always publish with a /cases/<slug>/filings/<file> link when possible
        preferredUrl: dir.endsWith(path.join('cases', slug, 'filings'))
          ? `/cases/${slug}/filings/${encodeURI(entry.name)}`
          : `/cases/${slug}/${encodeURI(entry.name)}`,
      });
    }
  }
  return candidates;
}

function preserveHeader(rawText) {
  const idx = rawText.search(/^\s*-\s+id\s*:/m);
  if (idx <= 0) return { header: '', body: rawText };
  return { header: rawText.slice(0, idx).replace(/\s+$/g, ''), body: rawText.slice(idx) };
}

function preferDecodeUrl(url) {
  try {
    return decodeURI(url);
  } catch {
    return url;
  }
}

async function repairOneDocketFile(absYmlPath, args) {
  const slug = path.basename(absYmlPath).replace(/\.yml$/i, '');
  if (args.only && slug !== args.only) return { slug, changed: false, fixes: 0, skipped: 0 };

  const raw = await fs.readFile(absYmlPath, 'utf8');
  const { header } = preserveHeader(raw);

  let data;
  try {
    data = yaml.load(raw, { schema: yaml.FAILSAFE_SCHEMA });
  } catch {
    return { slug, changed: false, fixes: 0, skipped: 0, error: 'yaml-parse-error' };
  }

  if (!Array.isArray(data)) return { slug, changed: false, fixes: 0, skipped: 0 };

  const candidates = await listPdfCandidates(slug);
  const candidateByAbs = new Map(candidates.map((c) => [c.absPath, c]));

  let fixes = 0;
  let skipped = 0;
  let changed = false;

  for (const entry of data) {
    if (!entry || typeof entry !== 'object') continue;
    if (typeof entry.file !== 'string' || !entry.file.toLowerCase().includes('.pdf')) continue;

    const fileUrl = entry.file;
    const normalizedUrl = fileUrl.startsWith('/') ? fileUrl : `/${fileUrl}`;
    const repoPath = path.join(repoRoot, preferDecodeUrl(normalizedUrl).slice(1));
    if (await exists(repoPath)) {
      // Already points to a real repo file (still might not be in _site, but Jekyll should copy it)
      continue;
    }

    const dv = dateVariants(entry.date);
    const oldTokens = normalizeTokens(entry.title || entry.file);

    let filtered = candidates;
    if (dv.length > 0) {
      filtered = candidates.filter((c) => dv.some((v) => c.filename.startsWith(v)));
      if (filtered.length === 0) filtered = candidates;
    }

    let best = null;
    let bestScore = -1;
    let secondScore = -1;

    for (const c of filtered) {
      const score = intersectionScore(oldTokens, normalizeTokens(c.filename));
      if (score > bestScore) {
        secondScore = bestScore;
        bestScore = score;
        best = c;
      } else if (score > secondScore) {
        secondScore = score;
      }
    }

    const uniqueByDate = dv.length > 0 && filtered.length === 1;
    const confidentByScore = best && bestScore >= 2 && bestScore > secondScore;

    if (uniqueByDate || confidentByScore) {
      entry.file = best.preferredUrl;
      fixes += 1;
      changed = true;
    } else {
      skipped += 1;
    }
  }

  if (!changed) return { slug, changed: false, fixes, skipped };

  const dumped = yaml.dump(data, { lineWidth: 1000, noRefs: true, sortKeys: false });
  const output = header ? `${header}\n\n${dumped}` : dumped;

  if (!args.dryRun) {
    await fs.writeFile(absYmlPath, output, 'utf8');
  }

  return { slug, changed: true, fixes, skipped };
}

async function main() {
  const args = parseArgs(process.argv);
  if (!(await exists(docketDir))) {
    console.error('Missing _data/docket directory');
    process.exitCode = 2;
    return;
  }

  const ymlFiles = (await fs.readdir(docketDir))
    .filter((f) => f.toLowerCase().endsWith('.yml'))
    .map((f) => path.join(docketDir, f));

  const results = [];
  for (const file of ymlFiles) {
    results.push(await repairOneDocketFile(file, args));
  }

  const changed = results.filter((r) => r.changed);
  const totalFixes = results.reduce((sum, r) => sum + (r.fixes || 0), 0);
  const totalSkipped = results.reduce((sum, r) => sum + (r.skipped || 0), 0);

  for (const r of changed) {
    console.log(`${r.slug}: fixed ${r.fixes}, skipped ${r.skipped}`);
  }

  console.log(`Total: changed ${changed.length}/${results.length} docket files; fixed ${totalFixes}; skipped ${totalSkipped}`);

  if (args.dryRun) {
    console.log('Dry run only (no files written).');
  }

  // non-zero if there were still skips (means review still needed)
  if (totalSkipped > 0) process.exitCode = 1;
}

await main();
