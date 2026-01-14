import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const repoRoot = process.cwd();

function parseArgs(argv) {
  const args = { site: '_site', outDir: path.join('_internal', 'reports') };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--site') {
      args.site = argv[i + 1];
      i += 1;
    } else if (arg === '--out') {
      args.outDir = argv[i + 1];
      i += 1;
    }
  }
  return args;
}

async function exists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function walkFiles(dirPath, extensions) {
  const results = [];
  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      results.push(...(await walkFiles(fullPath, extensions)));
      continue;
    }
    if (extensions.has(path.extname(entry.name).toLowerCase())) results.push(fullPath);
  }
  return results;
}

function decodeHtmlEntities(input) {
  return input.replaceAll('&amp;', '&');
}

function normalizeUrl(rawUrl) {
  const decoded = decodeHtmlEntities(rawUrl).trim();
  const withoutHash = decoded.split('#')[0];
  const withoutQuery = withoutHash.split('?')[0];
  return withoutQuery;
}

function safeDecodeUriPath(urlPath) {
  try {
    return decodeURI(urlPath);
  } catch {
    return urlPath;
  }
}

function isSkippableUrl(url) {
  const lower = url.toLowerCase();
  return (
    lower.startsWith('http://') ||
    lower.startsWith('https://') ||
    lower.startsWith('//') ||
    lower.startsWith('mailto:') ||
    lower.startsWith('tel:') ||
    lower.startsWith('data:')
  );
}

function toPosix(p) {
  return p.split(path.sep).join('/');
}

function resolveSiteTarget(siteDirAbs, fromFileAbs, urlPath) {
  // absolute site path
  if (urlPath.startsWith('/')) {
    const abs = path.join(siteDirAbs, urlPath.slice(1));
    // /foo/ -> /foo/index.html
    if (urlPath.endsWith('/')) return path.join(abs, 'index.html');
    // /foo -> /foo or /foo/index.html (try both later)
    return abs;
  }

  // relative path
  const abs = path.resolve(path.dirname(fromFileAbs), urlPath);
  if (urlPath.endsWith('/')) return path.join(abs, 'index.html');
  return abs;
}

async function targetExists(siteDirAbs, fromFileAbs, urlPath) {
  const resolved = resolveSiteTarget(siteDirAbs, fromFileAbs, urlPath);

  // Common patterns
  const candidates = [];
  // direct
  candidates.push(resolved);
  // if no extension and not trailing slash, allow index.html
  const ext = path.extname(resolved);
  if (!ext) {
    candidates.push(path.join(resolved, 'index.html'));
    candidates.push(`${resolved}.html`);
  }

  for (const c of candidates) {
    if (await exists(c)) return { ok: true, resolved: c };
  }
  return { ok: false, resolved };
}

async function main() {
  const args = parseArgs(process.argv);
  const siteDirAbs = path.isAbsolute(args.site) ? args.site : path.join(repoRoot, args.site);
  const outDirAbs = path.isAbsolute(args.outDir) ? args.outDir : path.join(repoRoot, args.outDir);

  if (!(await exists(siteDirAbs))) {
    console.error(`Site directory not found: ${siteDirAbs}`);
    process.exitCode = 2;
    return;
  }

  const pages = await walkFiles(siteDirAbs, new Set(['.html']));
  const linkRegex = /(?:href|src)=["']([^"']+?)["']/gi;

  const missing = [];

  for (const file of pages) {
    const content = await fs.readFile(file, 'utf8');
    let match;
    while ((match = linkRegex.exec(content)) !== null) {
      const raw = match[1];
      const normalized = safeDecodeUriPath(normalizeUrl(raw));
      if (!normalized || normalized === '#' || normalized.startsWith('#')) continue;
      if (normalized.includes('{{')) continue;
      if (isSkippableUrl(normalized)) continue;

      // Skip assets we don't care about
      const lower = normalized.toLowerCase();
      if (lower.endsWith('.pdf')) continue; // handled by check-pdf-links

      const check = await targetExists(siteDirAbs, file, normalized);
      if (!check.ok) {
        missing.push({
          url: normalized,
          referencedFrom: path.relative(siteDirAbs, file),
          resolvedAttempt: path.relative(repoRoot, check.resolved),
        });
      }
    }
  }

  await fs.mkdir(outDirAbs, { recursive: true });
  const outJson = path.join(outDirAbs, 'broken-site-links.json');
  const outMd = path.join(outDirAbs, 'broken-site-links.md');

  await fs.writeFile(
    outJson,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        siteDir: path.relative(repoRoot, siteDirAbs),
        pagesScanned: pages.length,
        missing,
      },
      null,
      2,
    ),
    'utf8',
  );

  const lines = [];
  lines.push('# Broken Internal Links Report');
  lines.push('');
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push(`Site dir: ${path.relative(repoRoot, siteDirAbs)}`);
  lines.push(`Pages scanned: ${pages.length}`);
  lines.push(`Missing links: ${missing.length}`);
  lines.push('');

  if (missing.length === 0) {
    lines.push('No missing internal links detected.');
  } else {
    for (const item of missing.slice(0, 200)) {
      lines.push(`- ${item.url}`);
      lines.push(`  - from: ${toPosix(item.referencedFrom)}`);
      lines.push(`  - attempted: ${toPosix(item.resolvedAttempt)}`);
    }
  }

  lines.push('');
  lines.push(`JSON: ${toPosix(path.relative(repoRoot, outJson))}`);
  lines.push(`MD: ${toPosix(path.relative(repoRoot, outMd))}`);

  await fs.writeFile(outMd, `${lines.join('\n')}\n`, 'utf8');

  console.log(`Wrote ${path.relative(repoRoot, outMd)}`);
  console.log(`Wrote ${path.relative(repoRoot, outJson)}`);

  if (missing.length > 0) process.exitCode = 1;
}

await main();
