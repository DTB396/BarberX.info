import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import yaml from 'js-yaml';

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

async function exists(filePath) {
  try {
    await fs.access(filePath);
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
    if (extensions.has(path.extname(entry.name).toLowerCase())) {
      results.push(fullPath);
    }
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

function toPosixUrlPath(filePath) {
  return filePath.split(path.sep).join('/');
}

async function scanBuiltSiteForPdfLinks(siteDirAbs) {
  const files = await walkFiles(siteDirAbs, new Set(['.html', '.xml', '.json']));

  const pdfRegex = /(?:href|src)=["']([^"']+?\.pdf(?:\?[^"']*)?(?:#[^"']*)?)["']/gi;

  const missing = [];
  const foundCountByUrl = new Map();

  for (const file of files) {
    const content = await fs.readFile(file, 'utf8');
    let match;
    while ((match = pdfRegex.exec(content)) !== null) {
      const rawUrl = match[1];
      const normalized = normalizeUrl(rawUrl);
      if (!normalized || isSkippableUrl(normalized) || normalized.includes('{{')) continue;

      const normalizedDecoded = safeDecodeUriPath(normalized);

      const count = foundCountByUrl.get(normalizedDecoded) ?? 0;
      foundCountByUrl.set(normalizedDecoded, count + 1);

      let resolved;
      if (normalizedDecoded.startsWith('/')) {
        resolved = path.join(siteDirAbs, normalizedDecoded.slice(1));
      } else {
        resolved = path.resolve(path.dirname(file), normalizedDecoded);
      }

      if (!(await exists(resolved))) {
        missing.push({
          url: normalizedDecoded,
          referencedFrom: path.relative(siteDirAbs, file),
          resolvedSitePath: path.relative(repoRoot, resolved),
        });
      }
    }
  }

  return {
    totalPagesScanned: files.length,
    uniquePdfUrls: foundCountByUrl.size,
    totalPdfReferences: Array.from(foundCountByUrl.values()).reduce((a, b) => a + b, 0),
    missing,
  };
}

async function scanDocketYamlForPdfLinks(siteDirAbs) {
  const docketDirAbs = path.join(repoRoot, '_data', 'docket');
  if (!(await exists(docketDirAbs))) return { docketsScanned: 0, missing: [] };

  const ymlFiles = (await fs.readdir(docketDirAbs)).filter((f) => f.toLowerCase().endsWith('.yml'));
  const missing = [];

  for (const ymlFile of ymlFiles) {
    const slug = ymlFile.replace(/\.yml$/i, '');
    const abs = path.join(docketDirAbs, ymlFile);
    const raw = await fs.readFile(abs, 'utf8');

    let entries;
    try {
      entries = yaml.load(raw);
    } catch (error) {
      missing.push({
        slug,
        yml: path.relative(repoRoot, abs),
        type: 'yaml-parse-error',
        error: String(error),
      });
      continue;
    }

    if (!Array.isArray(entries)) continue;

    for (const entry of entries) {
      const fileUrl = entry?.file;
      if (typeof fileUrl !== 'string' || !fileUrl.toLowerCase().includes('.pdf')) continue;
      const normalized = safeDecodeUriPath(normalizeUrl(fileUrl));
      if (isSkippableUrl(normalized)) continue;

      const urlForCheck = normalized.startsWith('/') ? normalized : `/${normalized}`;
      const repoPath = path.join(repoRoot, urlForCheck.slice(1));
      const sitePath = path.join(siteDirAbs, urlForCheck.slice(1));

      const repoExists = await exists(repoPath);
      const siteExists = await exists(sitePath);

      if (!repoExists || !siteExists) {
        missing.push({
          slug,
          yml: path.relative(repoRoot, abs),
          id: entry?.id ?? null,
          date: entry?.date ?? null,
          title: entry?.title ?? null,
          url: urlForCheck,
          repoPath: path.relative(repoRoot, repoPath),
          sitePath: path.relative(repoRoot, sitePath),
          repoExists,
          siteExists,
        });
      }
    }
  }

  return { docketsScanned: ymlFiles.length, missing };
}

async function ensureDir(dirPath) {
  await fs.mkdir(dirPath, { recursive: true });
}

function summarizeMissing(missing) {
  const byUrl = new Map();
  for (const item of missing) {
    const key = item.url;
    const existing = byUrl.get(key) ?? { url: key, count: 0, examples: [] };
    existing.count += 1;
    if (existing.examples.length < 5) existing.examples.push(item.referencedFrom ?? item.yml ?? 'unknown');
    byUrl.set(key, existing);
  }

  return Array.from(byUrl.values()).sort((a, b) => b.count - a.count);
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

  const siteScan = await scanBuiltSiteForPdfLinks(siteDirAbs);
  const docketScan = await scanDocketYamlForPdfLinks(siteDirAbs);

  await ensureDir(outDirAbs);
  const outJson = path.join(outDirAbs, 'broken-pdf-links.json');
  const outMd = path.join(outDirAbs, 'broken-pdf-links.md');

  const combinedMissing = [
    ...siteScan.missing.map((m) => ({ source: 'site', ...m })),
    ...docketScan.missing.map((m) => ({ source: 'docket', ...m })),
  ];

  await fs.writeFile(
    outJson,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        siteDir: path.relative(repoRoot, siteDirAbs),
        siteScan,
        docketScan,
        combinedMissing,
      },
      null,
      2,
    ),
    'utf8',
  );

  const summary = summarizeMissing(combinedMissing);
  const lines = [];
  lines.push('# Broken PDF Links Report');
  lines.push('');
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push(`Site dir: ${path.relative(repoRoot, siteDirAbs)}`);
  lines.push('');
  lines.push(`- Pages scanned: ${siteScan.totalPagesScanned}`);
  lines.push(`- PDF references: ${siteScan.totalPdfReferences}`);
  lines.push(`- Unique PDF URLs: ${siteScan.uniquePdfUrls}`);
  lines.push(`- Missing references (raw hits): ${siteScan.missing.length}`);
  lines.push(`- Docket YAML files scanned: ${docketScan.docketsScanned}`);
  lines.push(`- Docket entries with missing file in repo and/or _site: ${docketScan.missing.length}`);
  lines.push('');
  lines.push('## Top Missing URLs');
  lines.push('');

  if (summary.length === 0) {
    lines.push('No missing PDF links detected.');
  } else {
    for (const item of summary.slice(0, 100)) {
      lines.push(`- ${item.url} (count: ${item.count})`);
      for (const ex of item.examples) {
        lines.push(`  - e.g. ${toPosixUrlPath(String(ex))}`);
      }
    }
  }

  lines.push('');
  lines.push(`JSON: ${toPosixUrlPath(path.relative(repoRoot, outJson))}`);
  lines.push(`MD: ${toPosixUrlPath(path.relative(repoRoot, outMd))}`);

  await fs.writeFile(outMd, `${lines.join('\n')}\n`, 'utf8');

  console.log(`Wrote ${path.relative(repoRoot, outMd)}`);
  console.log(`Wrote ${path.relative(repoRoot, outJson)}`);

  if (combinedMissing.length > 0) {
    process.exitCode = 1;
  }
}

await main();
