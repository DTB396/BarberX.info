import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const repoRoot = process.cwd();

function parseArgs(argv) {
  const args = {
    minBytes: 4096,
    outDir: path.join('_internal', 'reports'),
  };
  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--min-bytes') {
      args.minBytes = Number(argv[i + 1]);
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

async function walk(dirPath) {
  const out = [];
  if (!(await exists(dirPath))) return out;

  const entries = await fs.readdir(dirPath, { withFileTypes: true });
  for (const entry of entries) {
    const full = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      out.push(...(await walk(full)));
    } else {
      out.push(full);
    }
  }
  return out;
}

function toPosix(p) {
  return p.split(path.sep).join('/');
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

async function walkFiles(dirPath, extensions) {
  const results = [];
  if (!(await exists(dirPath))) return results;

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

async function scanBuiltSiteForPdfRefs(siteDirAbs) {
  const files = await walkFiles(siteDirAbs, new Set(['.html', '.xml', '.json']));
  const pdfRegex = /(?:href|src)=["']([^"']+?\.pdf(?:\?[^"']*)?(?:#[^"']*)?)["']/gi;

  const linkedSiteRelPaths = new Set();
  const foundCountByUrl = new Map();

  for (const file of files) {
    const content = await fs.readFile(file, 'utf8');
    let match;
    while ((match = pdfRegex.exec(content)) !== null) {
      const rawUrl = match[1];
      const normalized = normalizeUrl(rawUrl);
      if (!normalized || isSkippableUrl(normalized) || normalized.includes('{{')) continue;

      const normalizedDecoded = safeDecodeUriPath(normalized);
      foundCountByUrl.set(normalizedDecoded, (foundCountByUrl.get(normalizedDecoded) ?? 0) + 1);

      const resolvedAbs = normalizedDecoded.startsWith('/')
        ? path.join(siteDirAbs, normalizedDecoded.slice(1))
        : path.resolve(path.dirname(file), normalizedDecoded);

      const siteRel = path.relative(siteDirAbs, resolvedAbs);
      if (siteRel.startsWith('..')) continue;
      linkedSiteRelPaths.add(toPosix(siteRel));
    }
  }

  return {
    totalPagesScanned: files.length,
    uniquePdfUrls: foundCountByUrl.size,
    totalPdfReferences: Array.from(foundCountByUrl.values()).reduce((a, b) => a + b, 0),
    linkedSiteRelPaths,
  };
}

async function loadBrokenTodoPaths() {
  const candidatePaths = [
    path.join(repoRoot, '_internal', 'BROKEN-PDFS-TODO.md'),
    path.join(repoRoot, '_docs', 'BROKEN-PDFS-TODO.md'),
  ];

  for (const todoPath of candidatePaths) {
    if (!(await exists(todoPath))) continue;
    const text = await fs.readFile(todoPath, 'utf8');

    const matches = new Set();
    // Backtick-wrapped paths
    for (const m of text.matchAll(/`([^`]+?\.pdf)`/gi)) matches.add(m[1]);
    // Plain-text case paths (e.g. cases/<slug>/filings/<file>.pdf)
    for (const m of text.matchAll(/\b(?:cases|assets)\/[^\s`]+?\.pdf\b/gi)) matches.add(m[0]);

    if (matches.size > 0) return Array.from(matches);
  }

  return [];
}

async function statPdf(filePath) {
  const st = await fs.stat(filePath);
  return {
    path: path.relative(repoRoot, filePath),
    bytes: st.size,
    mtime: st.mtime.toISOString(),
  };
}

async function main() {
  const args = parseArgs(process.argv);

  const siteDirAbs = path.join(repoRoot, '_site');
  const sitePdfRefs = (await exists(siteDirAbs))
    ? await scanBuiltSiteForPdfRefs(siteDirAbs)
    : {
        totalPagesScanned: 0,
        uniquePdfUrls: 0,
        totalPdfReferences: 0,
        linkedSiteRelPaths: new Set(),
      };

  const scanRoots = [
    path.join(repoRoot, 'cases'),
    path.join(repoRoot, 'assets', 'cases'),
  ];

  const allFiles = [];
  for (const root of scanRoots) {
    allFiles.push(...(await walk(root)));
  }

  const pdfFiles = allFiles.filter((p) => p.toLowerCase().endsWith('.pdf'));

  const stats = [];
  for (const pdf of pdfFiles) {
    const st = await statPdf(pdf);
    const relPosix = toPosix(st.path);
    stats.push({
      ...st,
      linkedFromSite: sitePdfRefs.linkedSiteRelPaths.has(relPosix),
    });
  }

  const tiny = stats
    .filter((s) => s.bytes < args.minBytes)
    .sort((a, b) => a.bytes - b.bytes);

  const todoRefs = await loadBrokenTodoPaths();
  const todoStatus = [];
  for (const rel of todoRefs) {
    const abs = path.join(repoRoot, rel);
    if (await exists(abs)) {
      const st = await statPdf(abs);
      const relPosix = toPosix(path.normalize(rel));
      todoStatus.push({
        ref: rel,
        exists: true,
        bytes: st.bytes,
        linkedFromSite: sitePdfRefs.linkedSiteRelPaths.has(relPosix),
      });
    } else {
      todoStatus.push({ ref: rel, exists: false, bytes: null, linkedFromSite: false });
    }
  }

  const outDirAbs = path.isAbsolute(args.outDir) ? args.outDir : path.join(repoRoot, args.outDir);
  await fs.mkdir(outDirAbs, { recursive: true });

  const outJson = path.join(outDirAbs, 'pdf-health.json');
  const outMd = path.join(outDirAbs, 'pdf-health.md');

  await fs.writeFile(
    outJson,
    JSON.stringify(
      {
        generatedAt: new Date().toISOString(),
        minBytes: args.minBytes,
        pdfCount: pdfFiles.length,
        tinyCount: tiny.length,
        siteScan: {
          siteDir: '_site',
          pagesScanned: sitePdfRefs.totalPagesScanned,
          pdfReferences: sitePdfRefs.totalPdfReferences,
          uniquePdfUrls: sitePdfRefs.uniquePdfUrls,
        },
        tiny,
        todoStatus,
      },
      null,
      2,
    ),
    'utf8',
  );

  const lines = [];
  lines.push('# PDF Health Report');
  lines.push('');
  lines.push(`Generated: ${new Date().toISOString()}`);
  lines.push(`Min bytes threshold: ${args.minBytes}`);
  lines.push(`PDF files scanned: ${pdfFiles.length}`);
  lines.push(`Tiny/placeholder PDFs (< threshold): ${tiny.length}`);
  lines.push('');

  lines.push('## TODO References Status');
  lines.push('');
  if (todoStatus.length === 0) {
    lines.push('No TODO references found.');
  } else {
    for (const item of todoStatus) {
      if (!item.exists) lines.push(`- MISSING: ${toPosix(item.ref)}`);
      else lines.push(`- OK (${item.bytes} bytes)${item.linkedFromSite ? ' [LINKED]' : ''}: ${toPosix(item.ref)}`);
    }
  }

  lines.push('');
  lines.push('## Tiny PDFs (Linked vs Not Linked)');
  lines.push('');
  const tinyLinked = tiny.filter((t) => t.linkedFromSite);
  const tinyUnlinked = tiny.filter((t) => !t.linkedFromSite);
  lines.push(`- Linked from built site: ${tinyLinked.length}`);
  lines.push(`- Not linked from built site: ${tinyUnlinked.length}`);

  lines.push('');
  lines.push('## Smallest PDFs');
  lines.push('');
  if (tiny.length === 0) {
    lines.push('No tiny PDFs detected.');
  } else {
    for (const item of tiny.slice(0, 200)) {
      lines.push(`- ${item.bytes} bytes${item.linkedFromSite ? ' [LINKED]' : ''}: ${toPosix(item.path)}`);
    }
  }

  lines.push('');
  lines.push(`JSON: ${toPosix(path.relative(repoRoot, outJson))}`);
  lines.push(`MD: ${toPosix(path.relative(repoRoot, outMd))}`);

  await fs.writeFile(outMd, `${lines.join('\n')}\n`, 'utf8');

  console.log(`Wrote ${path.relative(repoRoot, outMd)}`);
  console.log(`Wrote ${path.relative(repoRoot, outJson)}`);

  // Exit non-zero if any TODO refs missing OR tiny PDFs exist
  if (tiny.length > 0 || todoStatus.some((t) => !t.exists)) process.exitCode = 1;
}

await main();
