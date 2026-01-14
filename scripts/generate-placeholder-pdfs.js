import fs from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';

const repoRoot = process.cwd();

function parseArgs(argv) {
  const args = {
    minBytes: 4096,
    padToBytes: 8192,
    forceTiny: false,
    paths: [],
  };

  for (let i = 2; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === '--min-bytes') {
      args.minBytes = Number(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--pad-to') {
      args.padToBytes = Number(argv[i + 1]);
      i += 1;
      continue;
    }
    if (arg === '--force-tiny') {
      args.forceTiny = true;
      continue;
    }
    args.paths.push(arg);
  }

  return args;
}

function escapePdfString(s) {
  return s.replaceAll('\\', '\\\\').replaceAll('(', '\\(').replaceAll(')', '\\)');
}

function toLF(s) {
  return s.replaceAll('\r\n', '\n');
}

function buildPdf({ lines, padToBytes }) {
  const safeLines = lines.map((l) => escapePdfString(l));

  const baseStreamLines = [
    'BT',
    '/F1 14 Tf',
    '18 TL',
    '50 760 Td',
    ...safeLines.flatMap((l, idx) => (idx === 0 ? [`(${l}) Tj`] : ['T*', `(${l}) Tj`])),
    'ET',
  ];

  let streamBody = `${baseStreamLines.join('\n')}\n`;

  const objects = [];

  // 1: Catalog
  objects.push('1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n');
  // 2: Pages
  objects.push('2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n');
  // 3: Page
  objects.push(
    '3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n',
  );

  // We'll insert object 4 after we compute padded stream length.

  // 5: Font
  objects.push('5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n');

  const header = '%PDF-1.4\n%\xE2\xE3\xCF\xD3\n';

  function assembleWithStream(stream) {
    const obj4 = `4 0 obj\n<< /Length ${Buffer.byteLength(stream, 'utf8')} >>\nstream\n${stream}endstream\nendobj\n`;

    // Place object 4 between 3 and 5 for readability.
    const fullObjects = [objects[0], objects[1], objects[2], obj4, objects[3]];

    const parts = [header, ...fullObjects];

    // Build xref
    let offset = Buffer.byteLength(header, 'utf8');
    const offsets = [0]; // object 0 (special)

    for (const obj of fullObjects) {
      offsets.push(offset);
      offset += Buffer.byteLength(obj, 'utf8');
    }

    const xrefStart = offset;

    const xrefLines = [];
    xrefLines.push('xref');
    xrefLines.push(`0 ${offsets.length}`);
    xrefLines.push('0000000000 65535 f ');

    for (let i = 1; i < offsets.length; i += 1) {
      const o = String(offsets[i]).padStart(10, '0');
      xrefLines.push(`${o} 00000 n `);
    }

    const trailer = [
      'trailer',
      `<< /Size ${offsets.length} /Root 1 0 R >>`,
      'startxref',
      String(xrefStart),
      '%%EOF',
      '',
    ].join('\n');

    return Buffer.from(toLF([...parts, xrefLines.join('\n') + '\n', trailer].join('')), 'utf8');
  }

  // First assemble without padding.
  let pdf = assembleWithStream(streamBody);

  if (pdf.byteLength < padToBytes) {
    const needed = padToBytes - pdf.byteLength;
    // Add a comment pad inside the stream (counts toward /Length, harmless).
    const padLine = `% pad ${'x'.repeat(Math.max(0, needed - 8))}\n`;
    streamBody += padLine;
    pdf = assembleWithStream(streamBody);
  }

  return pdf;
}

async function exists(p) {
  try {
    await fs.access(p);
    return true;
  } catch {
    return false;
  }
}

async function shouldReplace(absPath, minBytes, forceTiny) {
  if (!(await exists(absPath))) return true;
  const st = await fs.stat(absPath);
  if (st.size >= minBytes) return false;

  if (forceTiny) return true;

  // If it's tiny, also check whether it even looks like a PDF.
  try {
    const head = await fs.readFile(absPath, { encoding: null });
    const snippet = head.subarray(0, Math.min(16, head.length)).toString('utf8');
    return !snippet.startsWith('%PDF-');
  } catch {
    return true;
  }
}

async function main() {
  const args = parseArgs(process.argv);

  const defaultTargets = [
    'cases/a-000313-25/filings/20251026-njsc-atl-22-002292-barber-petition-for-postconvictionrelief-with-certification-and-memorandum.pdf',
    'cases/atl-dc-007956-25/filings/2025-09-03_Answer-FirstAppearance.pdf',
    'cases/atl-dc-007956-25/filings/20250903-answer-firstappearance.pdf',
    'cases/a-000313-25/filings/20251029-notice-of-appeal.pdf',
    'cases/atl-l-002794-25/filings/20251003-verified-complaint-and-cis.pdf',
    'cases/atl-l-002794-25/filings/20251016-motion-track-change.pdf',
    'cases/atl-l-002794-25/filings/20251028-njsc-atl-l-002794-25-barber-firstamendedcomplaint-with-exhibits.pdf',
    'cases/atl-l-002794-25/filings/20251028-njsc-atl-l-002794-25-barber-pcr-insupport-of-motiontochangetrack.pdf',
  ];

  const targets = args.paths.length > 0 ? args.paths : defaultTargets;

  const replaced = [];
  const skipped = [];

  for (const relPath of targets) {
    const absPath = path.join(repoRoot, relPath);
    const replace = await shouldReplace(absPath, args.minBytes, args.forceTiny);

    if (!replace) {
      skipped.push(relPath);
      continue;
    }

    await fs.mkdir(path.dirname(absPath), { recursive: true });

    const pdf = buildPdf({
      padToBytes: args.padToBytes,
      lines: [
        'FaithFrontier PDF Placeholder',
        '',
        'This PDF is a placeholder to prevent broken links.',
        'The original filing PDF has not been uploaded yet or was replaced during reorganization.',
        '',
        `Path: ${relPath}`,
        '',
        'If you have the original file, replace this PDF in the repository.',
      ],
    });

    await fs.writeFile(absPath, pdf);
    replaced.push({ path: relPath, bytes: pdf.byteLength });
  }

  console.log(JSON.stringify({
    replaced,
    skipped,
    minBytes: args.minBytes,
    padToBytes: args.padToBytes,
    forceTiny: args.forceTiny,
  }, null, 2));
}

await main();
