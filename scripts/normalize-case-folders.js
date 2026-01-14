import fs from "fs/promises";
import { createReadStream } from "fs";
import path from "path";
import crypto from "crypto";
import yaml from "js-yaml";

const CASES_ROOT = path.join(process.cwd(), "cases");
const META_ROOT = path.join(process.cwd(), "_cases");

const DOC_EXTENSIONS = new Set([
  ".pdf",
  ".docx",
  ".png",
  ".jpg",
  ".jpeg",
  ".tiff",
]);

const REQUIRED_DIRS = ["filings", "evidence", "notes"];

const countyHints = {
  atl: "Atlantic County",
  cam: "Camden County",
  glu: "Gloucester County",
  bur: "Burlington County",
  mer: "Mercer County",
  mid: "Middlesex County",
  mon: "Monmouth County",
  usdj: "District of New Jersey",
  a: "Statewide (Appellate Division)",
};

const report = [];

const requestedCases = process.argv.slice(2);
const caseFilter = requestedCases.length ? new Set(requestedCases) : null;

const caseEntries = await fs.readdir(CASES_ROOT, { withFileTypes: true });
for (const entry of caseEntries) {
  if (!entry.isDirectory()) continue;
  const caseName = entry.name;
  if (caseFilter && !caseFilter.has(caseName)) continue;
  const casePath = path.join(CASES_ROOT, caseName);
  const metadata = await loadCaseMetadata(caseName);
  const caseReport = {
    docket: caseName,
    createdDirs: [],
    movedFiles: [],
    dedupedFiles: [],
    metadataUpdated: [],
    backups: [],
    removedDirs: [],
  };

  for (const dirName of REQUIRED_DIRS) {
    const dirPath = path.join(casePath, dirName);
    if (!(await pathExists(dirPath))) {
      await fs.mkdir(dirPath, { recursive: true });
      caseReport.createdDirs.push(path.relative(casePath, dirPath));
    }
  }

  const files = await walkCase(casePath);
  for (const file of files) {
    const ext = path.extname(file.fullPath).toLowerCase();
    if (!DOC_EXTENSIONS.has(ext)) continue;
    const normalizedBase = normalizeBaseName(path.basename(file.fullPath, ext));
    const targetName = `${normalizedBase}${ext}`;
    const filingsDir = path.join(casePath, "filings");
    let destination = path.join(filingsDir, targetName);
    const currentResolved = path.resolve(file.fullPath);
    let destResolved = path.resolve(destination);
    if (currentResolved === destResolved) continue;

    const destDir = path.dirname(destination);
    if (!(await pathExists(destDir))) {
      await fs.mkdir(destDir, { recursive: true });
    }

    let suffix = 1;
    while (await pathExists(destination)) {
      const identical = await filesAreIdentical(file.fullPath, destination);
      if (identical) {
        await fs.unlink(file.fullPath);
        caseReport.dedupedFiles.push({
          kept: path.relative(casePath, destination),
          removed: path.relative(casePath, file.relPath),
        });
        destination = null;
        break;
      }
      const candidateName = `${normalizedBase}-v${suffix}${ext}`;
      destination = path.join(filingsDir, candidateName);
      destResolved = path.resolve(destination);
      if (currentResolved === destResolved) {
        destination = null;
        break;
      }
      suffix++;
    }

    if (!destination) continue;

    await fs.rename(file.fullPath, destination);
    caseReport.movedFiles.push({
      from: file.relPath,
      to: path.relative(casePath, destination),
    });
  }

  await cleanupExtraDirs(casePath, caseReport);

  const filingsEntries = (await fs.readdir(path.join(casePath, "filings"))).sort(
    (a, b) => a.localeCompare(b)
  );

  const docketData = buildDocketData(caseName, metadata);
  const docketPath = path.join(casePath, "docket.yml");
  await writeFileSafely(
    docketPath,
    yaml.dump(docketData, { lineWidth: 100 }),
    caseReport,
    "docket.yml"
  );

  const readmePath = path.join(casePath, "README.md");
  const readmeContent = buildReadmeContent({
    caseName,
    metadata,
    docketData,
    filings: filingsEntries,
  });
  await writeFileSafely(readmePath, readmeContent, caseReport, "README.md");

  report.push(caseReport);
}

if (caseFilter) {
  for (const requested of caseFilter) {
    if (!report.find((entry) => entry.docket === requested)) {
      console.warn(`Warning: requested case "${requested}" was not found under /cases.`);
    }
  }
}

console.log("Case normalization complete.");
for (const entry of report) {
  console.log(
    `- ${entry.docket}: +${entry.createdDirs.length} dirs, ${entry.movedFiles.length} moved, ${entry.dedupedFiles.length} deduped, ${entry.metadataUpdated.length} metadata updates`
  );
}

async function loadCaseMetadata(caseName) {
  const metaPath = path.join(META_ROOT, caseName, "index.md");
  if (!(await pathExists(metaPath))) return {};
  const raw = await fs.readFile(metaPath, "utf8");
  const frontMatterMatch = raw.match(/^---\s*[\r\n]+([\s\S]*?)\s*---/);
  let data = {};
  if (frontMatterMatch) {
    try {
      data = yaml.load(frontMatterMatch[1]) ?? {};
    } catch {
      data = {};
    }
  }
  if (data.overview) {
    data.summary = data.overview.trim();
  } else {
    const body = raw.slice(frontMatterMatch?.[0]?.length ?? 0).trim();
    if (body) {
      const paragraph = body.split(/\n\s*\n/)[0]?.trim();
      data.summary = paragraph;
    }
  }
  return data;
}

async function walkCase(baseDir) {
  const results = [];
  await walkRecursive(baseDir, baseDir, results);
  return results;
}

async function walkRecursive(currentDir, baseDir, results) {
  const entries = await fs.readdir(currentDir, { withFileTypes: true });
  for (const entry of entries) {
    const fullPath = path.join(currentDir, entry.name);
    if (entry.isSymbolicLink()) continue;
    if (entry.isDirectory()) {
      await walkRecursive(fullPath, baseDir, results);
    } else if (entry.isFile()) {
      results.push({
        fullPath,
        relPath: path.relative(baseDir, fullPath),
      });
    }
  }
}

function normalizeBaseName(base) {
  let slug = base.toLowerCase();
  slug = slug.replace(/[\s_]+/g, "-");
  slug = slug.replace(/[^a-z0-9-]/g, "-");
  slug = slug.replace(/-+/g, "-").replace(/^-|-$/g, "");
  const { datePart, remainder } = extractDate(slug);
  const cleanedRemainder = remainder.replace(/-+/g, "-").replace(/^-|-$/g, "");
  const baseName = cleanedRemainder || "filing";
  return datePart ? `${datePart}-${baseName}` : baseName;
}

function extractDate(slug) {
  const patterns = [
    {
      regex: /(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)/,
      format: (m) => `${m[1]}${m[2]}${m[3]}`,
    },
    {
      regex: /(?<!\d)(\d{2})-(\d{2})-(\d{4})(?!\d)/,
      format: (m) => `${m[3]}${m[1]}${m[2]}`,
    },
    {
      regex: /(?<!\d)(\d{2})-(\d{2})-(\d{2})(?!\d)/,
      format: (m) => {
        const year = parseInt(m[3], 10);
        const yearFull = year >= 70 ? 1900 + year : 2000 + year;
        return `${yearFull.toString().padStart(4, "0")}${m[1]}${m[2]}`;
      },
    },
    {
      regex: /(?<!\d)(\d{8})(?!\d)/,
      format: (m) => m[1],
    },
    {
      regex: /(?<!\d)(\d{6})(?!\d)/,
      format: (m) => {
        const digits = m[1];
        const mm = digits.slice(0, 2);
        const dd = digits.slice(2, 4);
        const yy = digits.slice(4);
        const year = parseInt(yy, 10);
        const yearFull = year >= 70 ? 1900 + year : 2000 + year;
        return `${yearFull}${mm}${dd}`;
      },
    },
  ];

  for (const pattern of patterns) {
    const match = slug.match(pattern.regex);
    if (match) {
      const formatted = pattern.format(match);
      const remainder =
        slug.slice(0, match.index) +
        slug.slice(match.index + match[0].length);
      return { datePart: formatted, remainder };
    }
  }
  return { datePart: "", remainder: slug };
}

async function filesAreIdentical(a, b) {
  try {
    const [statA, statB] = await Promise.all([fs.stat(a), fs.stat(b)]);
    if (statA.size !== statB.size) return false;
  } catch {
    return false;
  }
  const hashA = await hashFile(a);
  const hashB = await hashFile(b);
  return hashA === hashB;
}

function hashFile(targetPath) {
  return new Promise((resolve, reject) => {
    const hash = crypto.createHash("sha256");
    const stream = createReadStream(targetPath);
    stream.on("error", reject);
    stream.on("data", (chunk) => hash.update(chunk));
    stream.on("end", () => resolve(hash.digest("hex")));
  });
}

async function cleanupExtraDirs(casePath, caseReport) {
  const allowed = new Set(REQUIRED_DIRS);
  const entries = await fs.readdir(casePath, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.isDirectory() && !allowed.has(entry.name)) {
      const dirPath = path.join(casePath, entry.name);
      const removed = await removeIfEmpty(dirPath);
      if (removed) {
        caseReport.removedDirs.push(path.relative(casePath, dirPath));
      }
    }
  }
}

async function removeIfEmpty(dirPath) {
  const entries = await fs.readdir(dirPath);
  if (entries.length === 0) {
    await fs.rmdir(dirPath);
    return true;
  }
  return false;
}

async function pathExists(target) {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
}

function buildDocketData(caseName, metadata) {
  const docket =
    (metadata.primary_docket ||
      (Array.isArray(metadata.dockets) && metadata.dockets[0]) ||
      caseName) ??
    caseName;
  return {
    docket: String(docket).toUpperCase(),
    county: determineCounty(caseName, metadata),
    type: determineType(metadata, caseName),
    parties: metadata.parties ?? [],
    last_updated: new Date().toISOString(),
  };
}

function determineCounty(caseName, metadata) {
  if (metadata?.venue) return metadata.venue;
  const key = caseName.split("-")[0];
  if (countyHints[key]) return countyHints[key];
  if (caseName.startsWith("usdj")) return "District of New Jersey";
  if (caseName.startsWith("a-")) return "Statewide (Appellate Division)";
  return "Unknown";
}

function determineType(metadata, caseName) {
  const hint = (metadata?.case_type || "").toLowerCase();
  if (hint.includes("special")) return "special civil";
  if (hint.includes("municipal")) return "municipal";
  if (hint.includes("criminal") || hint.includes("post-conviction")) {
    return "criminal";
  }
  if (caseName.includes("-dc-")) return "special civil";
  if (caseName.includes("-sc-")) return "special civil";
  if (caseName.startsWith("atl-") && !caseName.includes("-l-")) {
    return "criminal";
  }
  return "civil";
}

function buildReadmeContent({ caseName, metadata, docketData, filings }) {
  const title =
    sanitizeTitle(metadata?.title) ||
    metadata?.short_title ||
    docketData.docket ||
    caseName.toUpperCase();
  const summary =
    metadata?.summary ||
    "This case folder has been normalized for intake automation. Filings listed below reflect the latest uploads.";
  const lines = [];
  lines.push(`# ${title}`);
  lines.push("");
  lines.push(`- **Docket:** ${docketData.docket}`);
  lines.push(`- **County:** ${docketData.county}`);
  lines.push(`- **Type:** ${docketData.type}`);
  lines.push(`- **Filings:** [View folder](./filings/)`);
  lines.push(`- **Last Updated:** ${docketData.last_updated}`);
  lines.push("");
  lines.push("## Summary");
  lines.push("");
  lines.push(summary);
  lines.push("");
  lines.push("## Filings Index");
  lines.push("");
  if (filings.length === 0) {
    lines.push("_No filings have been cataloged yet._");
  } else {
    for (const file of filings) {
      lines.push(`- [${file}](./filings/${encodeURI(file)})`);
    }
  }
  lines.push("");
  lines.push("## Notes & Evidence");
  lines.push("");
  lines.push(
    "Use the `notes/` and `evidence/` directories for drafts, summaries, photos, or supplemental materials."
  );
  lines.push("");
  lines.push(
    "This README is autogenerated by `scripts/normalize-case-folders.js` to keep the intake structure predictable."
  );
  lines.push("");
  return lines.join("\n");
}

function sanitizeTitle(title) {
  if (!title) return "";
  return title.replace(/[^\x20-\x7E]/g, "").replace(/\s+/g, " ").trim();
}

async function writeFileSafely(targetPath, content, caseReport, label) {
  if (await pathExists(targetPath)) {
    const backupPath = `${targetPath}.bak`;
    await fs.copyFile(targetPath, backupPath);
    caseReport.backups.push(path.basename(backupPath));
  }
  await fs.writeFile(targetPath, content, "utf8");
  caseReport.metadataUpdated.push(label);
}
