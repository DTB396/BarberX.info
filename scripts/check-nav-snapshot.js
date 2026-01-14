import { createHash } from "crypto";
import { readFile, writeFile } from "fs/promises";
import path from "path";

const root = process.cwd();
const navPath = path.resolve(root, "_includes", "nav.component.html");
const snapshotPath = path.resolve(root, "scripts", "nav-snapshot.json");
const shouldUpdate = process.env.NAV_SNAPSHOT_UPDATE === "1";

async function fileHash(filePath) {
  const contents = await readFile(filePath, "utf8");
  const hash = createHash("sha256");
  hash.update(contents);
  return hash.digest("hex");
}

async function main() {
  const hash = await fileHash(navPath);
  let snapshot = { hash: null };

  try {
    const stored = await readFile(snapshotPath, "utf8");
    snapshot = JSON.parse(stored);
  } catch (err) {
    if (err.code !== "ENOENT") {
      throw err;
    }
  }

  if (shouldUpdate || !snapshot.hash) {
    await writeFile(
      snapshotPath,
      JSON.stringify({ hash }, null, 2) + "\n",
      "utf8",
    );
    console.log("Nav snapshot updated.");
    return;
  }

  if (snapshot.hash !== hash) {
    console.error("Nav snapshot mismatch detected.");
    console.error(
      "If this change is intentional, update the snapshot with `NAV_SNAPSHOT_UPDATE=1 node scripts/check-nav-snapshot.js`.",
    );
    process.exit(1);
  }

  console.log("Nav snapshot matched.");
}

main().catch((err) => {
  console.error("Nav snapshot check failed:", err);
  process.exit(1);
});
