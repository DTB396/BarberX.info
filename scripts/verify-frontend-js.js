import { spawnSync } from "node:child_process";
import { readdirSync, statSync } from "node:fs";
import path from "node:path";

function collectJsFiles(dirPath) {
  const out = [];
  for (const entry of readdirSync(dirPath)) {
    const fullPath = path.join(dirPath, entry);
    const st = statSync(fullPath);
    if (st.isDirectory()) {
      out.push(...collectJsFiles(fullPath));
      continue;
    }
    if (st.isFile() && entry.toLowerCase().endsWith(".js")) {
      out.push(fullPath);
    }
  }
  return out;
}

function main() {
  const repoRoot = process.cwd();
  const assetsJsDir = path.join(repoRoot, "assets", "js");

  const files = collectJsFiles(assetsJsDir);
  if (!files.length) {
    console.error("No frontend JS files found in assets/js");
    process.exit(1);
  }

  let failed = false;
  for (const filePath of files) {
    const res = spawnSync(process.execPath, ["--check", filePath], {
      encoding: "utf8",
    });

    if (res.status !== 0) {
      failed = true;
      console.error(`Syntax check failed: ${filePath}`);
      if (res.stderr) console.error(res.stderr.trim());
      if (res.stdout) console.error(res.stdout.trim());
    }
  }

  if (failed) {
    process.exit(1);
  }

  console.log(`Verified ${files.length} frontend JS file(s).`);
}

main();
