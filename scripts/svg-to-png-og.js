import fs from "fs";
import path from "path";
import sharp from "sharp";

const SRC_DIR = "assets/img/og";
const WIDTH = 1200;
const HEIGHT = 630;

if (!fs.existsSync(SRC_DIR)) {
    console.error("❌ Source directory not found:", SRC_DIR);
    process.exit(1);
}

const files = fs.readdirSync(SRC_DIR).filter(f => f.endsWith(".svg"));

if (files.length === 0) {
    console.warn("⚠️ No SVG files found in", SRC_DIR);
    process.exit(0);
}

(async () => {
    for (const file of files) {
        const inputPath = path.join(SRC_DIR, file);
        const outputPath = path.join(
            SRC_DIR,
            file.replace(".svg", ".png")
        );

        // ✅ SKIP if PNG already exists
        if (fs.existsSync(outputPath)) {
            console.log("↪️  PNG exists, skipping:", outputPath);
            continue;
        }

        try {
            await sharp(inputPath)
                .resize(WIDTH, HEIGHT, { fit: "cover" })
                .png({ quality: 90 })
                .toFile(outputPath);

            console.log("✅ Generated:", outputPath);
        } catch (err) {
            console.error("❌ Failed:", file, err.message);
        }
    }

    console.log("\n🎉 PNG fallback generation complete (non-destructive).");
})();
