// Node.js script: Extract court stamp from PDF using pdf-parse
import fs from 'fs';
import pdf from 'pdf-parse';

const pdfPath = process.argv[2];
if (!pdfPath) {
  console.error('Usage: node extract-court-stamp.js <pdfPath>');
  process.exit(1);
}

async function extractCourtStamp(pdfPath) {
  const dataBuffer = fs.readFileSync(pdfPath);
  const data = await pdf(dataBuffer);
  // Heuristic: look for lines with "Filed" and a date/court
  const lines = data.text.split('\n');
  for (const line of lines) {
    if (/Filed/i.test(line) && /Court/i.test(line)) {
      return line.trim();
    }
    // Also match common date patterns
    if (/Filed/i.test(line) && /\d{4}/.test(line)) {
      return line.trim();
    }
  }
  return null;
}

extractCourtStamp(pdfPath).then(stamp => {
  if (stamp) {
    console.log(stamp);
    process.exit(0);
  } else {
    process.exit(2); // Not found
  }
});
