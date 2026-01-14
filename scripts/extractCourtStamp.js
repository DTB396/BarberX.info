// Helper for docket-intake.js: Extract court stamp using Node.js and fallback to Python OCR
import { spawnSync } from 'node:child_process';
import path from 'path';

export function getCourtStamp(pdfPath) {
  // Try Node.js extractor first
  const nodeScript = path.resolve('tools/extract-court-stamp.js');
  let result = spawnSync('node', [nodeScript, pdfPath], { encoding: 'utf8' });
  if (result.status === 0 && result.stdout.trim()) {
    return result.stdout.trim();
  }
  // Fallback to Python OCR
  const pyScript = path.resolve('tools/extract_court_stamp_ocr.py');
  result = spawnSync('python', [pyScript, pdfPath], { encoding: 'utf8' });
  if (result.status === 0 && result.stdout.trim()) {
    return result.stdout.trim();
  }
  return null;
}
