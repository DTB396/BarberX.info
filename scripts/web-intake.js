// FaithFrontier Express Web Interface for PDF Upload & Admin Review (legacy)

import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';
import express from 'express';
import multer from 'multer';
import pdf from 'pdf-parse';
import yaml from 'js-yaml';

const app = express();
const PORT = 4001;
const repoRoot = process.cwd();
const UPLOAD_DIR = path.join(repoRoot, '_inbox');
const DOCKET_DATA_DIR = path.join(repoRoot, '_data', 'docket');
const CASES_DIR = path.join(repoRoot, 'cases');

if (!fs.existsSync(UPLOAD_DIR)) fs.mkdirSync(UPLOAD_DIR, { recursive: true });

const upload = multer({ dest: UPLOAD_DIR });

app.use(express.static('public'));
app.use(express.json());

// Helper: Extract metadata from filename
function parseFilename(filename) {
  const re = /^(\d{4}-\d{2}-\d{2})_([\w-]+)_(.+)\.pdf$/i;
  const match = filename.match(re);
  if (match) {
    return {
      date: match[1],
      type: match[2].replace(/-/g, ' '),
      short: match[3].replace(/-/g, ' '),
      complete: true
    };
  }
  return { complete: false };
}

// Helper: Extract metadata from PDF text
function extractMetaFromText(text) {
  const lines = text.split(/\r?\n/).slice(0, 20);
  let date = null, type = null, short = null;
  for (const line of lines) {
    if (!date) {
      const d = line.match(/(\d{4}-\d{2}-\d{2})/);
      if (d) date = d[1];
    }
    if (!type && /order|motion|complaint|notice|filing|judgment|appearance|petition|application/i.test(line)) {
      type = line.trim().split(' ')[0];
    }
    if (!short && line.length > 10 && line.length < 80) {
      short = line.trim();
    }
  }
  return { date, type, short };
}

// Upload endpoint
app.post('/upload', upload.single('pdf'), async (req, res) => {
  const file = req.file;
  if (!file) return res.status(400).json({ error: 'No file uploaded' });
  let meta = parseFilename(file.originalname);
  let text = '';
  if (!meta.complete) {
    try {
      const data = await pdf(fs.readFileSync(file.path));
      text = data.text;
      meta = { ...meta, ...extractMetaFromText(text) };
    } catch (e) {
      return res.status(500).json({ error: 'Failed to parse PDF' });
    }
  }
  res.json({ filename: file.filename, originalname: file.originalname, meta });
});

// Admin review endpoint (list all uploaded PDFs with extracted metadata)
app.get('/review', (req, res) => {
  const files = fs.readdirSync(UPLOAD_DIR).filter(f => f.toLowerCase().endsWith('.pdf'));
  const review = files.map(f => {
    const meta = parseFilename(f);
    return { file: f, meta };
  });
  res.json(review);
});

// Serve a simple upload/review HTML page
app.get('/', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>FaithFrontier PDF Intake</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: sans-serif; margin: 2em; }
    .upload-box { border: 2px dashed #aaa; padding: 2em; text-align: center; margin-bottom: 2em; }
    .review-list { margin-top: 2em; }
    .review-item { border-bottom: 1px solid #eee; padding: 1em 0; }
    @media (max-width: 600px) { body { margin: 0.5em; } }
  </style>
</head>
<body>
  <h1>FaithFrontier PDF Intake</h1>
  <div class="upload-box">
    <input type="file" id="pdf" accept="application/pdf">
    <button onclick="upload()">Upload</button>
    <div id="upload-status"></div>
  </div>
  <h2>Admin Review</h2>
  <div class="review-list" id="review-list"></div>
  <script>
    async function upload() {
      const file = document.getElementById('pdf').files[0];
      if (!file) return alert('Select a PDF first!');
      const form = new FormData();
      form.append('pdf', file);
      document.getElementById('upload-status').innerText = 'Uploading...';
      const res = await fetch('/upload', { method: 'POST', body: form });
      const data = await res.json();
      document.getElementById('upload-status').innerText = data.error ? data.error : 'Uploaded!';
      loadReview();
    }
    async function loadReview() {
      const res = await fetch('/review');
      const data = await res.json();
      const list = document.getElementById('review-list');
      list.innerHTML = '';
      data.forEach(item => {
        list.innerHTML +=
          "<div class='review-item'><b>" +
          String(item.file) +
          "</b><br>Meta: " +
          JSON.stringify(item.meta) +
          "</div>";
      });
    }
    loadReview();
  </script>
</body>
</html>`);
});

app.listen(PORT, () => {
  console.log(`FaithFrontier PDF Intake Web running at http://localhost:${PORT}`);
});
