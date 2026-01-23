---
layout: page
title: Document Analysis
subtitle: AI-powered legal document analysis and classification
description: Upload PDFs for OCR, constitutional violation scanning, and metadata extraction
permalink: /tools/document-analysis/
toc: false
---

<div class="tool-page-header">
  <div class="tool-page-header__icon">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="48" height="48">
      <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
  </div>
  <h1>Document Analysis</h1>
  <p>Upload legal documents for AI-powered OCR, classification, and constitutional violation scanning</p>
</div>

<div class="upload-zone" id="uploadZone">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="64" height="64">
    <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M17 8l-5-5-5 5M12 3v12"/>
  </svg>
  <h3>Drop PDF files here or click to browse</h3>
  <p>Supports PDF, DOCX, and image files (JPG, PNG)</p>
  <input type="file" id="fileInput" accept=".pdf,.docx,.doc,.jpg,.jpeg,.png" multiple hidden />
  <button class="btn-premium" onclick="document.getElementById('fileInput').click()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
      <path d="M12 5v14M5 12l7-7 7 7"/>
    </svg>
    Choose Files
  </button>
</div>

<div class="analysis-features">
  <div class="feature-card">
    <div class="feature-card__icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="32" height="32">
        <path d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
      </svg>
    </div>
    <h3>OCR & Text Extraction</h3>
    <p>Extract searchable text from scanned PDFs and images with high accuracy OCR</p>
  </div>

  <div class="feature-card">
    <div class="feature-card__icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="32" height="32">
        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
      </svg>
    </div>
    <h3>Constitutional Scanning</h3>
    <p>Automatically detect potential Fourth Amendment, Due Process, and Equal Protection violations</p>
  </div>

  <div class="feature-card">
    <div class="feature-card__icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="32" height="32">
        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
        <line x1="9" y1="9" x2="15" y2="15"/>
        <line x1="15" y1="9" x2="9" y2="15"/>
      </svg>
    </div>
    <h3>Bates Numbering</h3>
    <p>Add sequential Bates stamps to documents for discovery production</p>
  </div>

  <div class="feature-card">
    <div class="feature-card__icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="32" height="32">
        <path d="M7 7h10v3m-4 11H7a2 2 0 01-2-2V7a2 2 0 012-2h10a2 2 0 012 2v5m-9 6h10a2 2 0 002-2v-5a2 2 0 00-2-2H8a2 2 0 00-2 2v5a2 2 0 002 2z"/>
      </svg>
    </div>
    <h3>Document Classification</h3>
    <p>Auto-classify document types: complaints, motions, orders, discovery, etc.</p>
  </div>
</div>

<div class="info-notice">
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
    <circle cx="12" cy="12" r="10"/>
    <line x1="12" y1="16" x2="12.01" y2="16"/>
    <path d="M12 8v4"/>
  </svg>
  <div>
    <strong>Upload System Active</strong>
    <p>Documents are uploaded via the backend API (if running) or through the Git inbox workflow. AI analysis features coming in the next update.</p>
  </div>
</div>

<script src="/assets/js/api-config.js"></script>
<script src="/assets/js/document-upload.js"></script>

<style>
.upload-zone {
  border: 2px dashed rgb(212 165 116 / 30%);
  border-radius: 20px;
  padding: 4rem 2rem;
  text-align: center;
  background: linear-gradient(135deg, rgb(212 165 116 / 5%) 0%, transparent 100%);
  margin: 2rem 0;
  transition: all 0.3s ease;
  cursor: pointer;
}

.upload-zone:hover {
  border-color: rgb(212 165 116 / 50%);
  background: linear-gradient(135deg, rgb(212 165 116 / 8%) 0%, transparent 100%);
}

.upload-zone svg {
  color: rgb(212 165 116 / 50%);
  margin-bottom: 1.5rem;
}

.upload-zone h3 {
  font-size: 1.25rem;
  color: #f5f5f7;
  margin-bottom: 0.5rem;
}

.upload-zone p {
  color: rgb(255 255 255 / 50%);
  margin-bottom: 1.5rem;
}

.analysis-features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 1.5rem;
  margin: 3rem 0;
}

.feature-card {
  padding: 1.5rem;
  background: linear-gradient(135deg, rgb(255 255 255 / 5%) 0%, rgb(255 255 255 / 2%) 100%);
  border: 1px solid rgb(255 255 255 / 12%);
  border-radius: 16px;
  text-align: center;
}

.feature-card__icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 64px;
  height: 64px;
  background: linear-gradient(135deg, #d4a574 0%, #c49364 100%);
  border-radius: 16px;
  color: #0a0a0f;
  margin-bottom: 1rem;
}

.feature-card h3 {
  font-size: 1.125rem;
  color: #f5f5f7;
  margin-bottom: 0.5rem;
}

.feature-card p {
  font-size: 0.9rem;
  color: rgb(255 255 255 / 60%);
  line-height: 1.6;
}

/* AI Analysis Results */
.analysis-results {
  background: linear-gradient(135deg, rgb(255 255 255 / 8%) 0%, rgb(255 255 255 / 3%) 100%);
  border: 1px solid rgb(212 165 116 / 30%);
  border-radius: 16px;
  padding: 2rem;
  margin: 2rem 0;
}

.analysis-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 1px solid rgb(255 255 255 / 10%);
}

.confidence-badge {
  padding: 0.5rem 1rem;
  background: rgb(16 185 129 / 20%);
  color: #6ee7b7;
  border-radius: 8px;
  font-weight: 600;
}

.analysis-section {
  margin: 1.5rem 0;
}

.analysis-section h4 {
  font-size: 1.125rem;
  color: #d4a574;
  margin-bottom: 1rem;
}

.document-type-badge {
  display: inline-block;
  padding: 0.5rem 1rem;
  background: rgb(59 130 246 / 20%);
  color: #93c5fd;
  border-radius: 8px;
  font-weight: 600;
}

.violations-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.violation-card {
  padding: 1.25rem;
  border-radius: 12px;
  border-left: 4px solid;
}

.violation-card--high {
  background: rgb(239 68 68 / 10%);
  border-left-color: #ef4444;
}

.violation-card--medium {
  background: rgb(251 191 36 / 10%);
  border-left-color: #fbbf24;
}

.violation-card--low {
  background: rgb(59 130 246 / 10%);
  border-left-color: #3b82f6;
}

.violation-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
}

.violation-type {
  font-weight: 600;
  color: #f5f5f7;
  font-size: 1rem;
}

.severity-badge {
  padding: 0.25rem 0.75rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.severity-badge--high { background: rgb(239 68 68 / 30%); color: #fca5a5; }
.severity-badge--medium { background: rgb(251 191 36 / 30%); color: #fde68a; }
.severity-badge--low { background: rgb(59 130 246 / 30%); color: #93c5fd; }

.violation-description {
  color: rgb(255 255 255 / 80%);
  margin: 0.5rem 0;
  line-height: 1.6;
}

.violation-amendment,
.violation-caselaw {
  font-size: 0.875rem;
  color: rgb(255 255 255 / 60%);
  margin: 0.25rem 0;
}

.damages-estimate {
  margin-top: 0.75rem;
  padding: 0.75rem;
  background: rgb(16 185 129 / 10%);
  border-radius: 8px;
  color: #6ee7b7;
  font-size: 0.9rem;
}

.facts-list,
.citations-list,
.recommendations-list {
  list-style: none;
  padding: 0;
}

.facts-list li,
.citations-list li,
.recommendations-list li {
  padding: 0.75rem 0 0.75rem 1.5rem;
  border-bottom: 1px solid rgb(255 255 255 / 5%);
  position: relative;
  color: rgb(255 255 255 / 70%);
}

.facts-list li:before {
  content: '🔑';
  position: absolute;
  left: 0;
}

.citations-list li:before {
  content: '📚';
  position: absolute;
  left: 0;
}

.recommendations-list li:before {
  content: '💡';
  position: absolute;
  left: 0;
}

.analysis-actions {
  display: flex;
  gap: 1rem;
  margin-top: 2rem;
  padding-top: 2rem;
  border-top: 1px solid rgb(255 255 255 / 10%);
  flex-wrap: wrap;
}

.info-notice--success {
  background: rgb(16 185 129 / 10%);
  border-color: rgb(16 185 129 / 25%);
  color: rgb(167 243 208);
}
</style>

<div id="analysisResults" style="display: none; margin: 2rem 0;"></div>

<div id="aiFeatures" style="display: none;">
  <div class="info-notice info-notice--success">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
      <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
    </svg>
    <div>
      <strong>✅ AI Analysis Enabled</strong>
      <p>Upload your certifications, police reports, or discovery documents to automatically detect constitutional violations and generate verified complaints.</p>
    </div>
  </div>
</div>

<div id="offlineNotice" style="display: none;">
  <div class="info-notice">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="24" height="24">
      <circle cx="12" cy="12" r="10"/>
      <line x1="12" y1="16" x2="12.01" y2="16"/>
      <path d="M12 8v4"/>
    </svg>
    <div>
      <strong>Offline Mode</strong>
      <p>AI analysis requires backend server. Start with: <code>cd tillerstead-toolkit/backend && python -m uvicorn app.main:app --reload</code></p>
      <p>Basic upload via Git inbox is still available.</p>
    </div>
  </div>
</div>

<script src="/assets/js/api-config.js"></script>
<script src="/assets/js/document-upload.js"></script>
<script src="/assets/js/constitutional-analysis.js"></script>

<script>
// Enhanced upload with AI analysis
const originalHandleFiles = window.handleFiles;
window.handleFiles = async function(files) {
  if (files.length === 0) return;
  
  const isBackendAvailable = await checkBackendHealth();
  
  if (isBackendAvailable && typeof analyzeDocument === 'function') {
    showToast(`Uploading and analyzing ${files.length} file(s)...`, 'info');
    
    for (const file of files) {
      try {
        const analysis = await analyzeDocument(file);
        if (analysis && uploadedDocuments) {
          uploadedDocuments.push({ file, analysis });
        }
      } catch (error) {
        console.error('Analysis failed:', error);
      }
    }
  } else if (originalHandleFiles) {
    originalHandleFiles(files);
  }
};
</script>
