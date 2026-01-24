---
title: "BarberX Legal Technologies Documentation"
layout: default
permalink: /docs/
---

# BarberX Legal Technologies Documentation

Welcome to the official documentation for BarberX Legal Technologies, the professional-grade AI-powered eDiscovery and forensic analysis platform.

---

## For All Users

**Access BarberX via the main website:**
- Go to [https://barberx.info](https://barberx.info)
- No installation or download required for standard users.
- All features are available securely through the web app.

---

## For Enterprise Customers Only

**Enterprise customers may download and run the local library:**
- Contact support@barberx.info for access to the enterprise package and installation instructions.
- Local install instructions and source code are provided only to verified enterprise clients.
- All proprietary code and advanced AI modules are protected and not distributed publicly.

---

## Table of Contents

- [Overview](#overview)
- [User Guide](#user-guide)
- [API Reference](#api-reference)
- [FAQ](#faq)
- [Troubleshooting](#troubleshooting)
- [Support & Community](#support--community)

---

## Overview

BarberX is a local-first, privacy-focused legal tech platform for processing body-worn camera (BWC) footage, police reports, and legal documents. It features:

- 100% local AI (no cloud required)
- Multi-user authentication & role-based access
- Chain of custody, audit logging, and evidence management
- Advanced document and media analysis

---

## User Guide

### Logging In

- Use the password you set or reset via the admin panel.

### Uploading Evidence

- Go to the dashboard and use the upload form for BWC videos, PDFs, or images.
- Supported formats: MP4, MOV, PDF, JPG, PNG, CSV, JSON, DOCX, etc.

### AI-Powered Analysis

- Transcribe audio, extract text, and run entity recognition on uploaded files.
- All processing is local—no data leaves your machine.

### Search & Export

- Use semantic search to find relevant evidence.
- Export court-ready exhibits as PDF, DOCX, or JSON.

---

## API Reference

- RESTful endpoints for evidence upload, user management, and analysis.
- See [API documentation](../api-reference.html) for full details.

---

## FAQ

- [FAQ page](../faq/)

---

## Troubleshooting

- **Missing dependencies:** Run `pip install -r requirements.txt`.
- **Database errors:** Ensure your database URI is correct and migrations are applied.
- **AI features unavailable:** Install required AI dependencies (see `requirements.txt` and documentation).
- **Port in use:** Change the `PORT` environment variable or stop the conflicting process.

---

## Support & Community

- [GitHub Issues](https://github.com/DTB396/BarberX.info/issues)
- [Contact](mailto:support@barberx.info)
- [Changelog](../CHANGELOG.md)

---

BarberX Legal Technologies © 2026. All rights reserved.

<section class="system-cards-section">
  <div class="container">
    <div class="system-cards-grid">
      <!-- Minimum Requirements Card -->
      <div class="system-card">
        <h3>Minimum Requirements</h3>
        <ul>
          <li>Windows 10/11, macOS 12+, Ubuntu 20.04+</li>
          <li>8GB RAM (16GB recommended)</li>
          <li>50GB free disk space</li>
          <li>Python 3.8+</li>
          <li>CPU processing (slower)</li>
        </ul>
      </div>
      <!-- Recommended Requirements Card -->
      <div class="system-card">
        <h3>Recommended</h3>
        <ul>
          <li>16GB+ RAM</li>
          <li>NVIDIA GPU with 6GB+ VRAM</li>
          <li>SSD storage</li>
          <li>Multi-core CPU (8+ cores)</li>
          <li>10x faster processing</li>
        </ul>
      </div>
      <!-- Processing Speed Card -->
      <div class="system-card">
        <h3>Processing Speed</h3>
        <ul>
          <li>Whisper: 2-3 min/hour (GPU)</li>
          <li>pyannote: 2-4 min/hour (GPU)</li>
          <li>Tesseract OCR: 5-10 pages/sec</li>
          <li>Real-ESRGAN: 0.5-1 sec/image</li>
          <li>YOLOv8: 30+ fps video</li>
        </ul>
      </div>
      <!-- Open-Source Licenses Card -->
      <div class="system-card">
        <h3>Open-Source Licenses</h3>
        <ul>
          <li>Whisper: MIT</li>
          <li>pyannote.audio: MIT</li>
          <li>Tesseract: Apache 2.0</li>
          <li>Real-ESRGAN: BSD 3-Clause</li>
          <li>YOLOv8: AGPL-3.0</li>
        </ul>
      </div>
    </div>
  </div>
</section>
