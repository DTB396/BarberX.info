---
title: "BarberX Legal Technologies Documentation"
layout: default
permalink: /docs/
---

# BarberX Legal Technologies Documentation

Welcome to the official documentation for BarberX Legal Technologies, the professional-grade AI-powered eDiscovery and forensic analysis platform.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Installation Guide](#installation-guide)
- [User Guide](#user-guide)
- [Admin Guide](#admin-guide)
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

## Quick Start

1. **Clone the repository:**
   ```sh
   git clone https://github.com/DTB396/BarberX.info.git
   cd BarberX.info
   ```
2. **Set up Python environment:**
   ```sh
   python -m venv .venv
   & .venv/Scripts/Activate.ps1  # Windows
   # or
   source .venv/bin/activate     # macOS/Linux
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the app:**
   ```sh
   python app.py
   ```
5. **Access the web app:**
   Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## Installation Guide

### Requirements

- Python 3.9+
- Git
- (Optional) PostgreSQL for production
- (Optional) ffmpeg for media processing

### Environment Variables

Create a `.env` file in the project root with:

```
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///instance/barberx_auth.db  # or your PostgreSQL URI
CORS_ORIGINS=http://localhost:5000
```

---

## User Guide

### Logging In

- Default admin: `admin@barberx.info`
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

## Admin Guide

- Manage users, roles, and subscription tiers from the admin panel (`/admin`).
- View audit logs and chain of custody for all evidence.
- Configure system settings and integrations.

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
