# THIRD-PARTY LICENSES AND ATTRIBUTIONS

**BarberX Legal Technologies**

This document lists all third-party software, APIs, and open-source tools used in the BarberX platform, along with their licenses and required attributions.

---

## 🤖 AI/ML LIBRARIES

### OpenAI Whisper

**Purpose:** Audio transcription (BWC footage → text)  
**License:** MIT License  
**Copyright:** © 2022 OpenAI  
**Repository:** https://github.com/openai/whisper  
**Attribution Required:** YES

**License Text:**

```
MIT License

Copyright (c) 2022 OpenAI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

**Our Attribution:**

> Transcription powered by OpenAI Whisper (MIT License)  
> https://github.com/openai/whisper

---

### Pyannote.audio

**Purpose:** Speaker diarization (identify different speakers in audio)  
**License:** MIT License  
**Copyright:** © 2020-2024 CNRS  
**Repository:** https://github.com/pyannote/pyannote-audio  
**Attribution Required:** YES

**License Text:**

```
MIT License

Copyright (c) 2020-2024 CNRS

Permission is hereby granted, free of charge, to any person obtaining a copy...
```

**Our Attribution:**

> Speaker identification powered by pyannote.audio (MIT License)  
> https://github.com/pyannote/pyannote-audio

---

### spaCy

**Purpose:** Named entity recognition (extract names, locations, organizations)  
**License:** MIT License  
**Copyright:** © 2016-2024 ExplosionAI GmbH  
**Repository:** https://github.com/explosion/spaCy  
**Attribution Required:** YES

**Our Attribution:**

> Entity extraction powered by spaCy (MIT License)  
> https://spacy.io

---

### Transformers (Hugging Face)

**Purpose:** Advanced NLP models  
**License:** Apache License 2.0  
**Copyright:** © 2018 The HuggingFace Team  
**Repository:** https://github.com/huggingface/transformers  
**Attribution Required:** YES

**Our Attribution:**

> NLP models provided by Hugging Face Transformers (Apache 2.0)  
> https://huggingface.co/transformers

---

### SentenceTransformers

**Purpose:** Semantic similarity and embeddings  
**License:** Apache License 2.0  
**Copyright:** © 2019 Nils Reimers  
**Repository:** https://github.com/UKPLab/sentence-transformers  
**Attribution Required:** YES

**Our Attribution:**

> Semantic analysis powered by SentenceTransformers (Apache 2.0)  
> https://www.sbert.net

---

## 🌐 WEB FRAMEWORK

### Flask

**Purpose:** Web application framework  
**License:** BSD-3-Clause License  
**Copyright:** © 2010 Pallets  
**Repository:** https://github.com/pallets/flask  
**Attribution Required:** YES

**Our Attribution:**

> Built with Flask web framework (BSD-3-Clause)  
> https://flask.palletsprojects.com

---

### Flask-Login

**Purpose:** User authentication and session management  
**License:** MIT License  
**Copyright:** © 2011 Matthew Frazier  
**Repository:** https://github.com/maxcountryman/flask-login  
**Attribution Required:** YES

---

### Flask-SQLAlchemy

**Purpose:** Database ORM  
**License:** BSD-3-Clause License  
**Copyright:** © 2010 Pallets  
**Repository:** https://github.com/pallets/flask-sqlalchemy  
**Attribution Required:** YES

---

### Flask-CORS

**Purpose:** Cross-origin resource sharing  
**License:** MIT License  
**Copyright:** © 2013 Cory Dolphin  
**Repository:** https://github.com/corydolphin/flask-cors  
**Attribution Required:** YES

---

## 🗄️ DATABASE

### SQLAlchemy

**Purpose:** SQL toolkit and ORM  
**License:** MIT License  
**Copyright:** © 2005-2024 Michael Bayer and contributors  
**Repository:** https://github.com/sqlalchemy/sqlalchemy  
**Attribution Required:** YES

**Our Attribution:**

> Database management powered by SQLAlchemy (MIT License)  
> https://www.sqlalchemy.org

---

### PostgreSQL (Production)

**Purpose:** Production database  
**License:** PostgreSQL License (similar to MIT/BSD)  
**Copyright:** © 1996-2024 The PostgreSQL Global Development Group  
**Website:** https://www.postgresql.org  
**Attribution Required:** YES (in documentation)

---

## 🎨 FRONTEND LIBRARIES

### Bootstrap

**Purpose:** CSS framework for responsive design  
**License:** MIT License  
**Copyright:** © 2011-2024 The Bootstrap Authors  
**Repository:** https://github.com/twbs/bootstrap  
**Attribution Required:** YES

**Our Attribution:**

> UI components powered by Bootstrap 5.3 (MIT License)  
> https://getbootstrap.com

---

### Bootstrap Icons

**Purpose:** Icon library  
**License:** MIT License  
**Copyright:** © 2019-2024 The Bootstrap Authors  
**Repository:** https://github.com/twbs/icons  
**Attribution Required:** YES

---

### Font Awesome (if used)

**Purpose:** Icon library  
**License:** Font Awesome Free License (Icons: CC BY 4.0, Fonts: SIL OFL 1.1, Code: MIT)  
**Copyright:** © Fonticons, Inc.  
**Website:** https://fontawesome.com  
**Attribution Required:** YES

---

## 📊 DATA PROCESSING

### Pandas

**Purpose:** Data manipulation and analysis  
**License:** BSD-3-Clause License  
**Copyright:** © 2008-2024 AQR Capital Management, LLC, Lambda Foundry, Inc. and PyData Development Team  
**Repository:** https://github.com/pandas-dev/pandas  
**Attribution Required:** YES

---

### NumPy

**Purpose:** Numerical computing  
**License:** BSD-3-Clause License  
**Copyright:** © 2005-2024 NumPy Developers  
**Repository:** https://github.com/numpy/numpy  
**Attribution Required:** YES

---

### Pydub

**Purpose:** Audio file manipulation  
**License:** MIT License  
**Copyright:** © 2011 James Robert  
**Repository:** https://github.com/jiaaro/pydub  
**Attribution Required:** YES

---

### FFmpeg

**Purpose:** Audio/video processing  
**License:** LGPL 2.1 (GPL-free build) ✅  
**Copyright:** © 2000-2026 FFmpeg team  
**Website:** https://ffmpeg.org  
**Build:** BtbN/FFmpeg-Builds LGPL-only release  
**Installation:** C:\ffmpeg-lgpl\ (version N-122527-g4561fc5e48-20260122)  
**Attribution Required:** YES

**✅ GPL CONFLICT RESOLVED (January 22, 2026):**

- Using verified LGPL-only build (NO GPL components)
- Compatible with proprietary "ALL RIGHTS RESERVED" license
- Build verified GPL-free via `ffmpeg -version` (no `--enable-gpl` flag)
- Supports all required formats: MP3, MP4, WAV, WebM

---

## 🔐 SECURITY & ENCRYPTION

### Werkzeug

**Purpose:** Password hashing and security utilities  
**License:** BSD-3-Clause License  
**Copyright:** © 2007 Pallets  
**Repository:** https://github.com/pallets/werkzeug  
**Attribution Required:** YES

---

### cryptography

**Purpose:** Encryption and secure hashing  
**License:** Apache License 2.0 or BSD License  
**Copyright:** © Individual contributors  
**Repository:** https://github.com/pyca/cryptography  
**Attribution Required:** YES

---

## 💳 PAYMENT PROCESSING

### Stripe Python SDK

**Purpose:** Payment processing  
**License:** MIT License  
**Copyright:** © Stripe, Inc.  
**Repository:** https://github.com/stripe/stripe-python  
**Attribution Required:** YES

**Note:** Stripe API usage also subject to Stripe Terms of Service

---

## 📄 DOCUMENT GENERATION

### ReportLab (if used)

**Purpose:** PDF generation  
**License:** BSD-3-Clause License (open source) / Commercial (for proprietary fonts)  
**Copyright:** © 2000-2024 ReportLab Inc.  
**Repository:** https://www.reportlab.com  
**Attribution Required:** YES

---

### python-docx (if used)

**Purpose:** DOCX file generation  
**License:** MIT License  
**Copyright:** © 2013 Steve Canny  
**Repository:** https://github.com/python-openxml/python-docx  
**Attribution Required:** YES

---

## 🌐 LEGAL DATA SOURCES

### CourtListener

**Purpose:** Public domain case law  
**License:** CC0 1.0 Universal (Public Domain Dedication)  
**Copyright:** Public domain (U.S. government works)  
**Operated by:** Free Law Project  
**Website:** https://www.courtlistener.com  
**Attribution Required:** NO (but recommended)

**Recommended Attribution:**

> Case law from CourtListener (Free Law Project)  
> https://www.courtlistener.com

---

### Justia

**Purpose:** Public domain legal resources  
**License:** Public domain (U.S. government works)  
**Website:** https://www.justia.com  
**Attribution Required:** NO (but recommended)

---

## ⚠️ PROPRIETARY SERVICES (NOT FOR EXPORT)

### Westlaw (Thomson Reuters)

**Purpose:** Legal research (INTERNAL USE ONLY)  
**License:** Proprietary subscription  
**Copyright:** © Thomson Reuters  
**Export Allowed:** ❌ NO - Internal research only  
**Attribution Required:** N/A (cannot republish)

---

### LexisNexis

**Purpose:** Legal research (INTERNAL USE ONLY)  
**License:** Proprietary subscription  
**Copyright:** © LexisNexis  
**Export Allowed:** ❌ NO - Internal research only  
**Attribution Required:** N/A (cannot republish)

---

## 📋 COMPLETE ATTRIBUTION BLOCK

**For inclusion in exports, documentation, and About page:**

```
BarberX Legal Technologies uses the following open-source software and APIs:

AI/ML:
• OpenAI Whisper (MIT) - https://github.com/openai/whisper
• pyannote.audio (MIT) - https://github.com/pyannote/pyannote-audio
• spaCy (MIT) - https://spacy.io
• Hugging Face Transformers (Apache 2.0) - https://huggingface.co

Web Framework:
• Flask (BSD-3-Clause) - https://flask.palletsprojects.com
• SQLAlchemy (MIT) - https://www.sqlalchemy.org
• Bootstrap (MIT) - https://getbootstrap.com

Data Processing:
• Pandas (BSD-3-Clause) - https://pandas.pydata.org
• NumPy (BSD-3-Clause) - https://numpy.org
• FFmpeg (LGPL) - https://ffmpeg.org

Legal Data:
• CourtListener (CC0 Public Domain) - https://www.courtlistener.com

Security:
• Stripe (MIT) - https://stripe.com
• Werkzeug (BSD-3-Clause) - https://palletsprojects.com

All trademarks are property of their respective owners.
```

---

## 🔍 LICENSE COMPLIANCE CHECKLIST

### MIT License Requirements:

- [x] Include copyright notice in source code
- [x] Include license text in distributions
- [x] Attribute in documentation/About page
- [ ] **Include in export manifests**

### BSD License Requirements:

- [x] Include copyright notice
- [x] Include license text
- [x] Do not use project names to endorse without permission
- [ ] **Include in export manifests**

### Apache 2.0 Requirements:

- [x] Include copyright notice
- [x] Include license text
- [x] Include NOTICE file if provided
- [x] State modifications (if any)
- [ ] **Include in export manifests**

### GPL/LGPL Requirements (FFmpeg):

- [x] Use LGPL-licensed build only
- [x] Do not statically link GPL code
- [x] Provide access to source code (if distributing)
- [ ] **Verify FFmpeg build is LGPL**

---

## ⚠️ CRITICAL COMPLIANCE ACTIONS

### Immediate (Before Launch):

1. [ ] Create LICENSES/ directory with full license texts
2. [ ] Add attribution block to About page
3. [ ] Update export manifest to include ALL open-source tools
4. [ ] Verify FFmpeg is LGPL build (not GPL)
5. [ ] Create NOTICE.txt for Apache 2.0 libraries

### Ongoing:

1. [ ] Update this file when adding new dependencies
2. [ ] Include attribution in every export
3. [ ] Credit open-source projects in marketing materials
4. [ ] Contribute back to open-source projects (recommended)

---

## 📧 CONTACT

**License Compliance Questions:**  
legal@barberx.info  
BarberCamX@ProtonMail.com

**Open Source Contributions:**  
opensource@barberx.info

---

**Last Updated:** January 22, 2026  
**Review Schedule:** Quarterly (when dependencies change)
