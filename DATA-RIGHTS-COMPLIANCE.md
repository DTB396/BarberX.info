# COPYRIGHT AND DATA RIGHTS COMPLIANCE FRAMEWORK

**BarberX Legal Tech Platform**  
**Effective:** January 22, 2026

---

## CRITICAL COMPLIANCE PATTERNS

This document outlines mandatory compliance patterns to protect BarberX Legal Technologies from copyright infringement, licensing violations, and data misuse claims.

---

## PATTERN 1: POINTER, DON'T REPUBLISH

### Principle

**NEVER republish full copyrighted content.** Store citations, fair-use excerpts, and pointers to authoritative sources only.

### Implementation Rules

#### ✅ ALLOWED: Citations and Metadata

```python
# CORRECT - Store citation metadata only
case_data = {
    "citation": "Smith v. Jones, 123 F.3d 456 (3d Cir. 2024)",
    "court": "United States Court of Appeals, Third Circuit",
    "date": "2024-03-15",
    "docket": "23-1234",
    "source_url": "https://www.courtlistener.com/opinion/...",
    "westlaw_cite": "2024 WL 123456",  # Citation only, not content
    "lexis_cite": "2024 U.S. App. LEXIS 5678",
    "fair_use_excerpt": "The court held that...[50 words max]",
    "our_analysis": "This case supports our motion because..."
}
```

#### ❌ PROHIBITED: Full Text Republication

```python
# WRONG - Do NOT store full opinion text from Westlaw/Lexis
case_data = {
    "full_text": "...[20 pages of copyrighted opinion]...",  # ILLEGAL
    "westlaw_headnotes": "...",  # PROPRIETARY - DO NOT COPY
    "lexis_shepards": "...",  # PROPRIETARY - DO NOT COPY
}
```

### Fair Use Guidelines

- **Excerpts:** Maximum 200 words per source
- **Purpose:** Criticism, comment, news reporting, teaching, research
- **Attribution:** Always cite source and copyright holder
- **No Substitution:** Excerpt must not replace need for original

### Sources Requiring Extra Caution

| Source             | What's Protected                                | What You Can Use                                 |
| ------------------ | ----------------------------------------------- | ------------------------------------------------ |
| **Westlaw**        | Headnotes, KeyCite, synopses, enhanced features | Public domain case text, citations               |
| **LexisNexis**     | Shepard's, summaries, editorial enhancements    | Public domain case text, citations               |
| **Bloomberg Law**  | BCite, summaries, practice tools                | Public domain case text, citations               |
| **Pacer**          | Free (but terms limit republication)            | Case metadata, docket entries (with attribution) |
| **Police Reports** | Usually copyrighted by department               | Fair use excerpts for litigation only            |
| **BWC Footage**    | Copyright varies by jurisdiction                | Client-owned footage only; cite public records   |

---

## PATTERN 2: KEEP PROPRIETARY LAYERS SEPARATE

### Principle

**Isolate proprietary data** in non-exportable private tables. Never mix proprietary fields with public domain content.

### Database Architecture

#### Public Domain Table (SAFE TO EXPORT)

```sql
CREATE TABLE public_case_data (
    case_id UUID PRIMARY KEY,
    citation TEXT NOT NULL,  -- "Smith v. Jones, 123 F.3d 456"
    court TEXT,
    date DATE,
    docket_number TEXT,
    parties TEXT,
    public_domain_text TEXT,  -- From CourtListener, Justia, etc.
    our_analysis TEXT,  -- Our original work
    created_at TIMESTAMP
);
```

#### Proprietary Data Table (NEVER EXPORT)

```sql
CREATE TABLE proprietary_source_data (
    id UUID PRIMARY KEY,
    case_id UUID REFERENCES public_case_data(case_id),
    source TEXT CHECK (source IN ('westlaw', 'lexis', 'bloomberg')),
    westlaw_keycite JSONB,  -- PROPRIETARY - internal use only
    lexis_shepards JSONB,  -- PROPRIETARY - internal use only
    headnotes JSONB,  -- PROPRIETARY - internal use only
    editorial_enhancements JSONB,  -- PROPRIETARY - internal use only
    accessed_date TIMESTAMP,
    -- CRITICAL: Mark as non-exportable
    export_allowed BOOLEAN DEFAULT FALSE,
    internal_use_only BOOLEAN DEFAULT TRUE
);
```

#### Access Control

```python
class CaseData:
    def export_for_court(self):
        """Export only public domain and our original analysis."""
        return {
            "citation": self.citation,
            "court": self.court,
            "public_text": self.public_domain_text,
            "our_analysis": self.our_analysis,
            # NEVER include: westlaw_keycite, lexis_shepards, etc.
        }

    def internal_research_view(self, user):
        """Internal use only - requires special permission."""
        if not user.has_permission('view_proprietary_research'):
            raise PermissionDenied("Proprietary data access restricted")

        return {
            **self.export_for_court(),
            "westlaw_keycite": self.westlaw_keycite,  # Internal only
            "lexis_shepards": self.lexis_shepards,  # Internal only
            "_warning": "PROPRIETARY DATA - DO NOT EXPORT OR SHARE"
        }
```

### Storage Location Rules

| Data Type                  | Storage Location                | Export Allowed?                |
| -------------------------- | ------------------------------- | ------------------------------ |
| Public domain case text    | `public_case_data` table        | ✅ Yes                         |
| Our original analysis      | `public_case_data` table        | ✅ Yes                         |
| Client-provided evidence   | `user_evidence` table           | ✅ Yes (to client only)        |
| Westlaw/Lexis data         | `proprietary_source_data` table | ❌ NO - internal use only      |
| Police reports (full text) | Do not store - link only        | ❌ NO - fair use excerpts only |

---

## PATTERN 3: RIGHTS-AWARE EXPORTS

### Principle

**Every export must include only materials you're permitted to export, with proper attribution and a source manifest.**

### Export Manifest Template

```json
{
  "export_id": "exp_abc123",
  "created_at": "2026-01-22T10:30:00Z",
  "created_by": "attorney@lawfirm.com",
  "case_number": "ATL-L-002794-25",
  "export_type": "discovery_production",

  "materials_included": {
    "bwc_videos": [
      {
        "filename": "BWC_Officer_Smith_2025-01-15.mp4",
        "source": "Atlantic County Sheriff OPRA Request #2025-001",
        "rights": "Public record - New Jersey OPRA",
        "sha256": "abcd1234...",
        "acquired_by": "John Doe, Esq.",
        "acquired_date": "2025-01-18"
      }
    ],
    "transcripts": [
      {
        "filename": "BWC_transcript_2025-01-15.pdf",
        "source": "Generated by BarberX Legal Tech (Whisper AI)",
        "rights": "Original work product - attorney work product privilege",
        "based_on": "BWC_Officer_Smith_2025-01-15.mp4",
        "generated_date": "2025-01-20"
      }
    ],
    "case_law": [
      {
        "citation": "Tennessee v. Garner, 471 U.S. 1 (1985)",
        "excerpt": "The use of deadly force... [excerpt within fair use]",
        "source": "CourtListener (public domain)",
        "rights": "Public domain - U.S. Supreme Court opinion",
        "fair_use_purpose": "Legal argument in brief",
        "excerpt_length": "150 words"
      }
    ],
    "our_analysis": [
      {
        "filename": "Discrepancy_Report_ATL-L-002794-25.pdf",
        "source": "BarberX Legal Tech analysis",
        "rights": "Original work product - attorney work product privilege",
        "copyright": "© 2026 [Law Firm Name]. All rights reserved."
      }
    ]
  },

  "attribution_requirements": [
    "Whisper AI (OpenAI) used for transcription under MIT License",
    "CourtListener used for public domain case law under CC0 License",
    "BarberX Legal Tech platform (proprietary software)"
  ],

  "excluded_materials": {
    "reason": "Proprietary or restricted",
    "items": [
      "Westlaw KeyCite analysis (proprietary - internal research only)",
      "Police report full text (copyrighted - fair use excerpts only in brief)"
    ]
  },

  "verification": {
    "all_materials_permitted": true,
    "attorney_certification": "I certify all materials comply with copyright and licensing requirements",
    "certifying_attorney": "John Doe, Esq., Bar #12345",
    "certification_date": "2026-01-22"
  }
}
```

### Export Validation Code

```python
class RightsAwareExport:
    def __init__(self, case_id):
        self.case_id = case_id
        self.manifest = {
            "export_id": str(uuid.uuid4()),
            "created_at": datetime.utcnow().isoformat(),
            "materials_included": {},
            "attribution_requirements": [],
            "excluded_materials": {"reason": "Proprietary or restricted", "items": []}
        }

    def add_material(self, material):
        """Add material only if export rights verified."""
        if not material.export_allowed:
            self.manifest["excluded_materials"]["items"].append({
                "filename": material.filename,
                "reason": material.restriction_reason
            })
            logger.warning(f"Material excluded from export: {material.filename} - {material.restriction_reason}")
            return False

        # Verify copyright compliance
        if material.source_type == "proprietary_legal_database":
            raise ExportViolation(f"Cannot export proprietary data: {material.filename}")

        if material.source_type == "copyrighted_document" and not material.is_fair_use:
            raise ExportViolation(f"Full copyrighted document cannot be exported: {material.filename}")

        # Add to manifest with full attribution
        self.manifest["materials_included"].setdefault(material.category, []).append({
            "filename": material.filename,
            "source": material.source,
            "rights": material.rights_statement,
            "sha256": material.file_hash,
            "acquired_by": material.acquired_by,
            "acquired_date": material.acquired_date.isoformat()
        })

        # Add attribution requirement
        if material.attribution_required:
            self.manifest["attribution_requirements"].append(material.attribution_text)

        return True

    def finalize_export(self, attorney_email):
        """Generate final export package with manifest."""
        self.manifest["verification"] = {
            "all_materials_permitted": len(self.manifest["excluded_materials"]["items"]) == 0,
            "certifying_attorney": attorney_email,
            "certification_date": datetime.utcnow().isoformat()
        }

        # Write manifest to export directory
        manifest_path = f"exports/{self.manifest['export_id']}/RIGHTS_MANIFEST.json"
        with open(manifest_path, 'w') as f:
            json.dump(self.manifest, f, indent=2)

        return manifest_path
```

---

## ENFORCEMENT CHECKLIST

### Pre-Export Verification (MANDATORY)

- [ ] All materials have verified export rights
- [ ] No proprietary database content (Westlaw/Lexis) included
- [ ] Fair use excerpts under 200 words with attribution
- [ ] Public domain sources clearly identified
- [ ] Client-owned evidence properly attributed
- [ ] Our original work product clearly marked
- [ ] Attribution requirements documented
- [ ] Rights manifest generated and included
- [ ] Attorney certification obtained

### Code Review Requirements

- [ ] No full-text storage of copyrighted materials
- [ ] Proprietary data in separate, non-exportable tables
- [ ] Export functions check `export_allowed` flag
- [ ] Fair use excerpt length validation (max 200 words)
- [ ] Source attribution auto-generated in exports
- [ ] Manifest generation tested

### Audit Trail

- [ ] All exports logged with manifest ID
- [ ] Attorney certification recorded
- [ ] Source URLs preserved for verification
- [ ] Acquisition dates documented
- [ ] Chain of custody maintained

---

## CONSEQUENCES OF NON-COMPLIANCE

### Legal Risks

- **Copyright infringement:** Statutory damages up to $150,000 per work
- **License violations:** Westlaw/Lexis may terminate access, sue for damages
- **Malpractice:** Attorney discipline for unauthorized republication
- **Injunctions:** Court orders to cease operations

### Company Risks

- **Platform shutdown:** Hosting providers may suspend service
- **DMCA takedowns:** GitHub/hosting removal
- **Reputation damage:** Loss of attorney trust
- **License revocation:** Loss of access to legal research tools

---

## COMPLIANCE CONTACTS

**Legal Questions:**  
BarberCamX@ProtonMail.com

**Copyright Concerns:**  
Report immediately to: legal@barberx.info

**License Violations:**  
Escalate to: compliance@barberx.info

---

**LAST UPDATED:** January 22, 2026  
**REVIEW SCHEDULE:** Quarterly  
**NEXT REVIEW:** April 22, 2026
