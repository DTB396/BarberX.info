# Faith Frontier Repository Refactor Prompt

**Purpose:** Comprehensive repository audit and refactor using centralized AI governance architecture  
**Context:** All scripts in `/scripts/` directory + governance rules in `/.ai/`  
**Target:** Entire FaithFrontier repository  
**Date:** December 20, 2025

---

## TASK OVERVIEW

Execute a **marathon refactor session** to audit and improve the FaithFrontier repository using the newly implemented AI governance architecture located in `/.ai/`.

This is a **multi-stage process** that will systematically review all content, code, documentation, and automation to ensure strict compliance with:

- **SYSTEM.md** - Foundational behavioral rules
- **STYLE.md** - Writing and tone standards
- **DOMAIN.md** - FaithFrontier project context
- **COMPLIANCE.md** - Legal and ethical boundaries
- **OUTPUT_RULES.md** - Technical and format conventions

---

## GOVERNANCE ARCHITECTURE REFERENCE

Before beginning, load and internalize these instruction files **in order**:

```text
1. /.ai/SYSTEM.md           ← Foundational constraints (HIGHEST AUTHORITY)
2. /.ai/STYLE.md            ← Writing standards
3. /.ai/DOMAIN.md           ← Project-specific context
4. /.ai/COMPLIANCE.md       ← Legal/ethical boundaries (ABSOLUTE)
5. /.ai/OUTPUT_RULES.md     ← Technical specifications
6. /.ai/CODEX.md            ← Agent-specific safety protocols
```

**Critical:** No file overrides SYSTEM.md or COMPLIANCE.md. These are **non-negotiable constraints**.

---

## AVAILABLE AUTOMATION SCRIPTS

The following scripts exist in `/scripts/` and may be used or improved during this refactor:

### Content & Analysis Scripts

1. **`analyze-cases.js`** - OpenAI-powered case analysis (judicial + journalistic perspectives)
   - Generates structured YAML analysis in `_data/analysis/`
   - Optional API key (graceful degradation)
   - Uses dual-perspective prompts

2. **`refactor-with-governance.js`** - Content refactor tool (NEEDS UPDATE)
   - Currently references `.github/copilot-instructions.md`
   - **TODO:** Update to load from `/.ai/` structure
   - Validates content against governance rules
   - Provides audit and interactive modes

3. **`scan-all-cases.js`** - Case structure scanner
   - Analyzes all case files
   - Checks front matter completeness
   - Reports missing fields or structural issues

### Docket & Document Management Scripts

4. **`docket-intake.js`** - Automated PDF intake and organization
   - Processes PDFs from `_inbox/` or `assets/uploads/`
   - Normalizes filenames: `YYYY-MM-DD_Type_Description.pdf`
   - Updates `_data/docket/<slug>.yml`
   - Moves files to `cases/<slug>/filings/`

5. **`batch-pdf-intake.js`** - Bulk PDF processing
6. **`pdf-intake.js`** - Single PDF intake
7. **`web-intake.js`** - Web-based upload handler
8. **`normalize-case-folders.js`** - Case directory structure normalization
9. **`reorganize-docket-files.js`** - Docket file reorganization

### Docket Repair & Maintenance Scripts

10. **`repair-docket-system.js`** - Comprehensive docket repair
11. **`repair-docket.js`** - Legacy docket repair
12. **`repair-docket-simple.sh`** - Shell-based docket repair
13. **`repair-docket-file-links-to-filings.js`** - Fix broken filing links
14. **`resolve-docket-conflicts.js`** - Docket conflict resolution
15. **`resolve-docket-conflicts.py`** - Python docket conflict resolver
16. **`reorganize-dockets.py`** - Python docket reorganizer

### OPRA (Open Public Records Act) Scripts

17. **`opra_case_autotag.py`** - Auto-tag OPRA cases
18. **`rename-opra-files.js`** - OPRA file renaming (JavaScript)
19. **`rename-opra-files.sh`** - OPRA file renaming (Shell)

### Quality Assurance Scripts

20. **`check-pdf-health.js`** - PDF file health checker
21. **`check-pdf-links.js`** - PDF link validation
22. **`check-site-links.js`** - Site-wide link checker
23. **`check-markdown-css.js`** - Markdown/CSS validation
24. **`check-nav-snapshot.js`** - Navigation structure validation
25. **`validate-contrast.js`** - Accessibility contrast checker
26. **`verify-frontend-js.js`** - Frontend JavaScript validation

### Utility Scripts

27. **`generate-checksums.js`** - File integrity checksums
28. **`generate-placeholder-pdfs.js`** - Placeholder PDF generation
29. **`generate-weekly-changelog.js`** - Automated changelog generation
30. **`slugify.js`** - Slug generation utility
31. **`compress_pdfs.sh`** - PDF compression (shell script)
32. **`test-openai-connection.js`** - OpenAI API connectivity test

### Workflow Scripts

33. **`analyze-workflows.py`** - GitHub Actions workflow analyzer
34. **`rerun_failed_workflows.sh`** - Retry failed workflows

---

## REFACTOR MARATHON STAGES

Execute these stages **sequentially**, documenting progress and issues at each step.

### STAGE 1: GOVERNANCE ALIGNMENT AUDIT

**Objective:** Ensure all scripts and content understand the new `.ai/` governance structure.

**Tasks:**

1. **Update `refactor-with-governance.js`:**
   - Change from reading `.github/copilot-instructions.md`
   - Load all files from `/.ai/` in prescribed order
   - Concatenate SYSTEM → STYLE → DOMAIN → COMPLIANCE → OUTPUT_RULES
   - Update rule extraction to parse new file structure
   - Add validation for hierarchical inheritance (no overrides)

2. **Update `analyze-cases.js`:**
   - Review system prompt construction
   - Ensure prompts align with COMPLIANCE.md (no legal advice)
   - Verify dual-perspective approach matches STYLE.md tone
   - Check that analysis outputs follow OUTPUT_RULES.md YAML format

3. **Audit all other scripts:**
   - Identify any hardcoded instruction logic
   - Replace with references to `/.ai/` files where appropriate
   - Ensure error messages follow STYLE.md conventions
   - Verify file naming follows OUTPUT_RULES.md patterns

**Output:** Report documenting all scripts updated + any governance conflicts found

---

### STAGE 2: CONTENT COMPLIANCE SWEEP

**Objective:** Review all content collections for compliance with COMPLIANCE.md boundaries.

**Tasks:**

1. **Essays (`_essays/`):**
   - Check for prohibited economic language (alternative currencies, fiat replacement)
   - Verify religious content boundaries (no prophetic certainty, no divine mandate)
   - Ensure tone matches STYLE.md (calm, grounded, sober)
   - Validate front matter against OUTPUT_RULES.md schema

2. **Cases (`_cases/`):**
   - Ensure legal content includes disclaimers ("not legal advice")
   - Verify procedural focus (not substantive legal interpretation)
   - Check privacy compliance (no PII in public records beyond what's in docket)
   - Validate case front matter completeness

3. **Trust Documents (`_trust/`):**
   - Ensure lawful framing (no parallel government claims)
   - Check mission statements align with DOMAIN.md
   - Verify no financial sovereignty language
   - Confirm transparency commitments

4. **Manifestos (`_manifesto/`):**
   - High-risk category - extra scrutiny
   - Check for absolutist or extremist language
   - Verify separation of belief from legal compliance
   - Ensure no challenge to government legitimacy

5. **Pages (`_pages/`) and Posts (`_posts/`):**
   - General compliance check
   - Tone and style validation
   - Link validation

**Output:** Compliance report with:

- Files flagged for review
- Specific violations or concerns
- Suggested rewrites or removals
- Low/Medium/High risk categorization

---

### STAGE 3: TECHNICAL STANDARDS ENFORCEMENT

**Objective:** Ensure all files follow OUTPUT_RULES.md specifications.

**Tasks:**

1. **File Naming Audit:**
   - Check all Markdown files follow naming conventions
   - Verify PDF files use `YYYY-MM-DD_Type_Description.pdf` format
   - Ensure docket YAML files match case slugs
   - Identify and rename non-compliant files

2. **Front Matter Validation:**
   - Verify all case files have required fields
   - Check essay front matter completeness
   - Validate YAML syntax (use `js-yaml` for parsing)
   - Report missing or incorrect fields

3. **Markdown Structure Check:**
   - One H1 per document
   - Hierarchical headings (no skipping H2 → H4)
   - Proper list formatting
   - Code blocks with language identifiers
   - Link format (relative paths for internal, HTTPS for external)

4. **Directory Structure Compliance:**
   - Verify files in correct directories
   - Check PDF locations: `cases/<slug>/filings/`
   - Validate data files: `_data/docket/<slug>.yml`
   - Ensure no files in wrong locations

**Output:** Technical compliance report with fix script or manual corrections needed

---

### STAGE 4: DOCKET SYSTEM INTEGRITY

**Objective:** Ensure complete docket system health and consistency.

**Tasks:**

1. **Run Full Docket Repair:**

   ```bash
   node scripts/repair-docket-system.js
   ```

2. **Verify Docket-Case Mapping:**
   - Check `_data/cases-map.yml` completeness
   - Ensure all dockets map to correct slugs
   - Identify orphaned dockets or unmapped cases

3. **PDF Health Check:**

   ```bash
   node scripts/check-pdf-health.js
   ```

   - Verify all PDFs are valid and not corrupt
   - Check file sizes (min 4KB, flag >5MB)
   - Identify missing PDFs referenced in dockets

4. **Filing Link Validation:**

   ```bash
   node scripts/repair-docket-file-links-to-filings.js
   ```

   - Ensure all docket entries link to existing PDFs
   - Fix broken references
   - Report missing files

**Output:** Docket system health report + any automated fixes applied

---

### STAGE 5: LINK INTEGRITY & ACCESSIBILITY

**Objective:** Ensure site-wide link health and accessibility standards.

**Tasks:**

1. **Site Link Check:**

   ```bash
   node scripts/check-site-links.js
   ```

   - Identify broken internal links
   - Check external links (404s, redirects)
   - Validate anchor links within pages

2. **Navigation Validation:**

   ```bash
   node scripts/check-nav-snapshot.js
   ```

   - Verify navigation structure consistency
   - Check for orphaned pages
   - Ensure breadcrumb accuracy

3. **Accessibility Audit:**

   ```bash
   node scripts/validate-contrast.js
   ```

   - Check color contrast ratios (WCAG AA minimum)
   - Verify alt text on images
   - Check semantic HTML structure

4. **Frontend JavaScript Health:**

   ```bash
   node scripts/verify-frontend-js.js
   ```

   - Validate JavaScript syntax
   - Check for console errors
   - Verify functionality

**Output:** Link and accessibility report with prioritized fixes

---

### STAGE 6: AUTOMATION & WORKFLOW REVIEW

**Objective:** Ensure GitHub Actions workflows align with governance and function correctly.

**Tasks:**

1. **Workflow Analysis:**

   ```bash
   python scripts/analyze-workflows.py
   ```

   - Review all `.github/workflows/*.yml` files
   - Check for optional vs. required steps
   - Verify secret handling follows COMPLIANCE.md
   - Ensure error handling is graceful

2. **Case Analysis Workflow:**
   - Review `case-analysis.yml`
   - Verify it respects optional OpenAI API key
   - Check that analysis follows COMPLIANCE.md (no legal advice)
   - Validate output format matches OUTPUT_RULES.md

3. **Docket Intake Workflow:**
   - Review `docket-intake.yml`
   - Check file handling security
   - Verify staging and commit process
   - Ensure proper error reporting

4. **Build & Deploy Workflow:**
   - Review `jekyll.yml`
   - Check build steps for efficiency
   - Verify no secrets leaked in logs
   - Ensure proper caching

**Output:** Workflow recommendations report

---

### STAGE 7: AI ANALYSIS QUALITY REVIEW

**Objective:** Evaluate existing AI-generated content for quality and compliance.

**Tasks:**

1. **Review Existing Case Analyses:**
   - Check files in `_data/analysis/`
   - Verify judicial and journalistic perspectives are distinct
   - Ensure no legal advice language
   - Check that analysis follows STYLE.md tone (calm, grounded)

2. **Regenerate Problem Analyses:**
   - Identify any analyses that violate governance
   - Re-run `analyze-cases.js` with updated prompts
   - Compare before/after for improvement

3. **Analysis Template Update:**
   - Review prompt templates in `analyze-cases.js`
   - Ensure they incorporate `/.ai/` governance rules
   - Add explicit compliance guardrails
   - Test with one case before batch processing

**Output:** Analysis quality report + regeneration recommendations

---

### STAGE 8: DOCUMENTATION COMPLETENESS

**Objective:** Ensure all documentation is current, accurate, and complete.

**Tasks:**

1. **Core Documentation Review:**
   - `README.md` - Update for `.ai/` governance structure
   - `DOCKET-SYSTEM.md` - Verify current accuracy
   - `_docs/ANALYSIS-SYSTEM.md` - Update for new governance
   - `_docs/QUICKSTART-ANALYSIS.md` - Ensure setup steps work
   - `.ai/README.md` - Review public-facing governance overview

2. **Script Documentation:**
   - Ensure each script has usage comments at top
   - Verify examples are accurate
   - Check that prerequisites are documented
   - Add references to relevant `/.ai/` files

3. **Workflow Documentation:**
   - `.github/SETUP-OPENAI.md` - Verify setup instructions
   - Workflow README (if exists) - Update
   - Add workflow diagrams if helpful

4. **Missing Documentation:**
   - Identify undocumented features or systems
   - Create documentation following OUTPUT_RULES.md format
   - Use STYLE.md tone and structure

**Output:** Documentation completeness report + new/updated docs

---

### STAGE 9: STYLE & TONE HARMONIZATION

**Objective:** Ensure consistent voice across all written content.

**Tasks:**

1. **Tone Audit:**
   - Scan all content for tone violations:
     - Alarmist or urgent language
     - Speculative or prophetic claims
     - Absolutist statements
     - Fear-mongering or apocalyptic framing
   - Flag for rewrite

2. **Voice Consistency:**
   - Check for first-person vs. institutional voice
   - Ensure appropriate formality level
   - Verify neighbor-facing (not adversarial) language
   - Check cultural sensitivity

3. **Structural Harmonization:**
   - Ensure consistent heading hierarchy
   - Standardize list formats
   - Unify code block styles
   - Consistent link formatting

4. **Batch Improvements:**
   - Identify common patterns that need fixing
   - Create sed/awk scripts for mechanical fixes
   - Run `refactor-with-governance.js` in batch mode
   - Review and commit changes incrementally

**Output:** Style audit report + batch of harmonized content

---

### STAGE 10: FINAL VALIDATION & BUILD TEST

**Objective:** Ensure all changes build correctly and site functions properly.

**Tasks:**

1. **Jekyll Build Test:**

   ```bash
   bundle exec jekyll build
   ```

   - Verify no build errors
   - Check for broken Liquid templates
   - Validate generated HTML
   - Test locally: `bundle exec jekyll serve`

2. **Checksum Generation:**

   ```bash
   node scripts/generate-checksums.js
   ```

   - Generate file integrity checksums
   - Compare with previous checksums
   - Identify unexpected changes

3. **Comprehensive Link Check:**
   - Run full site link validation
   - Test all interactive features
   - Check form functionality (if any)
   - Verify PDF serving

4. **GitHub Actions Dry Run:**
   - Push to test branch
   - Watch all workflows execute
   - Verify no failures
   - Check deployed preview

**Output:** Final validation report + "Ready to Deploy" confirmation

---

## EXECUTION GUIDELINES

### Working Method

1. **Sequential Execution** - Complete each stage before moving to next
2. **Document Progress** - Create reports in `reports/refactor-marathon/`
3. **Incremental Commits** - Commit after each major stage with descriptive messages
4. **Issue Tracking** - Create GitHub issues for items requiring human decision
5. **Backup Before Changes** - Use `reports/refactor-marathon/backups/` for originals

### Safety Protocols

Per **CODEX.md** safety rules:

- **Hard Stops:**
  - Attempting to commit secrets
  - Violating COMPLIANCE.md boundaries
  - Generating harmful content
  - Exceeding authorized scope
- **Confirmation Required:**
  - Deleting files
  - Modifying critical configs (`.github/`, `_config.yml`)
  - Breaking changes to public content
  - Bulk automated rewrites (>10 files)

- **Rollback Ready:**
  - Keep Git history clean with clear commits
  - Tag before major changes: `git tag pre-refactor-stage-N`
  - Document rollback procedures in reports

### Quality Assurance

Before considering stage complete:

- [ ] Jekyll builds successfully
- [ ] No new errors introduced
- [ ] Changes align with governance files
- [ ] Documentation updated to reflect changes
- [ ] Commit messages follow convention: `type(scope): description`

---

## OUTPUT STRUCTURE

Create the following directory structure for reports:

```text
reports/refactor-marathon/
├── 00-OVERVIEW.md                    # This prompt + execution plan
├── 01-governance-alignment.md        # Stage 1 report
├── 02-content-compliance.md          # Stage 2 report
├── 03-technical-standards.md         # Stage 3 report
├── 04-docket-integrity.md            # Stage 4 report
├── 05-links-accessibility.md         # Stage 5 report
├── 06-automation-workflows.md        # Stage 6 report
├── 07-ai-analysis-quality.md         # Stage 7 report
├── 08-documentation.md               # Stage 8 report
├── 09-style-harmonization.md         # Stage 9 report
├── 10-final-validation.md            # Stage 10 report
├── FINAL-SUMMARY.md                  # Complete refactor summary
├── backups/                          # Original files before changes
└── scripts/                          # Any custom refactor scripts
```

---

## SUCCESS CRITERIA

The refactor marathon is complete when:

✅ All scripts reference `/.ai/` governance structure  
✅ All content complies with COMPLIANCE.md boundaries  
✅ All files follow OUTPUT_RULES.md specifications  
✅ Docket system is fully functional and validated  
✅ All links are valid and accessible  
✅ GitHub Actions workflows execute without errors  
✅ AI-generated content meets quality standards  
✅ Documentation is complete and current  
✅ Style and tone are consistent across site  
✅ Jekyll builds and deploys successfully

---

## ESTIMATED EFFORT

- **Stage 1:** 2-3 hours (script updates)
- **Stage 2:** 4-6 hours (content review, high priority)
- **Stage 3:** 3-4 hours (technical validation)
- **Stage 4:** 2-3 hours (docket system)
- **Stage 5:** 2-3 hours (links and accessibility)
- **Stage 6:** 2-3 hours (workflow review)
- **Stage 7:** 3-4 hours (AI analysis quality)
- **Stage 8:** 2-3 hours (documentation)
- **Stage 9:** 4-5 hours (style harmonization)
- **Stage 10:** 2-3 hours (final validation)

**Total:** 26-37 hours (estimate for autonomous agent or marathon session)

---

## AUTHORIZATION & CONSTRAINTS

This refactor operates under:

- **Authority:** SYSTEM.md and COMPLIANCE.md (non-negotiable)
- **Scope:** Entire FaithFrontier repository
- **Autonomy Level:** Medium (requires confirmation for critical changes)
- **Risk Tolerance:** Conservative (safety over speed)
- **Primary Goal:** Governance compliance + quality improvement

**If uncertain at any stage:** Stop, document the issue, and request human guidance.

---

## BEGIN EXECUTION

When ready to begin, start with **STAGE 1: GOVERNANCE ALIGNMENT AUDIT** and proceed sequentially through all stages.

Document your progress in `reports/refactor-marathon/` using the structure above.

Good luck. This is important work.

**END - REFACTOR PROMPT**
