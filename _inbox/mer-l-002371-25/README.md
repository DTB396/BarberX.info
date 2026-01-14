# MER-L-002371-25 Inbox

Drop PDFs here for automated intake into the MER-L-002371-25 case.

## Quick Upload

```bash
# Copy PDFs to this directory
cp ~/Downloads/*.pdf _inbox/mer-l-002371-25/

# Commit and push
git add _inbox/mer-l-002371-25/
git commit -m "intake: MER-L-002371-25 new filings"
git push
```

## What Happens Next

GitHub Actions will:
1. Detect PDFs in this directory
2. Move them to `cases/mer-l-002371-25/filings/`
3. Rename to standard format: `YYYYMMDD-description.pdf`
4. Add entries to `_data/docket/mer-l-002371-25.yml`
5. Create a PR for your review

## Case Info

- **Docket:** MER-L-002371-25
- **Court:** Superior Court of New Jersey - Mercer County
- **Case Type:** Civil
- **Status:** Active
- **Filed:** 2025-10-29

## Current Docket

- 20 filings indexed (2025-10-29 to 2025-12-16)
- Ready for additional PDFs

## Need Help?

See `_cases/mer-l-002371-25/INTAKE-GUIDE.md` for complete documentation.
