# PDF Batch Upload Inbox

This directory is the **staging area** for automated docket intake.

## 🚀 Quick Start

### For ATL-L-003252-25 (or any case):

**Option 1: Drop files directly here**
```bash
# Copy PDFs with docket number in filename
cp ~/Downloads/ATL-L-003252-25-*.pdf _inbox/

# Then commit and push
git add _inbox/
git commit -m "intake: ATL-L-003252-25 batch upload"
git push
```

**Option 2: Use case-specific subdirectory** (recommended for organization)
```bash
# Create case inbox if needed
mkdir -p _inbox/atl-l-003252-25

# Copy files there
cp ~/Downloads/*.pdf _inbox/atl-l-003252-25/

# Then commit and push
git add _inbox/
git commit -m "intake: ATL-L-003252-25 batch"
git push
```

## 📋 How It Works

1. **Drop PDFs here** (or in case subdirectories)
2. **Git commit and push**
3. **GitHub Actions runs** scripts/docket-intake.js
4. **Automated processing:**
   - Detects docket numbers from filenames
   - Routes to correct case folder (cases/<slug>/filings/)
   - Renames with standard format (YYYY-MM-DD_Type_description.pdf)
   - Updates _data/docket/<slug>.yml
   - Creates PR for review
5. **You merge the PR** → Done!

## 📝 Filename Best Practices

**Include docket number** for auto-routing:
- ✅ ATL-L-003252-25-motion-to-dismiss.pdf
- ✅ 2025-12-20-ATL-L-003252-25-order.pdf
- ✅ order-ATL-L-003252-25.pdf

**Optional: Include date** (YYYY-MM-DD or YYYYMMDD):
- ✅ 2025-12-20-ATL-L-003252-25-brief.pdf
- ✅ 20251220-certification.pdf

**Without docket number** → files go to cases/unassigned/filings/
- ⚠️ random-document.pdf (manual routing needed)

## 🗂️ Directory Structure

```
_inbox/
├── README.md                    # This file
├── atl-l-003252-25/            # Optional: case-specific inbox
├── a-000313-25/                # Optional: another case
└── *.pdf                        # Or drop directly here
```

## 🔧 Supported Docket Formats

The intake system recognizes:
- ATL-L-003252-25 (NJ Superior Court - Law Division)
- ATL-DC-007956-25 (NJ Superior Court - Special Civil)
- ATL-22-002292 (Legacy format)
- A-000313-25 (NJ Appellate Division)
- 1:25-cv-15641-RMB-MJS (Federal District Court)

## 📚 Related Documentation

- **BATCH-UPLOAD-QUICKSTART.md** - Complete batch upload guide
- **DOCKET-SYSTEM.md** - Full docket management documentation
- **scripts/docket-intake.js** - The automation script

## 🛠️ Troubleshooting

**Q: Files not being processed?**
- Check filename includes docket number
- Verify case exists in _data/cases-map.yml
- Check GitHub Actions logs

**Q: Files going to wrong case?**
- Update mapping in _data/cases-map.yml
- Use case subdirectory for explicit routing

**Q: Need to add new case?**
1. Create case in _cases/<slug>/index.md
2. Add to _data/cases-map.yml
3. Create _data/docket/<slug>.yml
4. Drop PDFs in _inbox/

---

**Note:** _inbox/ should be tracked in git but individual PDFs are removed after intake processing.
