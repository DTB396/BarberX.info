# njcivil_terminal_tool

A small Python CLI to help you (1) extract/search text from filed PDFs/DOCX and (2) generate NJ civil pleadings from YAML facts.

## Install (Windows PowerShell)

```powershell
py -m pip install --upgrade pypdf python-docx typer jinja2 pyyaml
```

## Extract text from your filings

```powershell
py .\njcivil.py extract .\filings .\filings_txt --overwrite
```

## Quick keyword search (no database)

```powershell
py .\njcivil.py grep .\filings_txt "tow|storage|notice|hearing"
py .\njcivil.py grep .\filings_txt "Ruiz" --literal
```

## Build a fast full-text index and query

```powershell
py .\njcivil.py index .\filings_txt .\index.sqlite
py .\njcivil.py query .\index.sqlite "tow NEAR/10 notice"
py .\njcivil.py query .\index.sqlite "body-worn camera"
```

## Generate drafts from YAML facts

1) Copy `facts.example.yml` to `facts.yml` and fill it in.

```powershell
py .\njcivil.py make verified_complaint --facts .\facts.yml --out .\drafts\Verified_Complaint.docx
py .\njcivil.py make motion_interim --facts .\facts.yml --out .\drafts\Notice_of_Motion_Interim_Relief.docx
py .\njcivil.py make certification --facts .\facts.yml --out .\drafts\Certification_Supporting_Interim_Relief.docx
```

## Merge exhibit PDFs

```powershell
py .\njcivil.py merge-pdfs .\drafts\Exhibits.pdf .\ExhibitA.pdf .\ExhibitB.pdf .\ExhibitC.pdf
```

## Notes

- Templates use `StrictUndefined`. If a required fact is missing, generation fails rather than inserting a guess.
- This tool is *format-light*. You should still review and finalize in Word before filing.
