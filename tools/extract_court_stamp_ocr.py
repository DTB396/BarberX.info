# Python script: Extract court stamp from PDF using OCR (pytesseract)
import sys
import pytesseract
from pdf2image import convert_from_path

if len(sys.argv) < 2:
    print('Usage: python extract_court_stamp_ocr.py <pdfPath>')
    sys.exit(1)

pdf_path = sys.argv[1]

# Convert first page of PDF to image
images = convert_from_path(pdf_path, first_page=1, last_page=1)
if not images:
    print('No images found in PDF')
    sys.exit(2)

text = pytesseract.image_to_string(images[0])

# Heuristic: look for lines with "Filed" and "Court"
for line in text.split('\n'):
    if 'Filed' in line and 'Court' in line:
        print(line.strip())
        sys.exit(0)
    if 'Filed' in line and any(char.isdigit() for char in line):
        print(line.strip())
        sys.exit(0)

sys.exit(2) # Not found
