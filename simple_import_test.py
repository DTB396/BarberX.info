"""
Simple test import - bypasses all verification
"""

import sys
import time
from legal_library import LegalLibraryService
from models_auth import db

# Test with one case
library = LegalLibraryService()

citation = "384 U.S. 436"  # Miranda v. Arizona

print(f"Testing import of {citation}...")
print(f"CourtListener API URL: {library.courtlistener_api}search/")

# Try to fetch
import requests
url = f"{library.courtlistener_api}search/"
params = {
    'citation': citation,
    'format': 'json'
}

print(f"\nAttempting API call...")
print(f"URL: {url}")
print(f"Params: {params}\n")

try:
    response = requests.get(url, params=params, timeout=10)
    print(f"Status: {response.status_code}")
    print(f"Response length: {len(response.text)} bytes")
    
    if response.status_code == 200:
        data = response.json()
        print(f"\nJSON response keys: {list(data.keys())}")
        
        if 'results' in data:
            print(f"Number of results: {len(data['results'])}")
            
            if data['results']:
                result = data['results'][0]
                print(f"\nFirst result:")
                print(f"  - Case name: {result.get('caseName', 'N/A')}")
                print(f"  - Court: {result.get('court', 'N/A')}")
                print(f"  - Date: {result.get('dateFiled', 'N/A')}")
                print(f"  - Docket: {result.get('docketNumber', 'N/A')}")
                
                # Try to import
                print(f"\nAttempting to import via library service...")
                doc = library.ingest_from_courtlistener(citation)
                
                if doc:
                    print(f"[OK] SUCCESS! Imported case id {doc.id}")
                    print(f"  - Title: {doc.title}")
                    print(f"  - Citation: {doc.citation}")
                    print(f"  - Court: {doc.court}")
                else:
                    print("[FAIL] Import returned None")
            else:
                print("[WARN] No results found for citation")
        else:
            print(f"[WARN] No 'results' key in response: {list(data.keys())}")
    else:
        print(f"[FAIL] HTTP {response.status_code}")
        print(f"Response: {response.text[:500]}")
        
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
