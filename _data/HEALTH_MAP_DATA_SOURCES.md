# Health Map Data Sources - Verification Report
**Date:** 2025-12-26  
**Purpose:** Document official sources for overmedication data used in holistic health map

## Primary Official Sources

### 1. CDC Opioid Dispensing Rates (2023 - Most Recent)
**Source:** https://www.cdc.gov/overdose-prevention/data-research/facts-stats/us-dispensing-rate-maps.html

**National Rate (2023):** 37.5 prescriptions per 100 people
- Down from 39.5 in 2022
- Down from 46.8 in 2019
- Data from IQVIA Xponent (94% of retail pharmacy coverage)

**Highest Rates by State (2023):**
- States with highest opioid prescribing remain in South/Southeast
- Approximately 4% of counties dispensed enough for every resident
- Some counties 9x the national average

**Note:** 2024 data will not be available until late 2025

### 2. SAMHSA NSDUH (National Survey on Drug Use and Health)
**Source:** https://www.samhsa.gov/data/report/2023-nsduh-detailed-tables

**Key Statistics:**
- Mental health medication trends
- Benzodiazepine use declining: 2.7% (2018) → 2.2% (2024)
- State-level breakdowns available in detailed tables

### 3. CDC FastStats - Therapeutic Drug Use
**Source:** https://www.cdc.gov/nchs/fastats/drug-use-therapeutic.htm

**National Statistics:**
- 49.9% of Americans used at least one prescription in last 30 days
- ~25% use three or more medications

### 4. GoodRx/SingleCare Most Popular Prescriptions by State (2023)
**Source:** https://www.singlecare.com/blog/most-popular-prescriptions-by-state-2023/
**Source:** https://www.goodrx.com/healthcare-access/research/most-popular-prescription-medications

**Top Prescriptions by State (2023):**
- **Vitamin D:** Most filled in 19 states + DC
- **Levothyroxine:** Top in 13 states
- **Amphetamine/dextroamphetamine (Adderall):** Top in 5 states
- **Amoxicillin:** Widely prescribed nationwide

**National Top 10 (2024 IQVIA data):**
1. Atorvastatin (cholesterol)
2. Amlodipine (blood pressure)
3. Levothyroxine (thyroid)
4. Lisinopril (blood pressure)
5. Losartan (blood pressure)
6. Rosuvastatin (cholesterol)
7. Metoprolol ER (blood pressure/heart)
8. Metformin (diabetes)
9. Gabapentin (nerve pain)
10. Pantoprazole (acid reflux/PPI)

### 5. Antidepressant Prescribing Trends
**Source:** https://www.truveta.com/blog/research/research-insights/mental-health-prescribing-trends/
**Source:** https://clincalc.com/DrugStats/TC/Antidepressants

**Key Findings:**
- SSRIs account for nearly 50% of antidepressant prescriptions
- Peaked 2020-2023, now stabilizing
- Concerns about long-term use for mild depression

### 6. PPI Overprescription
**Source:** https://jamanetwork.com/journals/jamainternalmedicine/fullarticle/2609952
**JAMA Internal Medicine 177(5): 657-659 (2017)**

**Key Finding:** Up to 70% of long-term PPI users lack valid indication

## Additional Research Sources Referenced

### Healthcare Management Degree Analysis
**Source:** https://www.healthcare-management-degree.net/overmedicated/
- Overview of overmedication patterns
- Regional analysis

### NCBI/PMC Research
**Source:** https://pmc.ncbi.nlm.nih.gov/articles/PMC6731049/
- Academic research on overprescription
- Clinical outcomes

### STAT News Investigation
**Source:** https://www.statnews.com/2019/04/02/overprescribed-americas-other-drug-problem/
- Investigative journalism on prescription trends

### Ripon Society Analysis
**Source:** https://riponsociety.org/article/america-the-overprescribed/
- Policy perspective on overprescription

### NIDA Drug Information
**Source:** https://nida.nih.gov/research-topics/drugs-a-to-z
- Scientific information on medications

## Data Quality Notes

### Current Data File Issues:
1. **overmedication.yml** - Contains some outdated statistics
   - Uses CDC 2022 data (43 per 100) - should update to 2023 (37.5 per 100)
   - State-specific rates need verification against official CDC maps
   
2. **medication_states_data.yml** - Needs verification
   - Percentage estimates need official source citations
   - Top medications by state should align with GoodRx/SingleCare data

### Recommended Updates:

1. **Update National Opioid Rate**
   - Change from "43 per 100" to "37.5 per 100 (2023 CDC data)"

2. **Add State-Specific Top Medications**
   - Integrate GoodRx state-by-state data (Vitamin D, Levothyroxine, etc.)

3. **Add Citation Links**
   - Include direct CDC map links
   - Link to SAMHSA detailed tables

4. **Update Benzodiazepine Data**
   - Note declining trend (2.7% → 2.2%)

5. **Add Newer Medication Trends**
   - GLP-1 agonists (Tirzepatide, etc.) for diabetes/weight loss
   - Growing mental health medication prescribing

## Verification Status

✅ **CDC Opioid Data:** Verified - 2023 official data available  
✅ **SAMHSA Mental Health Data:** Verified - 2023 NSDUH available  
✅ **GoodRx State Data:** Verified - 2023 analysis published  
✅ **Top 10 National Rx:** Verified - 2024 IQVIA data via GoodRx  
⚠️ **Site Data Files:** Need updates to match official 2023/2024 sources  

## Next Steps

1. Update overmedication.yml with 2023 CDC opioid rates
2. Update medication_states_data.yml with GoodRx state-specific top medications
3. Add source citations to both files
4. Consider adding disclaimer about data year and update frequency
5. Link to official CDC/SAMHSA pages for user verification

---

**Prepared by:** GitHub Copilot CLI  
**Date:** December 26, 2025  
**Purpose:** Ensure Faith Frontier health map uses accurate, verifiable, official data sources
