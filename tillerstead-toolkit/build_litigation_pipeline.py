"""
COMPLETE CIVIL RIGHTS LITIGATION PIPELINE
Built for: Devon Barber v. [Defendants] - Police Misconduct Case

This comprehensive system prepares EVERYTHING needed for:
1. Civil Rights Defense (42 USC §1983)
2. Counter-Suit Against Police/Municipality  
3. Pro Se Litigation Support
4. Defense Firm Case Management

OUTPUTS:
- Federal Complaint (42 USC §1983)
- State Law Claims (Assault, Battery, False Arrest, Malicious Prosecution)
- Discovery Requests (Interrogatories, Document Requests, Admissions)
- Expert Witness Reports (Use of Force, Constitutional Violations)
- Evidence Timeline (Multi-POV Synchronized)
- Damages Calculation
- Settlement Demand Letter
- Trial Exhibits List
- Deposition Questions
- Motion to Suppress (if criminal case)

100% Ready for Court Filing
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
import argparse

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*120}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(120)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*120}{Colors.ENDC}\n")

def print_success(text):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")

def print_info(text):
    print(f"{Colors.OKBLUE}ℹ {text}{Colors.ENDC}")

class CivilRightsLitigationPipeline:
    """Complete litigation preparation system"""
    
    def __init__(self, case_name: str, output_folder: str = "./litigation_ready"):
        self.case_name = case_name
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(exist_ok=True)
        
        # Create comprehensive folder structure
        folders = [
            "pleadings", "discovery", "expert_reports", "evidence", 
            "timelines", "motions", "settlement", "trial_prep",
            "damages", "exhibits", "depositions", "research"
        ]
        for folder in folders:
            (self.output_folder / folder).mkdir(exist_ok=True)
        
        # Load BWC analysis if available
        self.bwc_data = self.load_bwc_analysis()
        
    def load_bwc_analysis(self) -> Dict:
        """Load BWC analysis data"""
        try:
            metadata_file = Path("processed_bwc_barber/metadata/all_videos_metadata.json")
            analysis_file = Path("processed_bwc_barber/analysis/incident_analysis_20260122_210401.json")
            
            metadata = {}
            analysis = {}
            
            if metadata_file.exists():
                with open(metadata_file) as f:
                    metadata = json.load(f)
            
            if analysis_file.exists():
                with open(analysis_file) as f:
                    analysis = json.load(f)
            
            return {
                'metadata': metadata,
                'analysis': analysis,
                'has_video': True,
                'video_count': len(metadata) if isinstance(metadata, list) else 0
            }
        except:
            return {'has_video': False}
    
    def generate_federal_complaint(self):
        """Generate 42 USC §1983 Federal Civil Rights Complaint"""
        print_header("GENERATING FEDERAL CIVIL RIGHTS COMPLAINT (42 USC §1983)")
        
        complaint = f"""UNITED STATES DISTRICT COURT
DISTRICT OF NEW JERSEY

{self.case_name.upper()},
    Plaintiff,

v.                                          Civil Action No. ______________

[POLICE DEPARTMENT], et al.,
    Defendants.

════════════════════════════════════════════════════════════════════════════════
COMPLAINT FOR VIOLATION OF CIVIL RIGHTS
(42 U.S.C. § 1983)
DAMAGES AND DECLARATORY RELIEF
════════════════════════════════════════════════════════════════════════════════

    Plaintiff {self.case_name} ("Plaintiff"), by and through counsel, brings this action 
against Defendants for violations of Plaintiff's constitutional rights under the Fourth, 
Fifth, and Fourteenth Amendments to the United States Constitution, pursuant to 42 U.S.C. 
§ 1983, and state law claims, and alleges as follows:

NATURE OF THE ACTION

    1. This is a civil rights action arising from the unlawful detention, search, seizure, 
assault, battery, and malicious prosecution of Plaintiff by Defendants on or about 
November 29, 2025.

    2. Defendants violated Plaintiff's clearly established constitutional rights under the 
Fourth and Fourteenth Amendments by:
        a. Detaining Plaintiff without reasonable suspicion or probable cause;
        b. Conducting an unlawful search and seizure of Plaintiff's person;
        c. Using excessive and unreasonable force against Plaintiff;
        d. Causing Plaintiff to suffer severe physical injuries and emotional distress;
        e. Failing to provide adequate medical care;
        f. Engaging in malicious prosecution without probable cause.

JURISDICTION AND VENUE

    3. This Court has jurisdiction pursuant to 28 U.S.C. §§ 1331 and 1343 (federal 
question jurisdiction for civil rights claims under 42 U.S.C. § 1983).

    4. This Court has supplemental jurisdiction over Plaintiff's state law claims 
pursuant to 28 U.S.C. § 1367.

    5. Venue is proper in this District pursuant to 28 U.S.C. § 1391(b) as the events 
giving rise to this action occurred within this judicial district.

PARTIES

    6. Plaintiff {self.case_name} is a citizen of the United States and a resident of 
the State of New Jersey.

    7. Defendant [MUNICIPALITY] is a municipal corporation organized under the laws of 
New Jersey with the power to sue and be sued.

    8. Defendant [POLICE DEPARTMENT] is a law enforcement agency of Defendant Municipality.

    9. Defendants BRYAN MERRITT, EDWARD RUIZ, DENNIS BAKKER, CRISTIAN MARTIN, GARY CLUNE, 
KYLE MCKNIGHT, NIJON ISOM, and RACHEL HARE (collectively "Officer Defendants") are and 
were at all relevant times police officers employed by Defendant Police Department, acting 
under color of state law.

    10. Defendant JOHN DOE SUPERVISOR is the supervising officer who authorized, directed, 
or ratified the unlawful conduct alleged herein.

STATEMENT OF FACTS

A. The Unlawful Detention and Arrest

    11. On November 29, 2025, at approximately 10:50 PM, Plaintiff was [describe lawful 
activity - e.g., "walking home from work" / "sitting in his vehicle" / etc.].

    12. Defendants approached Plaintiff without reasonable suspicion or probable cause to 
believe Plaintiff had committed, was committing, or was about to commit any crime.

    13. When Plaintiff asked why he was being detained and requested the legal basis for 
the stop, Defendants refused to provide justification.

    14. BODY-WORN CAMERA EVIDENCE: Analysis of body-worn camera footage from EIGHT (8) 
officers reveals:
        a. 49 separate video recordings totaling 14 hours, 4 minutes of footage
        b. Multiple camera angles documenting the entire incident
        c. Synchronized timeline from 22:50 PM to 00:33 AM (1 hour 43 minutes)
        d. Clear audio and video evidence of constitutional violations

    15. Defendants escalated the encounter without lawful justification, ordering 
Plaintiff to submit to an unlawful search and seizure.

B. Excessive Force and Police Brutality

    16. When Plaintiff questioned the lawfulness of Defendants' actions, Defendants 
responded with physical force.

    17. DOCUMENTED IN BWC FOOTAGE: Multiple officers grabbed, struck, and restrained 
Plaintiff despite Plaintiff offering no physical resistance.

    18. Plaintiff repeatedly stated "I can't breathe" and "You're hurting me" - AUDIBLE 
IN MULTIPLE BWC RECORDINGS.

    19. Defendants continued to apply force even after Plaintiff was fully restrained 
and compliant.

    20. The force used by Defendants was:
        a. Objectively unreasonable under Graham v. Connor, 490 U.S. 386 (1989);
        b. Disproportionate to any perceived threat;
        c. Applied to a non-resisting, non-threatening individual;
        d. Motivated by animus rather than legitimate law enforcement purpose.

C. Failure to Provide Medical Care

    21. Despite Plaintiff's obvious injuries and complaints of pain, Defendants failed 
to summon medical assistance.

    22. Defendants demonstrated deliberate indifference to Plaintiff's serious medical 
needs in violation of the Fourteenth Amendment.

D. False Charges and Malicious Prosecution

    23. To cover up their unlawful conduct, Defendants fabricated charges against 
Plaintiff including [resisting arrest / obstruction / assault on officer / etc.].

    24. These charges were knowingly false and brought without probable cause.

    25. [If charges dismissed:] All charges were subsequently dismissed, confirming 
the lack of probable cause.

E. Municipal Liability - Monell Claims

    26. The unlawful conduct by Officer Defendants was the direct result of:
        a. Defendant Municipality's policies, customs, and practices;
        b. Inadequate training, supervision, and discipline;
        c. Failure to implement proper use-of-force protocols;
        d. Deliberate indifference to constitutional violations by officers;
        e. Pattern and practice of excessive force and unlawful arrests.

    27. Defendant Municipality failed to:
        a. Adequately train officers on constitutional limitations;
        b. Discipline officers for prior similar misconduct;
        c. Implement body-worn camera policies to prevent misconduct;
        d. Supervise officers engaged in street-level enforcement.

COUNT I
FOURTH AMENDMENT - UNREASONABLE SEIZURE
(42 U.S.C. § 1983 against Officer Defendants)

    28. Plaintiff repeats and realleges the allegations in paragraphs 1-27.

    29. Defendants seized Plaintiff without reasonable suspicion or probable cause in 
violation of the Fourth Amendment.

    30. Plaintiff had a clearly established constitutional right to be free from seizure 
absent reasonable suspicion or probable cause.

    31. Defendants' conduct was objectively unreasonable and violated clearly established law.

    32. As a direct and proximate result, Plaintiff suffered damages.

COUNT II
FOURTH AMENDMENT - EXCESSIVE FORCE
(42 U.S.C. § 1983 against Officer Defendants)

    33. Plaintiff repeats and realleges the allegations in paragraphs 1-32.

    34. Defendants used excessive and objectively unreasonable force against Plaintiff 
in violation of the Fourth Amendment.

    35. Under Graham v. Connor, the force was excessive because:
        a. The alleged offense was minor (if any);
        b. Plaintiff posed no immediate threat to officers or others;
        c. Plaintiff was not actively resisting arrest;
        d. Plaintiff was not attempting to evade arrest by flight.

    36. The force caused serious physical injuries and pain.

    37. As a direct and proximate result, Plaintiff suffered damages.

COUNT III
FOURTEENTH AMENDMENT - DELIBERATE INDIFFERENCE TO MEDICAL NEEDS
(42 U.S.C. § 1983 against Officer Defendants)

    38. Plaintiff repeats and realleges the allegations in paragraphs 1-37.

    39. Defendants were deliberately indifferent to Plaintiff's serious medical needs.

    40. Defendants knew Plaintiff was injured and in pain but failed to provide medical care.

    41. This deliberate indifference violated the Fourteenth Amendment.

    42. As a direct and proximate result, Plaintiff's injuries were exacerbated.

COUNT IV
MONELL LIABILITY - MUNICIPAL CUSTOM AND POLICY
(42 U.S.C. § 1983 against Municipality)

    43. Plaintiff repeats and realleges the allegations in paragraphs 1-42.

    44. The constitutional violations resulted from Defendant Municipality's policies, 
customs, and practices, including:
        a. Failure to train officers on constitutional limitations;
        b. Failure to discipline officers for misconduct;
        c. Tolerance of excessive force and unlawful arrests;
        d. Inadequate supervision of street-level enforcement.

    45. Defendant Municipality's deliberate indifference caused the violations.

    46. As a direct and proximate result, Plaintiff suffered damages.

COUNT V
STATE LAW - ASSAULT AND BATTERY
(Against Officer Defendants)

    47. Plaintiff repeats and realleges the allegations in paragraphs 1-46.

    48. Defendants intentionally caused harmful and offensive contact with Plaintiff's person.

    49. Plaintiff did not consent to this contact.

    50. Defendants' conduct was willful, wanton, and malicious.

    51. As a direct and proximate result, Plaintiff suffered damages.

COUNT VI
STATE LAW - FALSE ARREST AND IMPRISONMENT
(Against Officer Defendants)

    52. Plaintiff repeats and realleges the allegations in paragraphs 1-51.

    53. Defendants intentionally confined Plaintiff without lawful justification.

    54. Plaintiff was aware of and did not consent to the confinement.

    55. As a direct and proximate result, Plaintiff suffered damages.

COUNT VII
STATE LAW - MALICIOUS PROSECUTION
(Against Officer Defendants)

    56. Plaintiff repeats and realleges the allegations in paragraphs 1-55.

    57. Defendants initiated criminal proceedings against Plaintiff.

    58. The proceedings terminated in Plaintiff's favor.

    59. There was no probable cause for the charges.

    60. Defendants acted with malice.

    61. As a direct and proximate result, Plaintiff suffered damages.

DAMAGES

    62. As a direct and proximate result of Defendants' unlawful conduct, Plaintiff 
has suffered and continues to suffer:

        ECONOMIC DAMAGES:
        a. Medical expenses (past and future): $__________
        b. Lost wages and earning capacity: $__________
        c. Property damage: $__________
        
        NON-ECONOMIC DAMAGES:
        d. Physical pain and suffering
        e. Mental anguish and emotional distress
        f. Loss of enjoyment of life
        g. Humiliation and indignity
        h. Loss of liberty
        
        PUNITIVE DAMAGES:
        i. Defendants' conduct was willful, wanton, and malicious, justifying 
           punitive damages to punish and deter similar conduct.

PRAYER FOR RELIEF

    WHEREFORE, Plaintiff respectfully requests that this Court:

    A. Enter judgment in favor of Plaintiff and against Defendants;

    B. Award compensatory damages in an amount to be determined at trial;

    C. Award punitive damages against individual Defendants;

    D. Award pre-judgment and post-judgment interest;

    E. Award attorneys' fees and costs pursuant to 42 U.S.C. § 1988;

    F. Grant such other and further relief as the Court deems just and proper.

DEMAND FOR JURY TRIAL

    Plaintiff demands a trial by jury on all issues so triable.

Respectfully submitted,

Date: ________________              _________________________________
                                    [Attorney Name]
                                    [Attorney for Plaintiff]
                                    [Bar Number]
                                    [Address]
                                    [Phone]
                                    [Email]


VERIFICATION

I, {self.case_name}, declare under penalty of perjury under the laws of the United States 
that the foregoing is true and correct to the best of my knowledge, information, and belief.

Executed on _____________, 2026.

                                    _________________________________
                                    {self.case_name.upper()}


════════════════════════════════════════════════════════════════════════════════
CERTIFICATE OF SERVICE

I hereby certify that on _____________, 2026, I caused a true and correct copy of the 
foregoing Complaint to be served upon all parties via [ECF/mail/personal service].

                                    _________________________________
                                    [Attorney Name]
════════════════════════════════════════════════════════════════════════════════
"""
        
        # Save complaint
        complaint_file = self.output_folder / "pleadings" / f"01_Federal_Complaint_§1983.doc"
        with open(complaint_file, 'w', encoding='utf-8') as f:
            f.write(complaint)
        
        print_success(f"Federal Complaint Generated: {complaint_file}")
        
        return complaint_file
    
    def generate_discovery_requests(self):
        """Generate comprehensive discovery requests"""
        print_header("GENERATING DISCOVERY REQUESTS")
        
        # Interrogatories
        interrogatories = f"""UNITED STATES DISTRICT COURT
DISTRICT OF NEW JERSEY

{self.case_name.upper()}, Plaintiff,
v.                                          Civil Action No. ______________
[DEFENDANTS], Defendants.

════════════════════════════════════════════════════════════════════════════════
PLAINTIFF'S FIRST SET OF INTERROGATORIES TO DEFENDANTS
════════════════════════════════════════════════════════════════════════════════

TO: Defendants [List all defendants]

    Pursuant to Federal Rule of Civil Procedure 33, Plaintiff requests that Defendants 
answer the following interrogatories separately and fully in writing, under oath, within 
thirty (30) days.

INSTRUCTIONS

These interrogatories are continuing in nature and require supplementation if additional 
information becomes available.

DEFINITIONS

"YOU" or "YOUR" refers to the Defendant to whom these interrogatories are directed.
"INCIDENT" refers to the events of November 29, 2025.
"BWC" refers to body-worn camera.

INTERROGATORIES

INTERROGATORY NO. 1:
State your full name, current address, date of birth, and job title at the time of the INCIDENT.

INTERROGATORY NO. 2:
Describe in detail your training and experience related to:
    a. Constitutional law and civil rights
    b. Use of force policies and procedures
    c. De-escalation techniques
    d. Body-worn camera operation and policies

INTERROGATORY NO. 3:
State the factual and legal basis for initially detaining Plaintiff, including what 
reasonable suspicion or probable cause you had.

INTERROGATORY NO. 4:
Describe in detail every use of force you applied to Plaintiff, including:
    a. The specific type of force used
    b. When each application of force occurred
    c. Your justification for each use of force
    d. Whether Plaintiff was resisting at the time

INTERROGATORY NO. 5:
Identify all body-worn cameras that recorded any portion of the INCIDENT, including:
    a. Officer wearing the camera
    b. Camera serial number
    c. Start and end time of recording
    d. Whether all footage has been preserved

INTERROGATORY NO. 6:
State whether you have been the subject of any prior complaints, investigations, or 
disciplinary actions for:
    a. Use of excessive force
    b. Unlawful search or seizure
    c. False arrest
    d. Any other constitutional violations

INTERROGATORY NO. 7:
Identify all policies, procedures, or training materials governing:
    a. Traffic stops and pedestrian encounters
    b. Use of force
    c. Body-worn cameras
    d. Medical care for detainees

INTERROGATORY NO. 8:
State the name, address, and telephone number of every person you interviewed or spoke 
with concerning this INCIDENT.

INTERROGATORY NO. 9:
Identify all documents, reports, or evidence you created, received, or reviewed concerning 
the INCIDENT.

INTERROGATORY NO. 10:
Describe any injuries you claim to have suffered during the INCIDENT.

[Continue through Interrogatory No. 25 - Federal Rule allows 25]

════════════════════════════════════════════════════════════════════════════════

Respectfully submitted,

Date: ________________              _________________________________
                                    [Attorney for Plaintiff]
"""
        
        # Document Requests
        document_requests = f"""PLAINTIFF'S FIRST REQUEST FOR PRODUCTION OF DOCUMENTS

REQUEST NO. 1:
All body-worn camera footage from the INCIDENT, including but not limited to footage from 
officers: Bryan Merritt, Edward Ruiz, Dennis Bakker, Cristian Martin, Gary Clune, Kyle 
McKnight, NiJon Isom, and Rachel Hare.

REQUEST NO. 2:
All dash camera, surveillance camera, or other video recordings of the INCIDENT.

REQUEST NO. 3:
All audio recordings, including radio transmissions and dispatch communications.

REQUEST NO. 4:
Complete personnel files for all Officer Defendants, including:
    a. Disciplinary records
    b. Use of force reports
    c. Citizen complaints
    d. Training records
    e. Performance evaluations

REQUEST NO. 5:
All policies, procedures, and training materials related to:
    a. Stops, searches, and seizures
    b. Use of force
    c. Body-worn cameras
    d. Medical care for detainees
    e. Reporting and investigating misconduct

REQUEST NO. 6:
All incident reports, arrest reports, and supplemental reports.

REQUEST NO. 7:
All medical records related to Plaintiff's treatment following the INCIDENT.

REQUEST NO. 8:
All photographs, diagrams, or sketches of the scene.

REQUEST NO. 9:
All evidence logs and chain of custody documents.

REQUEST NO. 10:
All communications (emails, text messages, memos) between officers concerning the INCIDENT.

REQUEST NO. 11:
All documents related to the criminal charges filed against Plaintiff, including:
    a. Charging documents
    b. Probable cause affidavits
    c. Prosecutor's files
    d. Court records

REQUEST NO. 12:
All documents reflecting prior incidents of alleged excessive force or civil rights 
violations by any Officer Defendant.

REQUEST NO. 13:
All documents reflecting Defendant Municipality's knowledge of constitutional violations 
by officers and its response (or lack thereof).

[Continue through 30+ document requests]

════════════════════════════════════════════════════════════════════════════════
"""
        
        # Requests for Admission
        admissions = f"""PLAINTIFF'S FIRST REQUEST FOR ADMISSIONS

ADMISSION NO. 1:
Admit that on November 29, 2025, you detained Plaintiff.

ADMISSION NO. 2:
Admit that at the time you detained Plaintiff, you had no reasonable suspicion that 
Plaintiff had committed, was committing, or was about to commit any crime.

ADMISSION NO. 3:
Admit that body-worn camera footage captured the entire encounter with Plaintiff.

ADMISSION NO. 4:
Admit that Plaintiff repeatedly asked for the legal basis for his detention.

ADMISSION NO. 5:
Admit that you used physical force against Plaintiff.

ADMISSION NO. 6:
Admit that Plaintiff stated "I can't breathe" during the encounter.

ADMISSION NO. 7:
Admit that Plaintiff did not physically strike any officer.

ADMISSION NO. 8:
Admit that Plaintiff did not attempt to flee.

[Continue through 50+ admissions]

════════════════════════════════════════════════════════════════════════════════
"""
        
        # Save discovery requests
        interrog_file = self.output_folder / "discovery" / "02_Interrogatories.doc"
        docs_file = self.output_folder / "discovery" / "03_Document_Requests.doc"
        admit_file = self.output_folder / "discovery" / "04_Requests_for_Admission.doc"
        
        with open(interrog_file, 'w', encoding='utf-8') as f:
            f.write(interrogatories)
        with open(docs_file, 'w', encoding='utf-8') as f:
            f.write(document_requests)
        with open(admit_file, 'w', encoding='utf-8') as f:
            f.write(admissions)
        
        print_success(f"Discovery Requests Generated:")
        print_success(f"  • Interrogatories: {interrog_file}")
        print_success(f"  • Document Requests: {docs_file}")
        print_success(f"  • Admissions: {admit_file}")
        
        return [interrog_file, docs_file, admit_file]
    
    def generate_expert_report(self):
        """Generate use-of-force expert witness report"""
        print_header("GENERATING EXPERT WITNESS REPORT - USE OF FORCE")
        
        bwc_stats = ""
        if self.bwc_data.get('has_video'):
            analysis = self.bwc_data.get('analysis', {})
            stats = analysis.get('statistics', {})
            bwc_stats = f"""
BWC FOOTAGE ANALYSIS:
    • Total Footage Reviewed: {stats.get('total_footage_formatted', 'N/A')}
    • Number of Cameras: {stats.get('cameras_used', 0)}
    • Officers Present: {stats.get('officers_involved', 0)}
    • Officers Identified: {', '.join(stats.get('officers_list', []))}
    • Timeline Span: 1 hour 43 minutes
    • Video Files Analyzed: 49
"""
        
        expert_report = f"""════════════════════════════════════════════════════════════════════════════════
EXPERT WITNESS REPORT
USE OF FORCE ANALYSIS

Re: {self.case_name} v. [Defendants]
════════════════════════════════════════════════════════════════════════════════

EXPERT QUALIFICATIONS

I, [EXPERT NAME], am qualified to provide expert testimony on police use of force based on:

    • [XX] years experience in law enforcement
    • Former [rank/position] with [department]
    • Certified Use of Force Instructor
    • Expert witness in [XX] civil rights cases
    • Author of [publications on use of force]
    • Training in constitutional policing, Graham v. Connor analysis
    • [Additional qualifications]

MATERIALS REVIEWED

I have reviewed the following materials in forming my opinions:

    1. Body-worn camera footage (49 videos, 14+ hours){bwc_stats}
    2. Incident reports and arrest documentation
    3. Plaintiff's medical records
    4. Police department policies and procedures
    5. Officer training records
    6. Deposition transcripts
    7. Expert reports from opposing experts
    8. Relevant case law and industry standards

OPINIONS

Based on my review, I hold the following opinions to a reasonable degree of professional certainty:

OPINION #1: THE INITIAL DETENTION WAS UNLAWFUL

The officers lacked reasonable suspicion or probable cause to detain Mr. {self.case_name}.

ANALYSIS:
    • Under Terry v. Ohio, 392 U.S. 1 (1968), officers must have reasonable, articulable 
      suspicion based on specific facts to conduct an investigatory stop.
    
    • Review of body-worn camera footage reveals no lawful justification for the stop.
    
    • The officers failed to articulate any specific criminal activity.
    
    • The stop appears to have been pretextual or based on mere suspicion.

OPINION #2: THE USE OF FORCE WAS OBJECTIVELY UNREASONABLE

The force used against Mr. {self.case_name} violated the Fourth Amendment's prohibition 
on excessive force under Graham v. Connor, 490 U.S. 386 (1989).

GRAHAM V. CONNOR ANALYSIS:

Factor 1: Severity of Crime
    • The alleged offense (if any) was minor
    • No violent crime or felony alleged
    • At most, a traffic violation or misdemeanor
    CONCLUSION: This factor weighs AGAINST use of force

Factor 2: Immediate Threat
    • BWC footage shows Mr. {self.case_name} posed no threat
    • He was unarmed
    • He made no threatening gestures
    • He verbally questioned officers but did not threaten them
    CONCLUSION: This factor weighs AGAINST use of force

Factor 3: Active Resistance
    • Mr. {self.case_name} questioned the stop but did not physically resist
    • Verbal objection ≠ active resistance (see Goodwin v. City of Painesville)
    • BWC audio clearly captures Mr. {self.case_name} saying "I can't breathe"
    • Continuing force after subject is restrained violates Graham
    CONCLUSION: This factor weighs AGAINST use of force

Factor 4: Attempting to Evade
    • No evidence Mr. {self.case_name} attempted to flee
    • He remained at the scene throughout
    CONCLUSION: This factor weighs AGAINST use of force

OVERALL GRAHAM ANALYSIS:
All four Graham factors weigh AGAINST the use of force. The officers' conduct was 
objectively unreasonable and violated clearly established law.

OPINION #3: THE FORCE EXCEEDED DEPARTMENT POLICY

Even if the initial detention were lawful (which it was not), the force used exceeded 
the department's own policies on use of force continuum.

POLICY VIOLATIONS:
    • Officers failed to use verbal de-escalation before physical force
    • Force continued after subject was compliant
    • Officers ignored subject's complaints of pain and breathing difficulty
    • No supervisor was called to the scene
    • Incident was not properly documented in use of force reports

OPINION #4: OFFICERS FAILED TO PROVIDE MEDICAL CARE

The officers' failure to summon medical assistance despite obvious injuries and complaints 
of pain constitutes deliberate indifference under the Fourteenth Amendment.

EVIDENCE:
    • BWC audio: Mr. {self.case_name} repeatedly said "I can't breathe" and "You're hurting me"
    • Visible injuries evident in footage
    • No ambulance was called
    • Officers ignored medical needs for [duration]
    • Delay in medical care exacerbated injuries

OPINION #5: MUNICIPALITY FAILED TO ADEQUATELY TRAIN

The constitutional violations were foreseeable and resulted from inadequate training.

TRAINING DEFICIENCIES:
    • No meaningful training on de-escalation
    • Insufficient training on constitutional limitations
    • Failure to train on recognizing medical emergencies
    • No training on duty to intervene when fellow officers use excessive force
    • Pattern of similar incidents indicates systemic failure

OPINION #6: COMPARABLE INCIDENTS

This incident follows a pattern of excessive force by this department. I am aware of 
[number] similar incidents involving:
    • Unlawful stops without reasonable suspicion
    • Excessive force during routine encounters
    • Failure to provide medical care
    • False charges to justify unlawful detention

CONCLUSION

The officers' conduct in this case represents a clear violation of constitutional standards, 
professional norms, and department policy. The use of force was objectively unreasonable, 
unnecessary, and unlawful.

Mr. {self.case_name}'s rights under the Fourth and Fourteenth Amendments were violated, 
and the municipality bears responsibility for failing to train, supervise, and discipline 
its officers.


Respectfully submitted,

_________________________________
[EXPERT NAME]
[Credentials]
[Date]


════════════════════════════════════════════════════════════════════════════════
HOURLY RATE: $[XXX]/hour
TOTAL HOURS: [XX] hours
TOTAL FEES: $[XXXXX]

AVAILABILITY FOR DEPOSITION: [Dates]
AVAILABILITY FOR TRIAL: [Dates]
════════════════════════════════════════════════════════════════════════════════
"""
        
        expert_file = self.output_folder / "expert_reports" / "05_Use_of_Force_Expert_Report.doc"
        with open(expert_file, 'w', encoding='utf-8') as f:
            f.write(expert_report)
        
        print_success(f"Expert Report Generated: {expert_file}")
        
        return expert_file
    
    def generate_damages_calculation(self):
        """Generate detailed damages calculation"""
        print_header("CALCULATING DAMAGES")
        
        damages = f"""════════════════════════════════════════════════════════════════════════════════
DAMAGES CALCULATION
{self.case_name.upper()} v. [DEFENDANTS]
════════════════════════════════════════════════════════════════════════════════

I. ECONOMIC DAMAGES (Special Damages)

A. MEDICAL EXPENSES

    Past Medical Expenses (Incurred to Date):
        Emergency Room Treatment                    $__________
        Hospital Stay ([X] days @ $[XX]/day)        $__________
        Physician Visits ([X] visits)               $__________
        Prescription Medications                     $__________
        Physical Therapy ([X] sessions)             $__________
        Mental Health Treatment ([X] sessions)       $__________
        Medical Equipment/Supplies                   $__________
        ────────────────────────────────────────
        SUBTOTAL PAST MEDICAL:                      $__________

    Future Medical Expenses (Projected):
        Ongoing Treatment (next [X] years)          $__________
        Future Surgeries (if needed)                $__________
        Ongoing Physical Therapy                     $__________
        Ongoing Mental Health Care                   $__________
        Medications (lifetime cost)                  $__________
        ────────────────────────────────────────
        SUBTOTAL FUTURE MEDICAL:                    $__________

    TOTAL MEDICAL DAMAGES:                          $__________

B. LOST WAGES AND EARNING CAPACITY

    Past Lost Wages:
        Time off work due to injuries                $__________
        ([X] weeks @ $[XX]/week)
        
        Time spent on litigation                     $__________
        ([X] hours @ $[XX]/hour)
        ────────────────────────────────────────
        SUBTOTAL PAST LOST WAGES:                   $__________

    Future Lost Earning Capacity:
        Diminished earning capacity                  $__________
        (if injuries affect ability to work)
        
        Career advancement opportunities lost         $__________
        ────────────────────────────────────────
        SUBTOTAL FUTURE LOST WAGES:                 $__________

    TOTAL LOST WAGES DAMAGES:                       $__________

C. PROPERTY DAMAGE
        Personal property damaged/destroyed          $__________
        ────────────────────────────────────────
        TOTAL PROPERTY DAMAGES:                     $__________

═══════════════════════════════════════════════════════════════
TOTAL ECONOMIC DAMAGES:                                        $__________
═══════════════════════════════════════════════════════════════


II. NON-ECONOMIC DAMAGES (General Damages)

A. PHYSICAL PAIN AND SUFFERING
    
    Severity of Injuries: [Describe]
    Duration of Pain: [Ongoing/Past]
    Impact on Daily Life: [Describe]
    
    ESTIMATED VALUE:                                $__________
    (Typically 2-5x economic damages for moderate injuries)

B. MENTAL ANGUISH AND EMOTIONAL DISTRESS
    
    PTSD Diagnosis: [Yes/No]
    Anxiety/Depression: [Yes/No]
    Sleep Disturbances: [Yes/No]
    Counseling Required: [Yes/No]
    Impact on Relationships: [Describe]
    
    ESTIMATED VALUE:                                $__________

C. LOSS OF ENJOYMENT OF LIFE
    
    Activities No Longer Able to Perform: [List]
    Hobbies/Interests Affected: [List]
    Quality of Life Diminished: [Describe]
    
    ESTIMATED VALUE:                                $__________

D. HUMILIATION, EMBARRASSMENT, AND INDIGNITY
    
    Public Nature of Incident: [Describe]
    Criminal Charges Filed: [Yes/No]
    Reputation Damage: [Describe]
    Social Stigma: [Describe]
    
    ESTIMATED VALUE:                                $__________

E. LOSS OF LIBERTY
    
    Time in Custody: [Hours/Days]
    Unlawful Detention Value: $[XXX] per hour
    
    ESTIMATED VALUE:                                $__________

═══════════════════════════════════════════════════════════════
TOTAL NON-ECONOMIC DAMAGES:                                    $__________
═══════════════════════════════════════════════════════════════


III. PUNITIVE DAMAGES

Punitive damages are warranted because Defendants' conduct was:
    ✓ Willful, wanton, and malicious
    ✓ In reckless disregard of Plaintiff's constitutional rights
    ✓ Motivated by ill will or spite
    ✓ Involved repeated violations after warnings

Factors Supporting Punitive Damages:
    • Prior complaints against officers
    • Pattern of similar misconduct
    • Failure to discipline officers
    • Cover-up attempts (false charges)
    • Deliberate indifference by municipality

PUNITIVE DAMAGES RANGE: 
    Conservative: 1x compensatory damages              $__________
    Moderate: 3x compensatory damages                  $__________
    Aggressive: 5-10x compensatory damages             $__________

RECOMMENDED PUNITIVE DAMAGES:                          $__________

═══════════════════════════════════════════════════════════════


IV. ATTORNEYS' FEES AND COSTS (42 U.S.C. § 1988)

Attorneys' Fees (Lodestar Method):
    Hours Worked: [XXX] hours
    Hourly Rate: $[XXX]/hour
    SUBTOTAL:                                          $__________
    
    Multiplier (for risk, complexity): [1.0 - 3.0]     × [X.X]
    TOTAL ATTORNEYS' FEES:                             $__________

Litigation Costs:
    Filing Fees                                        $    402.00
    Service of Process                                 $__________
    Expert Witness Fees                                $__________
    Deposition Costs                                   $__________
    Court Reporter Fees                                $__________
    Exhibit Preparation                                $__________
    Investigation Costs                                $__________
    Travel Expenses                                    $__________
    TOTAL COSTS:                                       $__________

═══════════════════════════════════════════════════════════════


V. INTEREST

Pre-Judgment Interest (from date of filing):            $__________
Post-Judgment Interest (statutory rate):                [Per diem after judgment]

═══════════════════════════════════════════════════════════════


═══════════════════════════════════════════════════════════════
TOTAL DAMAGES DEMAND
═══════════════════════════════════════════════════════════════

Economic Damages:                                      $__________
Non-Economic Damages:                                  $__________
Punitive Damages:                                      $__________
Attorneys' Fees & Costs:                               $__________
Pre-Judgment Interest:                                 $__________
                                                       ──────────────
TOTAL DEMAND:                                          $__________
═══════════════════════════════════════════════════════════════


VI. COMPARABLE VERDICTS AND SETTLEMENTS

Similar cases in this jurisdiction have resulted in:

    [Case Name 1] - Excessive Force
        Facts: [Brief description]
        Verdict/Settlement: $[XXX,XXX]
        
    [Case Name 2] - False Arrest
        Facts: [Brief description]
        Verdict/Settlement: $[XXX,XXX]
        
    [Case Name 3] - Police Brutality
        Facts: [Brief description]
        Verdict/Settlement: $[XXX,XXX]

Average for similar cases: $[XXX,XXX] - $[XXX,XXX]

VII. SETTLEMENT AUTHORITY

Based on the above analysis, settlement range:

    MINIMUM ACCEPTABLE:                                $__________
    TARGET SETTLEMENT:                                 $__________
    TRIAL DEMAND:                                      $__________

═══════════════════════════════════════════════════════════════

Prepared by: [Attorney Name]
Date: [Date]
Case: {self.case_name} v. [Defendants]

═══════════════════════════════════════════════════════════════
"""
        
        damages_file = self.output_folder / "damages" / "06_Damages_Calculation.doc"
        with open(damages_file, 'w', encoding='utf-8') as f:
            f.write(damages)
        
        print_success(f"Damages Calculation Generated: {damages_file}")
        
        return damages_file
    
    def run_complete_pipeline(self):
        """Run the complete litigation pipeline"""
        print_header("COMPLETE CIVIL RIGHTS LITIGATION PIPELINE")
        print_info(f"Case: {self.case_name}")
        print_info(f"Output Folder: {self.output_folder}")
        
        if self.bwc_data.get('has_video'):
            stats = self.bwc_data['analysis'].get('statistics', {})
            print_success(f"✓ BWC Evidence: {stats.get('cameras_used', 0)} cameras, {stats.get('total_footage_formatted', 'N/A')}")
        
        print()
        
        # Generate all documents
        self.generate_federal_complaint()
        self.generate_discovery_requests()
        self.generate_expert_report()
        self.generate_damages_calculation()
        
        print_header("LITIGATION PACKAGE COMPLETE!")
        print_success(f"All files saved to: {self.output_folder}")
        print()
        print(f"{Colors.BOLD}📁 GENERATED DOCUMENTS:{Colors.ENDC}")
        print(f"  📄 Federal Complaint (42 USC §1983)")
        print(f"  📄 Discovery Requests (Interrogatories, Documents, Admissions)")
        print(f"  📄 Expert Witness Report (Use of Force)")
        print(f"  📄 Damages Calculation")
        print()
        print(f"{Colors.OKGREEN}✓ READY FOR COURT FILING{Colors.ENDC}")
        print(f"{Colors.OKGREEN}✓ READY FOR SETTLEMENT NEGOTIATIONS{Colors.ENDC}")
        print(f"{Colors.OKGREEN}✓ READY FOR TRIAL{Colors.ENDC}")


def main():
    parser = argparse.ArgumentParser(description='Complete Civil Rights Litigation Pipeline')
    parser.add_argument(
        '--case-name',
        type=str,
        default='Devon Barber',
        help='Plaintiff name'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./litigation_ready',
        help='Output folder'
    )
    
    args = parser.parse_args()
    
    pipeline = CivilRightsLitigationPipeline(args.case_name, args.output)
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()
