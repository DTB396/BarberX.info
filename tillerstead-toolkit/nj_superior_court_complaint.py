#!/usr/bin/env python3
"""
NEW JERSEY SUPERIOR COURT CIVIL COMPLAINT
Generates state court complaint for civil rights case
"""

import json
from pathlib import Path
from datetime import datetime


def generate_nj_complaint():
    """Generate NJ Superior Court Civil Complaint"""
    
    complaint = """
SUPERIOR COURT OF NEW JERSEY
LAW DIVISION: [COUNTY NAME] COUNTY
DOCKET NO.: L-________________


DEVON BARBER,

                                    Plaintiff,

        v.                                              CIVIL ACTION

[MUNICIPALITY NAME],                                    COMPLAINT
[POLICE DEPARTMENT NAME],
[POLICE CHIEF NAME], individually                      JURY TRIAL DEMANDED
and in official capacity,
[OFFICER 1 NAME], individually
and in official capacity,
[OFFICER 2 NAME], individually
and in official capacity,
[OFFICER 3 NAME], individually
and in official capacity,
[OFFICER 4 NAME], individually
and in official capacity,
[OFFICER 5 NAME], individually
and in official capacity,
[OFFICER 6 NAME], individually
and in official capacity,
[OFFICER 7 NAME], individually
and in official capacity,
[OFFICER 8 NAME], individually
and in official capacity,

                                    Defendants.


================================================================================
                            COMPLAINT AND JURY DEMAND
================================================================================


    Plaintiff DEVON BARBER, by and through undersigned counsel [or pro se], 
hereby complains against Defendants and alleges as follows:


                            NATURE OF THE ACTION

    1.  This is a civil action brought pursuant to the New Jersey Civil 
Rights Act, N.J.S.A. 10:6-1 et seq., and common law, seeking damages and 
other relief for violations of Plaintiff's constitutionally protected rights 
by Defendants.

    2.  Plaintiff was subjected to an unlawful detention, excessive force, 
false arrest, assault, battery, and intentional infliction of emotional 
distress by Defendant police officers.

    3.  The incident was captured on FORTY-NINE (49) BODY-WORN CAMERAS 
from EIGHT (8) different officers, providing over FOURTEEN (14) HOURS of 
video and audio evidence documenting the constitutional violations.

    4.  Plaintiff has obtained complete transcriptions of all body-worn 
camera audio recordings, synchronized across all cameras, providing a 
comprehensive real-time record of the entire incident.


                          JURISDICTION AND VENUE

    5.  This Court has subject matter jurisdiction pursuant to N.J. Const. 
art. VI, § 3, ¶ 2, and N.J.S.A. 2A:15-5.13.

    6.  Venue is proper in this County pursuant to R. 4:3-2(a)(1) as the 
events giving rise to this action occurred in this County.


                                PARTIES

    7.  Plaintiff DEVON BARBER is an individual residing in the State of 
New Jersey.

    8.  Defendant [MUNICIPALITY NAME] is a municipal corporation organized 
under the laws of the State of New Jersey.

    9.  Defendant [POLICE DEPARTMENT NAME] is a law enforcement agency 
within Defendant Municipality.

    10. Defendant [POLICE CHIEF NAME] is the Chief of Police of Defendant 
Police Department and is sued in both individual and official capacities.

    11. Defendants [OFFICER 1-8 NAMES] are police officers employed by 
Defendant Police Department and are sued in both their individual and 
official capacities.


                          FACTS COMMON TO ALL COUNTS

Date and Location of Incident

    12. On or about [DATE OF INCIDENT], at approximately [TIME], Plaintiff 
was [LOCATION DESCRIPTION].

    13. Defendant police officers [OFFICER NAMES] responded to the scene.

Body-Worn Camera Evidence

    14. All Defendant officers were equipped with body-worn cameras 
pursuant to New Jersey Attorney General Law Enforcement Directive No. 
2015-1 (Revised 2021).

    15. The incident was recorded by EIGHT (8) officers' body-worn cameras, 
resulting in FORTY-NINE (49) separate video files.

    16. The total duration of body-worn camera footage exceeds FOURTEEN 
(14) HOURS AND FOUR (4) MINUTES.

    17. Plaintiff has obtained and analyzed all body-worn camera footage, 
including:

        a. Officer Bryan Merritt: 9 camera files
        b. Officer Edward Ruiz: 8 camera files
        c. Officer NiJon Isom: 11 camera files
        d. Officer Gary Clune: 6 camera files
        e. Officer Cristian Martin: 5 camera files
        f. Officer Dennis Bakker: 4 camera files
        g. Officer Kyle McKnight: 4 camera files
        h. Officer Rachel Hare: 2 camera files

    18. Plaintiff has caused all body-worn camera audio to be transcribed 
using professional-grade automated speech recognition technology (Faster-
Whisper/OpenAI Whisper).

    19. The transcriptions have been synchronized across all cameras into 
a unified, real-time transcript showing the chronological sequence of all 
audible statements made during the incident.

    20. The synchronized transcript is attached hereto as Exhibit A and 
incorporated by reference.

    21. The certified transcript provides a comprehensive, minute-by-minute 
account of the incident from multiple simultaneous perspectives.

Initial Encounter

    22. At the time of the encounter, Plaintiff was [DESCRIPTION OF 
PLAINTIFF'S ACTIVITY].

    23. Plaintiff was not engaged in any criminal activity.

    24. Plaintiff had committed no violation of law warranting detention 
or arrest.

    25. Defendant officers approached Plaintiff without reasonable suspicion 
or probable cause.

Unlawful Detention and Excessive Force

    26. Without legal justification, Defendant officers detained Plaintiff.

    27. Plaintiff repeatedly asked why he was being detained.

    28. Defendant officers failed to provide any lawful basis for the 
detention.

    29. Plaintiff complied with all lawful orders.

    30. Despite Plaintiff's compliance, Defendant officers used physical 
force against Plaintiff.

    31. The force used was objectively unreasonable under the circumstances.

    32. As captured on body-worn camera audio and transcribed, Plaintiff 
repeatedly stated: "I CAN'T BREATHE" and "STOP!"

    33. Despite Plaintiff's clear statements of distress, Defendant officers 
continued to apply force.

    34. The body-worn camera transcripts document Plaintiff's repeated pleas 
for the officers to stop.

    35. Multiple officers applied force simultaneously.

    36. The force resulted in physical injury to Plaintiff.

Medical Needs

    37. Plaintiff requested medical attention.

    38. Defendant officers were deliberately indifferent to Plaintiff's 
obvious medical needs.

    39. Plaintiff was denied prompt medical care.

Unlawful Arrest and False Charges

    40. Plaintiff was arrested without probable cause.

    41. Charges were filed against Plaintiff that were false, fabricated, 
or otherwise unsupported by evidence.

    42. The charges were subsequently [DISMISSED/WITHDRAWN/ACQUITTED].

Municipal Liability

    43. The constitutional violations alleged herein were the direct and 
proximate result of Defendant Municipality's policies, practices, and 
customs.

    44. Defendant Municipality failed to adequately train police officers 
regarding:
        a. Constitutional limits on use of force
        b. Duty to intervene when fellow officers use excessive force
        c. Recognition of and response to medical emergencies
        d. Proper procedures for investigatory stops and arrests
        e. Body-worn camera policies and evidence preservation

    45. Defendant Municipality knew or should have known that its failure 
to train would likely result in constitutional violations.

    46. Defendant Police Chief, as the final policymaker, ratified and 
approved the officers' unconstitutional conduct.

Damages

    47. As a direct and proximate result of Defendants' actions, Plaintiff 
suffered:
        a. Physical injuries including [DESCRIPTION]
        b. Pain and suffering
        c. Emotional distress, humiliation, and mental anguish
        d. Loss of liberty
        e. Violation of constitutional rights
        f. Medical expenses
        g. Lost wages and earning capacity
        h. Damage to reputation
        i. Attorney's fees and costs of litigation


                                  COUNT ONE
                    NEW JERSEY CIVIL RIGHTS ACT (N.J.S.A. 10:6-2)
                              (Against All Defendants)

    48. Plaintiff repeats and realleges paragraphs 1-47 as if fully set 
forth herein.

    49. Plaintiff's detention, arrest, and subjection to excessive force 
violated rights secured by the New Jersey Constitution, including:
        a. Article I, Paragraph 1 (Natural and Unalienable Rights)
        b. Article I, Paragraph 7 (Right Against Unreasonable Searches and 
           Seizures)
        c. Article I, Paragraph 12 (Right to Remedy for Injuries)

    50. Defendants' conduct constituted a violation of the New Jersey Civil 
Rights Act, N.J.S.A. 10:6-2(c).

    51. Defendants acted under color of state law.

    52. Defendants' actions were intentional, reckless, and/or with actual 
malice.

    53. As a direct and proximate result, Plaintiff suffered damages as 
alleged above.

    WHEREFORE, Plaintiff demands judgment against Defendants, jointly and 
severally, for compensatory damages, punitive damages, attorney's fees and 
costs, and such other relief as the Court deems just and proper.


                                  COUNT TWO
                                FALSE ARREST
                              (Against All Defendants)

    54. Plaintiff repeats and realleges paragraphs 1-53 as if fully set 
forth herein.

    55. Defendant officers arrested Plaintiff without probable cause.

    56. Plaintiff was detained against his will.

    57. The arrest was unlawful.

    58. As a direct and proximate result, Plaintiff suffered damages as 
alleged above.

    WHEREFORE, Plaintiff demands judgment against Defendants, jointly and 
severally, for compensatory and punitive damages, and such other relief as 
the Court deems just and proper.


                                 COUNT THREE
                             ASSAULT AND BATTERY
                              (Against All Defendants)

    59. Plaintiff repeats and realleges paragraphs 1-58 as if fully set 
forth herein.

    60. Defendant officers intentionally applied unlawful force to 
Plaintiff's person.

    61. Plaintiff did not consent to the application of such force.

    62. The force caused harmful and offensive contact with Plaintiff.

    63. Plaintiff suffered physical injuries as a result.

    64. As a direct and proximate result, Plaintiff suffered damages as 
alleged above.

    WHEREFORE, Plaintiff demands judgment against Defendants, jointly and 
severally, for compensatory and punitive damages, and such other relief as 
the Court deems just and proper.


                                  COUNT FOUR
                    INTENTIONAL INFLICTION OF EMOTIONAL DISTRESS
                              (Against All Defendants)

    65. Plaintiff repeats and realleges paragraphs 1-64 as if fully set 
forth herein.

    66. Defendants' conduct was extreme and outrageous.

    67. Defendants acted intentionally or recklessly.

    68. Defendants' conduct caused severe emotional distress to Plaintiff.

    69. The emotional distress was severe and of a nature that no reasonable 
person could be expected to endure it.

    70. As a direct and proximate result, Plaintiff suffered damages as 
alleged above.

    WHEREFORE, Plaintiff demands judgment against Defendants, jointly and 
severally, for compensatory and punitive damages, and such other relief as 
the Court deems just and proper.


                                  COUNT FIVE
                              MALICIOUS PROSECUTION
                              (Against All Defendants)

    71. Plaintiff repeats and realleges paragraphs 1-70 as if fully set 
forth herein.

    72. Defendant officers instituted criminal proceedings against Plaintiff.

    73. The criminal proceedings terminated in Plaintiff's favor by 
[DISMISSAL/ACQUITTAL/WITHDRAWAL OF CHARGES].

    74. The proceedings were instituted without probable cause.

    75. The proceedings were instituted with malice.

    76. As a direct and proximate result, Plaintiff suffered damages as 
alleged above.

    WHEREFORE, Plaintiff demands judgment against Defendants, jointly and 
severally, for compensatory and punitive damages, and such other relief as 
the Court deems just and proper.


                                   COUNT SIX
                        NEGLIGENT HIRING, TRAINING, AND SUPERVISION
                          (Against Municipal Defendants)

    77. Plaintiff repeats and realleges paragraphs 1-76 as if fully set 
forth herein.

    78. Defendant Municipality and Police Chief owed a duty of care to 
Plaintiff to properly hire, train, and supervise police officers.

    79. Defendant Municipality and Police Chief breached that duty by:
        a. Failing to adequately screen officer candidates
        b. Failing to provide adequate training on use of force
        c. Failing to discipline officers for prior misconduct
        d. Failing to supervise officers to ensure constitutional policing

    80. The breach of duty was a proximate cause of Plaintiff's injuries.

    81. As a direct and proximate result, Plaintiff suffered damages as 
alleged above.

    WHEREFORE, Plaintiff demands judgment against Defendant Municipality 
and Police Chief for compensatory damages and such other relief as the 
Court deems just and proper.


                               PRAYER FOR RELIEF

    WHEREFORE, Plaintiff respectfully requests that this Court enter 
judgment in favor of Plaintiff and against Defendants, jointly and 
severally, as follows:

    A.  Compensatory damages in an amount to be determined at trial;

    B.  Punitive damages in an amount to be determined at trial;

    C.  Attorney's fees and costs of suit pursuant to N.J.S.A. 10:6-2(f);

    D.  Pre-judgment and post-judgment interest as allowed by law;

    E.  Such other and further relief as this Court deems just and proper.


                              DEMAND FOR JURY TRIAL

    Plaintiff demands a trial by jury on all issues so triable.


Dated: ______________________


                                        Respectfully submitted,


                                        _________________________________
                                        [ATTORNEY NAME / PRO SE PLAINTIFF]
                                        [ADDRESS]
                                        [PHONE]
                                        [EMAIL]


                            CERTIFICATION PURSUANT TO R. 4:5-1

    I certify that the matter in controversy is not the subject of any 
other action pending in any court or of any pending arbitration proceeding, 
nor is any other action or arbitration proceeding contemplated. I also 
certify that there are no other parties who should be joined in this action 
at this time.


                                        _________________________________
                                        [ATTORNEY NAME / PRO SE PLAINTIFF]


                        DESIGNATION OF TRIAL COUNSEL

    Pursuant to R. 4:25-4, the undersigned is hereby designated as trial 
counsel.


                                        _________________________________
                                        [ATTORNEY NAME / PRO SE PLAINTIFF]
    """
    
    return complaint


def main():
    complaint = generate_nj_complaint()
    
    # Save to file
    output_file = Path("litigation_ready") / "pleadings" / "01_NJ_Superior_Court_Complaint.doc"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(complaint)
    
    print("=" * 80)
    print("NEW JERSEY SUPERIOR COURT COMPLAINT GENERATED")
    print("=" * 80)
    print(f"\nFile saved: {output_file}")
    print("\nREADY FOR NEW JERSEY SUPERIOR COURT FILING")
    print("\nThis complaint includes:")
    print("  - Reference to 49 body-worn camera videos")
    print("  - Reference to complete transcriptions")
    print("  - 6 counts (Civil Rights Act, False Arrest, Assault/Battery,")
    print("    IIED, Malicious Prosecution, Negligent Training)")
    print("  - Jury trial demand")
    print("  - Attorney's fees provision")
    print("\nFILL IN THE BRACKETS WITH YOUR SPECIFIC INFORMATION:")
    print("  [COUNTY NAME], [MUNICIPALITY NAME], [POLICE DEPARTMENT NAME],")
    print("  [POLICE CHIEF NAME], [OFFICER 1-8 NAMES], [DATE OF INCIDENT],")
    print("  [TIME], [LOCATION DESCRIPTION], etc.")


if __name__ == "__main__":
    main()
