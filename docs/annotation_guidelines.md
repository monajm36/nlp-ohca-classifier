# OHCA Annotation Guidelines

## Overview

This document provides comprehensive guidelines for manually annotating discharge notes to identify Out-of-Hospital Cardiac Arrest (OHCA) cases.

## Definition of OHCA

**Out-of-Hospital Cardiac Arrest (OHCA)** is a cardiac arrest that occurs:
1. **Outside** a healthcare facility (home, workplace, public spaces, etc.)
2. As the **primary reason** for the current hospital admission
3. Requiring emergency medical intervention (CPR, defibrillation, etc.)

## Annotation Labels

- **1 = OHCA**: Clear case of out-of-hospital cardiac arrest
- **0 = Non-OHCA**: Everything else (in-hospital arrest, non-arrest conditions, etc.)

## Include as OHCA (Label = 1)

### Clear OHCA Cases
- "Found down at home, CPR initiated by family"
- "Cardiac arrest at work, bystander CPR given"
- "Collapsed in restaurant, EMS resuscitation"
- "Arrest in parking lot, ROSC achieved in field"
- "Witnessed collapse at gym, immediate CPR"

### Key Phrases Indicating OHCA
- "Found down at [location outside hospital]"
- "Cardiac arrest at home/work/public"
- "Bystander CPR"
- "EMS resuscitation"
- "Field resuscitation"
- "Collapse witnessed by [family/bystander]"
- "ROSC in field/ambulance"

## Exclude as Non-OHCA (Label = 0)

### In-Hospital Cardiac Arrests
- "Code blue called on ward"
- "Arrest during surgery"
- "Arrest in ICU"
- "Arrest during procedure"

### Historical/Previous Arrests
- "History of cardiac arrest 2 years ago"
- "Prior arrest in 2020"
- "Patient previously had arrest"

### Non-Arrest Conditions
- "Chest pain without arrest"
- "Heart attack (MI) without arrest"
- "Shortness of breath"
- "Syncope/fainting"
- "Near-drowning without arrest"

### Trauma-Related Arrests
- "Arrest secondary to car accident"
- "Traumatic arrest from gunshot"
- "Arrest due to overdose"

### Uncertain Location
- "Transfer from outside hospital for arrest"
- "Arrest of unknown location"

## Decision Tree

```
1. Did a cardiac arrest occur?
   └── NO → Label = 0
   └── YES → Continue to 2

2. Did the arrest happen OUTSIDE a healthcare facility?
   └── NO (hospital, clinic, etc.) → Label = 0
   └── YES → Continue to 3

3. Is OHCA the PRIMARY reason for this admission?
   └── NO (admitted for something else) → Label = 0
   └── YES → Label = 1
```

## Confidence Scale

Rate your confidence in the annotation (1-5 scale):

- **5 = Very Confident**: Clear, unambiguous case
- **4 = Confident**: Strong evidence, minor uncertainty
- **3 = Moderately Confident**: Some ambiguity but leaning toward decision
- **2 = Uncertain**: Significant ambiguity, difficult case
- **1 = Very Uncertain**: Unclear, may need expert review

## Difficult Cases

### Transfers from Other Hospitals
- **Include** if specifically transferred FOR OHCA treatment
- **Exclude** if transferred for other reasons, even if arrest mentioned

### Multiple Conditions
- Focus on the **primary reason** for admission
- If arrest is secondary to another condition, may be **excluded**

### Unclear Timeline
- If timing is unclear, err on the side of **excluding**
- Note uncertainty in comments

### Incomplete Information
- Base decision on available information
- Use lower confidence score
- Add notes about missing information

## Examples with Rationale

### Example 1: Clear OHCA (Label = 1)
**Text**: "Chief complaint: Cardiac arrest. Patient found down at home by spouse, immediate CPR initiated, EMS arrived and achieved ROSC."

**Rationale**: 
- ✅ Cardiac arrest occurred
- ✅ Outside hospital (at home)
- ✅ Primary reason for admission
- **Label**: 1, **Confidence**: 5

### Example 2: In-Hospital Arrest (Label = 0)
**Text**: "Patient admitted for pneumonia, developed cardiac arrest on day 3 of hospitalization."

**Rationale**:
- ✅ Cardiac arrest occurred
- ❌ Inside hospital
- **Label**: 0, **Confidence**: 5

### Example 3: Non-Arrest Condition (Label = 0)
**Text**: "Chief complaint: Chest pain. Patient presents with acute MI, underwent emergency PCI."

**Rationale**:
- ❌ No cardiac arrest occurred
- **Label**: 0, **Confidence**: 5

### Example 4: Historical Arrest (Label = 0)
**Text**: "Patient with history of cardiac arrest 1 year ago, now presents with chest pain."

**Rationale**:
- ❌ Current admission not for OHCA
- ❌ Historical arrest, not current
- **Label**: 0, **Confidence**: 4

### Example 5: Transfer Case - Include (Label = 1)
**Text**: "Transfer from community hospital. Patient had cardiac arrest at home, CPR by family, transferred for further care."

**Rationale**:
- ✅ Original arrest was out-of-hospital
- ✅ Primary reason for care
- **Label**: 1, **Confidence**: 4

### Example 6: Transfer Case - Exclude (Label = 0)
**Text**: "Transfer for cardiac catheterization. Patient had arrest during procedure at outside hospital."

**Rationale**:
- ❌ Arrest occurred in healthcare facility
- **Label**: 0, **Confidence**: 4

## Quality Control

### Before Submitting
1. **Double-check** each decision against the criteria
2. **Review** cases with confidence < 3
3. **Add notes** for any unusual or borderline cases
4. **Ensure consistency** in similar cases

### Notes Field
Use the notes field to document:
- Reasoning for difficult decisions
- Key phrases that influenced decision
- Uncertainties or missing information
- Questions for review

## Common Mistakes to Avoid

1. **Don't** include in-hospital arrests
2. **Don't** include historical arrests from previous admissions
3. **Don't** include trauma-related arrests unless clearly stated as cardiac
4. **Don't** include conditions that might lead to arrest but where no arrest occurred
5. **Don't** guess when information is unclear - use appropriate confidence scores

## Support

If you encounter cases that don't fit these guidelines:
1. Make your best judgment
2. Use a lower confidence score
3. Document your reasoning in notes
4. Flag for expert review if needed

Remember: Consistency is key for model training. When in doubt, err on the side of excluding (Label = 0) and document your uncertainty.
