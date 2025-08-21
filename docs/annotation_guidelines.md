# OHCA Annotation Guidelines

## Overview

This document provides comprehensive guidelines for manually annotating discharge notes to identify Out-of-Hospital Cardiac Arrest (OHCA) cases.

## Definition of OHCA

**Out-of-Hospital Cardiac Arrest (OHCA)** is a cardiac arrest that occurs **outside** a healthcare facility (home, workplace, public spaces, etc.).

## Annotation Labels

- **1 = OHCA**: Cardiac arrest that occurred outside a healthcare facility
- **0 = Non-OHCA**: Everything else

## Include as OHCA (Label = 1)

- Cardiac arrest at home
- Cardiac arrest at work
- Cardiac arrest in public spaces
- Cardiac arrest witnessed by bystanders
- Cardiac arrest with CPR given outside hospital

## Exclude as Non-OHCA (Label = 0)

### In-Hospital Cardiac Arrests
- Cardiac arrest occurring within any healthcare facility
- Code blue called on hospital ward
- Arrest during surgery or medical procedures
- Arrest in ICU, emergency department, or any hospital unit

### Transfer Cases
- **All transfer cases are excluded**, including:
- Patients transferred from outside hospitals for OHCA
- Patients transferred for cardiac catheterization after OHCA

### Historical Cardiac Arrests
- Patients with history of cardiac arrest who did not present with current cardiac arrest

### Non-Arrest Conditions
- Chest pain without cardiac arrest
- Heart attack (MI) without cardiac arrest
- Shortness of breath
- Syncope/fainting without arrest
- Any condition where cardiac arrest did not occur

## Decision Tree

```
1. Did a cardiac arrest occur during this admission?
   └── NO → Label = 0
   └── YES → Continue to 2

2. Did the arrest happen OUTSIDE a healthcare facility?
   └── NO → Label = 0
   └── YES → Continue to 3

3. Is this a transfer case?
   └── YES → Label = 0
   └── NO → Label = 1
```

## Confidence Scale

Rate your confidence in the annotation (1-5 scale):

- **5 = Very Confident**: Clear, unambiguous case
- **4 = Confident**: Strong evidence, minor uncertainty
- **3 = Moderately Confident**: Some ambiguity but leaning toward decision
- **2 = Uncertain**: Significant ambiguity, difficult case
- **1 = Very Uncertain**: Unclear, may need expert review

## Examples with Rationale

### Example 1: OHCA at Home (Label = 1)
**Text**: "Patient found down at home by spouse, immediate CPR initiated, EMS arrived and achieved ROSC."

**Rationale**: 
- ✅ Cardiac arrest occurred
- ✅ Outside hospital (at home)
- ✅ Not a transfer case
- **Label**: 1, **Confidence**: 5

### Example 2: In-Hospital Arrest (Label = 0)
**Text**: "Patient admitted for pneumonia, developed cardiac arrest on day 3 of hospitalization."

**Rationale**:
- ✅ Cardiac arrest occurred
- ❌ Inside hospital
- **Label**: 0, **Confidence**: 5

### Example 3: Transfer for OHCA (Label = 0)
**Text**: "Transfer from community hospital. Patient had cardiac arrest at home, CPR by family, transferred for further care."

**Rationale**:
- ✅ Cardiac arrest occurred outside hospital originally
- ❌ This is a transfer case
- **Label**: 0, **Confidence**: 5

### Example 4: Historical Arrest (Label = 0)
**Text**: "Patient with history of cardiac arrest 1 year ago, now presents with chest pain."

**Rationale**:
- ❌ No current cardiac arrest
- ❌ Historical arrest, not current admission
- **Label**: 0, **Confidence**: 5

### Example 5: Non-Arrest Condition (Label = 0)
**Text**: "Chief complaint: Chest pain. Patient presents with acute MI, underwent emergency PCI."

**Rationale**:
- ❌ No cardiac arrest occurred
- **Label**: 0, **Confidence**: 5

### Example 6: Workplace Cardiac Arrest (Label = 1)
**Text**: "Patient collapsed at work, coworkers initiated CPR, EMS transported to hospital."

**Rationale**:
- ✅ Cardiac arrest occurred
- ✅ Outside hospital (at work)
- ✅ Not a transfer case
- **Label**: 1, **Confidence**: 5

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
2. **Don't** include any transfer cases (even if transferred for OHCA)
3. **Don't** include historical arrests from previous admissions
4. **Don't** include conditions where no cardiac arrest occurred

## Difficult Cases

### Transfer Cases
- **All transfer cases are excluded (Label = 0)**
- This includes transfers specifically for OHCA management
- Focus on whether this is a transfer, not the original location of arrest

### Multiple Conditions
- Focus on whether cardiac arrest actually occurred
- If no arrest mentioned, label as 0

### Incomplete Information
- Base decision on available information
- Use lower confidence score
- Add notes about missing information

## Support

If you encounter cases that don't fit these guidelines:
1. Make your best judgment
2. Use a lower confidence score
3. Document your reasoning in notes
4. Flag for expert review if needed

Remember: Consistency is key for model training. When in doubt, err on the side of excluding (Label = 0) and document your uncertainty.

## Key Principles

1. **OHCA must occur outside healthcare facilities**
2. **All transfer cases are excluded**
3. **Only current cardiac arrests, not historical ones**
4. **No cardiac arrest = automatic exclusion**
5. **When uncertain, exclude and note reasons**
