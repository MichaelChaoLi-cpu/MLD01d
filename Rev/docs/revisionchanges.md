# Manuscript Revision Changes

Schema: `kila-revision-changes/v1`

## reviewer-1/comment-3

### part-01

- Location: Methods > Variables, paragraph beginning 'The core explanatory variable'
- Reason: Define multi-hazard exposure explicitly as a distinct-hazard-type count and state what the count does not measure.
- Kila decisions: KILA-D-20260826-001
- Mode: `replace`
- Timestamp: 2026-08-26T07:16:01Z
- Author: Kila
- Markup SHA-256 before: `1c65af3ec15ed427d4018fd1209be5c006fc0d6171d1ddef05bbabd45e676d58`
- Markup SHA-256 after: `bf8d1b94cd26c04dfdf5a69505144ffcb7776d7b670b9c6234f9205d3ec8deb6`
- Revision IDs: `1`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260826T161601809601.reviewer-1-comment-3.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
The core explanatory variable for environmental pressure is the number of natural disaster types experienced. This is a derived count variable that summarizes multi-hazard exposure.
~~~~

- After:

~~~~text
The core explanatory variable for environmental pressure is the number of natural disaster types experienced. This is a derived count variable that summarizes multi-hazard exposure. Specifically, the count is the number of distinct disaster types reported by a household (observed range, 0-15 across 19 survey categories), rather than a measure of disaster frequency, intensity, timing, or co-occurrence.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " Specifically, the count is the number of distinct disaster types reported by a household (observed range, 0-15 across 19 survey categories), rather than a measure of disaster frequency, intensity, timing, or co-occurrence."

### metadata-correction-01

- Location: Whole markup document > tracked-change metadata
- Reason: Human requested that the tracked-change author be anonymous rather than Kila.
- Kila decisions: none (non-substantive metadata correction)
- Mode: `metadata-only`
- Timestamp: 2026-08-26T07:46:57Z
- Author: anonymous
- Markup SHA-256 before: `bf8d1b94cd26c04dfdf5a69505144ffcb7776d7b670b9c6234f9205d3ec8deb6`
- Markup SHA-256 after: `aaf96b72cff99baba0392119906194640e9f28230e2b307b7c74238e56ffff18`
- Revision IDs: `1`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260826T164411753146.tracked-author.Kila-to-anonymous.docx`
- Tracked author before: `Kila`
- Tracked author after: `anonymous`
- Author-only XML verification: `true`
- Manuscript content unchanged: `true`
- Paragraph properties preserved: `true`
- Run content and styles preserved: `true`

## reviewer-2/comment-3

### part-01

- Location: Methods > Variables, paragraph beginning 'The primary dependent variable'
- Reason: Define the self-reported H07 outcome, its 25-year reference period, household respondent scope, and binary coding.
- Kila decisions: KILA-D-20260826-001
- Mode: `replace`
- Timestamp: 2026-08-26T08:35:30Z
- Author: anonymous
- Markup SHA-256 before: `aaf96b72cff99baba0392119906194640e9f28230e2b307b7c74238e56ffff18`
- Markup SHA-256 after: `280ec7da64f434abcc433e1c5e7168fab6051020ee47a28b480b5541f2ef3c5c`
- Revision IDs: `2, 3, 4, 5, 6, 7, 8, 9, 10`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260826T173530586216.reviewer-2-comment-3.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
The primary dependent variable is the Increase in Household Disease Incidence Dummy. This variable is a binary indicator from the pooled NCCIS dataset.
~~~~

- After:

~~~~text
The primary dependent variable is a respondent-reported binary indicator derived from the NCCIS H07 item. In 2016, H07 asks whether the incidence of illness due to any disease increased in the respondent's family over the previous 25 years; the 2022 item asks whether the respondent or household members experienced a higher incidence of disease than 25 years earlier. We code yes as 1 and no as 0.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "the"
     - After: "a"
  2. `replace`
     - Before: "Increase in Household Disease Incidence Dummy. This variable is a"
     - After: "respondent-reported"
  3. `insert`
     - Before: ""
     - After: "derived "
  4. `delete`
     - Before: "pooled "
     - After: ""
  5. `replace`
     - Before: "dataset"
     - After: "H07 item"
  6. `insert`
     - Before: ""
     - After: " In 2016, H07 asks whether the incidence of illness due to any disease increased in the respondent's family over the previous 25 years; the 2022 item asks whether the respondent or household members experienced a higher incidence of disease than 25 years earlier. We code yes as 1 and no as 0."

### part-02

- Location: Discussion > Limitations, paragraph beginning 'Several limitations should be acknowledged'
- Reason: Explain the 25-year recall, household proxy-reporting, reporting heterogeneity, and interpretation limits of the subjective outcome.
- Kila decisions: KILA-D-20260826-001
- Mode: `replace`
- Timestamp: 2026-08-26T08:43:43Z
- Author: anonymous
- Markup SHA-256 before: `280ec7da64f434abcc433e1c5e7168fab6051020ee47a28b480b5541f2ef3c5c`
- Markup SHA-256 after: `84d39980b7e2545fb7166d7bf894ecffe26136075d9a8da8b5fc0b631c44d613`
- Revision IDs: `11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260826T174344029289.reviewer-2-comment-3.part-02.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Second, both the outcome and key predictors are self-reported, introducing potential recall bias and common-method variance.
~~~~

- After:

~~~~text
Second, the outcome asks one respondent to compare family illness with conditions 25 years earlier, while the exposure and climate-knowledge measures are also self-reported. The outcome therefore captures perceived change rather than clinically verified incidence and may vary with memory, respondent age or education, proxy reporting for other household members, and interpretation of the reference period. Correlated reporting tendencies may also create common-method bias and influence the magnitude or direction of observed associations; consequently, the findings should not be interpreted as estimates of objective disease incidence.
~~~~

- Minimal tracked fragments:
  1. `delete`
     - Before: " both"
     - After: ""
  2. `insert`
     - Before: ""
     - After: "asks one respondent to compare family illness with conditions 25 years earlier, while the exposure "
  3. `replace`
     - Before: "key"
     - After: "climate-knowledge"
  4. `replace`
     - Before: "predictors"
     - After: "measures"
  5. `insert`
     - Before: ""
     - After: "also "
  6. `insert`
     - Before: ""
     - After: ". The outcome therefore captures perceived change rather than clinically verified incidence and may vary with memory"
  7. `replace`
     - Before: "introducing"
     - After: "respondent"
  8. `replace`
     - Before: "potential"
     - After: "age"
  9. `replace`
     - Before: "recall"
     - After: "or education, proxy reporting for other household members, and interpretation of the reference period. Correlated reporting tendencies may also create common-method"
  10. `replace`
     - Before: "common-method"
     - After: "influence"
  11. `replace`
     - Before: "variance"
     - After: "the magnitude or direction of observed associations; consequently, the findings should not be interpreted as estimates of objective disease incidence"

## reviewer-3/comment-3

### part-01

- Location: Methods > Variables, climate-change knowledge paragraph
- Reason: Disclose that the outcome, multi-hazard exposure, and climate-change knowledge measures share the same respondent and interview, as requested by Reviewer 3.
- Kila decisions: KILA-D-20260826-001
- Mode: `replace`
- Timestamp: 2026-08-26T12:54:43Z
- Author: anonymous
- Markup SHA-256 before: `b03b67b314409da68368985e4245bf6e53d71c3d1d807851bd7f6b51e620d532`
- Markup SHA-256 after: `73ba7329bc4e65d2dc4554a3daeb7dd05ed2a189525bcfe5711df904d3d918a4`
- Revision IDs: `29`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260826T215443976422.reviewer-3-comment-3.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
In the analytical framework, this indicator is treated as a potential moderator. It may mitigate the adverse health effects of disaster exposure, as indicated by the literature (Adom et al., 2025; Hossain, 2025; Liu et al., 2026).
~~~~

- After:

~~~~text
In the analytical framework, this indicator is treated as a potential moderator. It may mitigate the adverse health effects of disaster exposure, as indicated by the literature (Adom et al., 2025; Hossain, 2025; Liu et al., 2026). The survey measures used for the outcome, multi-hazard exposure, and climate-change knowledge are all reported by the same respondent in the same interview.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " The survey measures used for the outcome, multi-hazard exposure, and climate-change knowledge are all reported by the same respondent in the same interview."

### part-02

- Location: Discussion > limitations paragraph, boundary before the third limitation
- Reason: State reverse causation explicitly alongside the existing common-method-bias caveat, as requested by Reviewer 3.
- Kila decisions: KILA-D-20260826-001
- Mode: `replace`
- Timestamp: 2026-08-26T13:23:25Z
- Author: anonymous
- Markup SHA-256 before: `73ba7329bc4e65d2dc4554a3daeb7dd05ed2a189525bcfe5711df904d3d918a4`
- Markup SHA-256 after: `e6b5ce790cf3a3195fabfeb97d20a589e63b8bde903e82c25c69425dfa61d269`
- Revision IDs: `30`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260826T222325301416.reviewer-3-comment-3.part-02.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Third, the binary climate knowledge measure captures awareness but not the depth or behavioural translation of that information.
~~~~

- After:

~~~~text
Reverse causation also cannot be excluded because perceived household health deterioration may influence subsequent climate-change awareness or the reporting of past disaster exposure. Third, the binary climate knowledge measure captures awareness but not the depth or behavioural translation of that information.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: "Reverse causation also cannot be excluded because perceived household health deterioration may influence subsequent climate-change awareness or the reporting of past disaster exposure. "

## reviewer-2/comment-7

### part-01

- Location: Methods > Variables, paragraph beginning 'The model incorporates a comprehensive set'
- Reason: Clarify the existing housing and health-care-access proxies without changing the approved 64-feature model specification.
- Kila decisions: KILA-D-20260827-001, KILA-D-20260827-002
- Mode: `replace`
- Timestamp: 2026-08-27T01:39:14Z
- Author: anonymous
- Markup SHA-256 before: `e6b5ce790cf3a3195fabfeb97d20a589e63b8bde903e82c25c69425dfa61d269`
- Markup SHA-256 after: `6356d359fae0584c3ce90a5e78585c858e129cecc6104379e58825d468f848c5`
- Revision IDs: `31, 32, 33, 34, 35, 36`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260827T103915035135.reviewer-2-comment-7.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Indicators of economic status covering residence type, asset ownership, agricultural land, access to communication and transportation assets, and distance to services.
~~~~

- After:

~~~~text
Indicators of economic status cover residence ownership and type, asset ownership, agricultural land, access to communication and transportation assets, and distances to services, including the nearest health center.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "covering"
     - After: "cover"
  2. `insert`
     - Before: ""
     - After: " ownership and"
  3. `replace`
     - Before: "distance"
     - After: "distances"
  4. `insert`
     - Before: ""
     - After: ", including the nearest health center"

### part-02

- Location: Discussion > limitations paragraph, immediately before the fourth limitation
- Reason: Identify the reviewer-raised unmeasured domains and explain the residual-confounding boundary without changing the approved 64-feature model or numerical results.
- Kila decisions: KILA-D-20260827-001, KILA-D-20260827-002
- Mode: `replace`
- Timestamp: 2026-08-27T01:50:27Z
- Author: anonymous
- Markup SHA-256 before: `6356d359fae0584c3ce90a5e78585c858e129cecc6104379e58825d468f848c5`
- Markup SHA-256 after: `ea7eb3a592923f68293d14eccb0ae591ae733ddab5f55e57ccd3f2426ee943bf`
- Revision IDs: `37`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260827T105027253784.reviewer-2-comment-7.part-02.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Fourth, the XGBoost framework does not address potential endogeneity between climate knowledge and health outcomes; longitudinal or quasi-experimental designs would strengthen causal identification.
~~~~

- After:

~~~~text
Although the model includes distance to the nearest health center, residence characteristics, province, and ecological belt as proxies for access and structural or geographic context, it does not include direct measures of baseline health status, health-service affordability, quality or use, household water and sanitation conditions, or local disease epidemiology. Residual confounding by these factors may affect the magnitude or direction of the observed associations, which should therefore be interpreted as descriptive and predictive patterns rather than causally identified effects. Fourth, the XGBoost framework does not address potential endogeneity between climate knowledge and health outcomes; longitudinal or quasi-experimental designs would strengthen causal identification.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: "Although the model includes distance to the nearest health center, residence characteristics, province, and ecological belt as proxies for access and structural or geographic context, it does not include direct measures of baseline health status, health-service affordability, quality or use, household water and sanitation conditions, or local disease epidemiology. Residual confounding by these factors may affect the magnitude or direction of the observed associations, which should therefore be interpreted as descriptive and predictive patterns rather than causally identified effects. "
