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

## reviewer-2/comment-9

### part-01

- Location: Methods > Variables, paragraph beginning 'Climate Change Knowledge is measured'
- Reason: Use the exact two-wave F01 construct and coding, and avoid treating a single heard-about-climate-change item as a direct measure of adaptive capacity.
- Kila decisions: KILA-D-20260827-004
- Mode: `replace`
- Timestamp: 2026-08-27T02:49:41Z
- Author: anonymous
- Markup SHA-256 before: `ea7eb3a592923f68293d14eccb0ae591ae733ddab5f55e57ccd3f2426ee943bf`
- Markup SHA-256 after: `2a74fdd45c77cab0a9a536e5fb56f0bea6452bdc998c7a747cbba568413a9466`
- Revision IDs: `38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260827T114941555106.reviewer-2-comment-9.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
Climate Change Knowledge is measured using a binary indicator to evaluate the role of cognitive factors in health resilience, as in previous studies (Hossain, 2025; Iyer & Alphonsa Jose, 2025; Ramadani et al., 2025). This indicator represents whether the respondent has heard of climate change and its impacts.
~~~~

- After:

~~~~text
Climate Change Knowledge is measured using a binary awareness indicator, as in previous studies (Hossain, 2025; Iyer & Alphonsa Jose, 2025; Ramadani et al., 2025). In both survey waves, NCCIS item F01 asks whether the respondent has heard about climate change; yes is coded as 1 and no as 0.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: "awareness "
  2. `delete`
     - Before: " to evaluate the role of cognitive factors in health resilience"
     - After: ""
  3. `replace`
     - Before: "This"
     - After: "In"
  4. `replace`
     - Before: "indicator"
     - After: "both"
  5. `replace`
     - Before: "represents"
     - After: "survey waves, NCCIS item F01 asks"
  6. `replace`
     - Before: "of"
     - After: "about"
  7. `insert`
     - Before: ""
     - After: "; yes is coded as 1"
  8. `replace`
     - Before: "its"
     - After: "no"
  9. `replace`
     - Before: "impacts"
     - After: "as 0"

## reviewer-3/comment-5

### part-01a

- Location: Results, first sentence of the paragraph beginning 'Socio-demographic subgroup analysis'
- Reason: State accurately that Figure 8 shows heterogeneity in the climate-change-knowledge contrast across household human-capital levels.
- Kila decisions: KILA-D-20260828-003
- Mode: `replace`
- Timestamp: 2026-08-28T03:14:52Z
- Author: anonymous
- Markup SHA-256 before: `f35f51f18a581451239bad21681eec6c2f3fa0cd6dcd06a7d7cca9ba25aa50c3`
- Markup SHA-256 after: `e7b231143519aa4ac8f7f83fac6c55eb7ae795335e691d96d5c0f9e20d48e865`
- Revision IDs: `59, 60, 61, 62`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T121452652918.reviewer-3-comment-5.part-01a.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Socio-demographic subgroup analysis further reveals that human capital is a key moderator of the disaster-disease relationship, as illustrated in Figure 8.
~~~~

- After:

~~~~text
Socio-demographic subgroup analysis further shows that the estimated climate change knowledge contrast varies with household human capital, as illustrated in Figure 8.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "reveals"
     - After: "shows"
  2. `insert`
     - Before: ""
     - After: " the estimated climate change knowledge contrast varies with household"
  3. `delete`
     - Before: " is a key moderator of the disaster-disease relationship"
     - After: ""

### part-01b

- Location: Results, second and third sentences of the paragraph beginning 'Socio-demographic subgroup analysis'
- Reason: Define the Figure 8 prediction contrast and explain the observed literacy and education gradients.
- Kila decisions: KILA-D-20260828-003
- Mode: `replace`
- Timestamp: 2026-08-28T03:16:07Z
- Author: anonymous
- Markup SHA-256 before: `e7b231143519aa4ac8f7f83fac6c55eb7ae795335e691d96d5c0f9e20d48e865`
- Markup SHA-256 after: `00c217f07711e2f4778ff422bdfb462c137e8f9f6fb651066e2887910389494c`
- Revision IDs: `63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T121607454668.reviewer-3-comment-5.part-01b.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Households with higher literacy ratios and greater shares of members with 12 or more years of education demonstrate markedly attenuated disaster-related health risks, as formal education and literacy equip members with the cognitive tools needed to implement preventive measures, navigate post-disaster sanitary challenges, and access medical resources more effectively. Conversely, illiterate households and those in geographically remote areas face disproportionately higher health burdens, reflecting how poverty and isolation compound climate-driven vulnerability.
~~~~

- After:

~~~~text
For each household, this contrast is calculated as the difference between the predicted probability of increased household disease incidence when climate change knowledge is set to yes and the corresponding prediction when it is set to no; negative values therefore indicate a lower predicted probability under the knowledge condition. The contrast is most negative among households with lower literate-member ratios and lower shares of members with 12 or more years of education, and it attenuates toward zero as these ratios increase.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "Households"
     - After: "For each household, this contrast is calculated as the difference between the predicted probability of increased household disease incidence when climate change knowledge is set to yes and the corresponding prediction when it is set to no; negative values therefore indicate a lower predicted probability under the knowledge condition. The contrast is most negative among households"
  2. `replace`
     - Before: "higher"
     - After: "lower"
  3. `replace`
     - Before: "literacy"
     - After: "literate-member"
  4. `replace`
     - Before: "greater"
     - After: "lower"
  5. `delete`
     - Before: " demonstrate markedly attenuated disaster-related health risks, as formal education and literacy equip members with the cognitive tools needed to implement preventive measures, navigate post-disaster sanitary challenges"
     - After: ""
  6. `replace`
     - Before: "access"
     - After: "it"
  7. `replace`
     - Before: "medical"
     - After: "attenuates"
  8. `replace`
     - Before: "resources"
     - After: "toward"
  9. `replace`
     - Before: "more"
     - After: "zero"
  10. `replace`
     - Before: "effectively"
     - After: "as these ratios increase"
  11. `delete`
     - Before: " Conversely, illiterate households and those in geographically remote areas face disproportionately higher health burdens, reflecting how poverty and isolation compound climate-driven vulnerability."
     - After: ""

### part-01c

- Location: Results, final two sentences of the paragraph beginning 'Socio-demographic subgroup analysis'
- Reason: Explain that Figure 8 shows heterogeneity in the knowledge contrast rather than parallel effects of knowledge, literacy, and education.
- Kila decisions: KILA-D-20260828-003
- Mode: `replace`
- Timestamp: 2026-08-28T03:17:25Z
- Author: anonymous
- Markup SHA-256 before: `00c217f07711e2f4778ff422bdfb462c137e8f9f6fb651066e2887910389494c`
- Markup SHA-256 after: `4b4e24b133a112a3584df7de2625855cde6bfa3b86dc3d936d2ea357da6c4dc9`
- Revision IDs: `83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T121725177117.reviewer-3-comment-5.part-01c.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Age and sex composition show little moderating influence. The protective effect of climate knowledge itself follows a nonlinear pattern across the exposure gradient: attenuation strengthens as households transition from low to intermediate disaster counts, but becomes more variable at very high exposure levels, implying that structural constraints such as inadequate infrastructure and resource scarcity may limit the effectiveness of information-based adaptation under the most severe multi-hazard conditions.
~~~~

- After:

~~~~text
Figure 8 therefore shows that the additional predictive difference associated with climate change knowledge is larger where general human capital is more limited, rather than showing parallel effects of climate knowledge, literacy, and education. Household age and sex composition show comparatively little variation in the knowledge contrast.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "Age"
     - After: "Figure 8 therefore shows that the additional predictive difference associated with climate change knowledge is larger where general human capital is more limited, rather than showing parallel effects of climate knowledge, literacy, and education. Household age"
  2. `insert`
     - Before: ""
     - After: "comparatively "
  3. `replace`
     - Before: "moderating"
     - After: "variation"
  4. `replace`
     - Before: "influence."
     - After: "in"
  5. `replace`
     - Before: "The protective effect of climate"
     - After: "the"
  6. `replace`
     - Before: "itself follows a nonlinear pattern across the exposure gradient: attenuation strengthens as households transition from low to intermediate disaster counts, but becomes more variable at very high exposure levels, implying that structural constraints such as inadequate infrastructure and resource scarcity may limit the effectiveness of information-based adaptation under the most severe multi-hazard conditions"
     - After: "contrast"

### part-02

- Location: Figure 8 caption
- Reason: Define the quantity plotted in Figure 8 so that the subgroup gradients cannot be misread as separate literacy and education effects.
- Kila decisions: KILA-D-20260828-003
- Mode: `replace`
- Timestamp: 2026-08-28T03:18:29Z
- Author: anonymous
- Markup SHA-256 before: `4b4e24b133a112a3584df7de2625855cde6bfa3b86dc3d936d2ea357da6c4dc9`
- Markup SHA-256 after: `781905ae382e9b01b10c2e7f2cc494e8db8a1e911a0f63976bbf6c6d86602f4c`
- Revision IDs: `94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T121829573915.reviewer-3-comment-5.part-02.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
Figure 8: Heterogeneity of The Effects of Climate Change Knowledge among Different Groups
~~~~

- After:

~~~~text
Figure 8: Heterogeneity in the Predicted Probability Contrast Associated with Climate Change Knowledge across Household Subgroups. The plotted value is the mean difference between predictions with climate change knowledge set to yes and no; negative values indicate a lower predicted probability under the knowledge condition. Error bars represent 95% confidence intervals.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "of"
     - After: "in"
  2. `replace`
     - Before: "The"
     - After: "the"
  3. `replace`
     - Before: "Effects"
     - After: "Predicted"
  4. `replace`
     - Before: "of"
     - After: "Probability Contrast Associated with"
  5. `replace`
     - Before: "among"
     - After: "across"
  6. `replace`
     - Before: "Different"
     - After: "Household"
  7. `replace`
     - Before: "Groups"
     - After: "Subgroups. The plotted value is the mean difference between predictions with climate change knowledge set to yes and no; negative values indicate a lower predicted probability under the knowledge condition. Error bars represent 95% confidence intervals."

### part-03a

- Location: Discussion policy sentence, phrase before 'programmes'
- Reason: Present climate-health education as a targeted programme component while preserving the Word proofing boundary around 'programmes'.
- Kila decisions: KILA-D-20260828-003
- Mode: `replace`
- Timestamp: 2026-08-28T03:20:52Z
- Author: anonymous
- Markup SHA-256 before: `781905ae382e9b01b10c2e7f2cc494e8db8a1e911a0f63976bbf6c6d86602f4c`
- Markup SHA-256 after: `258d93d9a4362387e37fc4f913fd93b34c1e7d9b440d66842f0ed6dd76042ec3`
- Revision IDs: `108, 109, 110, 111, 112, 113`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T122052772193.reviewer-3-comment-5.part-03a.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
integrating climate education into community health
~~~~

- After:

~~~~text
targeted climate-health education delivered through community health
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "integrating"
     - After: "targeted"
  2. `replace`
     - Before: "climate"
     - After: "climate-health"
  3. `replace`
     - Before: "into"
     - After: "delivered through"

### part-03b

- Location: Discussion policy sentence, phrase after 'programmes' and before the proofed verb
- Reason: Connect the policy recommendation directly to the larger Figure 8 knowledge contrast in lower-human-capital groups.
- Kila decisions: KILA-D-20260828-003
- Mode: `replace`
- Timestamp: 2026-08-28T03:21:12Z
- Author: anonymous
- Markup SHA-256 before: `258d93d9a4362387e37fc4f913fd93b34c1e7d9b440d66842f0ed6dd76042ec3`
- Markup SHA-256 after: `46a4bdc01da3a186213b739f5fbd2aedf93a76388e4004b5b2aaa5bc68927aaf`
- Revision IDs: `114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T122112500640.reviewer-3-comment-5.part-03b.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
 offers a cost-effective complement to physical investments, given that climate knowledge 
~~~~

- After:

~~~~text
 may complement general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach 
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "offers a cost-effective"
     - After: "may"
  2. `replace`
     - Before: "to"
     - After: "general"
  3. `replace`
     - Before: "physical"
     - After: "education"
  4. `replace`
     - Before: "investments"
     - After: "and literacy"
  5. `replace`
     - Before: "given"
     - After: "particularly"
  6. `replace`
     - Before: "that"
     - After: "where"
  7. `replace`
     - Before: "climate"
     - After: "broader"
  8. `replace`
     - Before: "knowledge"
     - After: "human-capital"
  9. `insert`
     - Before: ""
     - After: "resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach "

### part-03c

- Location: Discussion policy sentence, proofed verb
- Reason: Replace the predictive-importance claim with the intended complementarity wording.
- Kila decisions: KILA-D-20260828-003
- Mode: `replace`
- Timestamp: 2026-08-28T03:21:47Z
- Author: anonymous
- Markup SHA-256 before: `46a4bdc01da3a186213b739f5fbd2aedf93a76388e4004b5b2aaa5bc68927aaf`
- Markup SHA-256 after: `9e2b444e1b2bfbc2473099b19494a469478781c3c2c7833edb4725ca1d5e7d4a`
- Revision IDs: `131, 132`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T122147603036.reviewer-3-comment-5.part-03c.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
rivals
~~~~

- After:

~~~~text
reinforces
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "rivals"
     - After: "reinforces"

### part-03d

- Location: Discussion policy sentence, final phrase
- Reason: Clarify that targeted climate-health education complements rather than replaces general education and structural protection.
- Kila decisions: KILA-D-20260828-003
- Mode: `replace`
- Timestamp: 2026-08-28T03:22:16Z
- Author: anonymous
- Markup SHA-256 before: `9e2b444e1b2bfbc2473099b19494a469478781c3c2c7833edb4725ca1d5e7d4a`
- Markup SHA-256 after: `ddc6bd322a538e3778c8eaa32f009599d573911fd2ea6e2f87911631f6f3dd91`
- Revision IDs: `133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T122217035391.reviewer-3-comment-5.part-03d.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
 traditional socioeconomic determinants in predictive importance.
~~~~

- After:

~~~~text
 rather than replaces general education and structural health protection.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "traditional"
     - After: "rather"
  2. `replace`
     - Before: "socioeconomic"
     - After: "than"
  3. `replace`
     - Before: "determinants"
     - After: "replaces"
  4. `replace`
     - Before: "in"
     - After: "general"
  5. `replace`
     - Before: "predictive"
     - After: "education"
  6. `replace`
     - Before: "importance"
     - After: "and structural health protection"

### part-01d

- Location: Results, Figure 8 interpretation paragraph beginning 'Socio-demographic subgroup analysis'
- Reason: Align the Results wording with the corrected Figure 8, in which the knowledge-related prediction difference approaches and crosses zero in the highest literacy and education groups.
- Kila decisions: KILA-D-20260828-004, KILA-D-20260828-006
- Mode: `replace`
- Timestamp: 2026-08-28T06:40:58Z
- Author: anonymous
- Markup SHA-256 before: `2c21c07573591febf23e49a72a5c05e5ab9254524b1cb2b37d77d1d8c76ea0c4`
- Markup SHA-256 after: `c9eff1f2e42723667c580fced84ccb77b5b1942bcf337858a7dd50311e901c5a`
- Revision IDs: `146`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T154058161456.reviewer-3-comment-5.part-01d.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
The contrast is most negative among households with lower literate-member ratios and lower shares of members with 12 or more years of education, and it attenuates toward zero as these ratios increase.
~~~~

- After:

~~~~text
The contrast is most negative among households with lower literate-member ratios and lower shares of members with 12 or more years of education, and it attenuates toward zero and crosses it in the highest groups as these ratios increase.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: "and crosses it in the highest groups "

### part-01e

- Location: Results, first sentence of the paragraph beginning 'Socio-demographic subgroup analysis further shows'
- Reason: Place the Figure 8 uncertainty explanation in the Results text while keeping the caption consistent with the manuscript's concise figure-title convention.
- Kila decisions: KILA-D-20260828-003, KILA-D-20260828-007
- Mode: `replace`
- Timestamp: 2026-08-28T07:09:23Z
- Author: anonymous
- Markup SHA-256 before: `c9eff1f2e42723667c580fced84ccb77b5b1942bcf337858a7dd50311e901c5a`
- Markup SHA-256 after: `d6837ce4355eba54aa53b195e89e5154dfe92af99224ed0df9f66ac23230d583`
- Revision IDs: `147`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T160923225801.reviewer-3-comment-5.part-01e.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Figure 8.
~~~~

- After:

~~~~text
Figure 8. Error bars represent 95% confidence intervals.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " Error bars represent 95% confidence intervals."

