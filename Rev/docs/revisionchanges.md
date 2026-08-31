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

### part-03

- Location: Discussion, climate change knowledge interpretation paragraph
- Reason: Carry the shared-reporting and reverse-causation caveat into the interpretation of the knowledge-related pattern, as approved in bundle part 1 of 3.
- Kila decisions: KILA-D-20260830-005
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-30T02:23:33Z
- Author: Kila
- Markup SHA-256 before: `c78de5e7e3ac1ae8f99c3ccf26f0f6a1443ccf566dd2809c263f6f731168a7ce`
- Markup SHA-256 after: `ffca20660f70286d12a09fe084b949961c61425961cfc5455af55f72296f3c0b`
- Revision IDs: `435`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260830T112333647253.reviewer-3-comment-3.part-03.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
The variation in predicted disease probability by climate change knowledge status aligns with theoretical frameworks linking cognitive awareness to proactive health-seeking behaviour, including pre-emptive protection of water sources, improved hygiene practices, and earlier care-seeking during periods of environmental stress (Hossain, 2025; Iyer & Alphonsa Jose, 2025). The larger knowledge-related prediction difference at high disaster counts indicates that the observed association is strongest when environmental conditions are most severe. This pattern is consistent with evidence from comparable low-resource settings, where information-based interventions have demonstrated measurable reductions in climate-related health risks even in the absence of strong material adaptive capacity (Dumitraşcu et al., 2026; Mantilla et al., 2025). The spatial heterogeneity in knowledge-related prediction differences across province-ecological belt units, and the attenuation of disaster-disease associations among more educated households, further indicate that the observed association varies with local infrastructure, socioeconomic resources, and human capital (Ali et al., 2026; Negi et al., 2025; Sandilya & Goswami, 2025).
~~~~

- After:

~~~~text
The variation in predicted disease probability by climate change knowledge status aligns with theoretical frameworks linking cognitive awareness to proactive health-seeking behaviour, including pre-emptive protection of water sources, improved hygiene practices, and earlier care-seeking during periods of environmental stress (Hossain, 2025; Iyer & Alphonsa Jose, 2025). The larger knowledge-related prediction difference at high disaster counts indicates that the observed association is strongest when environmental conditions are most severe. This pattern is consistent with evidence from comparable low-resource settings, where information-based interventions have demonstrated measurable reductions in climate-related health risks even in the absence of strong material adaptive capacity (Dumitraşcu et al., 2026; Mantilla et al., 2025). The spatial heterogeneity in knowledge-related prediction differences across province-ecological belt units, and the attenuation of disaster-disease associations among more educated households, further indicate that the observed association varies with local infrastructure, socioeconomic resources, and human capital (Ali et al., 2026; Negi et al., 2025; Sandilya & Goswami, 2025). At the same time, because climate change knowledge, disaster exposure, and household disease change are reported by the same respondent in the same interview, shared reporting tendencies and the possibility that perceived health deterioration shapes climate-change awareness or retrospective exposure reporting remain alternative explanations for part of the observed pattern.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " At the same time, because climate change knowledge, disaster exposure, and household disease change are reported by the same respondent in the same interview, shared reporting tendencies and the possibility that perceived health deterioration shapes climate-change awareness or retrospective exposure reporting remain alternative explanations for part of the observed pattern."

### part-04

- Location: Discussion, policy paragraph beginning 'These findings have direct policy implications'
- Reason: Carry the shared-reporting and temporal-ordering caveat into a constructive programme-evaluation recommendation while preserving the approved substantive policy priorities, as approved in bundle part 2 of 3.
- Kila decisions: KILA-D-20260830-005
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-30T02:24:00Z
- Author: Kila
- Markup SHA-256 before: `ffca20660f70286d12a09fe084b949961c61425961cfc5455af55f72296f3c0b`
- Markup SHA-256 after: `8d9ad60cee279519c8dee4ed267868d9f6d18ba87b9532786acdb22b20a8d02e`
- Revision IDs: `436, 437, 438, 439, 440`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260830T112400350693.reviewer-3-comment-3.part-04.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `18be3bb9140ea30c36fd7dfe87069732fa45498a7648c9ef61bdf41873cfd67b`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
These findings have direct policy implications. Infrastructure reinforcement and medical resource allocation should be prioritised in high-risk province-ecological belt units where multi-hazard intensity and disease burden coincide, particularly in the Terai and Hill regions. Simultaneously, targeted climate-health education delivered through community health programmes may complement general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach reinforces rather than replaces general education and structural health protection. Disaster risk reduction frameworks should move beyond single-hazard approaches to address the cumulative and nonlinear nature of multi-hazard health threats (Erdem (Erdem Okumus, 2025; O’Donnell & Sovacool, 2026).
~~~~

- After:

~~~~text
These findings have policy relevance for targeting and programme design. Infrastructure reinforcement and medical resource allocation should be prioritised in high-risk province-ecological belt units where multi-hazard intensity and disease burden coincide, particularly in the Terai and Hill regions. Simultaneously, targeted climate-health education delivered through community health programmes may complement general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach reinforces rather than replaces general education and structural health protection. Because the present survey measures climate change knowledge, reported disaster exposure, and reported disease change in the same interview, such programmes should be accompanied by prospective evaluation that separately measures climate-health knowledge, preparedness practices, and independently assessed health outcomes. This would clarify how information translates into action and health benefits. Disaster risk reduction frameworks should move beyond single-hazard approaches to address the cumulative and nonlinear nature of multi-hazard health threats (Erdem Okumus, 2025; O’Donnell & Sovacool, 2026).
~~~~

- Minimal tracked fragments:
  1. `delete`
     - Before: "direct "
     - After: ""
  2. `replace`
     - Before: "implications"
     - After: "relevance for targeting and programme design"
  3. `insert`
     - Before: ""
     - After: "Because the present survey measures climate change knowledge, reported disaster exposure, and reported disease change in the same interview, such programmes should be accompanied by prospective evaluation that separately measures climate-health knowledge, preparedness practices, and independently assessed health outcomes. This would clarify how information translates into action and health benefits. "
  4. `delete`
     - Before: "(Erdem "
     - After: ""

### part-03-reapply-01

- Location: Discussion, climate change knowledge interpretation paragraph
- Reason: Restore the already approved interpretation sentence after the human Word save for part 05 replaced a stale open copy and removed the earlier machine insertion.
- Kila decisions: KILA-D-20260830-005, KILA-D-20260830-006
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-30T02:35:02Z
- Author: Kila
- Markup SHA-256 before: `98b6787b9de6285b8a3b4c4a1ae72a36b3f1dd58a22a2724e226a8c7c54e9e24`
- Markup SHA-256 after: `c0c719cb3da67ffbf079e69d5053408480932de07643e3b5c19a1aa7245398d9`
- Revision IDs: `452`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260830T113502465515.reviewer-3-comment-3.part-03-reapply-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
The variation in predicted disease probability by climate change knowledge status aligns with theoretical frameworks linking cognitive awareness to proactive health-seeking behaviour, including pre-emptive protection of water sources, improved hygiene practices, and earlier care-seeking during periods of environmental stress (Hossain, 2025; Iyer & Alphonsa Jose, 2025). The larger knowledge-related prediction difference at high disaster counts indicates that the observed association is strongest when environmental conditions are most severe. This pattern is consistent with evidence from comparable low-resource settings, where information-based interventions have demonstrated measurable reductions in climate-related health risks even in the absence of strong material adaptive capacity (Dumitraşcu et al., 2026; Mantilla et al., 2025). The spatial heterogeneity in knowledge-related prediction differences across province-ecological belt units, and the attenuation of disaster-disease associations among more educated households, further indicate that the observed association varies with local infrastructure, socioeconomic resources, and human capital (Ali et al., 2026; Negi et al., 2025; Sandilya & Goswami, 2025).
~~~~

- After:

~~~~text
The variation in predicted disease probability by climate change knowledge status aligns with theoretical frameworks linking cognitive awareness to proactive health-seeking behaviour, including pre-emptive protection of water sources, improved hygiene practices, and earlier care-seeking during periods of environmental stress (Hossain, 2025; Iyer & Alphonsa Jose, 2025). The larger knowledge-related prediction difference at high disaster counts indicates that the observed association is strongest when environmental conditions are most severe. This pattern is consistent with evidence from comparable low-resource settings, where information-based interventions have demonstrated measurable reductions in climate-related health risks even in the absence of strong material adaptive capacity (Dumitraşcu et al., 2026; Mantilla et al., 2025). The spatial heterogeneity in knowledge-related prediction differences across province-ecological belt units, and the attenuation of disaster-disease associations among more educated households, further indicate that the observed association varies with local infrastructure, socioeconomic resources, and human capital (Ali et al., 2026; Negi et al., 2025; Sandilya & Goswami, 2025). At the same time, because climate change knowledge, disaster exposure, and household disease change are reported by the same respondent in the same interview, shared reporting tendencies and the possibility that perceived health deterioration shapes climate-change awareness or retrospective exposure reporting remain alternative explanations for part of the observed pattern.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " At the same time, because climate change knowledge, disaster exposure, and household disease change are reported by the same respondent in the same interview, shared reporting tendencies and the possibility that perceived health deterioration shapes climate-change awareness or retrospective exposure reporting remain alternative explanations for part of the observed pattern."

### part-04-reapply-01

- Location: Discussion, policy paragraph beginning 'These findings have direct policy implications'
- Reason: Restore the already approved policy revision after the human Word save for part 05 replaced a stale open copy and removed the earlier machine changes.
- Kila decisions: KILA-D-20260830-005, KILA-D-20260830-006
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-30T02:35:50Z
- Author: Kila
- Markup SHA-256 before: `c0c719cb3da67ffbf079e69d5053408480932de07643e3b5c19a1aa7245398d9`
- Markup SHA-256 after: `705e3f935946edb3266ba77cbb8259fbaf3717d96b5c3cc8b1c964a90a07a09c`
- Revision IDs: `453, 454, 455, 456, 457`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260830T113550356424.reviewer-3-comment-3.part-04-reapply-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `18be3bb9140ea30c36fd7dfe87069732fa45498a7648c9ef61bdf41873cfd67b`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
These findings have direct policy implications. Infrastructure reinforcement and medical resource allocation should be prioritised in high-risk province-ecological belt units where multi-hazard intensity and disease burden coincide, particularly in the Terai and Hill regions. Simultaneously, targeted climate-health education delivered through community health programmes may complement general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach reinforces rather than replaces general education and structural health protection. Disaster risk reduction frameworks should move beyond single-hazard approaches to address the cumulative and nonlinear nature of multi-hazard health threats (Erdem (Erdem Okumus, 2025; O’Donnell & Sovacool, 2026).
~~~~

- After:

~~~~text
These findings have policy relevance for targeting and programme design. Infrastructure reinforcement and medical resource allocation should be prioritised in high-risk province-ecological belt units where multi-hazard intensity and disease burden coincide, particularly in the Terai and Hill regions. Simultaneously, targeted climate-health education delivered through community health programmes may complement general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach reinforces rather than replaces general education and structural health protection. Because the present survey measures climate change knowledge, reported disaster exposure, and reported disease change in the same interview, such programmes should be accompanied by prospective evaluation that separately measures climate-health knowledge, preparedness practices, and independently assessed health outcomes. This would clarify how information translates into action and health benefits. Disaster risk reduction frameworks should move beyond single-hazard approaches to address the cumulative and nonlinear nature of multi-hazard health threats (Erdem Okumus, 2025; O’Donnell & Sovacool, 2026).
~~~~

- Minimal tracked fragments:
  1. `delete`
     - Before: "direct "
     - After: ""
  2. `replace`
     - Before: "implications"
     - After: "relevance for targeting and programme design"
  3. `insert`
     - Before: ""
     - After: "Because the present survey measures climate change knowledge, reported disaster exposure, and reported disease change in the same interview, such programmes should be accompanied by prospective evaluation that separately measures climate-health knowledge, preparedness practices, and independently assessed health outcomes. This would clarify how information translates into action and health benefits. "
  4. `delete`
     - Before: "(Erdem "
     - After: ""

### part-05

- Location: Discussion, complete limitations paragraph beginning 'Several limitations should be acknowledged'
- Reason: Reorganize the limitations without a deliberate shortening target, explicitly connect same-interview reporting to common-method bias and reverse causation, and preserve all limitation and contribution commitments approved for earlier reviewer comments.
- Kila decisions: KILA-D-20260830-004, KILA-D-20260830-005, KILA-D-20260830-006
- Mode: `human-manual-replace`
- Revises prior parts: `reviewer-2/comment-1#part-02`, `reviewer-2/comment-3#part-02`, `reviewer-3/comment-3#part-02`, `reviewer-2/comment-7#part-02`
- Timestamp: 2026-08-30T11:30:00+09:00
- Author: Jie MI
- Markup SHA-256 before: `8d9ad60cee279519c8dee4ed267868d9f6d18ba87b9532786acdb22b20a8d02e`
- Markup SHA-256 after human save: `98b6787b9de6285b8a3b4c4a1ae72a36b3f1dd58a22a2724e226a8c7c54e9e24`
- Revision IDs created or updated by the human save in this paragraph: `355, 357, 358, 360, 361, 363, 364, 366, 367, 369, 370, 372, 373, 375, 376, 378, 379, 381, 382, 384, 385, 387, 388, 390, 393, 394`
- Recovery source: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260830T112400350693.reviewer-3-comment-3.part-04.docx` (the limitations target is unchanged in this pre-part-04 backup)
- Paragraph properties preserved: pending consolidated fresh-clean visual review
- Run style verification: deferred to consolidated fresh-clean visual review
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
Several limitations should be acknowledged. The data capture conditions reported in the 2016 and 2022 survey waves. Given the time gap between the survey periods and the analysis, subsequent changes in hazard patterns, health-service access, climate information, adaptation practices, and disease conditions may have altered the magnitude and geographic distribution of the observed relationships. The findings should therefore be interpreted in relation to the survey periods. Within this temporal scope, the analysis identifies the nonlinear relationship between cumulative multi-hazard exposure and reported household disease change and shows how this relationship varies with climate change knowledge and across geographic and socioeconomic groups. These results provide specific priorities for current monitoring, including whether risk remains concentrated after multiple hazards, where knowledge-related differences persist, and which population groups continue to experience greater vulnerability. First, the pooled cross-sectional design precludes causal inference; findings reflect associations between perceived disaster exposure and perceived health deterioration. Second, the outcome asks one respondent to compare family illness with conditions 25 years earlier, while the exposure and climate-knowledge measures are also self-reported. The outcome therefore captures perceived change rather than clinically verified incidence and may vary with memory, respondent age or education, proxy reporting for other household members, and interpretation of the reference period. Correlated reporting tendencies may also create common-method bias and influence the magnitude or direction of observed associations; consequently, the findings should not be interpreted as estimates of objective disease incidence. Reverse causation also cannot be excluded because perceived household health deterioration may influence subsequent climate-change awareness or the reporting of past disaster exposure. Third, the binary climate knowledge measure captures basic awareness and may reflect access to climate-related information, but it does not measure the depth or accuracy of understanding, risk perception, preparedness, adaptive actions, or the resources needed to implement them. It should therefore be interpreted as an awareness indicator rather than as a direct measure of behavioural change, preparedness, or adaptive capacity. Accordingly, the observed moderation pattern should be interpreted as an association with reported awareness status, rather than as evidence that awareness translated into behavioural adaptation or improved preparedness. Although the model includes distance to the nearest health center, residence characteristics, province, and ecological belt as proxies for access and structural or geographic context, it does not include direct measures of baseline health status, health-service affordability, quality or use, household water and sanitation conditions, or local disease epidemiology. Residual confounding by these factors may affect the magnitude or direction of the observed associations, which should therefore be interpreted as descriptive and predictive patterns rather than causally identified effects. Fourth, the XGBoost framework does not address potential endogeneity between climate knowledge and health outcomes; longitudinal or quasi-experimental designs would strengthen causal identification. Fifth, spatial dependence and spillover effects are not explicitly modelled, and geographically weighted approaches could further illuminate regional drivers of climate-health vulnerability.
~~~~

- After:

~~~~text
Several limitations should be acknowledged. First, the data capture conditions reported in the 2016 and 2022 survey waves. Given the time gap between the survey periods and the analysis, subsequent changes in hazard patterns, health-service access, climate information, adaptation practices, and disease conditions may have altered the magnitude and geographic distribution of the observed relationships. The findings should therefore be interpreted in relation to the survey periods. Within this temporal scope, the analysis identifies the nonlinear relationship between cumulative multi-hazard exposure and reported household disease change and shows how this relationship varies with climate change knowledge and across geographic and socioeconomic groups. These results provide specific priorities for current monitoring, including whether risk remains concentrated after multiple hazards, where knowledge-related differences persist, and which population groups continue to experience greater vulnerability. Second, the pooled cross-sectional design precludes causal inference; findings reflect associations between perceived disaster exposure and perceived health deterioration. The outcome asks one respondent to compare family illness with conditions 25 years earlier, and the outcome, exposure, and climate-knowledge measures are reported by the same respondent in the same interview. The outcome therefore captures perceived change rather than clinically verified incidence and may vary with memory, respondent age or education, proxy reporting for other household members, and interpretation of the reference period. Correlated reporting tendencies may create common-method bias and influence the magnitude or direction of the observed associations; consequently, the findings should not be interpreted as estimates of objective disease incidence. Reverse causation also cannot be excluded because perceived household health deterioration may influence climate-change awareness or the retrospective reporting of disaster exposure. Third, the binary climate knowledge measure captures basic awareness and may reflect access to climate-related information, but it does not measure the depth or accuracy of understanding, risk perception, preparedness, adaptive actions, or the resources needed to implement them. It should therefore be interpreted as an awareness indicator rather than as a direct measure of behavioural change, preparedness, or adaptive capacity; accordingly, the observed moderation pattern is not evidence that awareness translated into behavioural adaptation or improved preparedness. Fourth, although the model includes distance to the nearest health center, residence characteristics, province, and ecological belt as proxies for access and structural or geographic context, it does not include direct measures of baseline health status, health-service affordability, quality or use, household water and sanitation conditions, or local disease epidemiology. Residual confounding by these factors may affect the magnitude or direction of the observed associations, which should therefore be interpreted as descriptive and predictive patterns rather than causally identified effects. Fifth, the XGBoost framework does not address potential endogeneity between climate knowledge and health outcomes; longitudinal or quasi-experimental designs would strengthen causal identification. Spatial dependence and spillover effects are also not explicitly modelled, and geographically weighted approaches could further illuminate regional drivers of climate-health vulnerability.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "Complete prior limitations paragraph"
     - After: "Complete approved content-preserving limitations paragraph"

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

## reviewer-2/comment-1

### part-01

- Location: Methods > Survey Data and Sample, paragraph beginning 'This study uses the NCCIS'
- Reason: Explain why the 2016 and 2022 waves were used and state the temporal coverage limitation in Methods.
- Kila decisions: KILA-D-20260828-014
- Mode: `replace`
- Timestamp: 2026-08-28T09:05:51Z
- Author: anonymous
- Markup SHA-256 before: `daec383bfeb821c3f82ccdaefc6307f9710b9bbbe36478c82cd5f3bb72a0faa0`
- Markup SHA-256 after: `7949f8756f641630c31b735f8c3d9ff265eb830c609a0f36ae3a74a3a564d5a2`
- Revision IDs: `158, 159, 160, 161`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T180551149296.reviewer-2-comment-1.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
Pooling the 2016 and 2022 survey waves yields a final analytic sample of 11,568 respondent-level observations, including 5,060 from 2016 and 6,508 from 2022.
~~~~

- After:

~~~~text
At the time of analysis, the 2016 and 2022 waves were the available nationally representative rounds of this government survey, with 2022 being the latest. Consequently, changes in hazard exposure, health conditions, or adaptation after the 2022 survey are not captured in the data and should be considered when interpreting the findings in relation to current conditions. Pooling these waves yields a final analytic sample of 11,568 respondent-level observations, including 5,060 from 2016 and 6,508 from 2022.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "Pooling"
     - After: "At the time of analysis,"
  2. `insert`
     - Before: ""
     - After: "waves were the available nationally representative rounds of this government "
  3. `insert`
     - Before: ""
     - After: ", with 2022 being the latest. Consequently, changes in hazard exposure, health conditions, or adaptation after the 2022 survey are not captured in the data and should be considered when interpreting the findings in relation to current conditions. Pooling these"

### part-02

- Location: Discussion > limitations, paragraph beginning 'Several limitations should be acknowledged'
- Reason: Explain how the time gap affects current interpretation and state the study's specific contribution and monitoring relevance.
- Kila decisions: KILA-D-20260828-014
- Mode: `replace`
- Timestamp: 2026-08-28T09:06:34Z
- Author: anonymous
- Markup SHA-256 before: `7949f8756f641630c31b735f8c3d9ff265eb830c609a0f36ae3a74a3a564d5a2`
- Markup SHA-256 after: `2570162f19448c859998072c3469e66504fd36ba203d3a28f0de31621c2b9346`
- Revision IDs: `162`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T180634126342.reviewer-2-comment-1.part-02.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Several limitations should be acknowledged.
~~~~

- After:

~~~~text
Several limitations should be acknowledged. The data capture conditions reported in the 2016 and 2022 survey waves. Given the time gap between the survey periods and the analysis, subsequent changes in hazard patterns, health-service access, climate information, adaptation practices, and disease conditions may have altered the magnitude and geographic distribution of the observed relationships. The findings should therefore be interpreted in relation to the survey periods. Within this temporal scope, the analysis identifies the nonlinear relationship between cumulative multi-hazard exposure and reported household disease change and shows how this relationship varies with climate change knowledge and across geographic and socioeconomic groups. These results provide specific priorities for current monitoring, including whether risk remains concentrated after multiple hazards, where knowledge-related differences persist, and which population groups continue to experience greater vulnerability.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " The data capture conditions reported in the 2016 and 2022 survey waves. Given the time gap between the survey periods and the analysis, subsequent changes in hazard patterns, health-service access, climate information, adaptation practices, and disease conditions may have altered the magnitude and geographic distribution of the observed relationships. The findings should therefore be interpreted in relation to the survey periods. Within this temporal scope, the analysis identifies the nonlinear relationship between cumulative multi-hazard exposure and reported household disease change and shows how this relationship varies with climate change knowledge and across geographic and socioeconomic groups. These results provide specific priorities for current monitoring, including whether risk remains concentrated after multiple hazards, where knowledge-related differences persist, and which population groups continue to experience greater vulnerability."

## reviewer-2/comment-2

### part-01

- Location: Title
- Reason: Replace the causal title verb with the human-approved directional association phrase.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:21Z
- Author: anonymous
- Markup SHA-256 before: `2570162f19448c859998072c3469e66504fd36ba203d3a28f0de31621c2b9346`
- Markup SHA-256 after: `2beee7f695e131708abc390472fca1c5e8cb573f04bba066b9a7ba9f61f17da8`
- Revision IDs: `163, 164`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184521627965.reviewer-2-comment-2.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `f0eb213d81be50b2c016f965ca140d56e8046aaf230b3de50339fd849b66046c`
- Formula verification: not applicable
- Before:

~~~~text
Climate Knowledge Mitigates Health Risks from Multi-Hazard Exposure: Evidence from Nepal
~~~~

- After:

~~~~text
Climate Knowledge Is Associated with Lower Health Risks from Multi-Hazard Exposure: Evidence from Nepal
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "Mitigates"
     - After: "Is Associated with Lower"

### part-02

- Location: Summary > Methods
- Reason: Describe subgroup results as heterogeneity in predicted associations rather than effects.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:22Z
- Author: anonymous
- Markup SHA-256 before: `2beee7f695e131708abc390472fca1c5e8cb573f04bba066b9a7ba9f61f17da8`
- Markup SHA-256 after: `c3b79d69784162ed85624c506614a2c58ae0a981ee72f3e817d587cf1a40421c`
- Revision IDs: `165, 166, 167, 168`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184522140248.reviewer-2-comment-2.part-02.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
Partial dependence and subgroup analyses examined effect heterogeneity across geographic and socioeconomic strata.
~~~~

- After:

~~~~text
Partial dependence and subgroup analyses examined heterogeneity in predicted associations across geographic and socioeconomic strata.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "effect"
     - After: "heterogeneity"
  2. `replace`
     - Before: "heterogeneity"
     - After: "in predicted associations"

### part-04

- Location: Summary > Interpretation
- Reason: State the knowledge result as a predicted association and temper the education recommendation.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:22Z
- Author: anonymous
- Markup SHA-256 before: `c3b79d69784162ed85624c506614a2c58ae0a981ee72f3e817d587cf1a40421c`
- Markup SHA-256 after: `2e8445eee66021bca3bc70273df41ce064fb62cdfb59eb2f33aec459af1c7e61`
- Revision IDs: `169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184522671619.reviewer-2-comment-2.part-04.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
Cumulative multi-hazard exposure is associated with increased household disease risk in a nonlinear pattern, and climate change knowledge is associated with attenuation of this risk. Integrating climate education into community health programmers may offer a cost-effective strategy for building health resilience in hazard-prone populations across South and Southeast Asia.
~~~~

- After:

~~~~text
Cumulative multi-hazard exposure is associated with increased household disease risk in a nonlinear pattern, and climate change knowledge is associated with a lower predicted probability of disease increase. Targeted climate-health education delivered through community health programmes may complement general education and structural health protection in hazard-prone populations across South and Southeast Asia.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "attenuation"
     - After: "a lower predicted probability"
  2. `replace`
     - Before: "this"
     - After: "disease"
  3. `replace`
     - Before: "risk"
     - After: "increase"
  4. `replace`
     - Before: "Integrating"
     - After: "Targeted"
  5. `replace`
     - Before: "climate"
     - After: "climate-health"
  6. `replace`
     - Before: "into"
     - After: "delivered through"
  7. `replace`
     - Before: "programmers"
     - After: "programmes"
  8. `replace`
     - Before: "offer"
     - After: "complement"
  9. `replace`
     - Before: "a"
     - After: "general"
  10. `replace`
     - Before: "cost-effective"
     - After: "education"
  11. `replace`
     - Before: "strategy"
     - After: "and"
  12. `replace`
     - Before: "for building"
     - After: "structural"
  13. `replace`
     - Before: "resilience"
     - After: "protection"

### part-05

- Location: Introduction, study objective paragraph beginning 'This study addresses these gaps'
- Reason: Replace moderating influence with variation by knowledge status.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:23Z
- Author: anonymous
- Markup SHA-256 before: `2e8445eee66021bca3bc70273df41ce064fb62cdfb59eb2f33aec459af1c7e61`
- Markup SHA-256 after: `e354b578a22e3b7b3502281b327ba40493035739e96d9dff7e0434f3eb8d3e28`
- Revision IDs: `195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184523208323.reviewer-2-comment-2.part-05.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
It quantifies the disaster-disease relationship and the moderating influence of climate knowledge.
~~~~

- After:

~~~~text
It characterizes the disaster-disease association and its variation by climate knowledge status.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "quantifies"
     - After: "characterizes"
  2. `replace`
     - Before: "relationship"
     - After: "association"
  3. `replace`
     - Before: "the"
     - After: "its"
  4. `replace`
     - Before: "moderating"
     - After: "variation"
  5. `replace`
     - Before: "influence of"
     - After: "by"
  6. `insert`
     - Before: ""
     - After: " status"

### part-06

- Location: Methods > Variables, climate change knowledge paragraph
- Reason: Frame the moderator rationale as variation in health patterns rather than mitigation.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:23Z
- Author: anonymous
- Markup SHA-256 before: `e354b578a22e3b7b3502281b327ba40493035739e96d9dff7e0434f3eb8d3e28`
- Markup SHA-256 after: `d790aad7b58dad65c429b681a96921b8cb1763fe1790b15ec9f7cfae33ca4438`
- Revision IDs: `206, 207, 208, 209, 210, 211, 212, 213, 214`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184523716462.reviewer-2-comment-2.part-06.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
In the analytical framework, this indicator is treated as a potential moderator. It may mitigate the adverse health effects of disaster exposure, as indicated by the literature (Adom et al., 2025; Hossain, 2025; Liu et al., 2026).
~~~~

- After:

~~~~text
In the analytical framework, this indicator is treated as a potential moderator based on literature suggesting that disaster-related health patterns may vary by climate change knowledge status (Adom et al., 2025; Hossain, 2025; Liu et al., 2026).
~~~~

- Minimal tracked fragments:
  1. `delete`
     - Before: "."
     - After: ""
  2. `replace`
     - Before: "It"
     - After: "based on literature suggesting that disaster-related health patterns"
  3. `replace`
     - Before: "mitigate the adverse health effects of disaster exposure, as indicated"
     - After: "vary"
  4. `replace`
     - Before: "the"
     - After: "climate"
  5. `replace`
     - Before: "literature"
     - After: "change knowledge status"

### part-07

- Location: Methods > Analytical Framework
- Reason: Describe subgroup PDPs as predicted-probability comparisons rather than a moderating role.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:24Z
- Author: anonymous
- Markup SHA-256 before: `d790aad7b58dad65c429b681a96921b8cb1763fe1790b15ec9f7cfae33ca4438`
- Markup SHA-256 after: `b08f89c6aec8e5334e3e6894bfe33a0550b6c093d38722fe9d4a11f06bea83ed`
- Revision IDs: `215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184524237433.reviewer-2-comment-2.part-07.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Subgroup PDPs stratified by climate knowledge status quantify the moderating role of climate change knowledge.
~~~~

- After:

~~~~text
Subgroup PDPs stratified by climate knowledge status characterize differences in predicted probabilities across knowledge groups.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "quantify"
     - After: "characterize"
  2. `replace`
     - Before: "the"
     - After: "differences"
  3. `replace`
     - Before: "moderating"
     - After: "in"
  4. `replace`
     - Before: "role"
     - After: "predicted"
  5. `replace`
     - Before: "of"
     - After: "probabilities"
  6. `replace`
     - Before: "climate change"
     - After: "across"
  7. `insert`
     - Before: ""
     - After: " groups"

### part-08

- Location: Results, Figure 6 paragraph beginning 'Climate change knowledge significantly reduces'
- Reason: Replace causal and protective interpretations with predicted-probability differences.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:24Z
- Author: anonymous
- Markup SHA-256 before: `b08f89c6aec8e5334e3e6894bfe33a0550b6c093d38722fe9d4a11f06bea83ed`
- Markup SHA-256 after: `4b76f40d48093e6fd17cfb6c1c00a2988803fff457ef11d977b464beaf5e2b0c`
- Revision IDs: `228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184524756110.reviewer-2-comment-2.part-08.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Climate change knowledge significantly reduces the negative effect of cumulative disaster exposure on household health. The y-axis of the PDPs is the mean predicted probability of new disease occurrence. As shown in Figure 6, across the full exposure gradient, households with climate change knowledge exhibit a consistently lower predicted probability of disease increase compared to those without. The protective gap widens as disaster counts increase, indicating that the benefits of climate awareness are most pronounced under high multi-hazard conditions. At lower exposure levels the gap is modest, but it expands substantially as households accumulate four or more distinct hazard types.
~~~~

- After:

~~~~text
Climate change knowledge is associated with differences in the predicted probability of increased household disease incidence across the cumulative disaster-exposure gradient. The y-axis of the PDPs is the mean predicted probability of new disease occurrence. As shown in Figure 6, across the full exposure gradient, households with climate change knowledge exhibit a consistently lower predicted probability of disease increase compared with those without. The prediction difference widens as disaster counts increase and is largest under high multi-hazard exposure. At lower exposure levels the difference is modest, but it expands substantially as households accumulate four or more distinct hazard types.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "significantly"
     - After: "is"
  2. `replace`
     - Before: "reduces"
     - After: "associated with differences in"
  3. `replace`
     - Before: "negative"
     - After: "predicted"
  4. `replace`
     - Before: "effect"
     - After: "probability"
  5. `replace`
     - Before: "cumulative disaster exposure on"
     - After: "increased"
  6. `replace`
     - Before: "health"
     - After: "disease incidence across the cumulative disaster-exposure gradient"
  7. `replace`
     - Before: "to"
     - After: "with"
  8. `replace`
     - Before: "protective"
     - After: "prediction"
  9. `replace`
     - Before: "gap"
     - After: "difference"
  10. `delete`
     - Before: ","
     - After: ""
  11. `replace`
     - Before: "indicating"
     - After: "and"
  12. `replace`
     - Before: "that"
     - After: "is"
  13. `replace`
     - Before: "the benefits of climate awareness are most pronounced"
     - After: "largest"
  14. `replace`
     - Before: "conditions"
     - After: "exposure"
  15. `replace`
     - Before: "gap"
     - After: "difference"

### part-09b

- Location: Results, Figure 7 paragraph beginning 'Spatial analysis reveals'
- Reason: Replace attenuation and buffering wording with negative prediction differences.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:25Z
- Author: anonymous
- Markup SHA-256 before: `4b76f40d48093e6fd17cfb6c1c00a2988803fff457ef11d977b464beaf5e2b0c`
- Markup SHA-256 after: `8d04f6836efe08860acbfd4e98042431b94662209d709e94899c67045e6ee65c`
- Revision IDs: `257, 258, 259, 260, 261, 262, 263, 264, 265, 266`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184525262267.reviewer-2-comment-2.part-09b.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
Stronger attenuation effects are concentrated in several western and eastern units, while parts of central Nepal show comparatively weaker buffering.
~~~~

- After:

~~~~text
Larger negative prediction differences are concentrated in several western and eastern units, while parts of central Nepal show comparatively smaller differences.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "Stronger"
     - After: "Larger"
  2. `replace`
     - Before: "attenuation"
     - After: "negative"
  3. `replace`
     - Before: "effects"
     - After: "prediction differences"
  4. `replace`
     - Before: "weaker"
     - After: "smaller"
  5. `replace`
     - Before: "buffering"
     - After: "differences"

### part-10a

- Location: Discussion, opening paragraph beginning 'This study provides evidence'
- Reason: Use knowledge-status variation and prediction-difference language in the main findings synthesis.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:25Z
- Author: anonymous
- Markup SHA-256 before: `8d04f6836efe08860acbfd4e98042431b94662209d709e94899c67045e6ee65c`
- Markup SHA-256 after: `d254cb7a7861725769c571aad114cfb20143764d63ac40bf740f9b10fce047d7`
- Revision IDs: `267, 268, 269`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184525777002.reviewer-2-comment-2.part-10a.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
and that climate change knowledge may buffer this association
~~~~

- After:

~~~~text
and that the predicted disaster-disease pattern varies by climate change knowledge status
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " the predicted disaster-disease pattern varies by"
  2. `replace`
     - Before: "may buffer this association"
     - After: "status"

### part-11

- Location: Discussion, climate change knowledge interpretation paragraph
- Reason: Remove protective-effect and adaptive-resource claims owned by this study while retaining the cited intervention evidence.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:26Z
- Author: anonymous
- Markup SHA-256 before: `d254cb7a7861725769c571aad114cfb20143764d63ac40bf740f9b10fce047d7`
- Markup SHA-256 after: `584b54a1e86e10a092716c651e3b6c4b6ecb7b5936735078a55ab6072cd71571`
- Revision IDs: `270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184526352064.reviewer-2-comment-2.part-11.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Before:

~~~~text
The moderating role of climate change knowledge aligns with theoretical frameworks linking cognitive awareness to proactive health-seeking behaviour, including pre-emptive protection of water sources, improved hygiene practices, and earlier care-seeking during periods of environmental stress (Hossain, 2025; Iyer & Alphonsa Jose, 2025). The finding that the protective effect is most pronounced at high disaster counts suggests that climate knowledge functions as a critical adaptive resource precisely when environmental conditions are most severe. This pattern is consistent with evidence from comparable low-resource settings, where information-based interventions have demonstrated measurable reductions in climate-related health risks even in the absence of strong material adaptive capacity (Dumitraşcu et al., 2026; Mantilla et al., 2025). The spatial heterogeneity in buffering effects across province-ecological belt units, and the attenuation of disaster-disease associations among more educated households, further indicate that the protective value of climate knowledge is conditioned by local infrastructure, socioeconomic resources, and human capital (Ali et al., 2026; Negi et al., 2025; Sandilya & Goswami, 2025).
~~~~

- After:

~~~~text
The variation in predicted disease probability by climate change knowledge status aligns with theoretical frameworks linking cognitive awareness to proactive health-seeking behaviour, including pre-emptive protection of water sources, improved hygiene practices, and earlier care-seeking during periods of environmental stress (Hossain, 2025; Iyer & Alphonsa Jose, 2025). The larger knowledge-related prediction difference at high disaster counts indicates that the observed association is strongest when environmental conditions are most severe. This pattern is consistent with evidence from comparable low-resource settings, where information-based interventions have demonstrated measurable reductions in climate-related health risks even in the absence of strong material adaptive capacity (Dumitraşcu et al., 2026; Mantilla et al., 2025). The spatial heterogeneity in knowledge-related prediction differences across province-ecological belt units, and the attenuation of disaster-disease associations among more educated households, further indicate that the observed association varies with local infrastructure, socioeconomic resources, and human capital (Ali et al., 2026; Negi et al., 2025; Sandilya & Goswami, 2025).
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "moderating"
     - After: "variation"
  2. `replace`
     - Before: "role"
     - After: "in"
  3. `replace`
     - Before: "of"
     - After: "predicted disease probability by"
  4. `insert`
     - Before: ""
     - After: " status"
  5. `replace`
     - Before: "finding"
     - After: "larger"
  6. `replace`
     - Before: "that"
     - After: "knowledge-related"
  7. `replace`
     - Before: "the"
     - After: "prediction"
  8. `replace`
     - Before: "protective effect is most pronounced"
     - After: "difference"
  9. `replace`
     - Before: "suggests"
     - After: "indicates"
  10. `replace`
     - Before: "climate"
     - After: "the"
  11. `replace`
     - Before: "knowledge"
     - After: "observed"
  12. `replace`
     - Before: "functions"
     - After: "association"
  13. `replace`
     - Before: "as"
     - After: "is"
  14. `replace`
     - Before: "a critical adaptive resource precisely"
     - After: "strongest"
  15. `replace`
     - Before: "buffering"
     - After: "knowledge-related"
  16. `replace`
     - Before: "effects"
     - After: "prediction differences"
  17. `replace`
     - Before: "protective"
     - After: "observed"
  18. `replace`
     - Before: "value"
     - After: "association"
  19. `replace`
     - Before: "of"
     - After: "varies"
  20. `replace`
     - Before: "climate knowledge is conditioned by"
     - After: "with"

### part-12

- Location: Discussion, concluding paragraph beginning 'Multi-hazard exposure and climate change knowledge'
- Reason: Replace independent-effect and cost-effectiveness claims with association and complementary-policy wording.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:26Z
- Author: anonymous
- Markup SHA-256 before: `584b54a1e86e10a092716c651e3b6c4b6ecb7b5936735078a55ab6072cd71571`
- Markup SHA-256 after: `b7a15ddb5da45935d03722fb329fbf28675a0377f35f8bab63ed7f5ea7c328eb`
- Revision IDs: `309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184526858321.reviewer-2-comment-2.part-12.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
Multi-hazard exposure and climate change knowledge are strong and independent predictors of household disease incidence in Nepal, with effects that are nonlinear and spatially heterogeneous. These findings underscore the need for disaster risk reduction strategies that account for cumulative hazard accumulation rather than isolated events, and support the integration of climate education into community health programmers as a cost-effective approach to building health resilience in hazard-prone populations across South and Southeast Asia.
~~~~

- After:

~~~~text
Multi-hazard exposure and climate change knowledge are associated with predicted household disease incidence in Nepal, with nonlinear and spatially heterogeneous patterns. These findings support disaster risk reduction strategies that account for cumulative hazard accumulation rather than isolated events and position targeted climate-health education as a potential complement to general education and structural health protection in hazard-prone populations.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "strong"
     - After: "associated"
  2. `replace`
     - Before: "and"
     - After: "with"
  3. `replace`
     - Before: "independent predictors of"
     - After: "predicted"
  4. `delete`
     - Before: "effects that are "
     - After: ""
  5. `insert`
     - Before: ""
     - After: " patterns"
  6. `replace`
     - Before: "underscore the need for"
     - After: "support"
  7. `delete`
     - Before: ","
     - After: ""
  8. `replace`
     - Before: "support"
     - After: "position"
  9. `replace`
     - Before: "the"
     - After: "targeted"
  10. `replace`
     - Before: "integration of climate"
     - After: "climate-health"
  11. `delete`
     - Before: " into community health programmers"
     - After: ""
  12. `replace`
     - Before: "cost-effective"
     - After: "potential"
  13. `replace`
     - Before: "approach"
     - After: "complement"
  14. `replace`
     - Before: "building"
     - After: "general education and structural"
  15. `replace`
     - Before: "resilience"
     - After: "protection"
  16. `delete`
     - Before: " across South and Southeast Asia"
     - After: ""

### part-13

- Location: Figure 6 caption
- Reason: Replace causal mediation wording with knowledge-status stratification wording.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:27Z
- Author: anonymous
- Markup SHA-256 before: `b7a15ddb5da45935d03722fb329fbf28675a0377f35f8bab63ed7f5ea7c328eb`
- Markup SHA-256 after: `1fb16c8d9a5288a2ed2bc96ed2bcb87bc8798e3229c2da54e97e9558d0227aa4`
- Revision IDs: `336, 337`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184527355016.reviewer-2-comment-2.part-13.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
Figure 6: Global Relationship between Natural Disaster Count and Disease Increase Probability Mediated by Climate Change Knowledge
~~~~

- After:

~~~~text
Figure 6: Global Relationship between Natural Disaster Count and Disease Increase Probability by Climate Change Knowledge Status
~~~~

- Minimal tracked fragments:
  1. `delete`
     - Before: "Mediated "
     - After: ""
  2. `insert`
     - Before: ""
     - After: " Status"

### part-14

- Location: Figure 7 caption
- Reason: Replace effect wording with prediction-difference wording.
- Kila decisions: KILA-D-20260828-018
- Mode: `replace`
- Timestamp: 2026-08-28T09:45:27Z
- Author: anonymous
- Markup SHA-256 before: `1fb16c8d9a5288a2ed2bc96ed2bcb87bc8798e3229c2da54e97e9558d0227aa4`
- Markup SHA-256 after: `c45b38a46b1fa6c9c50f51812880aae587188e71032d3f2016ee58f59b1e57d1`
- Revision IDs: `338, 339, 340`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260828T184527873427.reviewer-2-comment-2.part-14.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Before:

~~~~text
Figure 7: Spatial Heterogeneity of The Effects of Climate Change Knowledge
~~~~

- After:

~~~~text
Figure 7: Spatial Heterogeneity in Climate Change Knowledge Prediction Differences
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "of The Effects of"
     - After: "in"
  2. `insert`
     - Before: ""
     - After: " Prediction Differences"

## reviewer-3/comment-1

### part-01

- Location: Introduction, first paragraph, displacement and migration pathway discussion
- Reason: Present displacement as a proportionate general literature background pathway without implying that the study directly analyzes displaced or migrant populations.
- Kila decisions: KILA-D-20260829-001, KILA-D-20260829-002
- Mode: `replace`
- Timestamp: 2026-08-29T00:51:22Z
- Author: anonymous
- Markup SHA-256 before: `fdcecd4cb3960562fedf6664d9374cc522bee172a5c538f133bd14f39deb6c5b`
- Markup SHA-256 after: `3a689f7220d7c79ca3be6763d504e86e097568f70e87de33d8fd0420061d1e6a`
- Revision IDs: `353, 354, 355, 356`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T095122823973.reviewer-3-comment-1.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `6e3ceae216f0c4101517f9a6a20a8f53fe4c999a4d4cbef3a2f0c3b4acefa265`
- Formula verification: not applicable
- Before:

~~~~text
Extreme climate events also displace populations, disrupting livelihoods, fragmenting social networks, and placing additional pressure on already strained public health systems (Ali et al., 2026; Cai et al., 2024; Neira et al., 2023). Forced migration often results in overcrowded temporary settlements, inadequate sanitation infrastructure, and limited access to clean water and medical services, thereby increasing the risk of communicable disease transmission and long-term health deterioration.
~~~~

- After:

~~~~text
The broader literature also identifies population displacement and related disruptions to livelihoods and essential services as potential health pathways of extreme climate events (Ali et al., 2026; Cai et al., 2024; Neira et al., 2023).
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "Extreme"
     - After: "The broader literature also identifies population displacement and related disruptions to livelihoods and essential services as potential health pathways of extreme"
  2. `delete`
     - Before: " also displace populations, disrupting livelihoods, fragmenting social networks, and placing additional pressure on already strained public health systems"
     - After: ""
  3. `delete`
     - Before: " Forced migration often results in overcrowded temporary settlements, inadequate sanitation infrastructure, and limited access to clean water and medical services, thereby increasing the risk of communicable disease transmission and long-term health deterioration."
     - After: ""

## reviewer-3/comment-6

### part-01

- Location: Methods > Variables, paragraph beginning 'The model incorporates a comprehensive set'
- Reason: Report the implemented missing-data coding, the post-processing missingness rate, and whether XGBoost native missing-value routing was invoked.
- Kila decisions: KILA-D-20260829-005, KILA-D-20260829-006
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-29T05:22:32Z
- Author: anonymous
- Markup SHA-256 before: `b7d11cae405ae7dbe12f0c360aa5b861ddea2389bb058016fbcf7be73cfb4922`
- Markup SHA-256 after: `6d5f5b11e631202a9db517d18644ab8443687655593bdf9d18671d68e2d35253`
- Revision IDs: `362`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T142232690679.reviewer-3-comment-6.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
The model incorporates a comprehensive set of socio-demographic controls at both the individual and household levels. These controls help isolate the relationship between disaster exposure and health outcomes. Socio-demographic controls include respondent age, gender, literacy, and education; household demographic composition, such as shares of female, elderly, young, and educated members. Indicators of economic status cover residence ownership and type, asset ownership, agricultural land, access to communication and transportation assets, and distances to services, including the nearest health center. Spatial identifiers for province and ecological belt control for geographic and climatic heterogeneity. Descriptive statistics for all variables are reported in Table 1.
~~~~

- After:

~~~~text
The model incorporates a comprehensive set of socio-demographic controls at both the individual and household levels. These controls help isolate the relationship between disaster exposure and health outcomes. Socio-demographic controls include respondent age, gender, literacy, and education; household demographic composition, such as shares of female, elderly, young, and educated members. Indicators of economic status cover residence ownership and type, asset ownership, agricultural land, access to communication and transportation assets, and distances to services, including the nearest health center. Spatial identifiers for province and ecological belt control for geographic and climatic heterogeneity. Descriptive statistics for all variables are reported in Table 1. Missing values were handled during data preprocessing. Binary indicators followed an affirmative-only (‘yes-is-yes’) rule: a value of 1 was assigned only when the respondent explicitly reported the defining condition, while all other responses were coded as 0. For continuous variables, structurally inapplicable entries were assigned their logical zero value, such as agricultural-experience years for households without agricultural land, whereas observations with genuine missing continuous values were subject to complete-case exclusion. No applicable continuous values remained missing, so this step removed no observations. The final analytical dataset comprised 11,568 households with no missing values across the 64 predictors and outcome (0%); consequently, XGBoost’s native missing-value routing was not invoked.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " Missing values were handled during data preprocessing. Binary indicators followed an affirmative-only (‘yes-is-yes’) rule: a value of 1 was assigned only when the respondent explicitly reported the defining condition, while all other responses were coded as 0. For continuous variables, structurally inapplicable entries were assigned their logical zero value, such as agricultural-experience years for households without agricultural land, whereas observations with genuine missing continuous values were subject to complete-case exclusion. No applicable continuous values remained missing, so this step removed no observations. The final analytical dataset comprised 11,568 households with no missing values across the 64 predictors and outcome (0%); consequently, XGBoost’s native missing-value routing was not invoked."

## reviewer-2/comment-8

### part-01

- Location: Methods > Variables, paragraph beginning 'The model incorporates a comprehensive set of socio-demographic controls' (approved bundle part 1 of 8)
- Reason: Explicitly state that the pooled model controls survey-wave differences through a survey-year indicator.
- Kila decisions: KILA-D-20260829-008, KILA-D-20260829-009, KILA-D-20260829-010, KILA-D-20260829-011
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-29T07:00:52Z
- Author: Kila
- Markup SHA-256 before: `1bbd9ce5c5db1237b9d2db3f8d649784952b9863f89505556cd1537817cf87ee`
- Markup SHA-256 after: `98b6355f3cc0a664308e3058ff96cb9cc1f34a86d0c6d3dddb50d5f95491009a`
- Revision IDs: `363`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T160052961315.reviewer-2-comment-8.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
Spatial identifiers for province and ecological belt control for geographic and climatic heterogeneity.
~~~~

- After:

~~~~text
Spatial identifiers for province and ecological belt control for geographic and climatic heterogeneity. The pooled model also includes a survey-year indicator (2016 or 2022) to control for wave-level differences.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " The pooled model also includes a survey-year indicator (2016 or 2022) to control for wave-level differences."

### part-02

- Location: Methods > Analytical Framework, paragraph beginning 'The study uses XGBoost' (approved bundle part 2 of 8)
- Reason: Describe the same-specification wave-specific sensitivity analysis and the fixed-hyperparameter comparability design.
- Kila decisions: KILA-D-20260829-008, KILA-D-20260829-009, KILA-D-20260829-010, KILA-D-20260829-011
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-29T07:01:03Z
- Author: Kila
- Markup SHA-256 before: `98b6355f3cc0a664308e3058ff96cb9cc1f34a86d0c6d3dddb50d5f95491009a`
- Markup SHA-256 after: `119d7c14a1524d2757bfdebebade5a68f5476648ed8d869ee94504a4ce7d1ff2`
- Revision IDs: `364`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T160103759508.reviewer-2-comment-8.part-02.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
Hyperparameters were optimized via random search over 500 iterations, tuning learning rate, maximum tree depth, number of estimators, and subsampling ratios.
~~~~

- After:

~~~~text
Hyperparameters were optimized via random search over 500 iterations, tuning learning rate, maximum tree depth, number of estimators, and subsampling ratios. To examine whether pooling masks temporal differences, we fit the same XGBoost specification separately to the 2016 and 2022 samples. Survey year is omitted from these wave-specific models because it is constant within each wave; all other predictors, the pooled-model hyperparameters, and outcome-stratified 10-fold cross-validation are retained. Using common hyperparameters provides a direct same-specification comparison without wave-specific re-optimization.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " To examine whether pooling masks temporal differences, we fit the same XGBoost specification separately to the 2016 and 2022 samples. Survey year is omitted from these wave-specific models because it is constant within each wave; all other predictors, the pooled-model hyperparameters, and outcome-stratified 10-fold cross-validation are retained. Using common hyperparameters provides a direct same-specification comparison without wave-specific re-optimization."

### part-03

- Location: Results, paragraph beginning 'The XGBoost classification model identifies cumulative disaster exposure' (approved bundle part 3 of 8)
- Reason: Report concrete wave-specific AUC and accuracy values and show closely comparable predictive performance.
- Kila decisions: KILA-D-20260829-008, KILA-D-20260829-009, KILA-D-20260829-010, KILA-D-20260829-011
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-29T07:01:14Z
- Author: Kila
- Markup SHA-256 before: `119d7c14a1524d2757bfdebebade5a68f5476648ed8d869ee94504a4ce7d1ff2`
- Markup SHA-256 after: `1113889f04796c60e53682a569b2edbe3b32228415b4434dd4982a28f60efde1`
- Revision IDs: `365`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T160114377774.reviewer-2-comment-8.part-03.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
This performance outperformed the logistic regression baseline mean of 61.8% as listed in Supplementary Materials Table S2.
~~~~

- After:

~~~~text
This performance outperformed the logistic regression baseline mean of 61.8% as listed in Supplementary Materials Table S2. In the wave-specific sensitivity analysis, the 2016 and 2022 models achieve AUCs of 0.779 and 0.774 and accuracies of 70.89% and 72.36%, respectively, indicating closely comparable predictive performance.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " In the wave-specific sensitivity analysis, the 2016 and 2022 models achieve AUCs of 0.779 and 0.774 and accuracies of 70.89% and 72.36%, respectively, indicating closely comparable predictive performance."

## reviewer-2/comment-5

### part-01

- Location: Methods > Analytical Framework, paragraph beginning 'The study uses XGBoost' (approved integrated bundle part 1 of 4)
- Reason: Describe the fair ordinary-logistic comparator and explain the affirmative methodological rationale for retaining XGBoost with the prespecified high-dimensional covariate set.
- Kila decisions: KILA-D-20260829-013, KILA-D-20260829-014, KILA-D-20260829-016, KILA-D-20260829-017
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-29T12:40:32Z
- Author: Kila
- Markup SHA-256 before: `304137f448ee52ddc573c7f0fe763d14d7302945aae39d9a430fd354120d4317`
- Markup SHA-256 after: `f24f9631dbd3b97d95b4fdba7e970899f609637a1d964e61240a7ac8f1303805`
- Revision IDs: `379`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T214032908697.reviewer-2-comment-5.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
XGBoost captures nonlinear relationships and high-order interactions without imposing parametric constraints—properties particularly valuable when marginal effects may vary with exposure level, socioeconomic status, and geographic context (Chen & Guestrin, 2016; Li & Managi, 2025; Li et al., 2026).
~~~~

- After:

~~~~text
XGBoost captures nonlinear relationships and high-order interactions without imposing parametric constraints—properties particularly valuable when marginal effects may vary with exposure level, socioeconomic status, and geographic context (Chen & Guestrin, 2016; Li & Managi, 2025; Li et al., 2026). Unlike ordinary logistic regression, XGBoost does not require estimation of a full-rank coefficient vector, allowing the prespecified covariate set to be retained without outcome-driven variable screening; regularization and row and column subsampling constrain model complexity. For comparison, we fit an L2-penalized ordinary logistic regression using the same 11,568 households, outcome, 64 predictors, and outcome-stratified 10-fold splits, without added interaction terms. Detailed specifications and diagnostics are reported in Supplementary Materials Table S2.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " Unlike ordinary logistic regression, XGBoost does not require estimation of a full-rank coefficient vector, allowing the prespecified covariate set to be retained without outcome-driven variable screening; regularization and row and column subsampling constrain model complexity. For comparison, we fit an L2-penalized ordinary logistic regression using the same 11,568 households, outcome, 64 predictors, and outcome-stratified 10-fold splits, without added interaction terms. Detailed specifications and diagnostics are reported in Supplementary Materials Table S2."

### part-02

- Location: Results, paragraph beginning 'The XGBoost classification model identifies cumulative disaster exposure' (approved integrated bundle part 2 of 4)
- Reason: Replace the unqualified logistic-superiority statement with concrete same-fold performance and explicit non-convergence diagnostics.
- Kila decisions: KILA-D-20260829-013, KILA-D-20260829-014, KILA-D-20260829-016, KILA-D-20260829-017
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-29T12:40:44Z
- Author: Kila
- Markup SHA-256 before: `f24f9631dbd3b97d95b4fdba7e970899f609637a1d964e61240a7ac8f1303805`
- Markup SHA-256 after: `c5ea2422842f1bd273a0f4a036d8c9fddb03806a2239d5964d1d0b1616a24a0d`
- Revision IDs: `380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 394, 395`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T214044146361.reviewer-2-comment-5.part-02.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
This performance outperformed the logistic regression baseline mean of 61.8% as listed in Supplementary Materials Table S2.
~~~~

- After:

~~~~text
The corresponding out-of-fold AUC was 0.773. Under identical outcome-stratified 10-fold splits, the ordinary logistic regression yielded an AUC of 0.637 and accuracy of 61.87%, but reached its iteration limit with convergence warnings in all 10 folds. Increasing the maximum iterations from 100 to 5,000 did not resolve convergence; its performance values are therefore treated as diagnostic benchmarks, with full specifications and goodness-of-fit diagnostics reported in Supplementary Materials Table S2.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "This"
     - After: "The"
  2. `replace`
     - Before: "performance"
     - After: "corresponding"
  3. `replace`
     - Before: "outperformed"
     - After: "out-of-fold AUC was 0.773. Under identical outcome-stratified 10-fold splits,"
  4. `insert`
     - Before: ""
     - After: " ordinary"
  5. `replace`
     - Before: "baseline"
     - After: "yielded"
  6. `replace`
     - Before: "mean"
     - After: "an AUC of 0.637 and accuracy"
  7. `replace`
     - Before: "8"
     - After: "87"
  8. `insert`
     - Before: ""
     - After: ", but reached its iteration limit with convergence warnings in all 10 folds. Increasing the maximum iterations from 100 to 5,000 did not resolve convergence; its performance values are therefore treated"
  9. `replace`
     - Before: "listed"
     - After: "diagnostic benchmarks, with full specifications and goodness-of-fit diagnostics reported"

### part-01-reedit-01

- Location: Methods > Analytical Framework, sentence beginning 'For comparison, we fit an L2-penalized ordinary logistic regression'
- Reason: Keep the main-text comparator description concise while retaining the complete interaction-term specification in the Supplementary Materials.
- Kila decisions: KILA-D-20260829-019
- Mode: `reedit`
- Revises prior parts: reviewer-2/comment-5#part-01
- Timestamp: 2026-08-29T13:32:32Z
- Author: anonymous
- Markup SHA-256 before: `3bb236018dda151764aa9f0ae27603492bcc8fb4218fec9e96050afcc7227cd1`
- Markup SHA-256 after: `2fd9864f2402ae2fb91e74e41812d9c8dfa5eb3a7521b79a89d44145f8d7cea7`
- Revision IDs: `109`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T223232489578.reviewer-2-comment-5.part-01-reedit-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
For comparison, we fit an L2-penalized ordinary logistic regression using the same 11,568 households, outcome, 64 predictors, and outcome-stratified 10-fold splits, without added interaction terms.
~~~~

- After:

~~~~text
For comparison, we fit an L2-penalized ordinary logistic regression using the same 11,568 households, outcome, 64 predictors, and outcome-stratified 10-fold splits.
~~~~

- Minimal tracked fragments:
  1. `delete`
     - Before: ", without added interaction terms"
     - After: ""

## reviewer-2/comment-6

### part-01

- Location: Methods > Analytical Framework, paragraph beginning 'Feature importance is assessed using the gain metric' (approved bundle part 1 of 5)
- Reason: Add the held-out TreeSHAP calculation and define the direction of SHAP contributions alongside the existing gain metric.
- Kila decisions: KILA-D-20260829-015, KILA-D-20260829-021
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-29T14:02:58Z
- Author: anonymous
- Markup SHA-256 before: `2fd9864f2402ae2fb91e74e41812d9c8dfa5eb3a7521b79a89d44145f8d7cea7`
- Markup SHA-256 after: `9255729a7ba2dbebadf751a55bd5372b016ce8c8159b1b680839905b0f0f874a`
- Revision IDs: `400`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T230258728914.reviewer-2-comment-6.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
Feature importance is assessed using the gain metric.
~~~~

- After:

~~~~text
Feature importance is assessed using the gain metric. To complement this measure, exact TreeSHAP contributions are calculated for held-out observations in each validation fold and combined into one out-of-fold SHAP profile per household; positive and negative values indicate contributions toward higher and lower predicted log-odds, respectively.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " To complement this measure, exact TreeSHAP contributions are calculated for held-out observations in each validation fold and combined into one out-of-fold SHAP profile per household; positive and negative values indicate contributions toward higher and lower predicted log-odds, respectively."

### part-02b

- Location: Results, first paragraph, final sentence (approved bundle part 2 of 5, mechanically split 2 of 2)
- Reason: Replace the imprecise gain-only conclusion with the validated out-of-fold SHAP rankings, directions, and Supplementary Figures S3–S4 cross-reference.
- Kila decisions: KILA-D-20260829-015, KILA-D-20260829-021
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-29T14:04:03Z
- Author: anonymous
- Markup SHA-256 before: `9255729a7ba2dbebadf751a55bd5372b016ce8c8159b1b680839905b0f0f874a`
- Markup SHA-256 after: `4108f6c3748ac43e1fa8402bf6afc3ea728a5cea15ae7b8842dc1e561b301cfd`
- Revision IDs: `401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T230404071117.reviewer-2-comment-6.part-02b.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
The Natural Disaster Experience Indicator indicates a top driver of the model’s predictive performance.
~~~~

- After:

~~~~text
The out-of-fold TreeSHAP analysis ranks multi-hazard exposure count first and climate change knowledge second by mean absolute SHAP value (0.346 and 0.120, respectively; Supplementary Materials Figures S3–S4). Mean SHAP contributions for the climate-change knowledge indicator are 0.106 log-odds for no and −0.135 log-odds for yes; those for multi-hazard exposure increase from −0.534 in the lowest exposure quartile to 0.387 in the highest.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "Natural"
     - After: "out-of-fold"
  2. `replace`
     - Before: "Disaster"
     - After: "TreeSHAP"
  3. `replace`
     - Before: "Experience"
     - After: "analysis"
  4. `replace`
     - Before: "Indicator"
     - After: "ranks"
  5. `replace`
     - Before: "indicates"
     - After: "multi-hazard"
  6. `replace`
     - Before: "a"
     - After: "exposure"
  7. `replace`
     - Before: "top"
     - After: "count"
  8. `replace`
     - Before: "driver"
     - After: "first"
  9. `replace`
     - Before: "of"
     - After: "and climate change knowledge second by mean absolute SHAP value (0.346 and 0.120, respectively; Supplementary Materials Figures S3–S4). Mean SHAP contributions for"
  10. `replace`
     - Before: "model’s"
     - After: "climate-change"
  11. `replace`
     - Before: "predictive"
     - After: "knowledge"
  12. `replace`
     - Before: "performance"
     - After: "indicator are 0"
  13. `insert`
     - Before: ""
     - After: "106 log-odds for no and −0.135 log-odds for yes; those for multi-hazard exposure increase from −0.534 in the lowest exposure quartile to 0.387 in the highest."

## reviewer-2/comment-4

### part-01

- Location: Methods > Analytical Framework, paragraph beginning 'We use the XGBoost machine learning algorithm'
- Reason: Replace the generic train-test description with the approved outcome-stratified ten-fold out-of-fold evaluation design, class-balance handling, threshold, metric set, and validation boundary.
- Kila decisions: KILA-D-20260829-016, KILA-D-20260829-023
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-29T14:44:41Z
- Author: Kila
- Markup SHA-256 before: `b95a866b28e29e36b40b68571985cfa4d995daf8586a29058e9bb3c3af9a8191`
- Markup SHA-256 after: `0c604564dbf046e6df348f1ef717a676f17fe380a74b1009276df08a84e23e1d`
- Revision IDs: `431, 432`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T234441330443.reviewer-2-comment-4.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
The model minimizes a binary cross-entropy loss function and is trained on a reproducible train-test split. Hyperparameters were optimized via random search over 500 iterations, tuning learning rate, maximum tree depth, number of estimators, and subsampling ratios.
~~~~

- After:

~~~~text
The model minimizes a binary cross-entropy loss function. Hyperparameters were optimized via random search over 500 iterations, tuning learning rate, maximum tree depth, number of estimators, and subsampling ratios. Using the selected hyperparameters, predictive performance is evaluated through outcome-stratified 10-fold cross-validation with shuffling (seed 42). Each fold uses 90% of the sample for training and 10% for held-out testing, so every household contributes one out-of-fold prediction. The outcome prevalence is 39.17%, and stratification preserves this distribution across folds; no over- or undersampling or class weighting is applied. Threshold-based metrics use a probability threshold of 0.5. We report AUC, accuracy, balanced accuracy, sensitivity/recall, specificity, precision, F1 score, Brier score, and log loss. These estimates represent cross-validated out-of-fold performance rather than independent external validation.
~~~~

- Minimal tracked fragments:
  1. `delete`
     - Before: " and is trained on a reproducible train-test split"
     - After: ""
  2. `insert`
     - Before: ""
     - After: " Using the selected hyperparameters, predictive performance is evaluated through outcome-stratified 10-fold cross-validation with shuffling (seed 42). Each fold uses 90% of the sample for training and 10% for held-out testing, so every household contributes one out-of-fold prediction. The outcome prevalence is 39.17%, and stratification preserves this distribution across folds; no over- or undersampling or class weighting is applied. Threshold-based metrics use a probability threshold of 0.5. We report AUC, accuracy, balanced accuracy, sensitivity/recall, specificity, precision, F1 score, Brier score, and log loss. These estimates represent cross-validated out-of-fold performance rather than independent external validation."

### part-02

- Location: Results, paragraph beginning 'The XGBoost classification model identifies cumulative disaster exposure'
- Reason: Replace the accuracy-only statement with the approved complete out-of-fold performance metrics; the human performed this part in Word after the machine dry-run was blocked by incompatible run styles.
- Kila decisions: KILA-D-20260829-016, KILA-D-20260829-023, KILA-D-20260830-001
- Mode: `human-manual-replace`
- Revises prior parts: none
- Timestamp: 2026-08-30T09:01:52+09:00
- Author: Jie MI
- Markup SHA-256 before: `0c604564dbf046e6df348f1ef717a676f17fe380a74b1009276df08a84e23e1d`
- Markup SHA-256 after: `c78de5e7e3ac1ae8f99c3ccf26f0f6a1443ccf566dd2809c263f6f731168a7ce`
- Revision IDs: `128, 129`
- Recovery source: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260829T234441330443.reviewer-2-comment-4.part-01.docx` (the Results target is unchanged in this pre-part-01 backup)
- Paragraph properties preserved: `true`
- Run style verification: deferred to consolidated fresh-clean visual review
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
The XGBoost model achieved a mean classification accuracy of 71·4% with a standard deviation of 0.014.
~~~~

- After:

~~~~text
Across the out-of-fold predictions, the XGBoost model achieved an accuracy of 71.46%, balanced accuracy of 67.55%, sensitivity/recall of 49.55%, specificity of 85.56%, precision of 68.84%, and F1 score of 57.62%; the Brier score was 0.187 and log loss was 0.551.
~~~~

- Minimal tracked fragments:
  1. `replace`
     - Before: "The XGBoost model achieved a mean classification accuracy of 71·4% with a standard deviation of 0.014."
     - After: "Across the out-of-fold predictions, the XGBoost model achieved an accuracy of 71.46%, balanced accuracy of 67.55%, sensitivity/recall of 49.55%, specificity of 85.56%, precision of 68.84%, and F1 score of 57.62%; the Brier score was 0.187 and log loss was 0.551."

## reviewer-1/comment-4

### part-01

- Location: Methods > Analytical framework, sentence beginning 'Spatial aggregation across province-ecological belt units'
- Reason: Clarify that the spatial maps use the Government of Nepal administrative boundary while the ecological-belt layer is author-derived from JAXA global DSM elevation data.
- Kila decisions: KILA-D-20260830-011, KILA-D-20260830-012, KILA-D-20260830-013
- Mode: `replace`
- Revises prior parts: none
- Timestamp: 2026-08-30T07:05:40Z
- Author: Kila
- Markup SHA-256 before: `705e3f935946edb3266ba77cbb8259fbaf3717d96b5c3cc8b1c964a90a07a09c`
- Markup SHA-256 after: `a4d782632ae5539d8bf6df864aec554033f84c3cd174cddf4ec399a313d7f9c6`
- Revision IDs: `458`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260830T160540678130.reviewer-1-comment-4.part-01.docx`
- Paragraph properties preserved: `true`
- Run style source SHA-256: `4dc24241e26ab6183f0b6161c94260c36032c814a98259890184cf3be04ecae5`
- Formula verification: not applicable
- Endnote hyperlinks preserved: `true`
- Endnote hyperlink count: `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
Spatial aggregation across province-ecological belt units and socioeconomic subgroup analyses characterize heterogeneity in disaster-related health risks.
~~~~

- After:

~~~~text
Spatial aggregation across province-ecological belt units and socioeconomic subgroup analyses characterize heterogeneity in disaster-related health risks. The spatial maps use the Government of Nepal administrative-boundary shapefile. Ecological-belt boundaries (Mountain, Hill, and Terai) are derived by the authors through elevation-based classification of JAXA global DSM data because an official EcoBelt vector layer was not available from the Government of Nepal.
~~~~

- Minimal tracked fragments:
  1. `insert`
     - Before: ""
     - After: " The spatial maps use the Government of Nepal administrative-boundary shapefile. Ecological-belt boundaries (Mountain, Hill, and Terai) are derived by the authors through elevation-based classification of JAXA global DSM data because an official EcoBelt vector layer was not available from the Government of Nepal."

## reviewer-1/comment-1

### part-01

- Location: Research in Context > Implications, final sentence
- Reason: Distinguish the questions addressed by longitudinal, interrupted time-series or other natural/quasi-experimental, and cluster-randomised controlled designs.
- Kila decisions: `KILA-D-20260830-019`, `KILA-D-20260830-020`
- Mode: `human-manual-replace`
- Revises prior parts: none
- Timestamp: 2026-08-30T17:22:57+09:00
- Author: Jie MI
- Markup SHA-256 before human save: `2f20c5b17b071b942eb32b1029426ae49684333bc0450a098da1a28cdff12bd8`
- Markup SHA-256 after final human save: `a83634d56501ec56104f8e4bfb62434c1c48b5ed89eef90d2f57ef5071e0d29e`
- Revision IDs created by the human save in this paragraph: `39, 40`
- Backup: human Word edit; no new machine backup was created
- Paragraph properties preserved: verified in fresh clean and rendered markup
- Run style verification: verified in fresh clean and rendered markup
- Formula verification: not applicable
- Endnote hyperlinks preserved: effective state unchanged; `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
Future longitudinal and quasi-experimental studies are required to clarify causal pathways and test whether community-based climate-health education can reduce disaster-related disease burden.
~~~~

- After:

~~~~text
Future research should match study design to the question. Longitudinal studies can establish temporal ordering, while natural or quasi-experimental designs, including interrupted time-series analyses, can evaluate policy or programme implementation. Where feasible and ethically appropriate, cluster-randomised controlled trials can test whether community-based climate-health education improves preparedness practices and health outcomes.
~~~~

- Minimal tracked fragments:
  1. `human-manual-replace`
     - Before: "Future longitudinal and quasi-experimental studies are required to clarify causal pathways and test whether community-based climate-health education can reduce disaster-related disease burden."
     - After: "Future research should match study design to the question. Longitudinal studies can establish temporal ordering, while natural or quasi-experimental designs, including interrupted time-series analyses, can evaluate policy or programme implementation. Where feasible and ethically appropriate, cluster-randomised controlled trials can test whether community-based climate-health education improves preparedness practices and health outcomes."

### part-02

- Location: Discussion > limitations paragraph, sentence beginning 'Fifth, the XGBoost framework'
- Reason: Explain how future designs address temporal ordering, policy or programme implementation, and intervention effects, while preserving the separate spatial-dependence limitation.
- Kila decisions: `KILA-D-20260830-019`, `KILA-D-20260830-020`
- Mode: `human-manual-replace`
- Revises prior parts: `reviewer-3/comment-3#part-05`, `reviewer-2/comment-7#part-02`
- Timestamp: 2026-08-30T17:22:57+09:00
- Author: Jie MI
- Markup SHA-256 before human save: `2f20c5b17b071b942eb32b1029426ae49684333bc0450a098da1a28cdff12bd8`
- Markup SHA-256 after final human save: `a83634d56501ec56104f8e4bfb62434c1c48b5ed89eef90d2f57ef5071e0d29e`
- Human revision IDs created or updated in this paragraph: `364–366, 368–406`
- Correction history: the first human save removed the following spatial-dependence sentence; the final human save restored it before fresh-clean approval
- Backup: human Word edit; no new machine backup was created
- Paragraph properties preserved: verified in fresh clean and rendered markup
- Run style verification: verified in fresh clean and rendered markup
- Formula verification: not applicable
- Endnote hyperlinks preserved: effective state unchanged; `0`
- Endnote hyperlink XML SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- Endnote relationships SHA-256: `absent`
- Before:

~~~~text
Fifth, the XGBoost framework does not address potential endogeneity between climate knowledge and health outcomes; longitudinal or quasi-experimental designs would strengthen causal identification. Spatial dependence and spillover effects are also not explicitly modelled, and geographically weighted approaches could further illuminate regional drivers of climate-health vulnerability.
~~~~

- After:

~~~~text
Fifth, the XGBoost framework does not address potential endogeneity between climate knowledge and health outcomes. Future research should use designs matched to the question, including longitudinal studies to establish temporal ordering, interrupted time-series analyses or other natural or quasi-experimental approaches to evaluate policy or programme implementation, and, where feasible and ethically appropriate, cluster-randomised controlled trials to test the effects of community-based climate-health education on preparedness practices and health outcomes. Spatial dependence and spillover effects are also not explicitly modelled, and geographically weighted approaches could further illuminate regional drivers of climate-health vulnerability.
~~~~

- Minimal tracked fragments:
  1. `human-manual-replace`
     - Before: "Fifth, the XGBoost framework does not address potential endogeneity between climate knowledge and health outcomes; longitudinal or quasi-experimental designs would strengthen causal identification."
     - After: "Fifth, the XGBoost framework does not address potential endogeneity between climate knowledge and health outcomes. Future research should use designs matched to the question, including longitudinal studies to establish temporal ordering, interrupted time-series analyses or other natural or quasi-experimental approaches to evaluate policy or programme implementation, and, where feasible and ethically appropriate, cluster-randomised controlled trials to test the effects of community-based climate-health education on preparedness practices and health outcomes."
  2. `preserve-after-correction`
     - Text: "Spatial dependence and spillover effects are also not explicitly modelled, and geographically weighted approaches could further illuminate regional drivers of climate-health vulnerability."

## reviewer-1/comment-6

### part-01

- Location: Discussion, between the nonlinear multi-hazard interpretation and the climate-change-knowledge interpretation
- Reason: Add Nepal-specific epidemiological, entomological, synthesis, and modelling evidence for diarrhoeal and vector-borne disease patterns requested by the reviewer while distinguishing the analytical scales from the present household outcome.
- Kila decisions: `KILA-D-20260830-022`, `KILA-D-20260830-023`, `KILA-D-20260830-024`, `KILA-D-20260830-025`
- Mode: `human-manual-insert-with-endnote-fields`
- Revises prior parts: none
- Timestamp: 2026-08-30T18:34:49+09:00
- Author: Jie MI
- Markup SHA-256 before human implementation: `a83634d56501ec56104f8e4bfb62434c1c48b5ed89eef90d2f57ef5071e0d29e`
- Markup SHA-256 after final EndNote update: `4f72ecb292e5dd116ecbc144ba176bd39432e2bd1bc0a9ca493aabad5b25e1de`
- Revision IDs: `282` for the inserted Discussion prose; EndNote citation fields are embedded within the inserted paragraph
- Backup: human Word edit; no new machine backup was created
- Paragraph properties preserved: verified in fresh clean and rendered markup
- Run style verification: verified in fresh clean and rendered markup
- Formula verification: not applicable
- EndNote fields: final markup and clean each contain 60 `ADDIN EN.CITE` fields and one `ADDIN EN.REFLIST` field
- Before:

~~~~text
[No Nepal-specific disease-evidence paragraph at this location.]
~~~~

- After:

~~~~text
The reported household disease pattern is also consistent with Nepal-specific epidemiological and entomological evidence on climate-sensitive diseases. A national ecological time-series analysis found that childhood diarrhoeal incidence increased by 4.4% per 1 °C increase in mean temperature and by 0.28% per 1 cm increase in rainfall, with the largest estimated effects in mountain regions (Dhimal et al., 2022). Field studies documented vectors of dengue and lymphatic filariasis across Nepal’s elevation gradient and showed that temperature, rainfall, and relative humidity predicted vector abundance (Dhimal et al., 2015; Dhimal et al., 2014). A systematic synthesis of the Hindu Kush Himalayan region and recent Nepal-wide modelling further indicate expansion of vector-borne disease risk into highland areas and longer periods of thermal suitability for dengue in the mid-hills and major urban centres (Acharya et al., 2025; Dhimal et al., 2021). Although these studies use disease-specific outcomes at different analytical scales, they provide Nepal-specific scientific context for the perceived household disease changes identified in our analysis.
~~~~

- Minimal tracked fragments:
  1. `human-manual-insert`
     - Before: ""
     - After: complete paragraph shown above

### part-02

- Location: References, Acharya et al. (2025)
- Reason: Add the Nepal-wide dengue thermal-suitability study cited in the new Discussion paragraph.
- Kila decisions: `KILA-D-20260830-022`, `KILA-D-20260830-023`, `KILA-D-20260830-024`, `KILA-D-20260830-025`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T18:34:49+09:00
- Markup SHA-256 before human implementation: `a83634d56501ec56104f8e4bfb62434c1c48b5ed89eef90d2f57ef5071e0d29e`
- Markup SHA-256 after final EndNote update: `4f72ecb292e5dd116ecbc144ba176bd39432e2bd1bc0a9ca493aabad5b25e1de`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate `w:ins` wrapper
- Backup: human Word/EndNote edit; no new machine backup was created
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Acharya, B. K., Khanal, L., & Dhimal, M. (2025). Increased thermal suitability elevates the risk of dengue transmission across the mid hills of Nepal. PLOS ONE, 20(4), e0322031. https://doi.org/10.1371/journal.pone.0322031
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

## reviewer-2/comment-10

### part-01

- Location: Research in Context—Implications, sentence beginning “Public health and disaster risk reduction strategies...”
- Reason: Present climate-health education as a component to be evaluated alongside structural investments rather than as a programme that should already be adopted on the basis of the observational findings.
- Kila decisions: `KILA-D-20260830-032`
- Mode: `human-manual-replace`
- Revises prior parts: none at the target sentence; the same paragraph also contains text revised for `reviewer-1/comment-1`
- Timestamp: 2026-08-30T23:41:01+09:00
- Markup SHA-256 before human implementation: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Markup SHA-256 after final human save: `2b6778b677a9a9aa19d4ff241c676a2f2581c792f890d27da6e48b8d35fef7f3`
- Fresh clean SHA-256: `0f90b1bea04fc341194e89db8ff163c4129f4579e88e5246b4d5a98b2b3bcaec`
- Revision IDs: human Word tracked change; verified in markup and accepted fresh clean
- Backup: human Word edit; no new machine backup was created
- Before:

~~~~text
Public health and disaster risk reduction strategies in Nepal and similar hazard-prone settings should therefore combine climate-health education with targeted investments in health systems, water and sanitation, and local preparedness.
~~~~

- After:

~~~~text
Public health and disaster risk reduction strategies in Nepal and similar hazard-prone settings could therefore evaluate climate-health education as one component alongside targeted investments in health systems, water and sanitation, and local preparedness.
~~~~

- Minimal tracked fragments: `human-manual-replace`

### part-02

- Location: Discussion, recommendation paragraph beginning “Within these policy frameworks”
- Reason: Make the education recommendation conditional and complementary, and use Segala et al. (2024) only as contextual evidence about climate-health knowledge and curricular gaps among young health professionals.
- Kila decisions: `KILA-D-20260830-032`
- Mode: `human-manual-replace-with-endnote-field`
- Revises prior parts: `reviewer-1/comment-5#part-02`, including its disclosed overlaps with `reviewer-3/comment-3#part-04` and `reviewer-3/comment-5#part-03a`–`part-03d`
- Timestamp: 2026-08-30T23:41:01+09:00
- Markup SHA-256 before human implementation: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Markup SHA-256 after final human save: `2b6778b677a9a9aa19d4ff241c676a2f2581c792f890d27da6e48b8d35fef7f3`
- Fresh clean SHA-256: `0f90b1bea04fc341194e89db8ff163c4129f4579e88e5246b4d5a98b2b3bcaec`
- Revision IDs: human Word tracked change plus EndNote `ADDIN EN.CITE` field; verified in markup and accepted fresh clean
- Backup: human Word/EndNote edit; no new machine backup was created
- Before:

~~~~text
Simultaneously, targeted climate-health education delivered through community health programmes may complement general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach reinforces rather than replaces general education and structural health protection.
~~~~

- After:

~~~~text
Simultaneously, targeted climate-health education delivered through community health programmes  could be evaluated as a complement to general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach would reinforce rather than replace general education and structural health protection. An Italian national survey documenting climate-health knowledge gaps and limited curricular provision among young doctors and medical students provides additional context for health-workforce education (Segala et al., 2024).
~~~~

- Minimal tracked fragments: `human-manual-replace-with-endnote-field`

### part-03

- Location: References, Segala et al. (2024)
- Reason: Add the reviewer-suggested medical-education source cited in the revised Discussion.
- Kila decisions: `KILA-D-20260830-032`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T23:41:01+09:00
- Markup SHA-256 before human implementation: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Markup SHA-256 after final human save: `2b6778b677a9a9aa19d4ff241c676a2f2581c792f890d27da6e48b8d35fef7f3`
- Fresh clean SHA-256: `0f90b1bea04fc341194e89db8ff163c4129f4579e88e5246b4d5a98b2b3bcaec`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate `w:ins` wrapper
- Backup: human Word/EndNote edit; no new machine backup was created
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Segala, F. V., Di Gennaro, F., Giannini, L. A. A., Stroffolini, G., Colpani, A., De Vito, A., Di Gregorio, S., Frallonardo, L., Guido, G., Novara, R., Amendolara, A., Ritacco, I. A., Ferrante, F., Masini, L., Iannetti, I., Mazzeo, S., Marello, S., Veronese, N., Gobbi, F., . . . Saracino, A. (2024). Perspectives on climate action and the changing burden of infectious diseases among young Italian doctors and students: a national survey [Original Research]. Frontiers in Public Health, Volume 12 - 2024. https://doi.org/10.3389/fpubh.2024.1382505
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

## reviewer-1/comment-7

### part-01

- Location: Discussion, paragraph beginning "Nepal-specific research places"
- Reason: Add distinct Nepal-specific evidence on climate-risk perception, protection motivation, the perception–adaptation gap, adaptation constraints, livelihood assets, and ecological heterogeneity without duplicating the disease evidence or official policy documents added for preceding comments.
- Kila decisions: `KILA-D-20260830-029`, `KILA-D-20260830-030`
- Mode: `human-manual-insert-with-endnote-fields`
- Revises prior parts: none
- Timestamp: 2026-08-30T22:32:28+09:00
- Author: Jie MI
- Markup SHA-256 before human implementation: `8dfc5bddba9770525ca9d727babc206077d48b60125593c1f39f2eadd4fac474`
- Markup SHA-256 after final EndNote correction: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Revision IDs: human-created tracked insertion retained and visually verified in the rendered markup
- Backup: human Word/EndNote edit; no new machine backup was created
- Paragraph properties preserved: verified in fresh clean and rendered markup
- Run style verification: verified in fresh clean and rendered markup
- Formula verification: not applicable
- Before:

~~~~text
[No separate Nepal-specific risk-perception and adaptive-capacity paragraph at this location.]
~~~~

- After:

~~~~text
Nepal-specific research places these knowledge-related and geographic patterns in a broader context of risk perception and adaptive capacity. Studies across central Nepal and the Khumbu region show that perceptions of climate change and its health and environmental impacts vary by elevation and by perceived vulnerability, efficacy, and response costs  (Phuyal et al., 2025; Poudyal et al., 2021). Although more than 80% of surveyed households in the Koshi River Basin perceived climatic changes, only 32% reported agricultural adaptation (Hussain et al., 2018); studies elsewhere in Nepal similarly identify financial, informational, agency, and institutional constraints on adaptation  (Choquette-Levy et al., 2025; Gurung et al., 2021). Together with evidence that livelihood assets shape household vulnerability across ecological settings (Pandey & Bardsley, 2015), these findings support interpreting the present knowledge-related contrasts as context dependent rather than equating awareness with preparedness or adaptive action.
~~~~

- Minimal tracked fragments: `human-manual-insert-with-endnote-fields`; final Hussain citation is attached directly to the 32% adaptation statistic
- EndNote hyperlink guard: markup and fresh clean each contain 69 `ADDIN EN.CITE` fields and 55 hyperlinks

### part-02

- Location: References, Choquette-Levy et al. (2025)
- Reason: Add the Nepal subsistence-farming study cited as evidence on constraints shaping adaptation.
- Kila decisions: `KILA-D-20260830-029`, `KILA-D-20260830-030`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T22:32:28+09:00
- Markup SHA-256 after final EndNote correction: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate revision wrapper required
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Choquette-Levy, N., Ghimire, D., Oppenheimer, M., Ghimire, R., & Ck, D. (2025). Retrenchment under climate-driven risks in subsistence farming communities. Population and Environment, 47(2), 22. https://doi.org/10.1007/s11111-025-00493-8
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-03

- Location: References, Gurung et al. (2021)
- Reason: Add the Nepalese Himalaya adaptation study cited as evidence on informational and institutional constraints.
- Kila decisions: `KILA-D-20260830-029`, `KILA-D-20260830-030`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T22:32:28+09:00
- Markup SHA-256 after final EndNote correction: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate revision wrapper required
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Gurung, L. J., Miller, K. K., Venn, S., & Bryan, B. A. (2021). Climate change adaptation for managing non-timber forest products in the Nepalese Himalaya. Science of The Total Environment, 796, 148853. https://doi.org/https://doi.org/10.1016/j.scitotenv.2021.148853
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-04

- Location: References, Hussain et al. (2018)
- Reason: Add the Koshi River Basin study cited for the documented gap between perceived climatic change and reported agricultural adaptation.
- Kila decisions: `KILA-D-20260830-029`, `KILA-D-20260830-030`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T22:32:28+09:00
- Markup SHA-256 after final EndNote correction: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate revision wrapper required
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Hussain, A., Rasul, G., Mahapatra, B., Wahid, S., & Tuladhar, S. (2018). Climate change-induced hazards and local adaptations in agriculture: a study from Koshi River Basin, Nepal. Natural Hazards, 91(3), 1365-1383. https://doi.org/10.1007/s11069-018-3187-1
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-05

- Location: References, Pandey and Bardsley (2015)
- Reason: Add the Nepali Himalaya study cited for livelihood-asset and ecological-setting variation in household vulnerability.
- Kila decisions: `KILA-D-20260830-029`, `KILA-D-20260830-030`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T22:32:28+09:00
- Markup SHA-256 after final EndNote correction: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate revision wrapper required
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Pandey, R., & Bardsley, D. K. (2015). Social-ecological vulnerability to climate change in the Nepali Himalaya. Applied Geography, 64, 74-86. https://doi.org/https://doi.org/10.1016/j.apgeog.2015.09.008
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-06

- Location: References, Phuyal et al. (2025)
- Reason: Add the central Nepal climate-perception study cited for elevation-related variation in perceived climate, health, and environmental impacts.
- Kila decisions: `KILA-D-20260830-029`, `KILA-D-20260830-030`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T22:32:28+09:00
- Markup SHA-256 after final EndNote correction: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate revision wrapper required
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Phuyal, P., Kramer, I. M., Kadel, I., Wouters, E., Magdeburg, A., Groneberg, D. A., Kuch, U., Ahrens, B., Dhimal, M. L., Dhimal, M., & Müller, R. (2025). On people’s perceptions of climate change and its impacts in a hotspot of global warming. PLOS ONE, 20(2), e0317786. https://doi.org/10.1371/journal.pone.0317786
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-07

- Location: References, Poudyal et al. (2021)
- Reason: Add the Khumbu-region study cited for heterogeneity in risk perception and protection motivation.
- Kila decisions: `KILA-D-20260830-029`, `KILA-D-20260830-030`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T22:32:28+09:00
- Markup SHA-256 after final EndNote correction: `18af012475d28795fd3283065bc39a41ba322370eb21f0a5602418c65a2b4216`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate revision wrapper required
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Poudyal, N. C., Joshi, O., Hodges, D. G., Bhandari, H., & Bhattarai, P. (2021). Climate change, risk perception, and protection motivation among high-altitude residents of the Mt. Everest region in Nepal. Ambio, 50(2), 505-518. https://doi.org/10.1007/s13280-020-01369-x
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

## reviewer-1/comment-5

### part-01

- Location: Discussion, policy-linkage paragraph beginning "These findings have policy relevance for targeting and programme design"
- Reason: Link the study's cumulative-exposure, spatial-heterogeneity, and household-human-capital findings directly to relevant priorities in Nepal's NAP 2021–2050, HNAP 2023–2030, and NDC 3.0.
- Kila decisions: `KILA-D-20260830-027`
- Mode: `human-manual-restructure-with-endnote-fields`
- Revises prior parts: `reviewer-3/comment-3#part-04`, `reviewer-3/comment-3#part-04-reapply-01`, and the policy wording subsequently revised for `reviewer-3/comment-5`
- Timestamp: 2026-08-30T20:37:23+09:00
- Author: Jie MI
- Markup SHA-256 before initial human implementation: `4f72ecb292e5dd116ecbc144ba176bd39432e2bd1bc0a9ca493aabad5b25e1de`
- Markup SHA-256 after final human correction: `8dfc5bddba9770525ca9d727babc206077d48b60125593c1f39f2eadd4fac474`
- Revision IDs: human-created tracked revisions retained and visually verified in the rendered markup
- Backup: human Word/EndNote edit; no new machine backup was created
- Paragraph properties preserved: verified in fresh clean and rendered markup
- Run style verification: verified in fresh clean and rendered markup
- Formula verification: not applicable
- EndNote fields: final markup and clean each contain 63 `ADDIN EN.CITE` fields and one `ADDIN EN.REFLIST` field; both contain 49 hyperlink elements
- Before:

~~~~text
[No separate Nepal policy-linkage paragraph at this location.]
~~~~

- After:

~~~~text
These findings have policy relevance for targeting and programme design. The nonlinear increase in predicted disease probability as households accumulate distinct hazard types is directly relevant to the National Adaptation Plan (NAP) 2021–2050 and Nationally Determined Contribution (NDC) 3.0 priorities for multi-hazard early-warning systems and emergency preparedness, because it indicates that monitoring and response planning should consider cumulative exposure rather than isolated events (Government of Nepal, 2021, 2025). The spatial heterogeneity across province–ecological belt units is likewise relevant to the NAP and Health National Adaptation Plan (HNAP) 2023–2030 priorities for climate-sensitive disease surveillance, climate-resilient health infrastructure, and water, sanitation, and hygiene services (Government of Nepal, 2021, 2023). The larger knowledge-related prediction contrast among households with lower literacy and education ratios adds a population-targeting dimension to the HNAP’s public-awareness and capacity-building priorities and NDC 3.0’s health-workforce training agenda, indicating where community-level climate-health communication may warrant greater emphasis (Government of Nepal, 2023, 2025).
~~~~

- Minimal tracked fragments: `human-manual-restructure-with-endnote-fields`

### part-02

- Location: Discussion, recommendation paragraph beginning "Within these policy frameworks"
- Reason: Separate the manuscript's recommendations from the preceding policy-linkage paragraph while retaining earlier reviewer-approved infrastructure, education, prospective-evaluation, and multi-hazard recommendations and replacing the absolute prospective-evaluation wording with a cautious formulation.
- Kila decisions: `KILA-D-20260830-027`
- Mode: `human-manual-restructure`
- Revises prior parts: `reviewer-3/comment-3#part-04`, `reviewer-3/comment-3#part-04-reapply-01`, and the policy wording subsequently revised for `reviewer-3/comment-5`
- Timestamp: 2026-08-30T20:37:23+09:00
- Author: Jie MI
- Markup SHA-256 before latest human correction: `da481be3504018db73b28e69686ae97cfc6ead7bd234bd392af3d71fe542f1de`
- Markup SHA-256 after final human correction: `8dfc5bddba9770525ca9d727babc206077d48b60125593c1f39f2eadd4fac474`
- Revision IDs: human-created tracked revisions retained and visually verified in the rendered markup
- Backup: human Word edit; no new machine backup was created
- Paragraph properties preserved: verified in fresh clean and rendered markup
- Run style verification: verified in fresh clean and rendered markup
- Formula verification: not applicable
- Before:

~~~~text
These findings have policy relevance for targeting and programme design. Infrastructure reinforcement and medical resource allocation should be prioritised in high-risk province-ecological belt units where multi-hazard intensity and disease burden coincide, particularly in the Terai and Hill regions. Simultaneously, targeted climate-health education delivered through community health programmes may complement general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach reinforces rather than replaces general education and structural health protection. Because the present survey measures climate change knowledge, reported disaster exposure, and reported disease change in the same interview, such programmes should be accompanied by prospective evaluation that separately measures climate-health knowledge, preparedness practices, and independently assessed health outcomes. This would clarify how information translates into action and health benefits. Disaster risk reduction frameworks should move beyond single-hazard approaches to address the cumulative and nonlinear nature of multi-hazard health threats (Erdem Okumus, 2025; O’Donnell & Sovacool, 2026).
~~~~

- After:

~~~~text
Within these policy frameworks, infrastructure reinforcement and medical resource allocation should be prioritised in high-risk province-ecological belt units where multi-hazard intensity and disease burden coincide, particularly in the Terai and Hill regions. Simultaneously, targeted climate-health education delivered through community health programmes may complement general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach reinforces rather than replaces general education and structural health protection. Given that the present study measures climate change knowledge, disaster exposure, and household disease change in the same interview, prospective evaluations of such programmes could use separate measures of climate-health knowledge and preparedness practices alongside independently assessed health outcomes. This would clarify how information translates into action and health benefits. Disaster risk reduction frameworks should move beyond single-hazard approaches to address the cumulative and nonlinear nature of multi-hazard health threats  (Erdem Okumus, 2025; O’Donnell & Sovacool, 2026).
~~~~

- Minimal tracked fragments: `human-manual-restructure`; final cautious prospective-evaluation wording occurs once in the fresh clean manuscript

### part-03

- Location: References, Government of Nepal (2021)
- Reason: Add the official National Adaptation Plan cited in the revised Discussion.
- Kila decisions: `KILA-D-20260830-027`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T20:37:23+09:00
- Markup SHA-256 after final human correction: `8dfc5bddba9770525ca9d727babc206077d48b60125593c1f39f2eadd4fac474`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate revision wrapper required
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Government of Nepal. (2021). National Adaptation Plan of Nepal.
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-04

- Location: References, Government of Nepal (2023)
- Reason: Add the official Climate Change Health Adaptation Strategy and Action Plan cited in the revised Discussion.
- Kila decisions: `KILA-D-20260830-027`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T20:37:23+09:00
- Markup SHA-256 after final human correction: `8dfc5bddba9770525ca9d727babc206077d48b60125593c1f39f2eadd4fac474`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate revision wrapper required
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Government of Nepal. (2023). Climate Change Health Adaptation Strategy and Action Plan (2023-2030).
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-05

- Location: References, Government of Nepal (2025)
- Reason: Add Nepal's official Nationally Determined Contribution 3.0 cited in the revised Discussion.
- Kila decisions: `KILA-D-20260830-027`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T20:37:23+09:00
- Markup SHA-256 after final human correction: `8dfc5bddba9770525ca9d727babc206077d48b60125593c1f39f2eadd4fac474`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate revision wrapper required
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Government of Nepal. (2025). Nepal's Nationally Determined Contribution (NDC) 3.0.
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

## reviewer-1/comment-6

### part-03

- Location: References, Dhimal et al. (2022)
- Reason: Add the national ecological childhood-diarrhoeal study cited in the new Discussion paragraph.
- Kila decisions: `KILA-D-20260830-022`, `KILA-D-20260830-023`, `KILA-D-20260830-024`, `KILA-D-20260830-025`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T18:34:49+09:00
- Markup SHA-256 before human implementation: `a83634d56501ec56104f8e4bfb62434c1c48b5ed89eef90d2f57ef5071e0d29e`
- Markup SHA-256 after final EndNote update: `4f72ecb292e5dd116ecbc144ba176bd39432e2bd1bc0a9ca493aabad5b25e1de`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate `w:ins` wrapper
- Backup: human Word/EndNote edit; no new machine backup was created
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Dhimal, M., Bhandari, D., Karki, K. B., Shrestha, S. L., Khanal, M., Shrestha, R. R., Dahal, S., Bista, B., Ebi, K. L., Cissé, G., Sapkota, A., & Groneberg, D. A. (2022). Effects of Climatic Factors on Diarrheal Diseases among Children below 5 Years of Age at National and Subnational Levels in Nepal: An Ecological Study. International Journal of Environmental Research and Public Health, 19(10), 6138.
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-04

- Location: References, Dhimal et al. (2015)
- Reason: Add the high-altitude Aedes-vector field study cited in the new Discussion paragraph.
- Kila decisions: `KILA-D-20260830-022`, `KILA-D-20260830-023`, `KILA-D-20260830-024`, `KILA-D-20260830-025`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T18:34:49+09:00
- Markup SHA-256 before human implementation: `a83634d56501ec56104f8e4bfb62434c1c48b5ed89eef90d2f57ef5071e0d29e`
- Markup SHA-256 after final EndNote update: `4f72ecb292e5dd116ecbc144ba176bd39432e2bd1bc0a9ca493aabad5b25e1de`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate `w:ins` wrapper
- Backup: human Word/EndNote edit; no new machine backup was created
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Dhimal, M., Gautam, I., Joshi, H. D., O’Hara, R. B., Ahrens, B., & Kuch, U. (2015). Risk Factors for the Presence of Chikungunya and Dengue Vectors (Aedes aegypti and Aedes albopictus), Their Altitudinal Distribution and Climatic Determinants of Their Abundance in Central Nepal. PLOS Neglected Tropical Diseases, 9(3), e0003545. https://doi.org/10.1371/journal.pntd.0003545
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-05

- Location: References, Dhimal et al. (2014)
- Reason: Add the elevation-transect dengue and lymphatic-filariasis vector study cited in the new Discussion paragraph.
- Kila decisions: `KILA-D-20260830-022`, `KILA-D-20260830-023`, `KILA-D-20260830-024`, `KILA-D-20260830-025`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T18:34:49+09:00
- Markup SHA-256 before human implementation: `a83634d56501ec56104f8e4bfb62434c1c48b5ed89eef90d2f57ef5071e0d29e`
- Markup SHA-256 after final EndNote update: `4f72ecb292e5dd116ecbc144ba176bd39432e2bd1bc0a9ca493aabad5b25e1de`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate `w:ins` wrapper
- Backup: human Word/EndNote edit; no new machine backup was created
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Dhimal, M., Gautam, I., Kreß, A., Müller, R., & Kuch, U. (2014). Spatio-Temporal Distribution of Dengue and Lymphatic Filariasis Vectors along an Altitudinal Transect in Central Nepal. PLOS Neglected Tropical Diseases, 8(7), e3035. https://doi.org/10.1371/journal.pntd.0003035
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

### part-06

- Location: References, Dhimal et al. (2021)
- Reason: Add the Hindu Kush Himalayan vector and vector-borne-disease systematic synthesis cited in the new Discussion paragraph.
- Kila decisions: `KILA-D-20260830-022`, `KILA-D-20260830-023`, `KILA-D-20260830-024`, `KILA-D-20260830-025`
- Mode: `human-endnote-reference-insert`
- Revises prior parts: none
- Timestamp: 2026-08-30T18:34:49+09:00
- Markup SHA-256 before human implementation: `a83634d56501ec56104f8e4bfb62434c1c48b5ed89eef90d2f57ef5071e0d29e`
- Markup SHA-256 after final EndNote update: `4f72ecb292e5dd116ecbc144ba176bd39432e2bd1bc0a9ca493aabad5b25e1de`
- Revision IDs: bibliography output generated inside the `ADDIN EN.REFLIST` field; no separate `w:ins` wrapper
- Backup: human Word/EndNote edit; no new machine backup was created
- Before:

~~~~text
[Reference not present.]
~~~~

- After:

~~~~text
Dhimal, M., Kramer, I. M., Phuyal, P., Budhathoki, S. S., Hartke, J., Ahrens, B., Kuch, U., Groneberg, D. A., Nepal, S., Liu, Q.-Y., Huang, C.-R., CissÉ, G., Ebi, K. L., KlingelhÖfer, D., & Müller, R. (2021). Climate change and its association with the expansion of vectors and vector-borne diseases in the Hindu Kush Himalayan region: A systematic synthesis of the literature. Advances in Climate Change Research, 12(3), 421-429. https://doi.org/https://doi.org/10.1016/j.accre.2021.05.003
~~~~

- Minimal tracked fragments: `human-endnote-reference-insert`

## finalization/markup-metadata

### metadata-normalization-01

- Location: Whole markup document > tracked-change and proofing-language metadata
- Reason: Human required American English proofing metadata and anonymous tracked-change authors.
- Kila decisions: none (non-substantive metadata normalization)
- Mode: `metadata-only`
- Timestamp: 2026-08-31T02:58:18.493404+00:00
- Author: anonymous
- Markup SHA-256 before: `d5c8b65f89022e8dfdf6b417ebf24ec479042e3722f3674a3e4e54fa70a16937`
- Markup SHA-256 after: `dfbb245bf5952da074725d59f855053b1c78dc5c6023b92ccb83492e4a1cd4db`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260831T115818493537.metadata-normalization.docx`
- Tracked authors before: `{"Jie MI": 193, "Kila": 28, "anonymous": 327}`
- Tracked authors after: `{"anonymous": 548}`
- Western proofing languages before: `{"en-JP": 29, "en-US": 1}`
- Western proofing languages after: `{"en-US": 30}`
- Author attributes changed: `221`
- Language attributes changed: `29`
- Metadata-only canonical XML verification: `true`
- Manuscript text unchanged: `true`
- Revision IDs unchanged: `true`
- Unmodified package members byte-identical: `true`

## finalization/markup-metadata

### metadata-normalization-02

- Location: Whole markup document > tracked-change and proofing-language metadata
- Reason: Human required American English proofing metadata and anonymous tracked-change authors.
- Kila decisions: none (non-substantive metadata normalization)
- Mode: `metadata-only`
- Timestamp: 2026-08-31T03:12:28.215549+00:00
- Author: anonymous
- Markup SHA-256 before: `dfbb245bf5952da074725d59f855053b1c78dc5c6023b92ccb83492e4a1cd4db`
- Markup SHA-256 after: `351a61293ad6d52492e7b41a10a115bc0d41321d919a248d7d353685bc9278e6`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260831T121228215681.metadata-normalization.docx`
- Tracked authors before: `{"anonymous": 548}`
- Tracked authors after: `{"anonymous": 548}`
- Western proofing languages before: `{"en-US": 30}`
- Western proofing languages after: `{"en-US": 30}`
- Author attributes changed: `0`
- Language attributes changed: `0`
- Core document identity properties changed: `2`
- Metadata-only canonical XML verification: `true`
- Manuscript text unchanged: `true`
- Revision IDs unchanged: `true`
- Unmodified package members byte-identical: `true`

## editor/artwork-guidelines

### figure-legends-layout-01

- Location: Front matter page break; end matter Figure Legends section
- Reason: Comply with the editor's artwork instructions by listing figure legends without embedded artwork.
- Kila decisions: none (editorial packaging and layout only)
- Mode: `tracked-heading-plus-layout-object-cleanup`
- Timestamp: 2026-08-31T05:05:05.403100+00:00
- Before heading: `Figures:`
- After heading: `Figure Legends`
- Preserved captions: `["Figure 1: Spatial Distribution of Respondents in Each Wave", "Figure 2: Regional Average Probability of Household Disease Incidence in Each Wave", "Figure 3: Regional Average Type Counts of Natural Disasters in Each Wave", "Figure 4: Regional Average Percentage of Population with Climate Knowledge in Each Wave", "Figure 5: Global Relationship between Natural Disaster Count and Disease Increase Probability", "Figure 6: Global Relationship between Natural Disaster Count and Disease Increase Probability by Climate Change Knowledge Status", "Figure 7: Spatial Heterogeneity in Climate Change Knowledge Prediction Differences", "Figure 8: Heterogeneity in Climate Change Knowledge Prediction Differences among Different Groups"]`
- Removed artwork paragraphs: `8`
- Removed drawing objects, including prior tracked image versions: `15`
- Removed redundant/inter-figure layout paragraphs: `16`
- Removed unused image relationships: `["rId10", "rId11", "rId12", "rId13", "rId14", "rId15", "rId16", "rId17", "rId18", "rId19", "rId20", "rId21", "rId22", "rId9"]`
- Removed unused media parts: `["word/media/image1.png", "word/media/image10.png", "word/media/image11.png", "word/media/image12.png", "word/media/image13.png", "word/media/image14.png", "word/media/image2.jpg", "word/media/image3.png", "word/media/image4.png", "word/media/image5.png", "word/media/image6.png", "word/media/image7.png", "word/media/image8.png", "word/media/image9.png"]`
- Tracked heading revision IDs: `[1829, 1830]`
- Markup SHA-256 before: `23c555c426239c2e010e81d1f29707d2c72a2c12052e97ad0096d4c07ecb7520`
- Markup SHA-256 after: `56a6c3766b00c942aa7eac80464469af02eb1efabe12bc4c1c388800afd159be`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260831T140505403109.artwork-layout.docx`
- Table legends section unchanged: `true`
- Figure caption text and order unchanged: `true`
- Page break before References retained: `true`
- Endnote hyperlink parts unchanged: `true`
- Track revisions retained: `true`

## finalization/markup-metadata

### metadata-normalization-03

- Location: Whole markup document > tracked-change and proofing-language metadata
- Reason: Human required American English proofing metadata and anonymous tracked-change authors.
- Kila decisions: none (non-substantive metadata normalization)
- Mode: `metadata-only`
- Timestamp: 2026-08-31T05:05:31.325084+00:00
- Author: anonymous
- Markup SHA-256 before: `56a6c3766b00c942aa7eac80464469af02eb1efabe12bc4c1c388800afd159be`
- Markup SHA-256 after: `353eb531d889ecd10447e04b4c8e018a125450f161c669b81f5f03e06a9b80e1`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260831T140531325227.metadata-normalization.docx`
- Tracked authors before: `{"Jie MI": 1278, "anonymous": 539}`
- Tracked authors after: `{"anonymous": 1817}`
- Western proofing languages before: `{"en-US": 1}`
- Western proofing languages after: `{"en-US": 1}`
- Author attributes changed: `1278`
- Language attributes changed: `0`
- Core document identity properties changed: `1`
- Metadata-only canonical XML verification: `true`
- Manuscript text unchanged: `true`
- Revision IDs unchanged: `true`
- Unmodified package members byte-identical: `true`

## finalization/markup-metadata

### metadata-normalization-04

- Location: Whole markup document > tracked-change and proofing-language metadata
- Reason: Human required American English proofing metadata and anonymous tracked-change authors.
- Kila decisions: none (non-substantive metadata normalization)
- Mode: `metadata-only`
- Timestamp: 2026-08-31T05:37:51.898945+00:00
- Author: anonymous
- Markup SHA-256 before: `7a8ecabe130e115282681fd63c6eebaba69ceee292785cc0bfeb3c870a17f172`
- Markup SHA-256 after: `a1480f4c49bf2797272cdf4666d8fe1d542e68b3a6f090981bfd045dea024e5a`
- Backup: `Rev/revision/.kila-backups/MLD01d.rev.markup.20260831T143751899071.metadata-normalization.docx`
- Tracked authors before: `{"Jie MI": 62, "anonymous": 1814}`
- Tracked authors after: `{"anonymous": 1876}`
- Western proofing languages before: `{"en-US": 1}`
- Western proofing languages after: `{"en-US": 1}`
- Author attributes changed: `62`
- Language attributes changed: `0`
- Core document identity properties changed: `1`
- Metadata-only canonical XML verification: `true`
- Manuscript text unchanged: `true`
- Revision IDs unchanged: `true`
- Unmodified package members byte-identical: `true`
