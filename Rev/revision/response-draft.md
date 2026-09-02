# Response to reviewers and editors of manuscript MLD01d

# Revision Summary

We thank the editor and reviewers for their careful and constructive assessment. The revised manuscript has been updated throughout to improve methodological transparency, interpretive precision, and relevance to Nepal's scientific and policy context.

- The title now states the climate-knowledge relationship using associative language, and causal wording has been replaced consistently with associative and predictive language.
- The Methods and limitations now define the respondent-reported outcome and multi-hazard exposure, document missing-data preprocessing and the survey-year covariate, and clarify recall, shared-reporting, reverse-causation, temporal, and residual-confounding boundaries.
- The modeling report now provides complete out-of-fold performance metrics, logistic-regression specifications and diagnostics, TreeSHAP analyses, and wave-specific sensitivity analyses in the main text and Supplementary Materials.
- The Discussion now incorporates Nepal-specific disease and adaptation evidence, explains the household human-capital heterogeneity in Figure 8, and links the findings to Nepal's NAP, HNAP, and NDC priorities while presenting climate-health education as complementary to general education and structural protection.
- Spatial figures were regenerated using Government of Nepal administrative boundaries and the elevation-derived ecological-belt layer; figures, figure legends, tables, and Supplementary Materials were prepared as separate submission components.
- Reporting sections covering funding, acknowledgments, contributors, declarations, data sharing, ethics and information protection, and Vancouver-style references were added or revised in response to the editorial requirements.

# Editor

## Editor Message (verbatim)

Thank you for submitting your manuscript to The Lancet Regional Health – Southeast Asia.

Your submission has now been assessed by external advisers and discussed by the editorial team. We would like to invite you to REVISE your paper in light of the editorial and reviewers' comments below. Please carefully address the issues raised in the comments. Please submit your revision by Sep 02, 2026.

ALL authors are also required to sign a completed "Author Statement Form", and each individual author must separately provide one completed "ICMJE COI Form". The disclosure in the ICMJE COI form should match EXACTLY the statement in the manuscript.

The "Author Statement Form" and "ICMJE COI Form" are also available on the submission guidelines page here. If you need help obtaining the forms, or if there are any questions please do let us know.

EDITORS' GENERAL POINTS:

Editorial points - IMPORTANT:

The following points list items that must be included in a point-by-point response before being considered further. Addressing them at this stage reduces the risk of errors and delays later. 
Please read the requirements below carefully and consult me or Information for Authors, for further details or clarification if needed.
Please note that not every point below will be relevant to your manuscript.

**Response:**

Thank you for inviting us to revise the manuscript. We have carefully read the editorial points and reporting guidelines and have prepared the revised manuscript and accompanying submission materials in strict accordance with the applicable requirements. We have also addressed all reviewer comments in the point-by-point responses below. All authors will sign the completed Author Statement Form, and each author will separately provide a completed ICMJE COI Form. Before resubmission, we will verify that the disclosure in each ICMJE COI Form matches exactly the Declaration of interests in the revised manuscript.

# Review Process Instructions

## Instructions

Reviewers' comments:

Please note that reviewer numbers are allocated by the system at invitation and not completion of reviews, so some numbers might be missing.

In your point-by-point reply to the reviewers' comments, please indicate the text changes that have been made (if any) and the line number on the tracked changes manuscript at which your change can be found. [Line numbers can be added to your word document using the “page layout” tab. Please select continuous numbers.]

When interpreting editorial points made by reviewers, please remember that we will edit the final manuscript if accepted.

# Reviewer 1

## Overall Comment

Reviewer #1: Option 2 — Direct & Professional
This paper is highly relevant and well-timed, and I appreciated reading it. Congratulations to the authors for their strong contribution. I have, however, noted several shortcomings that should be addressed prior to acceptance. Please find my detailed comments below.

**Response:**

Thank you for your encouraging assessment and constructive comments. We have addressed each of the issues raised in the detailed responses below. In particular, we clarified the proposed designs for future causal evaluation and the definition of multi-hazard exposure; documented the statutory confidentiality and information-protection framework of the two official surveys and the anonymized secondary-data basis of this study; updated the spatial figures using Government of Nepal administrative boundaries and clarified the elevation-based derivation of the EcoBelt layer; and strengthened the Discussion by linking the findings to Nepal’s NAP, HNAP, and NDC priorities and incorporating additional Nepal-specific evidence on climate-sensitive diseases, risk perception, and adaptive capacity. We believe these revisions improve the manuscript’s methodological transparency, geographic accuracy, and relevance to Nepal’s scientific and policy context.

## Comment 1

In line 95-97, it is mentioned that "Future longitudinal and quasi-experimental studies are required to clarify causal pathways and test whether community-based climate-health education can reduce disaster-related disease burden". Why one quasi-experiment studies? There could be interrupted time series analysis or randomized control trial. It needs to be clarify.

**Response:**

Thank you for highlighting that the earlier wording grouped future study designs too broadly. We have revised both the Research in Context and Discussion sections to distinguish the questions these designs address. Longitudinal studies can help establish temporal ordering but do not by themselves eliminate confounding. Interrupted time-series analysis is now identified explicitly as one quasi-experimental option for evaluating policy or program implementation using repeated observations before and after implementation. Randomized controlled trials are distinct from quasi-experimental designs. If future climate-health education is delivered at the community level, a cluster-randomized design could be considered, where feasible and ethically appropriate, to evaluate its effects on preparedness practices and health outcomes. The revised language therefore presents these approaches as complementary rather than treating quasi-experimental research as the only or universally preferred option.

The revised text reads:

"Future research should match study design to the question. Longitudinal studies can establish temporal ordering, while natural or quasi-experimental designs, including interrupted time-series analyses, can evaluate policy or program implementation. Where feasible and ethically appropriate, cluster-randomized controlled trials can test whether community-based climate-health education improves preparedness practices and health outcomes."

(Research in Context, pages 2–3, lines 47–52.)

"Fifth, the XGBoost framework does not address potential endogeneity between climate knowledge and health outcomes."

(Page 22, lines 482–484.)

## Comment 2

It is important to know whether ethical approval was taken to conduct 2016 and 2022 household surveys or not

**Response:**

Thank you for raising this important point. The 2016 Nepal National Climate Change Impact Survey and the 2022 Climate Change Survey were official national statistical surveys conducted by the Government of Nepal through the Central Bureau of Statistics and the National Statistics Office, respectively. The questionnaires for both waves state that the information collected is confidential under the applicable Statistical Act, that individual information is not published, and that the data are used only for statistical purposes. These provisions establish the statutory confidentiality and information-protection framework governing the original data collection. The present study uses only anonymized secondary data and involves no participant recruitment, direct contact, intervention, or collection of identifiable information.

## Comment 3

Muti-hazard exposure is not defined in the manuscript

**Response:**

Thank you for this comment. The revised Methods section now explicitly defines multi-hazard exposure as a count of distinct disaster types and clarifies what this measure does not capture.

The revised text reads:

"Specifically, the count is the number of distinct disaster types reported by a household (observed range, 0-15 across 19 survey categories), rather than a measure of disaster frequency, intensity, timing, or co-occurrence."

(Page 7, lines 139–141.)

## Comment 4

All figures should be updated using the Government of Nepal approached shape file

**Response:**

Thank you for this comment. We have updated all five spatial figures (Figures 1–4 and 7) using the Government of Nepal administrative-boundary shapefile. During this update, we also corrected administrative-name matching issues in the original mapping workflow so that the survey units align consistently with the updated boundary layer. For the ecological-belt component, an official EcoBelt vector layer was not available from the Government of Nepal. We therefore use the study's Mountain, Hill, and Terai classification derived from elevation using JAXA global DSM data. The revised Methods section now states both the administrative-boundary source and the ecological-belt construction method.

The revised text and updated figure locations read (five representative locations shown from six changed locations):

"The spatial maps use the Government of Nepal administrative-boundary shapefile. Ecological-belt boundaries (Mountain, Hill, and Terai) are derived by the authors through elevation-based classification of JAXA global DSM data because an official EcoBelt vector layer was not available from the Government of Nepal."

(Pages 10–11, lines 224–228.)

"Figure 1: Spatial Distribution of Respondents in Each Wave"

(Page 28, line 548.)

"Figure 2: Regional Average Probability of Household Disease Incidence in Each Wave"

(Page 28, line 549.)

"Figure 4: Regional Average Percentage of Population with Climate Knowledge in Each Wave"

(Page 28, lines 551–552.)

"Figure 7: Spatial Heterogeneity in Climate Change Knowledge Prediction Differences"

(Page 28, line 557.)

## Comment 5

The discussion section should be strengthened linking with climate change and health policies and plans in Nepal including NAP, HNAP, NDC

**Response:**

Thank you for this constructive suggestion. We have strengthened the Discussion by linking the study's principal findings to Nepal's National Adaptation Plan (NAP) 2021–2050, Health National Adaptation Plan (HNAP) 2023–2030, and Nationally Determined Contribution (NDC) 3.0. Specifically, the revised policy-linkage paragraph relates the nonlinear cumulative-exposure pattern to multi-hazard early warning and emergency preparedness, the spatial heterogeneity to disease surveillance, climate-resilient health infrastructure, and water, sanitation, and hygiene services, and the larger knowledge-related prediction contrast among households with lower literacy and education ratios to public awareness, capacity building, and health-workforce training. We have placed our recommendations in a separate paragraph so that the relationship with existing national priorities is distinguished clearly from the study's proposed targeting and evaluation implications. We have also added the three Government of Nepal policy documents to the References.

The revised Discussion text and policy reference entries read:

"These findings have policy relevance for targeting and program design. The nonlinear increase in predicted disease probability as households accumulate distinct hazard types is directly relevant to the National Adaptation Plan (NAP) 2021–2050 and Nationally Determined Contribution (NDC) 3.0 priorities for multi-hazard early-warning systems and emergency preparedness, because it indicates that monitoring and response planning should consider cumulative exposure rather than isolated events (55, 56). The spatial heterogeneity across province–ecological belt units is likewise relevant to the NAP and Health National Adaptation Plan (HNAP) 2023–2030 priorities for climate-sensitive disease surveillance, climate-resilient health infrastructure, and water, sanitation, and hygiene services (55, 57). The larger knowledge-related prediction contrast among households with lower literacy and education ratios adds a population-targeting dimension to the HNAP’s public-awareness and capacity-building priorities and NDC 3.0’s health-workforce training agenda, indicating where community-level climate-health communication may warrant greater emphasis (56, 57). "

(Page 19, lines 411–424.)

"Within these policy frameworks, infrastructure reinforcement and medical resource allocation should be prioritized in high-risk province-ecological belt units where multi-hazard intensity and disease burden coincide, particularly in the Terai and Hill regions. Simultaneously, targeted climate-health education delivered through community health programs  could be evaluated as a complement to general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach would reinforce rather than replace general education and structural health protection. An Italian national survey documenting climate-health knowledge gaps and limited curricular provision among young doctors and medical students provides additional context for health-workforce education (58). Given that the present study measures climate change knowledge, disaster exposure, and household disease change in the same interview, prospective evaluations of such programs could use separate measures of climate-health knowledge and preparedness practices alongside independently assessed health outcomes. This would clarify how information translates into action and health benefits. Disaster risk reduction frameworks should move beyond single-hazard approaches to address the cumulative and nonlinear nature of multi-hazard health threats  (28, 59)."

(Pages 19–20, lines 425–441.)

"55.	Government of Nepal. National Adaptation Plan of Nepal. In: Environment MoFa, editor. Singhdurbar, Kathmandu Nepa2021."

(Page 32, lines 718–719.)

"57.	Government of Nepal. Climate Change Health Adaptation Strategy and Action Plan (2023-2030). In: Population MoHa, editor. 2023."

(Page 32, lines 722–723.)

"56.	Government of Nepal. Nepal's Nationally Determined Contribution (NDC) 3.0. 2025."

(Page 32, lines 720–721.)

## Comment 6

There is perception of people on increase risk of disease due to climate change. These perceptions are confirmed by scientific literature especially on diarrheal diseases and vector-borne diseases in Nepal e.g Dhimal et al 2014, 205, 2021, 2022, 2025 etc. Hence, I suggest to elaborate discussion section citing relevant literature.

**Response:**

Thank you for this helpful suggestion. We have expanded the Discussion by relating the respondent-reported household disease pattern to Nepal-specific epidemiological, entomological, synthesis, and modeling evidence on diarrheal and vector-borne diseases. The revised paragraph reports the national associations of childhood diarrheal incidence with temperature and rainfall, summarizes field evidence on the distribution and climatic determinants of disease vectors across Nepal's elevation gradient, and incorporates evidence on the expansion and thermal suitability of vector-borne disease risk in highland and mid-hill areas. We also clarify that these disease-specific studies provide scientific context for the perceived household disease changes identified in our analysis while operating at different analytical scales. Five supporting studies have been added to the References.

The revised text and all five added reference entries read:

"The reported household disease pattern is also consistent with Nepal-specific epidemiological and entomological evidence on climate-sensitive diseases. A national ecological time-series analysis found that childhood diarrheal incidence increased by 4.4% per 1 °C increase in mean temperature and by 0.28% per 1 cm increase in rainfall, with the largest estimated effects in mountain regions (42). Field studies documented vectors of dengue and lymphatic filariasis across Nepal’s elevation gradient and showed that temperature, rainfall, and relative humidity predicted vector abundance (43, 44). A systematic synthesis of the Hindu Kush Himalayan region and recent Nepal-wide modeling further indicate expansion of vector-borne disease risk into highland areas and longer periods of thermal suitability for dengue in the mid-hills and major urban centers (45, 46). Although these studies use disease-specific outcomes at different analytical scales, they provide Nepal-specific scientific context for the perceived household disease changes identified in our analysis."

(Page 17, lines 370–382.)

"45.	Acharya BK, Khanal L, Dhimal M. Increased thermal suitability elevates the risk of dengue transmission across the mid hills of Nepal. PLOS ONE. 2025;20(4):e0322031."

(Page 31, lines 689–690.)

"42.	Dhimal M, Bhandari D, Karki KB, Shrestha SL, Khanal M, Shrestha RR, et al. Effects of Climatic Factors on Diarrheal Diseases among Children below 5 Years of Age at National and Subnational Levels in Nepal: An Ecological Study. International Journal of Environmental Research and Public Health [Internet]. 2022; 19(10):[6138 p.]."

(Page 31, lines 678–681.)

"43.	Dhimal M, Gautam I, Kreß A, Müller R, Kuch U. Spatio-Temporal Distribution of Dengue and Lymphatic Filariasis Vectors along an Altitudinal Transect in Central Nepal. PLOS Neglected Tropical Diseases. 2014;8(7):e3035."

(Page 31, lines 682–684.)

"44.	Dhimal M, Gautam I, Joshi HD, O’Hara RB, Ahrens B, Kuch U. Risk Factors for the Presence of Chikungunya and Dengue Vectors (Aedes aegypti and Aedes albopictus), Their Altitudinal Distribution and Climatic Determinants of Their Abundance in Central Nepal. PLOS Neglected Tropical Diseases. 2015;9(3):e0003545."

(Page 31, lines 685–688.)

"46.	Dhimal M, Kramer IM, Phuyal P, Budhathoki SS, Hartke J, Ahrens B, et al. Climate change and its association with the expansion of vectors and vector-borne diseases in the Hindu Kush Himalayan region: A systematic synthesis of the literature. Advances in Climate Change Research. 2021;12(3):421-9."

(Page 31, lines 691–694.)

## Comment 7

Nepal specific more literature need to be reviewed and including in discussion section

**Response:**

Thank you for this suggestion. We have expanded the Discussion with six additional Nepal-specific primary studies that are distinct from the disease-focused evidence and official policy documents added in response to the preceding comments. The new paragraph reviews evidence on climate-risk perception across ecological settings, differences in protection motivation, the gap between perceived climatic change and reported adaptation, and the financial, informational, agency, institutional, and livelihood-resource constraints that shape adaptive capacity. This evidence strengthens the interpretation of our knowledge-related and geographic contrasts while preserving the distinction between basic climate awareness and actual preparedness or adaptive action.

The revised text and all six added reference entries read:

"Nepal-specific research places these knowledge-related and geographic patterns in a broader context of risk perception and adaptive capacity. Studies across central Nepal and the Khumbu region show that perceptions of climate change and its health and environmental impacts vary by elevation and by perceived vulnerability, efficacy, and response costs  (47, 48). Although more than 80% of surveyed households in the Koshi River Basin perceived climatic changes, only 32% reported agricultural adaptation (49); studies elsewhere in Nepal similarly identify financial, informational, agency, and institutional constraints on adaptation  (50, 51). Together with evidence that livelihood assets shape household vulnerability across ecological settings (52), these findings support interpreting the present knowledge-related contrasts as context dependent rather than equating awareness with preparedness or adaptive action."

(Page 18, lines 383–393.)

"50.	Choquette-Levy N, Ghimire D, Oppenheimer M, Ghimire R, Ck D. Retrenchment under climate-driven risks in subsistence farming communities. Population and Environment. 2025;47(2):22."

(Page 32, lines 704–706.)

"51.	Gurung LJ, Miller KK, Venn S, Bryan BA. Climate change adaptation for managing non-timber forest products in the Nepalese Himalaya. Science of The Total Environment. 2021;796:148853."

(Page 32, lines 707–709.)

"49.	Hussain A, Rasul G, Mahapatra B, Wahid S, Tuladhar S. Climate change-induced hazards and local adaptations in agriculture: a study from Koshi River Basin, Nepal. Natural Hazards. 2018;91(3):1365-83."

(Page 32, lines 701–703.)

"52.	Pandey R, Bardsley DK. Social-ecological vulnerability to climate change in the Nepali Himalaya. Applied Geography. 2015;64:74-86."

(Page 32, lines 710–711.)

"48.	Phuyal P, Kramer IM, Kadel I, Wouters E, Magdeburg A, Groneberg DA, et al. On people’s perceptions of climate change and its impacts in a hotspot of global warming. PLOS ONE. 2025;20(2):e0317786."

(Page 32, lines 698–700.)

"47.	Poudyal NC, Joshi O, Hodges DG, Bhandari H, Bhattarai P. Climate change, risk perception, and protection motivation among high-altitude residents of the Mt. Everest region in Nepal. Ambio. 2021;50(2):505-18."

(Page 31, lines 695–697.)

# Reviewer 2

## Overall Comment

Reviewer #2

Reviewer comments (Major Revision)

**Overall assessment**

This is a timely and potentially impactful manuscript addressing an important public health issue through the application of machine-learning methods to nationally representative survey data. The study is novel and the findings are potentially relevant for climate adaptation policies in South Asia. However, several methodological and reporting issues require clarification before the manuscript can be considered for publication.

**Response:**

Thank you for recognizing the timeliness, novelty, and potential policy relevance of this study, and for identifying the methodological and reporting issues that required clarification. We have addressed each concern in the detailed responses below. The revised manuscript clarifies the recency and temporal limitations of the survey data, consistently presents the findings as non-causal associations, and defines the respondent-reported outcome and its measurement limitations. We have also expanded the Analytical Framework, Results, and Supplementary Materials to provide outcome-stratified out-of-fold validation with comprehensive performance metrics, a fully specified logistic-regression comparison and diagnostics, out-of-fold TreeSHAP analyses, and wave-specific sensitivity analyses. Finally, the revision more clearly distinguishes measured proxies from residual confounding, basic climate-change awareness from preparedness or adaptive capacity, and contextual educational evidence from intervention effectiveness. These changes improve the manuscript’s methodological transparency, robustness, interpretability, and precision in communicating its policy implications.

## Comment 1

1.The study is based on survey waves conducted in 2016 and 2022, yet the manuscript is submitted in 2026. The authors should explicitly justify why these are the most recent available nationally representative data and discuss how the time lag may influence the interpretation and current policy relevance of the findings. This limitation deserves greater emphasis in both the Methods and Discussion sections.

**Response:**

Thank you for raising this point. At the time of analysis, the 2016 and 2022 waves were the available nationally representative rounds of the Government of Nepal's household climate-change survey, with 2022 being the latest. The revised Methods now states this basis for data selection and explains that changes after the 2022 survey are not captured in the data. The revised Discussion further explains how the time gap may affect the current magnitude and geographic distribution of the observed relationships, while clarifying the study's specific contribution and current monitoring relevance.

The revised text reads:

"At the time of analysis, the 2016 and 2022 waves were the available nationally representative rounds of this government survey, with 2022 being the latest. Consequently, changes in hazard exposure, health conditions, or adaptation after the 2022 survey are not captured in the data and should be considered when interpreting the findings in relation to current conditions."

(Page 6, lines 112–116.)

"First, the data capture conditions reported in the 2016 and 2022 survey waves. Given the time gap between the survey periods and the analysis, subsequent changes in hazard patterns, health-service access, climate information, adaptation practices, and disease conditions may have altered the magnitude and geographic distribution of the observed relationships. The findings should therefore be interpreted in relation to the survey periods. Within this temporal scope, the analysis identifies the nonlinear relationship between cumulative multi-hazard exposure and reported household disease change and shows how this relationship varies with climate change knowledge and across geographic and socioeconomic groups. These results provide specific priorities for current monitoring, including whether risk remains concentrated after multiple hazards, where knowledge-related differences persist, and which population groups continue to experience greater vulnerability."

(Pages 20–21, lines 442–453.)

## Comment 2

2. Although the Discussion acknowledges that causal inference cannot be established, some parts of the manuscript still imply a protective effect of climate knowledge. The wording should consistently emphasize that the study identifies associations rather than causal relationships, particularly given the cross-sectional pooled design and the potential for reverse causality.

**Response:**

Thank you for raising this important point. We conducted a manuscript-wide language audit and revised the title, Summary, study objective, Methods, Results, Discussion, conclusion, and Figures 6–8 captions so that the findings are consistently described as associations, predicted-probability differences, or knowledge-status contrasts rather than protective causal effects. The limitations section also explicitly retains the pooled cross-sectional design and reverse-causality boundaries.

The revised text reads:

"Climate Knowledge Is Associated with Lower Health Risks from Multi-Hazard Exposure: Evidence from Nepal"

(Page 1, lines 1–3.)

"Findings: The model achieved 71.4% validation accuracy. Overall, 39.2% of households reported increased disease incidence, and 42.2% had climate change knowledge. The predicted probability of household disease increase was positively and nonlinearly associated with the number of disaster types experienced: risk rose steeply as households transitioned from zero to three distinct hazard types, then gradually plateaued beyond eight types. Households with climate change knowledge showed a consistently lower predicted disease probability across the full exposure gradient; the prediction difference widened at higher disaster counts. Substantial spatial heterogeneity was observed across province–ecological belt units, with larger knowledge-related prediction differences in western and eastern regions. Households with higher literacy and formal education ratios also demonstrated attenuated disaster–disease associations."

(Pages 1–2, lines 20–30.)

"Interpretation: Cumulative multi-hazard exposure is associated with increased household disease risk in a nonlinear pattern, and climate change knowledge is associated with a lower predicted probability of disease increase. Targeted climate-health education delivered through community health programs may complement general education and structural health protection in hazard-prone populations across South and Southeast Asia."

(Page 2, lines 31–35.)

"To characterize the marginal association between each predictor and the predicted outcome, we compute partial dependence plots (PDPs) that average model predictions over the observed distribution of all other covariates(39). Subgroup PDPs stratified by climate knowledge status characterize differences in predicted probabilities across knowledge groups."

(Page 10, lines 218–223.)

"Climate change knowledge is associated with differences in the predicted probability of increased household disease incidence across the cumulative disaster-exposure gradient. The y-axis of the PDPs is the mean predicted probability of new disease occurrence. As shown in Figure 6, across the full exposure gradient, households with climate change knowledge exhibit a consistently lower predicted probability of disease increase compared with those without. The prediction difference widens as disaster counts increase and is largest under high multi-hazard exposure. At lower exposure levels the difference is modest, but it expands substantially as households accumulate four or more distinct hazard types."

(Page 14, lines 303–311.)

"Spatial analysis shows that the climate-knowledge prediction difference varies across Nepal’s province-ecological belt units, as shown in Figure 7. Larger negative prediction differences are concentrated in several western and eastern units, while parts of central Nepal show comparatively smaller differences. This spatial heterogeneity indicates that the association between climate change knowledge and predicted disease risk varies across local environmental, infrastructural, and socioeconomic contexts."

(Page 15, lines 322–327.)

"This study provides evidence that cumulative multi-hazard exposure is associated with increased household disease risk in Nepal, and that the predicted disaster-disease pattern varies by climate change knowledge status. Three central findings emerge. Multi-hazard exposure is the dominant predictor of increased disease incidence, with a nonlinear exposure-response pattern characterized by rapid risk escalation during initial hazard accumulation and saturation at high exposure levels. Climate change knowledge is consistently associated with lower predicted disease probability across the full exposure gradient, with the prediction difference widening at higher disaster counts. Substantial spatial and socioeconomic heterogeneity indicates that these predictive patterns vary by geographic context and household characteristics."

(Page 16, lines 343–352.)

"The variation in predicted disease probability by climate change knowledge status aligns with theoretical frameworks linking cognitive awareness to proactive health-seeking behavior, including pre-emptive protection of water sources, improved hygiene practices, and earlier care-seeking during periods of environmental stress (24, 31). The larger knowledge-related prediction difference at high disaster counts indicates that the observed association is strongest when environmental conditions are most severe. This pattern is consistent with evidence from comparable low-resource settings, where information-based interventions have demonstrated measurable reductions in climate-related health risks even in the absence of strong material adaptive capacity (23, 53). The spatial heterogeneity in knowledge-related prediction differences across province-ecological belt units, and the attenuation of disaster-disease associations among more educated households, further indicate that the observed association varies with local infrastructure, socioeconomic resources, and human capital (13, 32, 54). At the same time, because climate change knowledge, disaster exposure, and household disease change are reported by the same respondent in the same interview, shared reporting tendencies and the possibility that perceived health deterioration shapes climate-change awareness or retrospective exposure reporting remain alternative explanations for part of the observed pattern."

(Pages 18–19, lines 394–410.)

"Second, the pooled cross-sectional design precludes causal inference; findings reflect associations between perceived disaster exposure and perceived health deterioration."

(Page 21, lines 453–455.)

"Reverse causation also cannot be excluded because perceived household health deterioration may influence climate-change awareness or the retrospective reporting of disaster exposure."

(Page 21, lines 463–465.)

"Multi-hazard exposure and climate change knowledge are associated with predicted household disease incidence in Nepal, with nonlinear and spatially heterogeneous patterns. These findings support disaster risk reduction strategies that account for cumulative hazard accumulation rather than isolated events and position targeted climate-health education as a potential complement to general education and structural health protection in hazard-prone populations. "

(Page 22, lines 492–497.)

"Figure 6: Global Relationship between Natural Disaster Count and Disease Increase Probability by Climate Change Knowledge Status"

(Page 28, lines 555–556.)

"Figure 7: Spatial Heterogeneity in Climate Change Knowledge Prediction Differences"

(Page 28, line 557.)

"Figure 8: Heterogeneity in Climate Change Knowledge Prediction Differences among Different Groups"

(Page 28, lines 558–559.)

## Comment 3

3. The primary outcome is based on respondents reporting whether household disease incidence has increased over the previous 25 years. This subjective measure is susceptible to substantial recall bias and reporting heterogeneity. More discussion is needed regarding the validity of this outcome and its implications for interpretation.

**Response:**

Thank you for highlighting this important validity concern. The revised Variables section now defines the outcome as respondent-reported, gives the questionnaire wording and binary coding for both survey waves, and the revised limitations paragraph explicitly explains recall error, proxy reporting, reporting heterogeneity, common-method bias, and the boundary between perceived change and clinically verified incidence.

The revised text reads:

"The primary dependent variable is a respondent-reported binary indicator derived from the NCCIS questionnaire. In 2016, respondents are asked whether the incidence of illness due to any disease increased in their family over the previous 25 years; the 2022 item asks whether the respondent or household members experienced a higher incidence of disease than 25 years earlier. We code yes as 1 and no as 0."

(Page 6, lines 127–131.)

"Second, the pooled cross-sectional design precludes causal inference; findings reflect associations between perceived disaster exposure and perceived health deterioration. The outcome asks one respondent to compare family illness with conditions 25 years earlier, and the outcome, exposure, and climate-knowledge measures are reported by the same respondent in the same interview. The outcome therefore captures perceived change rather than clinically verified incidence and may vary with memory, respondent age or education, proxy reporting for other household members, and interpretation of the reference period. Correlated reporting tendencies may create common-method bias and influence the magnitude or direction of the observed associations; consequently, the findings should not be interpreted as estimates of objective disease incidence."

(Page 21, lines 453–463.)

## Comment 4

4The statistical section requires additional detail. Please report:

* train/test split procedure;
* cross-validation strategy;
* class imbalance assessment;
* AUC, sensitivity, specificity, precision, recall and F1-score, rather than accuracy alone.
Reporting only an overall accuracy of 71.4% provides an incomplete evaluation of model performance.

**Response:**

Thank you for this helpful request. We have expanded the Analytical Framework, Results, and Supplementary Materials to describe the model-evaluation procedure and report a complete set of performance measures. The tuned XGBoost model is evaluated using outcome-stratified 10-fold cross-validation with shuffling, so each household contributes one prediction from a fold in which it was held out and each fold uses 90% of the sample for training and 10% for testing. We now report the outcome prevalence, the absence of over- or undersampling and class weighting, the probability threshold, and discrimination, classification, and probabilistic performance metrics. The Summary reports validation accuracy, while Supplementary Materials Table S2 provides the complete performance and diagnostic results. We also clarify that these values represent cross-validated out-of-fold performance rather than independent external validation.

The revised main-text passages read:

"The model minimizes a binary cross-entropy loss function. Hyperparameters were optimized via random search over 500 iterations, tuning learning rate, maximum tree depth, number of estimators, and subsampling ratios. Using the selected hyperparameters, predictive performance is evaluated through outcome-stratified 10-fold cross-validation with shuffling (seed 42). Each fold uses 90% of the sample for training and 10% for held-out testing, so every household contributes one out-of-fold prediction. The outcome prevalence is 39.17%, and stratification preserves this distribution across folds; no over- or undersampling or class weighting is applied. Threshold-based metrics use a probability threshold of 0.5. We report AUC, accuracy, balanced accuracy, sensitivity/recall, specificity, precision, F1 score, Brier score, and log loss. These estimates represent cross-validated out-of-fold performance rather than independent external validation."

(Pages 9–10, lines 198–209.)

"Across the out-of-fold predictions, the XGBoost model achieved an accuracy of 71.46%, balanced accuracy of 67.55%, sensitivity/recall of 49.55%, specificity of 85.56%, precision of 68.84%, and F1 score of 57.62%; the Brier score was 0.187 and log loss was 0.551. The corresponding out-of-fold AUC was 0.773."

(Page 12, lines 256–260.)

The revised Supplementary Materials text reads:

"We conduct a systematic hyperparameter tuning process using 500 random-search iterations across a predefined parameter space. The tuned parameters include learning rate, maximum tree depth, number of estimators, and row and column subsampling ratios. Using the selected hyperparameters, we evaluate the XGBoost model with outcome-stratified 10-fold cross-validation shuffled with seed 42. Each fold uses 90% of households for training and 10% for held-out testing, and the fold-specific predictions are combined so that each household contributes one out-of-fold prediction. The outcome prevalence is 39.17%; stratification preserves this distribution across folds, and no over- or undersampling or class weighting is applied. Threshold-based metrics use a probability threshold of 0.5. We report AUC, accuracy, balanced accuracy, sensitivity/recall, specificity, precision, F1 score, Brier score, and log loss (Table S2). These values describe cross-validated out-of-fold performance rather than independent external validation."

(Supplementary Materials, page 2, lines 12–23.)

## Comment 5

**5. Comparison with conventional regression**
The manuscript compares XGBoost with logistic regression but provides limited information regarding the specification of the regression model. The logistic regression should be fully described (covariates, interaction terms, diagnostics, goodness-of-fit) to ensure a fair comparison between approaches.

**Response:**

Thank you for this important suggestion. We have expanded the Analytical Framework, Results, and Supplementary Materials to provide a like-for-like comparison between XGBoost and conventional logistic regression. The logistic model uses the same 11,568 households, outcome, 64 predictors, and outcome-stratified 10-fold splits as XGBoost, with the lbfgs solver, L2 regularization (C=1), the existing analytical matrix without standardization, and no added interaction terms. In addition to the 100-iteration specification, we ran a diagnostic refit that changed only the maximum number of iterations to 5,000. Table S2 now reports discrimination, threshold-based performance, Brier score, log loss, calibration intercept and slope, the Hosmer–Lemeshow statistic, convergence warnings, and whether the iteration limit was reached.

The ordinary logistic model reached its iteration limit with convergence warnings in all 10 folds under both iteration settings. Its performance values are therefore reported transparently as diagnostic benchmarks rather than as estimates from a converged fit. The 100-iteration model yielded an AUC of 0.637 and accuracy of 61.87%, while the 5,000-iteration diagnostic refit yielded an AUC of 0.670 and accuracy of 64.50%; the corresponding XGBoost values were 0.773 and 71.46%. We have also clarified the methodological rationale for XGBoost: it retains the prespecified covariate information without outcome-driven screening, accommodates nonlinear relationships and high-order interactions without estimating a full-rank coefficient vector, and controls model complexity through regularization and row and column subsampling. Finally, we replaced the unqualified baseline-superiority wording in the Summary and Results with the validated out-of-fold metric and explicit convergence diagnostics.

The revised main-text passages read:

"Findings: The model achieved 71.4% validation accuracy."

(Page 1, line 20.)

"Unlike ordinary logistic regression, XGBoost does not require estimation of a full-rank coefficient vector, allowing the prespecified covariate set to be retained without outcome-driven variable screening; regularization and row and column subsampling constrain model complexity. For comparison, we fit an L2-penalized ordinary logistic regression using the same 11,568 households, outcome, 64 predictors, and outcome-stratified 10-fold splits. Detailed specifications and diagnostics are reported in Supplementary Materials Table S2."

(Page 9, lines 192–198.)

"The corresponding out-of-fold AUC was 0.773. Under identical outcome-stratified 10-fold splits, the ordinary logistic regression yielded an AUC of 0.637 and accuracy of 61.87%, but reached its iteration limit with convergence warnings in all 10 folds. Increasing the maximum iterations from 100 to 5,000 did not resolve convergence; its performance values are therefore treated as diagnostic benchmarks, with full specifications and goodness-of-fit diagnostics reported in Supplementary Materials Table S2."

(Page 12, lines 259–265.)

The following detailed description has also been added to the Supplementary Materials:

"To provide a conventional benchmark, we fit ordinary binary logistic regression using the same 11,568 households, outcome, 64 predictors, and identical outcome-stratified 10-fold splits used for XGBoost. The model used the lbfgs solver with L2 regularization (C=1), the existing analytical matrix without standardization, and no added interaction terms. The default fit used a maximum of 100 iterations; a diagnostic refit changed only the maximum to 5,000. Convergence was assessed from fold-specific warnings and whether the iteration limit was reached. Predictive diagnostics included AUC, accuracy, balanced accuracy, sensitivity, specificity, precision, F1 score, Brier score, and log loss. Calibration intercept and slope and the Hosmer–Lemeshow statistic were also calculated; given the large sample, the Hosmer–Lemeshow test was interpreted together with the Brier and calibration measures."

(Supplementary Materials, pages 2–3, lines 24–34.)

## Comment 6

6.Given the clinical and public health audience of The Lancet Regional Health, the manuscript would benefit from additional explainability analyses. SHAP values (or similar approaches) would be more informative than gain-based feature importance alone and would improve transparency regarding the contribution of each predictor.

**Response:**

Thank you for this helpful suggestion. We have expanded the Analytical Framework, Results, and Supplementary Materials with an exact out-of-fold TreeSHAP analysis that complements the existing gain-based feature importance. SHAP contributions were calculated for held-out observations in each outcome-stratified validation fold and combined to provide one out-of-fold profile for every household. Contributions were calculated for all 64 predictors, while the two supplementary figures display the 20 predictors with the largest mean absolute SHAP values for readability. We also report the SHAP scale and direction, verify additivity, and provide numerical results for multi-hazard exposure and climate change knowledge.

The revised Analytical Framework text reads:

"To complement this measure, exact TreeSHAP contributions are calculated for held-out observations in each validation fold and combined into one out-of-fold SHAP profile per household; positive and negative values indicate contributions toward higher and lower predicted log-odds, respectively."

(Page 10, lines 215–218.)

The revised Results text reads:

"The out-of-fold TreeSHAP analysis ranks multi-hazard exposure count first and climate change knowledge second by mean absolute SHAP value (0.346 and 0.120, respectively; Supplementary Materials Figures S3-S4). Mean SHAP contributions for the climate-change knowledge indicator are 0.106 log-odds for no and −0.135 log-odds for yes; those for multi-hazard exposure increase from −0.534 in the lowest exposure quartile to 0.387 in the highest."

(Pages 12–13, lines 269–274.)

The following methodological description has also been added to the Supplementary Materials:

"To complement gain-based importance, we calculate exact TreeSHAP contributions for held-out observations in each outcome-stratified validation fold using XGBoost’s built-in pred_contribs output. Fold-specific contributions are combined to obtain one out-of-fold SHAP profile for each of the 11,568 households across all 64 predictors. SHAP values are expressed on the raw prediction margin (log-odds) scale, where positive and negative values indicate contributions toward higher and lower model predictions, respectively. The sum of each household’s feature contributions and bias term reconstructs the corresponding raw margin, with a maximum absolute additivity error of 6.95 × 10−6. SHAP contributions are calculated for all 64 predictors; Figure S3 displays the 20 predictors with the largest mean absolute SHAP values, and Figure S4 displays their distributions and directions for readability. These values describe contributions within the fitted model rather than causal effects."

(Supplementary Materials, pages 3–4, lines 51–62.)

The added supplementary figures are titled:

"Figure S3: Global SHAP Feature Importance"

(Supplementary Materials, page 11, line 101.)

"Figure S4: SHAP Summary Plot"

(Supplementary Materials, page 12, line 103.)

## Comment 7

7. Several important determinants of health outcomes (baseline health status, healthcare access, environmental sanitation, local disease epidemiology) may not be fully captured in the model. The possibility of residual confounding should be discussed more extensively.

**Response:**

Thank you for highlighting this important concern. The revised Variables section now identifies the health-care-access, housing, and geographic proxies already included in the current model. The revised limitations paragraph distinguishes these proxies from unmeasured baseline health status, non-geographic dimensions of health-care access, household water and sanitation conditions, and local disease epidemiology. It also explains that residual confounding may affect the magnitude or direction of the observed associations and limits causal interpretation while preserving their descriptive and predictive value.

The revised text reads:

"Indicators of economic status cover residence ownership and type, asset ownership, agricultural land, access to communication and transportation assets, and distances to services, including the nearest health center."

(Page 8, lines 164–166.)

"Fourth, although the model includes distance to the nearest health center, residence characteristics, province, and ecological belt as proxies for access and structural or geographic context, it does not include direct measures of baseline health status, health-service affordability, quality or use, household water and sanitation conditions, or local disease epidemiology. Residual confounding by these factors may affect the magnitude or direction of the observed associations, which should therefore be interpreted as descriptive and predictive patterns rather than causally identified effects."

(Page 22, lines 476–482.)

## Comment 8

8.Pooling the 2016 and 2022 surveys increases statistical power but potentially masks temporal differences. Please clarify whether survey year was included in the model and consider presenting stratified or sensitivity analyses by survey wave to demonstrate consistency of findings.

**Response:**

Thank you for this important suggestion. Survey year is included in the pooled model to account for wave-level differences. We have also added a wave-specific sensitivity analysis that fits the same XGBoost specification separately to the 2016 and 2022 samples. The two models show closely comparable discrimination and accuracy, and both central findings retain the same direction across the supported exposure range: predicted disease probability increases with cumulative disaster exposure, while the climate-knowledge prediction difference remains negative. The numerical magnitudes vary somewhat between waves. To make the comparison direct, we retain the pooled-model hyperparameters rather than re-optimizing them separately by wave. The revised Methods, Results, and Discussion report these findings, and Supplementary Table S3 and Figure S2 provide the detailed wave-specific metrics and curves.

The revised text reads (five representative quotations from nine revised locations):

"The pooled model also includes a survey-year indicator (2016 or 2022) to control for wave-level differences."

(Page 8, lines 168–169.)

"To examine whether pooling masks temporal differences, we fit the same XGBoost specification separately to the 2016 and 2022 samples. Survey year is omitted from these wave-specific models because it is constant within each wave; all other predictors, the pooled-model hyperparameters, and outcome-stratified 10-fold cross-validation are retained. Using common hyperparameters provides a direct same-specification comparison without wave-specific re-optimization."

(Page 10, lines 209–214.)

"In the wave-specific sensitivity analysis, the 2016 and 2022 models achieve AUCs of 0.779 and 0.774 and accuracies of 70.89% and 72.36%, respectively, indicating closely comparable predictive performance."

(Page 12, lines 265–268.)

"Across disaster counts from 1 to 10, where both waves contain at least 30 observations at each count, predicted disease probability increases by 16.11 percentage points in 2016 and 22.72 percentage points in 2022. The climate-knowledge prediction difference remains negative at all ten supported counts in both waves and averages −6.01 percentage points in 2016 and −2.51 percentage points in 2022. The corresponding disaster-exposure and knowledge-difference curves are strongly correlated across waves (r = 0.920 and r = 0.924, respectively). Thus, the direction of both central patterns is stable across survey waves, although their numerical magnitudes and the locations of nonlinear changes vary. Detailed wave-specific performance metrics and prediction curves are presented in Supplementary Materials Table S3 and Figure S2, respectively."

(Pages 14–15, lines 312–321.)

"The wave-specific sensitivity analysis further supports the stability of these central patterns. Model discrimination and accuracy are closely comparable, cumulative exposure is associated with higher predicted disease probability, and the climate-knowledge prediction difference remains negative throughout the supported exposure range in both waves. The numerical variation may reflect differences in wave composition and the use of common pooled-model hyperparameters for direct comparability rather than separate wave-specific re-optimization; importantly, neither core pattern reverses."

(Page 16, lines 353–359.)

## Comment 9

9. Climate change knowledge is measured using a single binary variable ("having heard about climate change"). This is a rather crude proxy for adaptive capacity. The authors should discuss more explicitly the limitations of using awareness as a surrogate for actual behavioural change or preparedness.

**Response:**

Thank you for highlighting this important measurement limitation. The revised Variables section identifies the construct as a binary awareness indicator and reports its question content and coding without using an internal questionnaire item label. The revised Discussion distinguishes basic awareness from depth or accuracy of understanding, risk perception, preparedness, adaptive actions, resources for action, behavioral change, and adaptive capacity. It also clarifies that the observed moderation pattern is an association with reported awareness status, not evidence that awareness translated into behavioral adaptation or improved preparedness.

The revised text reads:

"Climate Change Knowledge is measured using a binary awareness indicator, as in previous studies (24, 25, 31). In both survey waves, NCCIS asks whether the respondent has heard about climate change; yes is coded as 1 and no as 0."

(Page 7, lines 149–151.)

"Third, the binary climate knowledge measure captures basic awareness and may reflect access to climate-related information, but it does not measure the depth or accuracy of understanding, risk perception, preparedness, adaptive actions, or the resources needed to implement them. It should therefore be interpreted as an awareness indicator rather than as a direct measure of behavioral change, preparedness, or adaptive capacity; accordingly, the observed moderation pattern is not evidence that awareness translated into behavioral adaptation or improved preparedness."

(Page 21, lines 465–471.)

## Comment 10

10. The Discussion proposes integrating climate education into community health programmes. While plausible, these recommendations should be presented more cautiously given the observational nature of the study and the absence of evidence that improving climate knowledge alone would reduce disease incidence. (please see and cite if you think appropriate this paper that treat on the role of medical education in climate change doi: 10.3389/fpubh.2024.1382505.)

**Response:**

Thank you for this important point and for suggesting the medical-education study. We have revised the Research in Context and Discussion to present climate-health education as a component to be evaluated alongside general education and structural protection, rather than as an intervention already shown to reduce disease incidence. We also cite Segala et al. (2024) only as contextual evidence of climate-health knowledge gaps and limited curricular provision among young doctors and medical students. Because that study is cross-sectional and does not evaluate an educational intervention or disease outcomes, we do not use it as evidence that education alone reduces disease incidence. The manuscript continues to recommend prospective evaluation using separate measures of knowledge, preparedness practices, and independently assessed health outcomes.

The revised text and added reference read:

"Public health and disaster risk reduction strategies in Nepal and similar hazard-prone settings could therefore evaluate climate-health education as one component alongside targeted investments in health systems, water and sanitation, and local preparedness."

(Research in Context, page 2, lines 43–46.)

"Simultaneously, targeted climate-health education delivered through community health programs  could be evaluated as a complement to general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach would reinforce rather than replace general education and structural health protection."

(Pages 19–20, lines 428–432.)

"58.	Segala FV, Di Gennaro F, Giannini LAA, Stroffolini G, Colpani A, De Vito A, et al. Perspectives on climate action and the changing burden of infectious diseases among young Italian doctors and students: a national survey. Frontiers in Public Health. 2024;Volume 12 - 2024."

(Page 32, lines 724–727.)

# Reviewer 3

## Overall Comment

Reviewer #3: This study uses a pooled cross-sectional sample of 11,568 households aged 45 and older with at least 25 years of residence in their community, drawn from the 2016 and 2022 waves of the Nepal Climate Change Impact Survey, and applies an XGBoost classification model to examine whether cumulative exposure to multiple types of natural disasters is associated with self-reported long-term increases in household disease incidence, and whether climate change knowledge moderates that association. I had a few conceptual and technical comments for the authors to consider. These are below.

**Response:**

Thank you for your careful summary of the study and for the conceptual and technical comments provided below. We have addressed each point in the detailed responses. Specifically, the revised manuscript clarifies the scope of the long-term-resident sample and the limits of interpreting displacement and migration pathways; states explicitly how survey year is incorporated; and carries the implications of same-respondent reporting, retrospective judgment, shared reporting tendencies, and reverse causation through the Methods, interpretation, policy, and limitations sections. We have also revised the manuscript to use consistently non-causal language, clarified that Figure 8 presents heterogeneity in the climate-change-knowledge prediction contrast across levels of household human capital, and documented the missing-data coding rules, missingness rate, and resulting analytical matrix. These changes strengthen the transparency of the study’s sample boundaries, temporal specification, measurement and interpretation limits, human-capital analysis, and data preprocessing.

## Comment 1

1. The survey, by design, includes people who have lived in the same community for at least 25 years, a stable, never-displaced population. The introduction devotes substantial space to displacement and migration as a health pathway, but I don't see how those included in this survey design can speak to that group, since anyone who left after a major disaster is missing from the sample, right? Am I missing something obvious?

**Response:**

Thank you for raising this important distinction. The 25-year residence criterion defines a long-term-resident sample, but it does not, by itself, establish uninterrupted physical presence or rule out temporary disaster-related evacuation followed by return. At the same time, our analysis does not include a measure of displacement history, so it cannot identify which respondents experienced temporary displacement or estimate that pathway; people who left their communities after a disaster and consequently did not meet the residence criterion are outside the analytic sample. The displacement and migration discussion in the Introduction was intended to summarize a general health pathway identified in the broader climate-health literature, rather than to imply that the present study analyzes health outcomes among displaced or migrant populations. We agree that the previous two-sentence discussion gave this pathway disproportionate prominence relative to the study scope. We have therefore condensed it to a single background sentence in the Introduction.

The revised text reads:

"The broader literature also identifies that extreme climate events may displace populations, disrupt livelihoods and social networks, and place additional pressure on public health systems (13-15)."

(Page 3, lines 51–53.)

## Comment 2

2. The two survey waves are pooled into a single sample, but I couldn't tell whether the wave is included as a covariate, which is necessary for statistical inference.

**Response:**

Thank you for requesting this clarification. Yes, survey wave is included in the pooled model as a survey-year indicator distinguishing the 2016 and 2022 observations, and the revised Variables section states this explicitly. The indicator is included to account for wave-level differences when the two nationally representative samples are analyzed together.

We also examine whether pooling obscures temporal differences through a complementary wave-specific sensitivity analysis. The same XGBoost specification is fitted separately to the 2016 and 2022 samples. Survey year is omitted only from these wave-specific models because it has no within-wave variation; all other predictors, the pooled-model hyperparameters, and the outcome-stratified 10-fold cross-validation procedure are retained. This design allows the two waves to be compared under a common specification rather than under separately optimized models. The resulting discrimination and accuracy are closely comparable across waves. Detailed wave-specific performance metrics and prediction curves are reported in Supplementary Materials Table S3 and Figure S2. Thus, survey year is controlled in the pooled analysis, while the stratified sensitivity analysis provides an additional check that the central patterns are not produced solely by pooling the two survey waves.

The revised text reads:

"The pooled model also includes a survey-year indicator (2016 or 2022) to control for wave-level differences."

(Page 8, lines 168–169.)

"To examine whether pooling masks temporal differences, we fit the same XGBoost specification separately to the 2016 and 2022 samples. Survey year is omitted from these wave-specific models because it is constant within each wave; all other predictors, the pooled-model hyperparameters, and outcome-stratified 10-fold cross-validation are retained. Using common hyperparameters provides a direct same-specification comparison without wave-specific re-optimization."

(Page 10, lines 209–214.)

"In the wave-specific sensitivity analysis, the 2016 and 2022 models achieve AUCs of 0.779 and 0.774 and accuracies of 70.89% and 72.36%, respectively, indicating closely comparable predictive performance."

(Page 12, lines 265–268.)

## Comment 3

3. As the components of the analyses, ie, exposure, moderator, and outcome, all come from the same respondent in the same interview, and the outcome itself asks one person to summarize disease trends for the whole household over 25 years considerable retrospective judgment riding on one person's memory and reporting style, and it opens the door to reverse causation and shared reporting bias as alternative explanations for the pattern. I think that is ok.. or at least a reality of such data, but the limitations paragraph mentions this once, but it would be good to carry that caveat through the interpretation and policy sections too, not just the closing paragraph.

**Response:**

Thank you for this important point. The Methods section now states explicitly that the outcome, multi-hazard exposure, and climate-change knowledge measures are reported by the same respondent in the same interview. We have also carried this caveat through the Discussion rather than confining it to the limitations paragraph. The revised interpretation presents shared reporting tendencies and reverse causation as alternative explanations for part of the observed knowledge-related pattern, while the revised policy section recommends prospective evaluation that measures knowledge, preparedness practices, and health outcomes separately. The limitations paragraph retains and consolidates the corresponding boundaries concerning retrospective judgment, reporting heterogeneity, common-method bias, and reverse causation.

The revised text reads:

"The survey measures used for the outcome, multi-hazard exposure, and climate-change knowledge are all reported by the same respondent in the same interview."

(Page 8, lines 158–159.)

"At the same time, because climate change knowledge, disaster exposure, and household disease change are reported by the same respondent in the same interview, shared reporting tendencies and the possibility that perceived health deterioration shapes climate-change awareness or retrospective exposure reporting remain alternative explanations for part of the observed pattern."

(Page 19, lines 406–410.)

"Given that the present study measures climate change knowledge, disaster exposure, and household disease change in the same interview, prospective evaluations of such programs could use separate measures of climate-health knowledge and preparedness practices alongside independently assessed health outcomes. This would clarify how information translates into action and health benefits."

(Page 20, lines 435–439.)

"Second, the pooled cross-sectional design precludes causal inference; findings reflect associations between perceived disaster exposure and perceived health deterioration. The outcome asks one respondent to compare family illness with conditions 25 years earlier, and the outcome, exposure, and climate-knowledge measures are reported by the same respondent in the same interview. The outcome therefore captures perceived change rather than clinically verified incidence and may vary with memory, respondent age or education, proxy reporting for other household members, and interpretation of the reference period. Correlated reporting tendencies may create common-method bias and influence the magnitude or direction of the observed associations; consequently, the findings should not be interpreted as estimates of objective disease incidence. Reverse causation also cannot be excluded because perceived household health deterioration may influence climate-change awareness or the retrospective reporting of disaster exposure."

(Page 21, lines 453–465.)

## Comment 4

4. The title and much of the language in the Interpretation and Discussion ("mitigates," "buffers," "reduces") is too causal, but nothing in the design supports such robust conclusions.

**Response:**

Thank you for raising this concern. We agree that the pooled cross-sectional design supports associative and predictive interpretation rather than claims that climate change knowledge mitigates, buffers, or reduces disease risk. The manuscript-wide language revision now uses association, predicted-probability difference, and knowledge-status contrast consistently across the title, Summary, Methods, Results, Discussion, conclusion, and figure captions. The revised limitations also state explicitly that the design precludes causal inference and that reverse causation cannot be excluded.

The revised text reads (10 representative locations shown from 15 changed locations):

"Climate Knowledge Is Associated with Lower Health Risks from Multi-Hazard Exposure: Evidence from Nepal"

(Page 1, lines 1–3.)

"Interpretation: Cumulative multi-hazard exposure is associated with increased household disease risk in a nonlinear pattern, and climate change knowledge is associated with a lower predicted probability of disease increase. Targeted climate-health education delivered through community health programs may complement general education and structural health protection in hazard-prone populations across South and Southeast Asia."

(Page 2, lines 31–35.)

"To characterize the marginal association between each predictor and the predicted outcome, we compute partial dependence plots (PDPs) that average model predictions over the observed distribution of all other covariates(39). Subgroup PDPs stratified by climate knowledge status characterize differences in predicted probabilities across knowledge groups."

(Page 10, lines 218–223.)

"Climate change knowledge is associated with differences in the predicted probability of increased household disease incidence across the cumulative disaster-exposure gradient. The y-axis of the PDPs is the mean predicted probability of new disease occurrence. As shown in Figure 6, across the full exposure gradient, households with climate change knowledge exhibit a consistently lower predicted probability of disease increase compared with those without. The prediction difference widens as disaster counts increase and is largest under high multi-hazard exposure. At lower exposure levels the difference is modest, but it expands substantially as households accumulate four or more distinct hazard types."

(Page 14, lines 303–311.)

"Spatial analysis shows that the climate-knowledge prediction difference varies across Nepal’s province-ecological belt units, as shown in Figure 7. Larger negative prediction differences are concentrated in several western and eastern units, while parts of central Nepal show comparatively smaller differences. This spatial heterogeneity indicates that the association between climate change knowledge and predicted disease risk varies across local environmental, infrastructural, and socioeconomic contexts."

(Page 15, lines 322–327.)

"This study provides evidence that cumulative multi-hazard exposure is associated with increased household disease risk in Nepal, and that the predicted disaster-disease pattern varies by climate change knowledge status. Three central findings emerge. Multi-hazard exposure is the dominant predictor of increased disease incidence, with a nonlinear exposure-response pattern characterized by rapid risk escalation during initial hazard accumulation and saturation at high exposure levels. Climate change knowledge is consistently associated with lower predicted disease probability across the full exposure gradient, with the prediction difference widening at higher disaster counts. Substantial spatial and socioeconomic heterogeneity indicates that these predictive patterns vary by geographic context and household characteristics."

(Page 16, lines 343–352.)

"The variation in predicted disease probability by climate change knowledge status aligns with theoretical frameworks linking cognitive awareness to proactive health-seeking behavior, including pre-emptive protection of water sources, improved hygiene practices, and earlier care-seeking during periods of environmental stress (24, 31). The larger knowledge-related prediction difference at high disaster counts indicates that the observed association is strongest when environmental conditions are most severe. This pattern is consistent with evidence from comparable low-resource settings, where information-based interventions have demonstrated measurable reductions in climate-related health risks even in the absence of strong material adaptive capacity (23, 53). The spatial heterogeneity in knowledge-related prediction differences across province-ecological belt units, and the attenuation of disaster-disease associations among more educated households, further indicate that the observed association varies with local infrastructure, socioeconomic resources, and human capital (13, 32, 54). At the same time, because climate change knowledge, disaster exposure, and household disease change are reported by the same respondent in the same interview, shared reporting tendencies and the possibility that perceived health deterioration shapes climate-change awareness or retrospective exposure reporting remain alternative explanations for part of the observed pattern."

(Pages 18–19, lines 394–410.)

"Second, the pooled cross-sectional design precludes causal inference; findings reflect associations between perceived disaster exposure and perceived health deterioration."

(Page 21, lines 453–455.)

"Reverse causation also cannot be excluded because perceived household health deterioration may influence climate-change awareness or the retrospective reporting of disaster exposure."

(Page 21, lines 463–465.)

"Multi-hazard exposure and climate change knowledge are associated with predicted household disease incidence in Nepal, with nonlinear and spatially heterogeneous patterns. These findings support disaster risk reduction strategies that account for cumulative hazard accumulation rather than isolated events and position targeted climate-health education as a potential complement to general education and structural health protection in hazard-prone populations. "

(Page 22, lines 492–497.)

## Comment 5

5. Climate knowledge appears to do much of the same work as literacy and education in the subgroup results (Figure 8 shows a very similar attenuation pattern for all three), and it is likely correlated with both in this kind of survey. Do the authors think th separate a climate-specific effect from a general human capital effect, which matters a lot for the policy recommendation, since it specifically calls for climate education rather than education or literacy more broadly.

**Response:**

Thank you for raising this important distinction. The model includes respondent-level literacy and education together with household-level shares of literate members and members with 12 or more years of education. When the climate change knowledge prediction contrast is calculated, these measured human-capital characteristics remain at their observed values. Figure 8 presents this contrast within household literacy and education subgroups and therefore provides a limited predictive separation from the measured human-capital characteristics, although it cannot fully distinguish climate-specific knowledge from unmeasured dimensions of general human capital. The revised Results section now explains the calculation and uncertainty of the contrast and clarifies that the literacy and education panels show heterogeneity in the knowledge-related prediction difference rather than parallel effects of the three variables. We have retained a concise Figure 8 title consistent with the other figure captions and have also revised the policy recommendation to present targeted climate-health education as a complement to general education and literacy, particularly where broader human-capital resources are limited, rather than as a substitute for them.

The revised text reads:

"Socio-demographic subgroup analysis further shows that the estimated climate change knowledge contrast varies with household human capital, as illustrated in Figure 8. Error bars represent 95% confidence intervals. For each household, this contrast is calculated as the difference between the predicted probability of increased household disease incidence when climate change knowledge is set to yes and the corresponding prediction when it is set to no; negative values therefore indicate a lower predicted probability under the knowledge condition. The contrast is most negative among households with lower literate-member ratios and lower shares of members with 12 or more years of education, and it attenuates toward zero and crosses it in the highest groups as these ratios increase. Figure 8 therefore shows that the additional predictive difference associated with climate change knowledge is larger where general human capital is more limited, rather than showing parallel effects of climate knowledge, literacy, and education. Household age and sex composition show comparatively little variation in the knowledge contrast."

(Pages 15–16, lines 328–341.)

"Figure 8: Heterogeneity in Climate Change Knowledge Prediction Differences among Different Groups"

(Page 28, lines 558–559.)

"Simultaneously, targeted climate-health education delivered through community health programs  could be evaluated as a complement to general education and literacy, particularly where broader human-capital resources are limited and the knowledge-related prediction contrast in Figure 8 is larger; this targeted approach would reinforce rather than replace general education and structural health protection."

(Pages 19–20, lines 428–432.)

## Comment 6

6. Missing data isn't discussed anywhere, and XGBoost's default way of handling it (learning which direction to send missing values at each split) treats absence as informative without saying so. If missingness differs by wave, region, or wealth, which seems plausible, that could quietly shift both the accuracy numbers and the interpretive plots. A missingness rate and a sentence on how it was handled should be added and confirm with best statistical practices.

**Response:**

Thank you for identifying this issue. We have revised the manuscript to distinguish response status in the source survey from completeness of the post-processing model matrix. In both survey waves, the primary climate-change knowledge item and disease-change outcome contained only yes/no responses and had 0% item nonresponse. The household-by-hazard exposure fields likewise contained no unrecorded values; explicit no and structurally inapplicable responses were coded as 0. For other binary predictors, an explicit affirmative response was coded as 1, whereas negative, structurally skipped, or otherwise nonaffirmative responses were assigned to the reference category. Among the continuous predictors, agricultural-experience years were fully observed in 2016. In 2022, 1,894 blank entries (29.1% of that wave) corresponded exactly to households reporting no agricultural land and were therefore treated as structurally inapplicable and assigned a logical value of 0. The remaining applicable continuous predictors had no missing values, and the complete-case safeguard used during model preparation consequently removed no households. The final analytical matrix therefore contained no NA values, and XGBoost’s native missing-value routing was not invoked. The Discussion also addresses the potential for assigning uncertain or unrecorded responses to the reference category to introduce misclassification and influence model performance and interpretation.

The revised text reads:

"Response status in the source survey was distinguished from completeness of the post-processing model matrix. In both survey waves, the primary climate change knowledge item and disease-change outcome contained only yes/no responses and had 0% item nonresponse. The household-by-hazard exposure fields likewise contained no unrecorded values; explicit no and structurally inapplicable responses were coded as 0. For binary predictors defined by an affirmative response, explicit affirmative responses were coded as 1, whereas negative, structurally skipped, or otherwise nonaffirmative responses were assigned to the reference category. Among the continuous predictors, agricultural-experience years were fully observed in 2016. In 2022, 1,894 blank entries (29.1% of that wave) corresponded exactly to households reporting no agricultural land and were therefore treated as structurally inapplicable and assigned a logical value of 0. The remaining applicable continuous predictors had no missing values, and the complete-case safeguard used during model preparation consequently removed no households. The final analytical matrix comprised 11,568 households with no missing values across the 64 predictors and outcome; therefore, XGBoost’s native missing-value routing was not invoked."

(Pages 8–9, lines 170–185.)

"Separately, assigning uncertain or unrecorded responses to the reference category during preprocessing may introduce misclassification; if such misclassification is nondifferential, it could attenuate affected associations, whereas differential response patterns across survey waves, regions, or socioeconomic groups may influence model performance and interpretation in either direction."

(Pages 21–22, lines 471–476.)
