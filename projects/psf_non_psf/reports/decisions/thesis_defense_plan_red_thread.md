# Euclidifying Gaia: Thesis Plan and Red Thread (3-Week Version)

## Scope right now
- Degree context: Bachelor thesis (University of Cambridge).
- Time to submission/defense: about 3 weeks.
- Current technical center: PSF vs non-PSF as core problem.
- Secondary work: representation analysis and cVAE as exploratory extension.
- Evidence status: mixed progress across modeling, statistics, and astrophysical interpretation.

## Core thesis claim (red thread)
This thesis argues that **measurement mismatch is physically informative**: by combining Gaia DR3 catalog statistics (point-source-optimized measurements) with Euclid imaging context, we can identify and characterize sources that are not simple point sources.

Operationally:
1. Build reliable Euclid-Gaia matched data.
2. Learn predictive mapping from Gaia statistics to morphology-related labels.
3. Validate with strict evaluation and stratified error analysis.
4. Interpret failures and successes in astrophysical terms (PSF-like vs extended/complex environment).

## Thought process behind report structure
The report should not read like a list of methods. It should read like a sequence of scientific questions:

1. Why this problem matters scientifically?
2. Why Gaia alone is ambiguous for non-point-like sources?
3. How Euclid provides missing context and weak supervision?
4. Whether the ML pipeline predicts meaningful morphology signals?
5. Whether results are robust under real observational constraints?
6. What is physically learned vs what remains uncertain?

Each chapter should answer one of these questions with evidence.

## Proposed chapter architecture (merged, with subsections)
1. Framing the Problem: Euclidifying Gaia  
2. From Measurements to Dataset: Instruments, Domain Gap, and Label Construction  
3. Modeling Strategy: Baseline Core and Exploratory Extensions  
4. Results: Predictive Performance, Robustness, and Representation Structure  
5. Scientific Interpretation and Astrophysical Relevance  
6. Limitations, Future Program, and Conclusion  

## Chapter/subsection blueprint and minimum evidence
### 1) Framing the Problem: Euclidifying Gaia
Suggested subsections:
- 1.1 Scientific context and motivation (Gaia/Euclid complementarity)
- 1.2 Thesis question and scope (PSF vs non-PSF core)
- 1.3 Contributions and expected outcomes

Minimum evidence:
- One end-to-end workflow figure.
- One explicit primary research question and 2-3 supporting sub-questions.

### 2) From Measurements to Dataset: Instruments, Domain Gap, and Label Construction
Suggested subsections:
- 2.1 Gaia measurement logic (AF, G PSF-fit, BP/RP prism flux)
- 2.2 Why mismatch carries morphology information
- 2.3 Euclid-Gaia cross-match protocol and quality filtering
- 2.4 Bright/faint stress points (Euclid saturation, Gaia depth non-uniformity)
- 2.5 Label reliability and ambiguity management

Minimum evidence:
- Compact conceptual diagram for Gaia vs Euclid measurement modes.
- Data-flow table with sample counts after each selection step.
- One sensitivity analysis on match radius/ambiguity handling.

### 3) Modeling Strategy: Baseline Core and Exploratory Extensions
Suggested subsections:
- 3.1 Problem setup and target definition (binary PSF/non-PSF now)
- 3.2 Feature design and preprocessing (Gaia statistics, PCA if used)
- 3.3 Baseline model (XGBoost) and training protocol
- 3.4 Score calibration/threshold strategy (if available)
- 3.5 Exploratory models (UMAP analyses, cVAE extension framing)

Minimum evidence:
- Reproducible train/validation protocol with leakage control statement.
- Clear primary metric set and model-selection rule.
- Short rationale for why cVAE is extension, not central claim.

### 4) Results: Predictive Performance, Robustness, and Representation Structure
Suggested subsections:
- 4.1 Main predictive results (overall performance)
- 4.2 Regime robustness (brightness, scan-depth/transits, imbalance)
- 4.3 Error taxonomy (typical false positives/false negatives)
- 4.4 Representation diagnostics (UMAP neighborhood behavior)

Minimum evidence:
- Confusion matrix + PR/F1/ROC summary.
- Regime-stratified results table.
- One reproducible representation finding linked to model behavior.

### 5) Scientific Interpretation and Astrophysical Relevance
Suggested subsections:
- 5.1 What model decisions correspond to physically
- 5.2 Case studies (isolated stars, blends, galaxy nuclei, SF knots)
- 5.3 Where catalog-only inference fails and Euclid context resolves it
- 5.4 Relevance for future Gaia catalog exploitation

Minimum evidence:
- Curated case-study panel with Euclid stamps and Gaia-feature signatures.
- Explicit interpretation of at least 2 failure modes as astrophysical, not only statistical.

### 6) Limitations, Future Program, and Conclusion
Suggested subsections:
- 6.1 Known limitations and mitigation status
- 6.2 Near-term roadmap (post-thesis continuation)
- 6.3 Final thesis contribution statement

Minimum evidence:
- Structured limitations table (issue, impact, mitigation, residual risk).
- Prioritized future-work plan with time horizon (1 month, 3 months, longer-term).

## Future directions you can realistically include
### A) Scientific extensions
- Move from binary PSF/non-PSF to hierarchical labels (first binary gate, then subclass typing).
- Add physically motivated uncertainty reporting to support catalog-level scientific use.
- Study transfer across sky regions with different Gaia scanning histories.

### B) Data and labeling improvements
- Improve cross-match confidence with probabilistic or multi-candidate matching in crowded fields.
- Add explicit saturation flags and bright-star handling policies.
- Build cleaner benchmark subsets with manual/visual audit for high-confidence evaluation.

### C) Modeling upgrades
- Keep XGB as reference baseline; compare with one calibrated neural baseline only if time permits.
- Use cVAE latents as auxiliary descriptors for error analysis before using them in final prediction.
- Explore cost-sensitive learning/weighting strategies for rare but important non-PSF cases.

### D) Evaluation and reproducibility
- Add calibration diagnostics and threshold-stability analysis for operational decisions.
- Report all key metrics by astrophysically meaningful slices, not only global averages.
- Publish a compact reproducibility package: dataset manifest, config, and script map.

## 3-week execution plan (realistic)
### Week 1: Freeze scope + evidence inventory
- Lock thesis core objective to PSF/non-PSF.
- Freeze dataset version and baseline evaluation protocol.
- Produce "evidence inventory" table:
  - what result exists,
  - what figure/table supports which chapter,
  - what is missing.
- Draft Chapters 1-3 (near-final text).

### Week 2: Final core results + write Methods/Results
- Run/confirm final baseline and robustness slices.
- Generate final plots/tables for Chapters 4-5.
- Draft Chapters 4-5 fully.
- Prepare one exploratory section (UMAP and/or cVAE) with explicit "extension" framing.

### Week 3: Interpretation, polishing, and defense packaging
- Write Chapter 6 with strong limitation/future-work clarity and finalize conclusions.
- Unify notation, captions, and figure references.
- Build defense-oriented summary:
  - 1-page executive abstract,
  - 10-12 slide backbone derived directly from chapter claims.
- Reserve final 2-3 days for edits only (no major new experiments).

## What to de-prioritize (to finish on time)
- Large method expansion without clear evaluation.
- Full cVAE maturity claims if diagnostics are incomplete.
- Too many label classes before binary task is stable.

## Suggested writing style rules for this thesis
- Every section starts with a scientific question.
- Every claim points to one figure/table.
- Every figure/table ends with a one-line takeaway sentence.
- Separate clearly:
  - what is measured,
  - what is inferred,
  - what remains uncertain.

## Clarification on "uncertainty calibration"
If included, this is a short subsection in evaluation:
- Goal: check whether predicted scores behave like probabilities.
- Practical output: reliability plot / calibration error / threshold stability note.
- Keep it lightweight if time is tight; do not let it block thesis completion.

## Immediate next actions (today)
1. Confirm final thesis question in one sentence (PSF/non-PSF core).
2. Create evidence inventory table mapping existing scripts/results to chapters.
3. Draft Introduction + Data Construction first (fastest stable chapters).
4. Freeze final experiment list before new exploratory runs.
