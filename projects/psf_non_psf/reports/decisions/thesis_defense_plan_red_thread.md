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

## Proposed chapter architecture (reconstruction-first, committee-aligned)
1. Scientific Framing and Positioning  
2. Measurement Domain Gap and Dataset Construction  
3. Reconstruction Framework (Core Technical Contribution)  
4. Morphology-Aware Modeling and Failure Mitigation  
5. Representation Geometry and Error Topology  
6. Interpretation: Model Limits vs Gaia Information Ceiling  
7. Limitations, Future Program, and Conclusion  

## Chapter/subsection blueprint and minimum evidence
### 1) Scientific Framing and Positioning
Suggested subsections:
- 1.1 Scientific context and motivation (Gaia/Euclid complementarity)
- 1.2 Related work and gap statement (mandatory)
- 1.3 Core question and scope definition
- 1.4 Contributions and expected outcomes

Minimum evidence:
- One end-to-end workflow figure.
- One explicit primary research question and 2-3 supporting sub-questions.
- One precise positioning paragraph: what exists, what is missing, what this thesis adds.

### 2) Measurement Domain Gap and Dataset Construction
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
- One cross-match uncertainty estimate (ambiguity rate and/or contamination upper bound).

### 3) Reconstruction Framework (Core Technical Contribution)
Suggested subsections:
- 3.1 Cross-domain regression setup (Gaia stats -> normalized Euclid morphology representation)
- 3.2 PCA basis construction and target space definition
- 3.3 XGBoost regression for PCA coefficients
- 3.4 Evaluation in physical units (denormalization + chi2 protocol)
- 3.5 Required ablations (minimum three clean controls)

Minimum evidence:
- Reproducible train/validation protocol with leakage control statement.
- Clear primary metric set and model-selection rule.
- Three ablations:
  - without PCA (or reduced basis),
  - feature-group reduction,
  - simpler baseline (linear/shallow tree).

### 4) Morphology-Aware Modeling and Failure Mitigation
Suggested subsections:
- 4.1 PSF vs non-PSF split and weak-label constraints
- 4.2 Specialist models and MoE setup
- 4.3 Weighting and trade-off behavior (tail improvement vs central degradation)
- 4.4 Controlled hypothesis test: does specialization reduce extreme tails?

Minimum evidence:
- Incremental comparison table: baseline -> specialist -> MoE.
- Tail-focused statistics and central-statistics side-by-side.
- Explicit statement of gains, regressions, and instability.

### 5) Representation Geometry and Error Topology
Suggested subsections:
- 5.1 UMAP structure in Gaia feature space
- 5.2 Cluster behavior, double-star structure, and neighborhood effects
- 5.3 Link between representation geometry and reconstruction error tails
- 5.4 Case studies with pre-registered selection criteria

Minimum evidence:
- One reproducible geometry-to-error linkage.
- Case-study panel selected with fixed rules (not ad hoc/cherry-picked).
- Failure taxonomy tied to tail behavior and morphology collapse patterns.

### 6) Interpretation: Model Limits vs Gaia Information Ceiling
Suggested subsections:
- 6.1 What Gaia features demonstrably encode
- 6.2 Evidence for model underfitting reduction with added capacity
- 6.3 Residual structure that persists despite capacity increases
- 6.4 Information-ceiling hypothesis and falsifiable alternatives

Minimum evidence:
- Capacity ladder evidence (baseline -> richer models) with diminishing/partial gains.
- Persistent regime-specific residual patterns after model upgrades.
- Explicit alternative explanations and what future tests would disprove them.

### 7) Limitations, Future Program, and Conclusion
Suggested subsections:
- 7.1 Known limitations and mitigation status
- 7.2 Near-term roadmap (post-thesis continuation)
- 7.3 Final thesis contribution statement

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

## Mandatory evidence gates before final writing lock
1. Related-work positioning paragraph that names specific methodological gaps.
2. Cross-match uncertainty estimate (ambiguity/radius sensitivity/contamination bound).
3. Three reconstruction ablations with one comparison table.
4. Pre-registered case-study selection protocol.
5. Model-limit vs information-ceiling subsection with explicit competing explanations.

## 3-week execution plan (realistic)
### Week 1: Freeze scope + evidence inventory
- Lock thesis core objective to reconstruction-first claim.
- Freeze dataset version and baseline evaluation protocol.
- Produce "evidence inventory" table:
  - what result exists,
  - what figure/table supports which chapter,
  - what is missing.
- Draft Chapters 1-2 (near-final text) and Chapter 3 methods skeleton.

### Week 2: Final core results + write Methods/Results
- Run/confirm baseline + mandatory ablations + morphology-aware comparisons.
- Generate final plots/tables for Chapters 3-5.
- Draft Chapters 3-5 fully.
- Keep cVAE as clearly exploratory evidence only.

### Week 3: Interpretation, polishing, and defense packaging
- Write Chapters 6-7 with strong limitation/future-work clarity and finalize conclusions.
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
1. Confirm final thesis question in one sentence (reconstruction-first core claim).
2. Create evidence inventory table mapping existing scripts/results to chapters.
3. Draft Introduction + Data Construction first (fastest stable chapters).
4. Freeze final experiment list before new exploratory runs.
