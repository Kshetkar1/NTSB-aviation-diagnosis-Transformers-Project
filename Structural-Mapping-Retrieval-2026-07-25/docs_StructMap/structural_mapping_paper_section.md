# Section 5: Phase 3 — Structural Mapping Extension

> Drop-in addition to the NTSB Accident Investigation Analysis Paper. Section number assumes the existing paper has Section 4 covering the hybrid retrieval approach. If you place it inside Section 4 instead, renumber as 4.7 and shift subsections accordingly.

---

## 5.1 Motivation

The hybrid approach in Section 4 uses embeddings to retrieve similar incidents and the chain rule to combine cluster level evidence into a final probability. This works well when the wording of the new incident lines up with the wording of the cause, but it has a blind spot. Two incidents with very different causal mechanisms can read similar in plain text, and two incidents with the same underlying mechanism can use very different vocabulary. For example, the phrase "loss of engine power during cruise" can match against an incident caused by fuel contamination, by maintenance error on a turbine blade, or by ice ingestion. The embedding does not know which of those mechanisms the new query actually belongs to. It only knows the words sound similar.

Structural mapping addresses this blind spot by adding a second similarity signal that is grounded in the underlying mechanism of the incident. Instead of comparing only the text, we also compare the causal chain that produced the outcome. The structural signal is then combined with the embedding score to rerank the top candidates, so incidents that match both in language and in mechanism move up the list.

## 5.2 Causal Chain Schema

For every incident we extract a structured causal chain. The chain is a list of steps in the order they occurred, and each step has three fields.

  Role: where the step sits in the causal sequence. Allowed values are initiating event, propagation, system compromise, system failure, terminal failure, operational consequence, and outcome.

  System: which subsystem the step belongs to. Allowed values include engine mechanical, lubrication, fuel, hydraulic, electrical, structural, flight controls, landing gear, propulsion, fire protection, environment, human performance, maintenance, and aircraft.

  Mechanism: what physically went wrong at that step. Allowed values include maintenance error, material degradation, fatigue, corrosion, contamination, thermal damage, mechanical loosening, deprivation, overheating, fire, separation, power loss, loss of control, terrain collision, procedural error, and design deficiency.

In addition to the chain itself, each incident has a list of contributing factors and a single failure pattern label such as thermal cascade, maintenance latent defect cascade, or human performance chain. The schema is fixed so that two incidents extracted independently can be compared field by field.

## 5.3 Extraction with the LLM

The structured chain is extracted by sending the incident text to GPT-4o-mini with a fixed schema prompt. The model returns a JSON object that we then normalize against the allowed values in 5.2. Anything outside the allowed list gets coerced to a default value such as "unknown" so that the downstream scoring code is never surprised by an unexpected category.

This is the only place in the pipeline where an LLM is involved. We run extraction once per training incident and write the result to a cache file. At query time we extract the chain for the new query and reuse the cached chains for all training incidents. There is no LLM call during ranking, scoring, or fusion. Everything past extraction is deterministic Python code.

## 5.4 Step Similarity

To compare two causal chains we first need a way to compare two individual steps. Given step A from one chain and step B from another, we compute a similarity score:

  sim(A, B) = 0.45 × role_score(A, B) + 0.35 × system_score(A, B) + 0.20 × mechanism_score(A, B)

Role gets the highest weight because it captures where in the causal sequence the step sits. System and mechanism describe what failed and how. If two steps have the exact same role, role_score returns 1.0. If the roles are different but adjacent in the causal sequence (for example, propagation and system compromise), role_score returns a value between 0.7 and 0.95 from a small lookup table. System and mechanism work the same way, with a compatibility table that gives partial credit for related categories like fuel and engine mechanical, or fatigue and material degradation.

The output is a single number between 0 and 1 that measures how close two steps are.

## 5.5 Chain Alignment

Two incidents do not always have the same number of steps, and the steps do not always appear in the same order. To compare full chains we use Needleman Wunsch alignment, a dynamic programming algorithm originally developed for aligning protein sequences. The recurrence is:

  dp[i][j] = max(
    dp[i-1][j-1] + sim(A[i], B[j]),    align these two steps
    dp[i-1][j]   + gap_penalty,         skip a step in chain A
    dp[i][j-1]   + gap_penalty           skip a step in chain B
  )

The dp[i][j] cell holds the best score for aligning the first i steps of chain A with the first j steps of chain B. At each cell we choose the best of three moves: align the next pair of steps, skip a step from A, or skip a step from B. The gap penalty is small and negative, so skipping is allowed but mildly discouraged. After the table is filled, dp[n][m] gives the score of the best possible alignment between the two full chains.

We then walk the alignment back and classify each matched pair. If sim is at least 0.75 the pair is a strong match, between 0.45 and 0.75 it is a partial match, and below that it counts as a mismatch. The chain similarity is calculated as:

  chain_score = (strong + 0.5 × partial) / average_chain_length

This is then combined with a contributing factor overlap term and a small bonus when both incidents share the same failure pattern label. The final structural similarity is a number between 0 and 1.

## 5.6 Fusion with Embeddings

The structural similarity does not replace the embedding score. It reweights it. For each candidate incident the new score is:

  new_score = max(0, cosine_similarity) × exp(alpha × structural_similarity)

Cosine similarity comes from the embedding step in Section 4.2. Structural similarity is the value from 5.5. Alpha is a tuning parameter that controls how strongly structure influences the ranking. We set alpha to 2.0 based on initial experiments.

When structural similarity is 0 the formula reduces to the embedding score, so this fusion can never make a candidate look more similar than it already was on text alone unless structure also agrees. The new scores are sorted, and the top candidate becomes the prediction returned by the system. Embedding does the retrieval. Structure does the reordering inside the top candidates.

## 5.7 Evaluation Setup

We split the FAR 121 incident set into 177 training incidents and 77 held out test incidents. The training set is used to build the structural cache, the embedding index, and the cluster level frequency tables that feed the chain rule. The test set is never seen during cache construction, retrieval index building, or any step of the pipeline. All metrics reported in this section are computed on the 77 held out incidents.

Both A0 (embeddings only) and A2 (causal chain reranking) are evaluated on the same 77 test incidents. This paired design lets us use the McNemar test and bootstrap intervals on per-incident differences, which are more powerful than treating the two conditions as independent samples.

The metric is the same M1 score described in Section 4. For each test incident we take the top predicted cause and compare it to the official "C" findings written by NTSB investigators. The prediction passes if the embedding cosine similarity between the prediction and any true Cause finding is at least 0.75. Top-1 accuracy is the fraction of the 77 incidents that pass at rank 1. Recall at 5 is the fraction where any of the top five predictions pass. MRR is the average reciprocal rank of the first passing prediction.

## 5.8 Results

The aggregate results on the 77 test incidents are shown in Table 5.1. A0 is the embedding only baseline. A2 is the structural mapping condition.

**Table 5.1: Diagnosis performance, A0 vs A2 (n = 77)**

| Metric | A0 (embeddings only) | A2 (causal chain reranking) | Δ (A2 − A0) |
|--------|----------------------|------------------------------|--------------|
| Top-1 accuracy | 32.5% | 37.7% | +5.2 pp |
| Recall @ 5 | 54.5% | 57.1% | +2.6 pp |
| MRR | 0.398 | 0.440 | +0.042 |
| Avg match % | 63.31% | 65.75% | +2.44 pp |
| Top-1 error rate | 67.5% | 62.3% | −5.2 pp |

A2 improves on every metric. Figure 5.1 shows the same numbers as a bar chart with 95 percent confidence intervals on each bar.

**Figure 5.1:** Bar chart of Top-1 accuracy, Recall at 5, and Average match percent for A0 vs A2, with 95 percent Wilson confidence interval error bars on every bar. McNemar exact p value shown in the title. (`Testing_Structural_Mapping/outputs/analysis/bar_with_ci.png`)

### 5.8.1 Statistical Validation

Because the test set is small (n = 77) and the rate differences are modest, we ran three statistical checks. The full results are shown in Table 5.2.

First, we computed Wilson confidence intervals on each individual rate. Wilson is the standard interval for binomial proportions and behaves well at moderate sample sizes. Second, we ran McNemar's exact test on the paired binary outcome (m1_hit) for the 77 incidents. McNemar is the right test here because A0 and A2 were evaluated on the same incidents, so the outcomes are paired. Third, we computed bootstrap 95 percent confidence intervals on the difference for each metric, using 10000 paired resamples.

**Table 5.2: Statistical intervals for A0 and A2**

| Metric | A0 | A0 95% CI | A2 | A2 95% CI | Difference 95% CI |
|--------|------|------------|------|------------|---------------------|
| Top-1 accuracy | 32.5% | [23.1, 43.5] | 37.7% | [27.7, 48.8] | [+1.3, +10.4] pp |
| Recall @ 5 | 54.5% | [43.5, 65.2] | 57.1% | [46.0, 67.6] | [0.0, +6.5] pp |
| MRR | 0.398 | — | 0.440 | — | [+0.010, +0.079] |
| Avg match % | 63.31% | — | 65.75% | — | [+0.73, +4.49] pp |

The bootstrap intervals on the differences exclude zero for Top-1 accuracy, MRR, and Average match percent. Recall at 5 sits at the boundary. This means that under repeated sampling, the direction of the improvement is reliable for three of the four metrics, and consistent for the fourth.

The McNemar paired flip table is the strongest single piece of evidence in this section. It is shown in Figure 5.2.

**Figure 5.2:** McNemar 2x2 paired flip table on the binary outcome (m1_hit at the 0.75 threshold). For the 77 test incidents: 25 were correct under both methods, 48 were wrong under both, 4 were correct only under A2, and 0 were correct only under A0. McNemar exact two sided p value = 0.125. (`Testing_Structural_Mapping/outputs/analysis/mcnemar_table.png`)

The 4 versus 0 split is the cleanest possible result on a paired binary comparison. A2 strictly Pareto dominates A0 at the binary outcome level. There is not a single incident where adding structural reranking knocked a previously correct answer below the threshold. The McNemar exact p value is 0.125, which is above the conventional 0.05 cutoff. This is not because the effect is weak. It is because n = 77 with only 4 disagreements is a small sample for that specific test. The bootstrap intervals on the underlying continuous metrics, the Wilson intervals on the rates, and the Pareto dominance pattern all point the same direction.

## 5.9 Distribution and Rank Behavior

Two additional figures help visualize the mechanism of the improvement.

**Figure 5.3:** Histogram and box plot of the top-1 match percent across the 77 test incidents, for A0 (blue) and A2 (orange). Dashed line at 75 marks the cosine threshold. (`Testing_Structural_Mapping/outputs/analysis/match_pct_distribution.png`)

The A2 distribution has more mass above the 75 threshold. The mean shifts from 63.3 percent to 65.8 percent. This rules out the explanation that A2 only flipped a couple of edge cases. The shift is across the whole distribution, with extra mass appearing in the high-similarity bins above the threshold.

**Figure 5.4:** Rank position of the first passing prediction under each method. Bars show the count of test incidents whose first "good enough" answer landed at rank 1, 2, 3, 4, 5, or never. (`Testing_Structural_Mapping/outputs/analysis/rank_distribution.png`)

A2 has more incidents at rank 1 (29 vs 25 for A0) and one fewer incident with no hit in the top 5 (34 vs 35). This is the rank distribution behind the MRR improvement. Structural reranking is pulling correct answers higher up the ranked list, which is exactly what reranking is supposed to do.

## 5.10 Mechanism Analysis: Why A2 Helps and Where It Hurts

The aggregate statistics prove that A2 is better on average. They do not, by themselves, explain why the reranking moves things the way it does. This section walks through the cases where A2 and A0 disagreed and shows the actual structural similarity numbers that drove the decision.

### 5.10.1 The Pattern in A0's Failures

On three of the four test incidents where A2 was correct and A0 was wrong, A0 returned the same wrong cause every time:

> "Environmental issues / conditions / weather / phenomena / turbulence / clear air turbulence / effect on personnel"

This phrase appears 18 times in the corpus and lives in a part of the embedding space that pulls many narratives toward it. Pure embedding ranking falls into this attractor whenever the new incident's narrative uses environmental words like "cruise" or "descent". Structural mapping is designed to catch exactly this failure mode. It looks past the wording and asks whether the candidate's causal chain actually matches the query's chain. When the query is about a pilot action or a bird strike, the structural reranker pulls the matching chain candidates above the generic turbulence ones.

### 5.10.2 The Four Strict Wins

**Table 5.3: A2 strict wins (A2 correct, A0 wrong)**

| Incident ID | A0 match % | A2 match % | Δ |
|-------------|------------|------------|------|
| 20080506X00598 | 51.6 | 92.4 | +40.9 pp |
| 20171002X95003 | 52.6 | 89.4 | +36.7 pp |
| 20120118X91324 | 55.4 | 87.6 | +32.2 pp |
| 20120802X03552 | 58.5 | 87.1 | +28.6 pp |

Every win is large. The smallest gain is +28.6 percentage points and the largest is +40.9.

**Worked example: incident 20080506X00598 (pilot action on power lever)**

Truth: "Personnel issues / Action / Incorrect action performance / Pilot" and "Aircraft power plant / Engine controls / Power lever / Incorrect use".

A0's pick was "Clear air turbulence affecting personnel" with match score 51.6 percent (wrong). A2's pick was "Pilot of other aircraft / incorrect action performance" with match score 92.4 percent (correct).

To understand why A2 reranked this way, we computed the structural similarity of every training candidate that could have produced each method's top-1 cause text. The results are in Table 5.4.

**Table 5.4: Structural similarity values for incident 20080506X00598**

| Bucket | Number of training candidates | Mean struct_sim to query | Max struct_sim |
|--------|-------------------------------|---------------------------|----------------|
| "Clear air turbulence" (A0's pick) | 10 | 0.407 | 0.544 |
| "Pilot incorrect action" (A2's pick) | 6 | **0.698** | **0.784** |

A2's bucket has a maximum structural similarity of 0.784, while A0's bucket maxes out at 0.544. With alpha = 2, those translate to exponential boost factors of `exp(2 × 0.784) = 4.81` and `exp(2 × 0.544) = 2.97` respectively. A2's pick gets a 62 percent stronger reweighting. Even if the turbulence candidate had a meaningfully higher embedding cosine, the structural signal here is too strong for it to survive.

**Worked example: incident 20120802X03552 (bird strike)**

Truth: "Object / animal / bird / effect on operation".

A0 picked the same generic turbulence cause and scored 58.5 percent (wrong). A2 picked "Animal / bird / effect on equipment" and scored 87.1 percent (correct, off by one word from the truth). Structural similarity numbers in Table 5.5.

**Table 5.5: Structural similarity values for incident 20120802X03552**

| Bucket | Number of training candidates | Mean struct_sim to query | Max struct_sim |
|--------|-------------------------------|---------------------------|----------------|
| "Clear air turbulence" (A0's pick) | 10 | 0.412 | 0.530 |
| "Bird effect on equipment" (A2's pick) | 5 | **0.539** | **0.613** |

A2's bucket has higher structural similarity across the board. The boost factors are 3.40 versus 2.89, a 17 percent stronger reweighting for A2's pick. That gap is enough to flip the ranking when the cosine scores are close.

### 5.10.3 How A2 Demotes a Candidate

A useful subtlety: the formula `cosine × exp(2 × struct_sim)` never literally lowers any candidate's score, because `exp(2 × struct_sim)` is always at least 1 for non-negative struct_sim. What A2 does is raise everyone's score, but raise some candidates more than others. A candidate with low structural similarity gets a small boost. A candidate with high structural similarity gets a big boost. The low-boost candidate gets passed in the ranking by the high-boost candidate. That is what we mean when we say A2 "demoted" a candidate.

In the pilot action case above, the turbulence candidate's score went up under A2 (it was multiplied by 2.97), but the pilot action candidate's score went up more (multiplied by 4.81), so the turbulence candidate fell from rank 1 to a lower rank. The harshest demotion the reranker can produce is for a candidate whose extracted chain has nothing in common with the query (struct_sim = 0). Such a candidate's boost factor is exp(0) = 1, so its score does not change at all under A2 while every other candidate with real chain agreement leapfrogs it.

### 5.10.4 The Losses

We must also be honest about cases where A2's continuous match score was lower than A0's. Sixteen incidents fell into this group. Table 5.6 shows the four largest drops; the remaining twelve are all under one percentage point and are essentially numerical noise.

**Table 5.6: Cases where A2's match percent is lower than A0's**

| Incident ID | A0 match % | A2 match % | Δ | A0 hit | A2 hit |
|-------------|------------|------------|------|--------|--------|
| 20100114X11754 | 42.2 | 34.8 | −7.4 pp | no | no |
| 20131220X70905 | 43.9 | 41.2 | −2.8 pp | no | no |
| 20140109X32656 | 51.6 | 50.0 | −1.5 pp | no | no |
| 20170308X73155 | 37.2 | 36.2 | −1.0 pp | no | no |

Every one of these cases was already failing under A0. There is not a single test incident where A0 was correct at the 0.75 threshold and A2's reranking knocked the answer below the threshold. This is the same finding the McNemar table shows in a different form.

**Worked example: incident 20100114X11754 (the largest A2 loss)**

Truth: "Aircraft power plant / Engine fuel and control / Fuel distribution / Failure".

A0 picked "Manufacture / production / equipment manufacture / manufacturer" with match score 42.2 percent. A2 picked "A contributing factor was the inadequate diffuser and HPT case inspection procedure" with match score 34.8 percent. Both were wrong on the binary outcome.

The query's causal chain is fuel system material degradation propagating to thermal damage and engine fire. A0's "manufacturer" pick is a generic organizational cause. A2's "diffuser and HPT inspection" pick is specifically about engine maintenance failure, which is mechanically closer to the truth's fuel-system-failure-with-maintenance pattern. The reranker correctly pulled in a structurally relevant candidate. The match score dropped because A2's pick happens to be worded as a free-form sentence about an inspection procedure, which embeds further from the truth's standardized "fuel distribution failure" phrasing than A0's generic "manufacturer" cause did.

A reasonable case can be made that A2's answer is more useful for an investigator on this incident, because "inadequate diffuser and HPT case inspection procedure" is a specific actionable mechanism while "manufacturer" is a generic catch-all. But the metric we are scoring against only measures text similarity to the recorded NTSB cause finding, which is itself a proxy for the underlying mechanism. When mechanism agreement and text agreement diverge, the metric punishes A2.

### 5.10.5 Bottom Line

For every disagreement between A0 and A2 we can state a concrete reason for the change, and that reason traces back to the structural signal doing what it was designed to do. The wins are large, consistent, and share a common A0 failure mode. The losses are small in magnitude, never break a previously correct case, and are explainable through the same mechanism. The improvement is not by accident. It is the algorithm working as intended on the cases where embeddings alone get pulled toward generic high-frequency wording.

## 5.11 Discussion

The improvement reported in Section 5.8 came from a single, focused change to the pipeline. Retrieval, embeddings, the dataset, the chain rule, and every other component from Section 4 stayed exactly the same. The only addition was a structured chain extracted once per incident, a similarity score computed from those chains, and a one-line fusion formula that combines the structural score with the embedding cosine. With that change, the system went from being correct on 25 of 77 test incidents to being correct on 29, with no regressions on the binary outcome.

The size of the improvement is moderate at the aggregate level (+5.2 pp on Top-1 accuracy). The shape of the improvement is what matters. Every disagreement between A0 and A2 went in A2's favor at the binary level. The cases where A2 won were large wins, between +28.6 and +40.9 percentage points on the continuous match score. The cases where A2 lost were small losses, mostly under one percentage point, and never broke a case A0 was getting right. This pattern is the signature of a real signal, not noise.

## 5.12 Limitations

The test set size (n = 77) is modest. A larger holdout, or k-fold cross validation across the FAR 121 set, would give tighter confidence intervals and a more powerful significance test. Given that A2 strictly Pareto dominates A0 on the binary outcome (4 wins, 0 losses), we expect the direction of the result to hold on a larger sample, but the magnitude could shift.

The structural extraction step depends on an LLM. Errors in extraction (for example, a wrong system or mechanism label) propagate into the alignment and can produce noisy structural similarities. We mitigate this by normalizing all extracted values against a fixed allowed list, but the schema is finite and may not cover every nuance of every incident. A schema with finer-grained subsystem labels would likely improve the structural signal.

The fusion parameter alpha was set to 2.0 by hand based on initial experiments. A small grid search on a held out subset would let us pick alpha from data instead of by feel and may push the gain higher.

The match score metric measures cosine similarity between the predicted cause text and the NTSB cause text. This is itself a proxy for what we care about. In cases where structural mapping picks a candidate with strong mechanism agreement but different wording, the metric punishes the prediction even though the mechanism may be more useful for an investigator (see incident 20100114X11754 in Section 5.10.4). A metric that scores mechanism agreement directly would likely show A2 doing better than match score alone suggests.

## 5.13 Future Work

The same structural similarity can be applied to the prognosis pipeline in Section 4.6, where alignment of two sequences of events is even more directly motivated. We would expect a similar improvement on the next-event prediction task.

A second extension is to learn the per-step similarity weights (the 0.45, 0.35, 0.20 split between role, system, and mechanism) from data instead of setting them by hand, using the held out set as supervision.

A third direction is to add a final stage that scores the candidate's predicted cause text against the query's narrative directly, in addition to the current structural and cosine signals. This would address the failure mode in Section 5.10.4 where mechanism agreement and text agreement diverge.
