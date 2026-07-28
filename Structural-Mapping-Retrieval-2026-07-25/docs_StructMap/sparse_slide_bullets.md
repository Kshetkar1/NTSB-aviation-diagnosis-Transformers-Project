# Sparse forward CPT slide — revised bullets (post–Maha meeting)

Use these on the slide. Say the **speaker note** aloud if he asks for detail.

---

## Slide bullets

- **Problem:** Some forward cells P(outcome | cause) have almost no data → raw counting gives 0% or 100%.
- **Zhang fix:** Beta-CDF smooths sparse ratios (we match Zhang here).
- **Our test:** Can narrative smoothing (similar incident stories) beat Beta-CDF?
- **How we tested:**
  - **158 cause→outcome pairs** where we trust the full-data answer (e.g. P(fire | fuel) over all years).
  - For each pair, **pretend** we only saw **n = 1, 2, 3, 5, or 10** incidents (hide the rest).
  - Repeat **400 random subsamples** per n; compare **4 methods:** raw count, Zhang cap, **Beta-CDF**, narrative.
  - **Held-out test:** hide each incident so narrative cannot peek at the full database.
- **Result:** Narrative did **not clearly beat** Beta-CDF on the fair held-out test → **use Beta-CDF** for sparse forward cells.
- **Table 7 unchanged:** still plain counting over 102 fires (diagnosis, not forward CPT).

---

## Speaker note (~45 sec)

> “For sparse **forward** probabilities — not Table 7 — some cells only have one or two incidents, so plain counting gives zero or one hundred percent. Zhang smooths those with a **Beta-CDF** curve.
>
> We wanted to see if **narrative smoothing** could do better. We couldn’t score real one-incident cells without knowing the truth, so we used about **158 cause-and-outcome pairs** where we **do** trust the full-data rate.
>
> Then we **pretended** we only had one, two, three, five, or ten incidents — **four hundred random subsamples** each — and asked four methods to guess against that known answer.
>
> On a **strict held-out test** — hiding each incident so narrative can’t peek — narrative **did not clearly beat** Beta-CDF. So we’re using **Beta-CDF** to match Zhang. **Table 7 stays plain counting** over 102 fires.”

---

## If he asks “what’s n?”

> “**n** is how many incidents we **pretend** to have for that one probability question after randomly hiding the rest — not 158, not 400. **158** is how many questions we tested; **400** is how many times we repeat each subsample so it’s not luck.”

---

## If he asks “what’s a pair?”

> “One probability question — one **cause** and one **outcome**, like P(fire | fuel).”
