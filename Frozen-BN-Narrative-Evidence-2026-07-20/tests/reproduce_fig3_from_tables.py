#!/usr/bin/env python3
"""Maha's question (Jul 15 meeting): "If you take the same CPT values Zhang used
in Tables 3, 4 and 5, do you get the same answers he is getting?"

Answer: YES. Plugging the PRINTED Table 3/4/5 values into plain Bayes' rule
reproduces every number in Fig. 3 (pages 4-5):

  Fig 3(b)  evidence: brake wear = Yes
      P(fire)   -> 93%      P(damage) -> 86%
  Fig 3(c)  evidence: fire = Yes
      P(wiring) -> 67%      P(brake)  -> 33%     P(damage) -> 92%

This proves the Section 3 tutorial is INTERNALLY consistent -- the outputs
follow from the printed inputs -- while the inputs themselves (Table 3 priors,
Table 4 CPT) are not derivable from the NTSB data (see build_table4_analogue.py
and the Eq. 9 vs Table 4 contradiction). Offline, no dependencies. Exit 0 = all
Fig. 3 numbers reproduced.
"""

# ---- the paper's printed inputs (verbatim) ---------------------------------
# Table 3 (p.5): marginal priors
P_B = 0.0001    # P(x1=1)  landing gear normal brake system wear
P_W = 0.0002    # P(x2=1)  electrical system wiring overheating

# Table 4 (p.5): CPT of fire  P(x3=1 | brake, wiring)
P_FIRE = {(1, 1): 0.99, (1, 0): 0.93, (0, 1): 0.95, (0, 0): 2e-9}

# Table 5 (p.5): CPT of damage  P(x4=1 | fire)
P_DMG_GIVEN_FIRE, P_DMG_GIVEN_NOFIRE = 0.92, 0.0

# Fig. 3 published outputs (p.5 text + figure panels)
PUBLISHED = {
    "3b P(fire | brake=Yes)":    0.93,
    "3b P(damage | brake=Yes)":  0.86,
    "3c P(wiring | fire=Yes)":   0.67,
    "3c P(brake | fire=Yes)":    0.33,
    "3c P(damage | fire=Yes)":   0.92,
}


def joint(b, w):
    """P(brake=b, wiring=w) from Table 3 (independent roots)."""
    return (P_B if b else 1 - P_B) * (P_W if w else 1 - P_W)


def main():
    # ---- Fig 3(b): condition on brake wear = Yes ----------------------------
    p_fire_b = sum(P_FIRE[(1, w)] * (P_W if w else 1 - P_W) for w in (0, 1))
    p_dmg_b = P_DMG_GIVEN_FIRE * p_fire_b + P_DMG_GIVEN_NOFIRE * (1 - p_fire_b)

    # ---- Fig 3(c): condition on fire = Yes (Bayes' rule, Eq. 5) -------------
    p_fire = sum(P_FIRE[(b, w)] * joint(b, w) for b in (0, 1) for w in (0, 1))
    p_w_fire = sum(P_FIRE[(b, 1)] * joint(b, 1) for b in (0, 1)) / p_fire
    p_b_fire = sum(P_FIRE[(1, w)] * joint(1, w) for w in (0, 1)) / p_fire
    p_dmg_fire = P_DMG_GIVEN_FIRE  # damage depends on fire only (Table 5)

    computed = {
        "3b P(fire | brake=Yes)":    p_fire_b,
        "3b P(damage | brake=Yes)":  p_dmg_b,
        "3c P(wiring | fire=Yes)":   p_w_fire,
        "3c P(brake | fire=Yes)":    p_b_fire,
        "3c P(damage | fire=Yes)":   p_dmg_fire,
    }

    print("Plugging Zhang's PRINTED Table 3/4/5 values into Bayes' rule:\n")
    print(f"  (prior P(fire) from his tables = {p_fire:.4e} -- Fig 3(a) scale)\n")
    ok = True
    for k, pub in PUBLISHED.items():
        got = computed[k]
        # Fig. 3 reports whole percentages -> match after rounding to 2 dp
        good = abs(round(got, 2) - pub) < 1e-9
        ok = ok and good
        print(f"  {'OK ' if good else 'XX '} {k:28} computed={got:.4f}  "
              f"published={pub:.2f}")

    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} -- his tutorial outputs follow "
          "exactly from his printed inputs.")
    print("=> The Section 3 example is internally consistent; the QUESTION was "
          "never the math,\n   it is where the Table 3/4 INPUT values came from "
          "(not derivable from the data).")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
