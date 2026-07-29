# Redaction residual-leak probe

n = 296 held-out narratives (2007-2019); 241 changed by redaction. Probe training pool: 1286 window narratives (1982-2006).

Probe 1 (window-trained, the clean audit): TF-IDF (1-2 grams) + logistic regression fit ONLY on window narratives, scored once on the held-out set. Probe 2 (in-sample 5-fold CV on the held-out set) is a worst-case diagnostic upper bound.

## Injury

| Text | Window-trained probe (clean) | In-sample CV probe (upper bound) |
|---|---|---|
| majority class | 58.4% | 58.4% |
| full narrative | 92.2% | 89.5% |
| redacted narrative | 92.2% | 89.2% |

Top-weight tokens on REDACTED text (leak check -- these must be mechanism words, not outcome words):

- **FATL**: and postcrash, forces and, impact forces, crashed, 2013 about, postcrash, postcrash fire, cargo, and first, cargo flight, 2013, forces
- **SERS**: turbulence, her, attendant, flight attendant, encountered, fa, the flight, attendants, flight attendants, flight, she, passenger
- **MINR**: touchdown, brake, main, gear, nut, landing gear, nlg, the nut, emergency, landing, ring, the nlg
- **NONE**: engine, the left, tug, taxiway, wing, the tug, fatigue, left, struck, gate, the right, ground

## Damage

| Text | Window-trained probe (clean) | In-sample CV probe (upper bound) |
|---|---|---|
| majority class | 42.6% | 42.6% |
| full narrative | 75.7% | 74.0% |
| redacted narrative | 73.3% | 72.3% |

Top-weight tokens on REDACTED text (leak check -- these must be mechanism words, not outcome words):

- **DEST**: cargo, cargo flight, fire the, captain and, and postcrash, crashed, forces and, impact forces, fire, 2013 about, postcrash fire, postcrash
- **SUBS**: tug, fuselage, accident, the tug, the accident, were to, pushback, skin, ramp, gate, the ramp, struck
- **MINR**: revealed, gear, landing gear, examination of, examination, main landing, right, engine, the engine, crack, failure, the right
- **NONE**: turbulence, flight attendant, attendant, her, encountered, flight, the flight, she, cabin, fa, galley, the turbulence
