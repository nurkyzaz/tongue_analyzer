# Color calibration — measured tradeoff & decision (WS3b, 2026-07-13)

Does Shades-of-Gray white balance (`infer.color_calibrate`, env `TIH_COLOR_CALIB`) help the colour
features? Re-tested now that v5 is WB-augmented. We lack diverse real-phone photos, so we **simulated
phone colour casts**: applied warm (tungsten) and cool (daylight) illuminants to each human-40 image and
measured coating-colour (`tai`) and body-colour (`zhi`) accuracy vs the human labels, off vs on, across
correction strengths (`evaluation/eval_color_calib.py`).

## Results (human-40, tai/zhi accuracy)

**tai (coating colour)** — calibration only helps:
| config | none | warm | cool |
|--------|-----:|-----:|-----:|
| off | 66% | 58% | 68% |
| on@0.35 | 66% | **61%** | **71%** |
| on@0.6 | 68% | **66%** | 71% |

**zhi (body colour)** — a real tradeoff:
| config | none | warm | cool |
|--------|-----:|-----:|-----:|
| off | **62%** | 54% | 41% |
| on@0.35 | 54% | 57% | **54%** |
| on@0.6 | 54% | 57% | 57% |

## Reading it
- **Calibration clearly helps `tai` everywhere** and **rescues both colours under casts** (cool-cast
  `zhi` 41→57). This is real: a colour cast otherwise over-reads yellow/red → spurious Damp-Heat.
- **But it costs `zhi` ~8pp on already-clean images** (62→54). Cause is fundamental: a face/tongue scene
  averages warm/pink, not grey, so Shades-of-Gray over-corrects and washes red out of the body colour.
  This happens at *every* strength (even 0.25 costs 5pp) — it's inherent to the grey-world assumption,
  not a tuning issue.
- Lower strength trims the clean-`zhi` damage a little and keeps most cast recovery; **0.35 is the knee**
  (tai never regresses, casts recover, clean-`zhi` cost bounded). The old default 0.6 costs the same
  clean-`zhi` for more cast recovery.

## Decision
**Keep calibration OFF by default** — by the project's rule (only change production if it clearly beats
the honest human metric), it doesn't: it trades away clean-body-colour, our known-weak axis. **But**:
- Default strength lowered **0.6 → 0.35** (`TIH_CC_STRENGTH`) so that *when* enabled it's the better
  operating point.
- **Recommendation: enable it (`TIH_COLOR_CALIB=1`, strength 0.35) once we have real-phone photos** that
  confirm the cast-recovery outweighs the clean-`zhi` cost on true inputs. Real phone captures have casts
  (the whole point), so calibration is likely net-positive in production — but that needs the real-phone
  eval to prove, not synthetic casts.

## What would make this decisive (WS3b part b — blocked on data)
A **real-phone eval set** across skin tones / lighting, human-labeled for `tai`/`zhi`. Then re-run the
off-vs-on comparison on real casts. This is the only way to settle the enable/disable question; synthetic
casts got us the strength and the shape of the tradeoff, not the real-world mix.

## Better long-term fix (roadmap)
Shades-of-Gray is the wrong prior for faces. Options: (a) estimate the illuminant from a **near-neutral
reference** in-frame (teeth/sclera, or an optional grey card); (b) a small **learned white-balance** head
trained on paired cast/clean tongue images; (c) cast-magnitude-gated correction. All need data; parked.
