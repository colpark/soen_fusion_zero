# Spurious "end of signal" cue — risk and diagnosis

## Is this a real risk?

**Yes, when training disrupt-only.** In that setup:

- Every training sample comes from a shot that **eventually disrupts**.
- The **target** is 0 before disruption and 1 from disruption to the end of the window (`first_disrupt` to end).
- So the model only ever sees sequences where "the right part" is labeled 1. It can learn shortcuts such as:
  - **Position:** "later in the 7813-step window → more likely disrupt."
  - **End of shot:** "signal ending / drop near the end" (flattop end or disruption) as a proxy for the label.

Those cues are **spurious** for real-time use: at inference we do not know where the shot will end, and we care about *precursor* content, not "we're near the end of the segment."

## How to diagnose

### 1. **Evaluate on clear (non-disruptive) shots**

- Use the same pipeline but **only clear shots** (no disruption).
- Run the model on full-shot or long windows from clear shots.
- **Healthy:** Predictions stay near 0 over the whole shot.
- **Spurious:** Predictions ramp up or spike **near the end of the shot** (where the signal simply ends). That suggests the model is using "end of signal" rather than true precursors.

### 2. **Position-of-disruption ablation**

- For **disrupt** windows, compute where the disruption falls in the window:  
  `frac = first_disrupt / T` (e.g. 0–0.2, 0.2–0.4, …, 0.8–1.0).
- Evaluate accuracy/F1 (or your metric) **per bin**.
- **Healthy:** Similar performance across bins (model uses content, not position).
- **Spurious:** Much better when disruption is in the **second half** (or last 20%) of the window; weak when disruption is in the first half.

### 3. **Mask or corrupt the tail at test time**

- At evaluation, **zero out** (or replace with a constant) the **last K timesteps** of each window.
- Compare metric with and without masking.
- **Healthy:** Small drop (model uses full context).
- **Spurious:** Large drop when the tail is masked → model is leaning on the end of the window.

### 4. **Train with clear shots**

- Add **clear-shot** data (e.g. `--clear-decimated-root` and a clear shot list) so the model sees many windows with **all-zero** labels (no disruption).
- Then "end of signal" is no longer a reliable cue (clear shots also end). Retrain and re-run diagnostics 1–3.

---

## Quick reference

| Check | What to do | Spurious if … |
|-------|------------|----------------|
| Clear-shot eval | Predict on clear shots only | Predictions ↑ near shot end |
| Position ablation | Metric vs. `first_disrupt/T` bin | Good only when disrupt in 2nd half |
| Tail masking | Zero last K steps at test | Big metric drop with masking |
| Clear in training | Add clear shots to training | (Mitigation, not a test) |
