# The 300 ms (Twarn) window and false positives

## What is Twarn?

**Twarn** is the *warning window*: the interval **(t_disrupt − Twarn, t_disrupt]** in samples at 1 MHz.

- Default: **300,000 samples** = **300 ms** before disruption.
- It defines the “at-risk” region we care about. By default we **label every timestep in that window as 1 (disruptive)** and every timestep **before** it as 0 (clear), and we train on both.
- Because we *assume* that whole 300 ms is “disruptive,” the model is forced to predict 1 there; if the true precursors only appear in the last 50–100 ms, the rest of the window can look like clear plasma and lead to **false positives** when the model sees similar clear regions elsewhere.

## Default: training on the 300 ms window

- Every timestep **within** the Twarn window is labeled **1**; every timestep **before** it is **0**.
- We train on both; the receptive field is ~30 ms, so the model uses preceding context.

## Why you might see many false positives

1. **300 ms may be too long**  
   If precursors only show up in the last 50–100 ms, the first 200–250 ms of the window is still “clear” plasma. We label it 1 anyway, so the model learns to predict 1 there. At test time, any similar-looking clear region can then be predicted as 1 → **false positives**.

2. **Sharp boundary**  
   The label switches from 0 to 1 at exactly `t_disruption - 300 ms`. Physics may change more gradually, so the model sees similar signals on both sides and can fire 1 a bit early → **FPs near the boundary**.

3. **Class balance**  
   Even with stratified batching, most *timesteps* are clear. If the model is uncertain, it may tend to predict 1 in clear regions that look slightly like the 300 ms window.

## Options implemented

### 0. **Ignore the Twarn window** (`ignore_twarn`, recommended to reduce FPs)

- **Do not train on the 300 ms window at all:** set **weight = 0** for every timestep in (t_disrupt − Twarn, t_disrupt].
- Only timesteps **before** the window are labeled 0 (clear) and used for loss; clear shots are still full 0.
- **Rationale:** Whether that pre-disruption signal is “disruptive” or “clear” is **learned** from the data (and from clear shots) rather than assumed. This avoids teaching the model “this whole 300 ms is 1” when much of it may still be clear.
- **Usage:**  
  `--ignore-twarn`  
  or in Python: `ECEiTCNDataset(..., ignore_twarn=True)`  
- Effect: no positive (1) labels from disruptive shots; only clear (0) is trained. The model can still output high values in the Twarn region at inference if it learns that the signal there is different from clear.

### 1. **Exclude the last N ms** (`exclude_last_ms`, default 0)

- Do **not** label the last N ms before disruption as 1; leave that segment out of the loss (weight 0).
- **Rationale (Churchill et al.):** “Not training on the last 30 ms before a disruption since this is a minimum amount of time needed to trigger mitigation systems.” Training on “too late” signal can also confuse the model and contribute to FPs.
- **Usage:**  
  `--exclude-last-ms 30`  
  or in Python: `ECEiTCNDataset(..., exclude_last_ms=30.0)`  
- Effect: label 1 only in **(t_disrupt - 300 ms, t_disrupt - 30 ms]**; the last 30 ms is not used as a positive target.

### 2. **Shorten Twarn** (e.g. 150 ms)

- Label as 1 only the last 150 ms before disruption.
- **Pros:** Tighter “disruptive” definition; fewer clear-like timesteps labeled 1 → can reduce FPs.
- **Cons:** Less warning time; may increase false negatives.
- **Usage:** `--twarn 150000` (150,000 samples at 1 MHz).

### 3. **Lengthen Twarn** (e.g. 500 ms)

- Label as 1 the last 500 ms.
- **Pros:** More pre-disruption context.
- **Cons:** More of the shot is labeled 1; if precursors are only in the last 100 ms, the extra 400 ms can look like “clear” and encourage the model to predict 1 in similar clear regions → **can increase FPs**.

## Recommendation

- To **avoid assuming** the 300 ms is disruptive and let the model learn the boundary: use **`--ignore-twarn`**. Only clear (0) is trained; the Twarn window is masked from the loss.
- If you prefer to keep explicit “1” labels but reduce FPs: try **`--exclude-last-ms 30`** (Churchill-style) and/or **shorten Twarn** (e.g. `--twarn 150000`).
- Keep **clear (non-disruptive) shots** in the dataset so the model sees true clear plasma; that is the most important lever for reducing FPs.
