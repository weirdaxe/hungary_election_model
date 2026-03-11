# Turnout models (detailed)

This document mirrors the **Methodology → Turnout models** section in the Streamlit app.

All turnout models produce a station-level vote total `votes_i` such that:

- `votes_i = turnout_i × registered_i`
- `Σ_i votes_i = turnout_target × Σ_i registered_i` (exactly, after the final rescale step)
- `turnout_i` is clipped to `[turnout_clip_low, turnout_clip_high]`

Notation used below:

- `i` = polling station (2026 station universe)
- `g` = group at the selected granularity (station / location / settlement / evk)
- `registered_i` = 2026 registered voters at station `i`
- `T` = `turnout_target` (national turnout rate)
- `t_ref(i)` = turnout rate at station `i` in the selected **reference election** (aligned to 2026 station IDs via settlement fallback)
- `logit(x) = log(x/(1-x))`, `sigmoid(z)=1/(1+e^{-z})`

A final common post-processing step is applied to every model:

- Clip all station turnout rates to `[clip_lo, clip_hi]`
- Rescale all station vote totals by a common factor so that the national total matches `T × total_registered`
- Repeat a few iterations until both constraints are satisfied

Below is what each model specifically does before that final enforcement step.

---

## 1) `uniform`

Same turnout rate everywhere.

- `turnout_i = T`
- `votes_i = T × registered_i`

---

## 2) `scaled_reference`

Preserves the **shape** of a reference turnout map (e.g., 2022 list), scaled to match the target national turnout.

1) `base_votes_i = t_ref(i) × registered_i`
2) `k = (T × Σ registered_i) / (Σ base_votes_i)`
3) `votes_i = k × base_votes_i`

---

## 3) `logit_slope_original`

Station-specific turnout elasticity in logit space.

Estimation:

`logit(t_{i,e}) = α_i + β_i × logit(t_{nat,e})`

Prediction:

`turnout_i = sigmoid( α_i + β_i × logit(T) )`

---

## 4) `logit_slope_ep_offset`

Grouped elasticity + EP dummy.

Estimation:

`logit(t_{g,e}) = α_g + β_g × logit(t_{nat,e}) + γ_g × I(e is EP)`

Prediction for parliamentary 2026 (EP dummy = 0):

- `turnout_g = sigmoid( α_g + β_g × logit(T) )`
- `votes_g = turnout_g × Σ_{i∈g} registered_i`

Station distribution:

- `w_i = registered_i × t_ref(i)`
- `votes_i = votes_g × w_i / Σ_{k∈g} w_k`

---

## 5) `relative_offset_parl`

Parliamentary-only relative logit offsets.

- `offset_g = mean_parl [ logit(t_{g,e}) - logit(t_{nat,e}) ]`
- `logit(turnout_g) = logit(T) + offset_g`

---

## 6) `logit_slope_reserve_adjusted`

Starts from `logit_slope_ep_offset`, then shifts the change vs baseline toward groups with more headroom.

- baseline: `turnout_g_base` from reference map
- raw: `turnout_g_raw` from elasticity prediction
- `Δ_g = turnout_g_raw - turnout_g_base`

Headroom:

- upturn: `reserve_g = clip_hi - turnout_g_base`
- downturn: `reserve_g = turnout_g_base - clip_lo`

Reweight:

- `scale_g = (reserve_g / mean(reserve)) ^ reserve_adjust_power`
- `turnout_g_adj = clip(turnout_g_base + Δ_g × scale_g)`

---

## 7) `baseline_plus_marginal`

Baseline from reference map + allocate net change using marginal propensity.

- `base_votes_i = t_ref(i) × registered_i`
- `Δ_total = target_votes - Σ base_votes_i`
- `w_i = (mp_i+ε)^marginal_concentration × (cap_i+ε)`
- `votes_i = base_votes_i + Δ_total × w_i / Σ w_i`

---

## 8) `baseline_ref_plus_elastic_delta`

Anchor a baseline turnout level `B` using the reference map, then allocate the surplus/deficit using elasticity-implied deltas.

- baseline votes scaled to `B × total_registered`
- group delta pattern:
  - `turnout_g(B) = sigmoid( α_g + β_g × logit(B) )`
  - `turnout_g(T) = sigmoid( α_g + β_g × logit(T) )`
  - `Δ_votes_g_raw = (turnout_g(T) - turnout_g(B)) × registered_g`
- rescale `Δ_votes_g_raw` to sum to `Δ_total`
- distribute to stations within group and add to baseline
