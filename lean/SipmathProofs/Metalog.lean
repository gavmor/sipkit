module

public import Mathlib.Analysis.SpecialFunctions.Log.Deriv
public import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Metalog basis and monotonicity

Keelin (2016) metalog basis on `(0, 1)`: `g₁ = 1`, `g₂ = logit`, `g₃ = (y - 1/2) · logit`.
A metalog quantile function is feasible iff it is strictly increasing on `(0, 1)`.
-/

@[expose] public section

namespace SipmathProofs

/-- `logit y = ln (y / (1 - y))`; only meaningful on `(0, 1)`. -/
noncomputable def logit (y : ℝ) : ℝ := Real.log (y / (1 - y))

/-- Metalog basis functions `g₁`, `g₂`, `g₃`. -/
noncomputable def g₁ (_ : ℝ) : ℝ := 1
noncomputable def g₂ (y : ℝ) : ℝ := logit y
noncomputable def g₃ (y : ℝ) : ℝ := (y - 1/2) * logit y

/-- Unbounded 3-term metalog quantile function. -/
noncomputable def metalog3 (a₁ a₂ a₃ : ℝ) (y : ℝ) : ℝ :=
  a₁ * g₁ y + a₂ * g₂ y + a₃ * g₃ y

/-- `d/dy ln (y / (1 - y)) = 1 / (y (1 - y))` on `(0, 1)`. -/
theorem hasDerivAt_logit {y : ℝ} (hy0 : 0 < y) (hy1 : y < 1) :
    HasDerivAt logit (1 / (y * (1 - y))) y := by
  have h0 : y ≠ 0 := hy0.ne'
  have h1 : 1 - y ≠ 0 := sub_ne_zero.mpr hy1.ne'
  have hq : HasDerivAt (fun x => x / (1 - x)) ((1 * (1 - y) - y * -1) / (1 - y) ^ 2) y :=
    (hasDerivAt_id' y).fun_div ((hasDerivAt_id' y).const_sub 1) h1
  show HasDerivAt (fun x => Real.log (x / (1 - x))) _ y
  convert hq.log (div_ne_zero h0 h1) using 1
  field_simp
  ring

theorem logit_strictMonoOn : StrictMonoOn logit (Set.Ioo 0 1) := by
  have hd : ∀ y ∈ Set.Ioo (0 : ℝ) 1, HasDerivAt logit (1 / (y * (1 - y))) y :=
    fun y hy => hasDerivAt_logit hy.1 hy.2
  refine strictMonoOn_of_deriv_pos (convex_Ioo 0 1)
    (fun y hy => (hd y hy).continuousAt.continuousWithinAt) fun y hy => ?_
  rw [interior_Ioo] at hy
  rw [(hd y hy).deriv]
  exact one_div_pos.mpr (mul_pos hy.1 (sub_pos.mpr hy.2))

/-- A 2-term metalog is strictly increasing on `(0, 1)` iff its spread coefficient `a₂` is
positive. -/
theorem metalog2_strictMonoOn_iff (a₁ a₂ : ℝ) :
    StrictMonoOn (fun y => a₁ * g₁ y + a₂ * g₂ y) (Set.Ioo 0 1) ↔ 0 < a₂ := by
  constructor
  · intro h
    have hq : (1 / 4 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 := by constructor <;> norm_num
    have hh : (1 / 2 : ℝ) ∈ Set.Ioo (0 : ℝ) 1 := by constructor <;> norm_num
    have hlt : (1 / 4 : ℝ) < 1 / 2 := by norm_num
    have hl := logit_strictMonoOn hq hh hlt
    have h0 : logit (1 / 2) = 0 := by norm_num [logit]
    have hf := h hq hh hlt
    simp only [g₁, g₂, h0] at hf hl
    nlinarith
  · intro ha y hy z hz hyz
    have hl := logit_strictMonoOn hy hz hyz
    simp only [g₁, g₂]
    nlinarith

theorem metalog2_strictly_monotonic (a₁ a₂ : ℝ) (ha₂ : 0 < a₂) :
    StrictMonoOn (fun y => a₁ * g₁ y + a₂ * g₂ y) (Set.Ioo 0 1) :=
  (metalog2_strictMonoOn_iff a₁ a₂).mpr ha₂

end SipmathProofs
