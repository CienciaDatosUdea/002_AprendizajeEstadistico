import Mathlib.Data.Set.Basic

theorem union_comm {α : Type _} (A B : Set α) : A ∪ B = B ∪ A := by
  apply Set.ext
  intro x
  constructor
  · intro hx
    cases hx with
    | inl hA => exact Or.inr hA
    | inr hB => exact Or.inl hB
  · intro hx
    cases hx with
    | inl hB => exact Or.inr hB
    | inr hA => exact Or.inl hA