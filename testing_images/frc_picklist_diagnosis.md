
# FRC Picklist Generator Analysis – Diagnostic Summary

## ✅ Step-by-Step Breakdown of Ranking Logic (Used in External Validation)

### 1. Load and Filter Data
- Loaded the `unified_event_2025lake.json` file.
- Excluded teams that were already selected:  
  `8044, 5653, 4087, 16, 456, 2036, 3526, 8808`

### 2. Metrics and Weights Used for Ranking
| Metric Description              | Field in Data                      | Weight |
|--------------------------------|------------------------------------|--------|
| Teleop Coral L1                | `_where_(teleop)_[coral_l1]`      | 3.0    |
| Defensive Capability           | `driver_skill`                    | 2.0    |
| Endgame Score                  | `endgame_score`                   | 2.0    |
| Auto Mobility Score            | `auto_mobility_score`             | 2.0    |
| Auto Score                     | `auto_score`                      | 1.5    |

### 3. Per-Team Average Calculation
- For each team: calculated the average of the metrics listed above across all their matches.

### 4. Weighted Score Calculation
- Formula used:
```text
weighted_score = (
    avg_teleop_coral_l1 × 3.0 +
    avg_driver_skill × 2.0 +
    avg_endgame_score × 2.0 +
    avg_auto_mobility_score × 2.0 +
    avg_auto_score × 1.5
)
```

### 5. Sort and Rank
- Teams were sorted in descending order of their `weighted_score`.

---

## 🔍 Diagnosis & Feedback on GPT-Based Generator

### 1. GPT is Ranking, But Shouldn’t Be Calculating
GPT is asked to *interpret and weight metrics*, which risks imprecision. GPT struggles with math-heavy logic across long lists.

**✔ Recommendation:**  
Precompute weighted scores in Python and pass them to GPT. Instruct GPT to use them as primary sort unless overridden by synergy.

---

### 2. Weight Application Lacks Formality
Prompt says:
> “Weight 3.0: These metrics are the PRIMARY deciding factors…”

But GPT has no internal formula engine to enforce this.

**✔ Recommendation:**  
Include a calculation example, or precompute score yourself and supply it as a key field.

---

### 3. Short Reason Limit Can Mislead Model
> “Each reason must be ≤ 12 words and cite ≥1 metric value.”

This forces superficial focus (e.g., “Teleop avg 6”) and may skew the ranking.

**✔ Recommendation:**  
Allow longer reasons (15–18 words) during testing to increase rationale diversity.

---

### 4. Risk of Token Limit Truncation
With 60+ teams and full stat profiles, the prompt may exceed GPT token limits—leading to cutoffs and incomplete ranking logic.

**✔ Recommendation:**  
Log prompt length, trim unused data, or batch into 20–25 team sets.

---

### 5. Validation Rules Are Great – Keep Them
Your system prompt correctly enforces:
- No duplicates
- All teams must appear once
- Must follow JSON schema

These are strong safeguards—retain them.

---

## ✅ Action Plan

1. Modify `_prepare_team_data_for_gpt()` to compute `weighted_score` for each team.
2. Add `weighted_score` to the prompt.
3. Instruct GPT:
   > “Sort primarily by weighted_score unless synergy/strategy dictates otherwise.”
4. Test output against manual calculation (like the one already validated).

---

*Prepared by ChatGPT for Daniel Eiland (May 2025)*
