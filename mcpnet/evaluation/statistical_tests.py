"""
Paired statistical significance testing for the ablation tables.

Wilcoxon signed-rank test is used for pairwise comparisons (rather than a
paired t-test) since Dice scores are bounded in [0, 1] and often
non-normally distributed / skewed near 1.0 — Wilcoxon doesn't assume
normality.

For tables with 3+ variants (Table A: 3 variants, Table C: 4 variants), run
`friedman_omnibus` first as an overall test, and only follow up with
pairwise `compare_variants` (preferably with a multiple-comparisons
correction) if the omnibus result is significant. Running several pairwise
Wilcoxon tests without this first step inflates false-positive risk
(multiple comparisons problem).
"""

from scipy.stats import friedmanchisquare, wilcoxon
import pandas as pd


def friedman_omnibus(df, metric_col, variant_col, case_id_col="case_id"):
    """Omnibus test across ALL variants at once (3+ variants). Ranks each
    case's variants against each other, then tests whether the ranking
    pattern is consistent across cases or effectively random.

    Returns dict with statistic, p_value, and the variant list tested.
    Follow up with pairwise `compare_variants` only if p_value < 0.05.
    """
    pivot = df.pivot(index=case_id_col, columns=variant_col, values=metric_col).dropna()
    variants = list(pivot.columns)
    if len(variants) < 3:
        raise ValueError(
            f"Friedman test needs 3+ variants, got {len(variants)}: {variants}. "
            "Use compare_variants (Wilcoxon) directly for a 2-variant comparison instead."
        )
    stat, p = friedmanchisquare(*[pivot[v] for v in variants])
    return {"statistic": stat, "p_value": p, "n_cases": len(pivot), "variants": variants}


def compare_variants(df, metric_col, variant_col, baseline_variant, case_id_col="case_id", correction=None):
    """Compares `baseline_variant` against every other variant present in df,
    on a single metric column, using per-case paired values (matched on
    case_id_col so the pairing is correct even if row order differs).

    `correction`: None, or "bonferroni" — if set, multiplies each p-value by
    the number of comparisons made (use this when following up a significant
    Friedman result with 2+ pairwise tests, e.g. Table C's 2x2).

    Returns a DataFrame: one row per non-baseline variant, with the Wilcoxon
    statistic and p-value.
    """
    pivot = df.pivot(index=case_id_col, columns=variant_col, values=metric_col)

    if baseline_variant not in pivot.columns:
        raise ValueError(f"Baseline variant '{baseline_variant}' not found in data.")

    other_variants = [v for v in pivot.columns if v != baseline_variant]
    n_comparisons = len(other_variants)

    if n_comparisons == 0:
        print(f"Warning: no other variants to compare against baseline '{baseline_variant}' "
              f"(only variant present). Returning an empty comparison table — check your config "
              f"if this wasn't intentional (e.g. Table B needs snapshot_counts with 2+ values "
              f"and/or include_single_fixed_lr_baseline=True to have anything to compare).")
        return pd.DataFrame(columns=["variant", "n_cases", "statistic", "p_value"])

    results = []
    for variant in other_variants:
        paired = pivot[[baseline_variant, variant]].dropna()
        if len(paired) < 2:
            results.append({"variant": variant, "n_cases": len(paired), "statistic": None, "p_value": None})
            continue
        stat, p = wilcoxon(paired[baseline_variant], paired[variant])
        if correction == "bonferroni":
            p = min(p * n_comparisons, 1.0)
        results.append({"variant": variant, "n_cases": len(paired), "statistic": stat, "p_value": p})

    return pd.DataFrame(results)


def annotate_significance(p_value, thresholds=((0.001, "***"), (0.01, "**"), (0.05, "*"))):
    """Returns the significance-star string for a table footnote/column."""
    if p_value is None:
        return "n/a"
    for threshold, stars in thresholds:
        if p_value < threshold:
            return stars
    return "ns"
