# %% [markdown]
# ## Quarterly visitor spending: recovery and composition
# This supplementary experiment joins 16 published DSEC spending estimates to
# calendar-quarter sums of daily administrative arrivals. The selected years are
# 2019, 2023, 2024 and 2025. All amounts are nominal MOP and exclude gaming.
# The calculations are descriptive ratios, shares, grouped comparisons and a
# consistency audit. Survey estimates and administrative counts remain distinct.

# %%
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SPENDING_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
SPENDING_DATA = SPENDING_ROOT / "data"
SPENDING_RESULTS = SPENDING_ROOT / "results"
SPENDING_FIGURES = SPENDING_ROOT / "figures"
for spending_dir in (SPENDING_DATA, SPENDING_RESULTS, SPENDING_FIGURES):
    spending_dir.mkdir(exist_ok=True)
SPENDING_INPUT_NAME = "macao_quarterly_spending_2019_2023_2025.csv"
# Read the fixed input files included in this share package.

SPENDING_YEARS = [2019, 2023, 2024, 2025]
SPENDING_KINDS = ["total", "overnight", "same_day"]
SPENDING_YEAR_COLORS = {2019: "#595959", 2023: "#0072B2", 2024: "#009E73", 2025: "#D55E00"}
SPENDING_YEAR_MARKERS = {2019: "o", 2023: "s", 2024: "^", 2025: "D"}
plt.rcParams.update({"figure.dpi": 135, "savefig.dpi": 180, "font.size": 10,
                     "axes.titlesize": 12, "axes.spines.top": False,
                     "axes.spines.right": False})

def spending_save_table(frame, filename):
    frame.to_csv(SPENDING_RESULTS / filename, index=False, encoding="utf-8-sig")

def spending_save_figure(fig, filename, footnote):
    fig.text(.015, .015, footnote, fontsize=8, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, .085, 1, .93))
    fig.savefig(SPENDING_FIGURES / filename, bbox_inches="tight")
    plt.close(fig)
    if "get_ipython" in globals():
        from IPython.display import Image, display
        display(Image(filename=str(SPENDING_FIGURES / filename)))

# %% [markdown]
# ### Input and join checks
# Every spending row is one individual quarter, not a cumulative period. Daily
# records are unique and continuous, and each selected quarter must contain all
# calendar dates. Existing same-day imputations and one-person discrepancies are
# counted explicitly. No rows or published values are adjusted to force equality.

# %%
spending = pd.read_csv(SPENDING_DATA / SPENDING_INPUT_NAME)
spending_daily = pd.read_csv(
    SPENDING_DATA / "macao_daily_total_overnight_sameday_2017_2025.csv",
    parse_dates=["date"])
assert len(spending) == 16 and not spending.duplicated(["year", "quarter"]).any()
assert set(zip(spending.year, spending.quarter)) == {
    (year, quarter) for year in SPENDING_YEARS for quarter in range(1, 5)}
assert spending.period_basis.eq("individual_quarter").all()
assert spending.price_basis.eq("nominal_current_prices").all()
assert spending_daily.date.is_unique
assert spending_daily.date.sort_values().tolist() == pd.date_range("2017-01-01", "2025-12-31").tolist()
assert spending_daily[["total", "overnight", "same_day_analysis"]].notna().all().all()
assert spending_daily[["total", "overnight", "same_day_analysis"]].ge(0).all().all()
spending_daily["year"] = spending_daily.date.dt.year
spending_daily["quarter"] = spending_daily.date.dt.quarter
spending_daily["raw_same_day_missing"] = spending_daily.same_day_raw.isna()
spending_daily["raw_type_additivity_flag"] = spending_daily.raw_additivity_residual.fillna(0).ne(0)
spending_daily_selected = spending_daily[spending_daily.year.isin(SPENDING_YEARS)].copy()
spending_quarter_arrivals = spending_daily_selected.groupby(["year", "quarter"]).agg(
    total_arrivals=("total", "sum"), overnight_arrivals=("overnight", "sum"),
    same_day_arrivals=("same_day_analysis", "sum"),
    same_day_raw_known_sum=("same_day_raw", "sum"),
    observed_days=("date", "size"), first_date=("date", "min"), last_date=("date", "max"),
    raw_same_day_missing_days=("raw_same_day_missing", "sum"),
    raw_type_additivity_flag_days=("raw_type_additivity_flag", "sum")
).reset_index()
spending_quarter_arrivals["expected_days"] = [
    pd.Period(f"{row.year}Q{row.quarter}", freq="Q").end_time.normalize().dayofyear -
    pd.Period(f"{row.year}Q{row.quarter}", freq="Q").start_time.dayofyear + 1
    for row in spending_quarter_arrivals.itertuples()]
assert spending_quarter_arrivals.observed_days.eq(spending_quarter_arrivals.expected_days).all()
spending_quarter_arrivals["analysis_type_sum_minus_total_arrivals"] = (
    spending_quarter_arrivals.overnight_arrivals + spending_quarter_arrivals.same_day_arrivals -
    spending_quarter_arrivals.total_arrivals)
spending_joined = spending.merge(spending_quarter_arrivals, on=["year", "quarter"],
                                 how="left", validate="one_to_one", indicator=True)
assert spending_joined._merge.eq("both").all()
spending_joined = spending_joined.drop(columns="_merge").sort_values(["year", "quarter"]).reset_index(drop=True)
assert pd.to_datetime(spending_joined.period_start).eq(spending_joined.first_date).all()
assert pd.to_datetime(spending_joined.period_end).eq(spending_joined.last_date).all()

# %% [markdown]
# ### Recovery ratios and visitor-type shares
# Each recovery index equals 100 × the current-quarter value divided by the
# corresponding 2019-quarter value. The three displayed indices retain their
# own published/administrative source definitions. Overnight shares use the
# reported overall total as denominator; rounding discrepancies are preserved.

# %%
spending_baseline = spending_joined[spending_joined.year.eq(2019)].set_index("quarter")
spending_recovery_columns = []
for spending_kind in SPENDING_KINDS:
    for spending_measure, spending_suffix in (
        (f"{spending_kind}_arrivals", "arrivals"),
        (f"{spending_kind}_spending_million_mop", "nominal_spending"),
        (f"{spending_kind}_per_capita_spending_mop", "official_per_capita_spending"),
    ):
        spending_index_name = f"{spending_kind}_{spending_suffix}_index_2019q100"
        spending_joined[spending_index_name] = (
            100 * spending_joined[spending_measure] /
            spending_joined.quarter.map(spending_baseline[spending_measure]))
        spending_recovery_columns.append(spending_index_name)
spending_joined["overnight_arrivals_share_pct"] = 100 * spending_joined.overnight_arrivals / spending_joined.total_arrivals
spending_joined["overnight_spending_share_pct"] = 100 * spending_joined.overnight_spending_million_mop / spending_joined.total_spending_million_mop
spending_joined["same_day_arrivals_share_pct"] = 100 * spending_joined.same_day_arrivals / spending_joined.total_arrivals
spending_joined["same_day_spending_share_pct"] = 100 * spending_joined.same_day_spending_million_mop / spending_joined.total_spending_million_mop
spending_joined["overnight_to_same_day_official_per_capita_ratio"] = spending_joined.overnight_per_capita_spending_mop / spending_joined.same_day_per_capita_spending_mop
for spending_share in ("overnight_arrivals_share_pct", "overnight_spending_share_pct"):
    spending_share_base = spending_joined[spending_joined.year.eq(2019)].set_index("quarter")[spending_share]
    spending_joined[spending_share.replace("_pct", "_change_pp_vs_2019q")] = (
        spending_joined[spending_share] - spending_joined.quarter.map(spending_share_base))
assert np.allclose(spending_joined.loc[spending_joined.year.eq(2019), spending_recovery_columns], 100)

# %% [markdown]
# ### Official versus derived per-capita spending
# The consistency calculation is published spending (million MOP × 1,000,000)
# divided by the corresponding daily-arrival sum. It is labelled "derived" and
# never replaces official per-capita spending. Signed differences can reflect
# published rounding, revisions and differences between source series. This
# audit does not assume the calculation must match exactly.

# %%
spending_consistency_records = []
for spending_row in spending_joined.to_dict("records"):
    for spending_kind in SPENDING_KINDS:
        spending_official = spending_row[f"{spending_kind}_per_capita_spending_mop"]
        spending_derived = spending_row[f"{spending_kind}_spending_mop"] / spending_row[f"{spending_kind}_arrivals"]
        spending_consistency_records.append({
            "year": spending_row["year"], "quarter": spending_row["quarter"],
            "period": spending_row["period"], "visitor_type": spending_kind,
            "reported_spending_million_mop": spending_row[f"{spending_kind}_spending_million_mop"],
            "daily_arrivals_sum": spending_row[f"{spending_kind}_arrivals"],
            "official_per_capita_spending_mop": spending_official,
            "derived_per_capita_spending_mop": spending_derived,
            "derived_minus_official_mop": spending_derived - spending_official,
            "derived_minus_official_pct": 100 * (spending_derived / spending_official - 1),
            "raw_same_day_missing_days": spending_row["raw_same_day_missing_days"],
            "raw_type_additivity_flag_days": spending_row["raw_type_additivity_flag_days"],
        })
spending_consistency = pd.DataFrame(spending_consistency_records)
spending_save_table(spending_quarter_arrivals, "spending_quarterly_arrivals.csv")
spending_save_table(spending_joined, "spending_quarterly_joined.csv")
spending_save_table(spending_joined[["year", "quarter", "period"] + spending_recovery_columns],
                    "spending_quarterly_recovery_indices.csv")
spending_structure_columns = ["year", "quarter", "period", "total_arrivals", "overnight_arrivals", "same_day_arrivals",
    "total_spending_million_mop", "overnight_spending_million_mop", "same_day_spending_million_mop",
    "total_per_capita_spending_mop", "overnight_per_capita_spending_mop", "same_day_per_capita_spending_mop",
    "overnight_arrivals_share_pct", "overnight_spending_share_pct", "same_day_arrivals_share_pct",
    "same_day_spending_share_pct", "overnight_to_same_day_official_per_capita_ratio",
    "overnight_arrivals_share_change_pp_vs_2019q", "overnight_spending_share_change_pp_vs_2019q"]
spending_save_table(spending_joined[spending_structure_columns], "spending_visitor_type_comparison.csv")
spending_save_table(spending_consistency, "spending_per_capita_consistency.csv")

# %% [markdown]
# ### Annual cross-checks and comparisons
# Quarter sums and published annual totals are stored in separate columns. The
# annual official per-capita estimate comes directly from the Q4 publication's
# annual table. It is not an unweighted average of four quarterly estimates.
# A separately named derived measure uses quarter-summed spending / daily-summed
# arrivals. Changes compare 2025 with 2019 and with 2024 using each stated basis.

# %%
spending_annual_total_controls = pd.read_csv(SPENDING_DATA / "spending_annual_controls.csv")
spending_annual_pc_controls = pd.read_csv(SPENDING_DATA / "spending_annual_per_capita_controls.csv")
spending_annual_records = []
for spending_year in SPENDING_YEARS:
    spending_year_rows = spending_joined[spending_joined.year.eq(spending_year)]
    for spending_kind in SPENDING_KINDS:
        spending_annual_total = spending_annual_total_controls[
            spending_annual_total_controls.year.eq(spending_year) &
            spending_annual_total_controls.visitor_type.eq(spending_kind)].iloc[0]
        spending_annual_pc = spending_annual_pc_controls[
            spending_annual_pc_controls.year.eq(spending_year) &
            spending_annual_pc_controls.visitor_type.eq(spending_kind)].iloc[0]
        spending_annual_arrivals = spending_year_rows[f"{spending_kind}_arrivals"].sum()
        spending_quarter_sum = spending_year_rows[f"{spending_kind}_spending_million_mop"].sum()
        assert spending_quarter_sum == spending_annual_total.quarterly_sum_million_mop
        spending_annual_records.append({
            "year": spending_year, "visitor_type": spending_kind,
            "daily_summed_annual_arrivals": int(spending_annual_arrivals),
            "quarter_summed_spending_million_mop": float(spending_quarter_sum),
            "published_annual_spending_million_mop": float(spending_annual_total.published_annual_million_mop),
            "quarter_sum_minus_published_annual_million_mop": float(spending_annual_total.difference_million_mop),
            "published_annual_per_capita_mop": float(spending_annual_pc.published_annual_per_capita_mop),
            "derived_annual_per_capita_from_quarter_sum_mop": float(spending_quarter_sum * 1e6 / spending_annual_arrivals),
            "derived_annual_per_capita_from_annual_control_mop": float(spending_annual_total.published_annual_million_mop * 1e6 / spending_annual_arrivals),
            "annual_total_source_url": spending_annual_total.source_url,
            "annual_total_source_location": spending_annual_total.location,
            "annual_per_capita_source_url": spending_annual_pc.source_url,
            "annual_per_capita_source_location": spending_annual_pc.location,
        })
spending_annual = pd.DataFrame(spending_annual_records)
assert spending_annual.quarter_sum_minus_published_annual_million_mop.abs().le(1).all()
spending_annual_comparison_records = []
spending_annual_metric_names = ["daily_summed_annual_arrivals", "quarter_summed_spending_million_mop",
    "published_annual_spending_million_mop", "published_annual_per_capita_mop",
    "derived_annual_per_capita_from_quarter_sum_mop"]
for spending_base_year in [2019, 2024]:
    for spending_kind in SPENDING_KINDS:
        spending_base_annual = spending_annual[(spending_annual.year.eq(spending_base_year)) &
                                              spending_annual.visitor_type.eq(spending_kind)].iloc[0]
        spending_target_annual = spending_annual[spending_annual.year.eq(2025) &
                                                spending_annual.visitor_type.eq(spending_kind)].iloc[0]
        spending_annual_comparison_record = {"base_year": spending_base_year, "target_year": 2025,
                                             "visitor_type": spending_kind}
        for spending_metric in spending_annual_metric_names:
            spending_annual_comparison_record[f"{spending_metric}_change_pct"] = float(
                100 * (spending_target_annual[spending_metric] / spending_base_annual[spending_metric] - 1))
        spending_annual_comparison_records.append(spending_annual_comparison_record)
spending_annual_comparisons = pd.DataFrame(spending_annual_comparison_records)
spending_save_table(spending_annual, "spending_annual_crosscheck.csv")
spending_save_table(spending_annual_comparisons, "spending_annual_comparisons.csv")

# %% [markdown]
# ### Figures
# No interpolation is used for 2020–2022. Quarterly recovery uses a matched
# seasonal baseline. Visitor-type figures compare all four observed years.
# Spending remains nominal and the lines describe aggregate observations.

# %%
spending_fig, spending_axes = plt.subplots(1, 3, figsize=(13.3, 4.8), sharey=True)
spending_line_specs = [
    ("total_arrivals_index_2019q100", "Visitor arrivals", "#0072B2", "o"),
    ("total_nominal_spending_index_2019q100", "Nominal spending", "#D55E00", "s"),
    ("total_official_per_capita_spending_index_2019q100", "Official spending per visit", "#009E73", "^")]
for spending_ax, spending_year in zip(spending_axes, [2023, 2024, 2025]):
    spending_subset = spending_joined[spending_joined.year.eq(spending_year)]
    for spending_column, spending_label, spending_color, spending_marker in spending_line_specs:
        spending_ax.plot(spending_subset.quarter, spending_subset[spending_column],
                         color=spending_color, marker=spending_marker, label=spending_label, lw=2)
    spending_ax.axhline(100, color="#666666", ls="--", lw=1)
    spending_ax.set(title=str(spending_year), xticks=[1, 2, 3, 4], xticklabels=["Q1", "Q2", "Q3", "Q4"])
    spending_ax.grid(axis="y", alpha=.18)
spending_axes[0].set_ylabel("Same quarter in 2019 = 100")
spending_axes[0].legend(loc="upper right", fontsize=8, frameon=False)
spending_fig.suptitle("Visitor arrivals and non-gaming spending: quarterly recovery", fontsize=15, y=.99)
spending_save_figure(spending_fig, "11_quarterly_spending_recovery.png",
    "Sources: DSEC daily administrative arrivals and Visitor Expenditure Survey. Spending is nominal MOP.\n"
    "Each quarter has its own 2019 baseline; survey estimates and administrative counts are not forced to reconcile.")

# %%
spending_fig, spending_axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
for spending_ax, spending_kind, spending_title in zip(spending_axes, ["overnight", "same_day"], ["Overnight visitors", "Same-day visitors"]):
    for spending_year in SPENDING_YEARS:
        spending_subset = spending_joined[spending_joined.year.eq(spending_year)]
        spending_ax.plot(spending_subset.quarter, spending_subset[f"{spending_kind}_per_capita_spending_mop"],
                         color=SPENDING_YEAR_COLORS[spending_year], marker=SPENDING_YEAR_MARKERS[spending_year],
                         label=str(spending_year), lw=2)
    spending_ax.set(title=spending_title, xticks=[1, 2, 3, 4], xticklabels=["Q1", "Q2", "Q3", "Q4"], ylim=(0, 5100))
    spending_ax.grid(axis="y", alpha=.18)
spending_axes[0].set_ylabel("Official non-gaming spending per visit (nominal MOP)")
spending_axes[1].legend(title="Year", frameon=False, ncol=2, loc="upper right")
spending_fig.suptitle("Official spending per visit by visitor type", fontsize=15, y=.99)
spending_save_figure(spending_fig, "12_spending_per_visit_by_type.png",
    "Source: DSEC Visitor Expenditure Survey, 16 published quarters. Quota-sample estimates; no price adjustment.\n"
    "Overnight visitors are broader than hotel guests. Values describe visits, not unique people or survey microdata.")

# %%
spending_fig, spending_axes = plt.subplots(1, 2, figsize=(11.5, 5), sharey=True)
for spending_ax, spending_column, spending_title in zip(spending_axes,
        ["overnight_arrivals_share_pct", "overnight_spending_share_pct"],
        ["Share of all visitor arrivals", "Share of all non-gaming spending"]):
    for spending_year in SPENDING_YEARS:
        spending_subset = spending_joined[spending_joined.year.eq(spending_year)]
        spending_ax.plot(spending_subset.quarter, spending_subset[spending_column],
                         color=SPENDING_YEAR_COLORS[spending_year], marker=SPENDING_YEAR_MARKERS[spending_year],
                         label=str(spending_year), lw=2)
    spending_ax.set(title=spending_title, xticks=[1, 2, 3, 4], xticklabels=["Q1", "Q2", "Q3", "Q4"], ylim=(0, 100))
    spending_ax.grid(axis="y", alpha=.18)
spending_axes[0].set_ylabel("Overnight visitors' share (%)")
spending_axes[1].legend(title="Year", frameon=False, ncol=2, loc="lower right")
spending_fig.suptitle("Overnight visitors: arrival share and spending share", fontsize=15, y=.99)
spending_save_figure(spending_fig, "13_overnight_arrival_spending_shares.png",
    "Sources: daily administrative arrivals (left); Visitor Expenditure Survey (right). Denominators are reported totals.\n"
    "Shares are descriptive; rounding and original one-person daily discrepancies remain visible in the exported audit.")

# %% [markdown]
# ### Saved findings and interpretation
# The JSON contains exact calculated results and provenance hashes; the text
# interpretation gives a concise account of observed differences and limits.

# %%
spending_2025 = spending_joined[spending_joined.year.eq(2025)]
spending_consistency_summary = []
for spending_kind in SPENDING_KINDS:
    spending_group = spending_consistency[spending_consistency.visitor_type.eq(spending_kind)]
    spending_largest = spending_group.loc[spending_group.derived_minus_official_pct.abs().idxmax()]
    spending_consistency_summary.append({
        "visitor_type": spending_kind,
        "max_absolute_difference_mop": float(spending_group.derived_minus_official_mop.abs().max()),
        "max_absolute_difference_pct": float(spending_group.derived_minus_official_pct.abs().max()),
        "period_of_max_absolute_pct_difference": spending_largest.period,
        "signed_difference_mop_at_max_pct": float(spending_largest.derived_minus_official_mop),
    })
spending_summary = {
    "executed_at_utc": datetime.now(timezone.utc).isoformat(),
    "python": sys.version.split()[0],
    "coverage": {"years": SPENDING_YEARS, "quarter_rows": len(spending_joined),
        "daily_rows_in_selected_years": len(spending_daily_selected),
        "all_selected_quarters_complete": True,
        "same_day_raw_missing_days": int(spending_daily_selected.raw_same_day_missing.sum()),
        "raw_type_additivity_flag_days": int(spending_daily_selected.raw_type_additivity_flag.sum()),
        "quarterly_arrival_type_residuals": spending_joined[["period", "analysis_type_sum_minus_total_arrivals"]].to_dict("records")},
    "measurement": {"spending_currency": "MOP", "spending_price_basis": "nominal_current_prices",
        "spending_excludes": "gaming", "spending_source": "DSEC quota-sample Visitor Expenditure Survey",
        "arrivals_source": "DSEC daily administrative visitor arrivals",
        "observation_unit": "city-quarter", "recovery_index_baseline": "same calendar quarter in 2019 = 100",
        "inference": "descriptive only; no causal attribution, confidence intervals, price deflation or decomposition"},
    "recovery_2025": spending_2025[["period"] + spending_recovery_columns].to_dict("records"),
    "type_comparison_2025": spending_2025[spending_structure_columns].to_dict("records"),
    "annual_crosscheck": spending_annual.to_dict("records"),
    "annual_comparisons_2025_vs_2019_and_2024": spending_annual_comparisons.to_dict("records"),
    "per_capita_consistency": spending_consistency_summary,
    "quarterly_spending_type_residuals_million_mop": spending_joined[["period", "component_sum_minus_reported_total_million_mop"]].to_dict("records"),
    "inputs": [{"file": str(path.relative_to(SPENDING_ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in (
        SPENDING_DATA / SPENDING_INPUT_NAME,
        SPENDING_DATA / "macao_daily_total_overnight_sameday_2017_2025.csv",
        SPENDING_DATA / "spending_annual_controls.csv",
        SPENDING_DATA / "spending_annual_per_capita_controls.csv")],
    "csv_outputs": [str(path.relative_to(SPENDING_ROOT)) for path in sorted(SPENDING_RESULTS.glob("spending_*.csv"))],
    "figures": ["figures/11_quarterly_spending_recovery.png", "figures/12_spending_per_visit_by_type.png",
                "figures/13_overnight_arrival_spending_shares.png"],
}
(SPENDING_ROOT / "spending_summary.json").write_text(json.dumps(spending_summary, ensure_ascii=False, indent=2), encoding="utf-8")
spending_findings = [
    "Quarterly visitor-spending experiment — executed results",
    "16 individual quarters: Q1–Q4 in 2019, 2023, 2024 and 2025. All selected daily quarters are complete.",
    "All spending is nominal MOP, excluding gaming. Recovery indices use the corresponding 2019 quarter = 100.",
    "", "2025 matched-quarter recovery (arrivals / nominal spending / official spending per visit):"]
for spending_row in spending_2025.to_dict("records"):
    spending_findings.append(
        f"{spending_row['period']}: {spending_row['total_arrivals_index_2019q100']:.1f} / "
        f"{spending_row['total_nominal_spending_index_2019q100']:.1f} / "
        f"{spending_row['total_official_per_capita_spending_index_2019q100']:.1f}; "
        f"overnight arrival share {spending_row['overnight_arrivals_share_pct']:.1f}%, "
        f"spending share {spending_row['overnight_spending_share_pct']:.1f}%; "
        f"overnight/same-day official per-visit spending ratio {spending_row['overnight_to_same_day_official_per_capita_ratio']:.2f}.")
spending_findings += ["", "Annual total-visitor comparisons (nominal; official annual per-capita values):"]
for spending_item in spending_annual_comparisons[spending_annual_comparisons.visitor_type.eq("total")].to_dict("records"):
    spending_findings.append(
        f"2025 vs {spending_item['base_year']}: arrivals "
        f"{spending_item['daily_summed_annual_arrivals_change_pct']:+.4f}%; "
        f"quarter-summed spending {spending_item['quarter_summed_spending_million_mop_change_pct']:+.4f}%; "
        f"published annual spending {spending_item['published_annual_spending_million_mop_change_pct']:+.4f}%; "
        f"official annual spending per visit {spending_item['published_annual_per_capita_mop_change_pct']:+.4f}%.")
spending_findings += [
    "2019 annual spending: quarter sum 64,078 vs published annual control 64,077 million MOP; "
    "2024: 75,357 vs 75,358; 2025: 80,122 vs 80,122. Annual comparisons retain both bases.",
    "Annual per-capita estimates are taken from annual source tables, never obtained by averaging quarterly per-capita values.",
    "", "Official-versus-derived per-visit audit:"]
for spending_item in spending_consistency_summary:
    spending_findings.append(
        f"{spending_item['visitor_type']}: largest absolute discrepancy "
        f"{spending_item['max_absolute_difference_pct']:.4f}% "
        f"({spending_item['period_of_max_absolute_pct_difference']}); "
        f"maximum absolute MOP discrepancy across quarters {spending_item['max_absolute_difference_mop']:.4f}.")
spending_findings += [
    "", "Interpretation and limits:",
    "Compare each 2025 quarter with its matching 2019 quarter: a return of arrival counts need not imply an identical spending index. These are separate descriptive measurements, not a factor decomposition.",
    "Overnight visitors account for a much larger fraction of non-gaming spending than of arrivals. Official spending per overnight visit exceeds same-day spending in every observed quarter.",
    "The spending survey uses quota sampling and extrapolation; arrival counts are administrative. They are separate data sources, not statistically independent variables or visitor-level linked records.",
    "Published total and type spending have rounding differences of up to 1 million MOP. Existing daily type anomalies are retained and counted. No arithmetic residual is assigned to a visitor type.",
    "Derived spending per visit is total reported spending divided by summed daily arrivals; the exported differences document rounding/revision/source compatibility, without assuming exact equality.",
    "No claim is made about real purchasing power, individual spending distributions, tourism GDP, gaming revenue, profit or policy effects. No TPI, Shapley, confidence intervals or respondent-level inference is used.",
    "Source definitions and the historical wording change for overnight visitors are documented in data/README.txt. Original per-capita precision is preserved; round to whole MOP in presentations.",
]
(SPENDING_ROOT / "spending_findings.txt").write_text("\n".join(spending_findings) + "\n", encoding="utf-8")
print("\n".join(spending_findings))
