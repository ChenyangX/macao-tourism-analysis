# %% [markdown]
# # Macao visitor arrivals: recovery, composition and calendar estimation
# This notebook/script applies CISC7204 methods to official historical daily
# arrivals. Read protocol.txt first: calendar-only candidates and chronological
# splits were specified before model execution. All dates and unusual high values
# are retained. Inferential p-values and ordinary iid bootstrap are not used.

# %%
from pathlib import Path
import hashlib
import json
import sys
import importlib.metadata
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA, RESULTS, FIGURES = ROOT / "data", ROOT / "results", ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 135, "savefig.dpi": 180,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 10, "axes.titlesize": 12})
COLORS = {2019: "#0072B2", 2025: "#D55E00"}
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

def save_table(frame, name):
    frame.to_csv(RESULTS / name, index=False, encoding="utf-8-sig")

def save_figure(name):
    plt.gcf().text(.01, .005, "Source: DSEC daily visitor arrivals; descriptive historical analysis.", fontsize=7, color="#555555")
    plt.tight_layout(rect=(0, .025, 1, 1))
    plt.savefig(FIGURES / name, bbox_inches="tight")
    plt.close()
    # Display the same exported PNG when this script is split into notebook cells.
    if "get_ipython" in globals():
        from IPython.display import Image, display
        display(Image(filename=str(FIGURES / name)))

def describe(values):
    return pd.Series({"days": len(values), "total": values.sum(),
                      "mean": values.mean(), "median": values.median(),
                      "q25": values.quantile(.25), "q75": values.quantile(.75),
                      "iqr": values.quantile(.75)-values.quantile(.25),
                      "p90": values.quantile(.9), "min": values.min(), "max": values.max()})

def group_describe(frame, keys, column="visitor_arrivals"):
    return frame.groupby(keys, observed=True)[column].apply(describe).unstack(-1).reset_index()

# %% [markdown]
# ## 1. Input audit and transparent type flags
# Total, overnight and same-day are administrative counts, not observations of
# individual travellers. One missing same-day value was derived upstream and is
# explicitly labelled; eight 1-person discrepancies remain unchanged.

# %%
total = pd.read_csv(DATA / "macao_daily_total_2017_2025.csv", parse_dates=["date"]).sort_values("date")
sources = pd.read_csv(DATA / "macao_daily_source_2017_2025.csv", parse_dates=["date"], dtype={"source_code": str})
types = pd.read_csv(DATA / "macao_daily_total_overnight_sameday_2017_2025.csv", parse_dates=["date"])
assert len(total) == 3287 and total.date.is_unique
assert total.date.tolist() == pd.date_range("2017-01-01", "2025-12-31").tolist()
assert total.visitor_arrivals.notna().all() and total.visitor_arrivals.ge(0).all()
assert sources.groupby("date").size().eq(17).all()
assert not sources.duplicated(["date", "source_code"]).any()
assert sources.visitor_arrivals.notna().all() and sources.visitor_arrivals.ge(0).all()
assert set(sources.date) == set(total.date)
source_sums = sources.groupby("date").visitor_arrivals.sum()
assert np.array_equal(source_sums.reindex(total.date).to_numpy(), total.visitor_arrivals.to_numpy())
types = types.sort_values("date")
assert types.date.tolist() == total.date.tolist()
assert np.array_equal(types.total.to_numpy(), total.visitor_arrivals.to_numpy())
types["type_anomaly_flag"] = types.same_day_raw.isna() | types.raw_additivity_residual.fillna(0).ne(0)
types["flag_reason"] = np.select([types.same_day_raw.isna(), types.raw_additivity_residual.fillna(0).ne(0)],
                                 ["missing_same_day_raw_derived_by_difference", "raw_additivity_discrepancy_1_person"], default="none")
assert types.type_anomaly_flag.sum() == 9
save_table(types.loc[types.type_anomaly_flag], "type_anomaly_dates.csv")
save_table(types, "type_data_with_flags.csv")
total["year"], total["month"], total["weekday"] = total.date.dt.year, total.date.dt.month, total.date.dt.dayofweek
total["stage"] = pd.cut(total.year, [2016, 2019, 2022, 2025], labels=["2017-2019", "2020-2022", "2023-2025"])
audit = {"total_days": len(total), "source_rows": len(sources), "source_groups": sources.source_code.nunique(),
         "total_missing": int(total.visitor_arrivals.isna().sum()), "source_additivity_verified": True,
         "type_flagged_dates": int(types.type_anomaly_flag.sum()),
         "type_missing_raw_days": int(types.same_day_raw.isna().sum()),
         "type_nonzero_additivity_days": int(types.raw_additivity_residual.fillna(0).ne(0).sum()),
         "removed_dates": 0, "removed_high_values": 0}
print("Data audit:", audit)

# %% [markdown]
# ## 2. Descriptive recovery and distribution comparison
# Stage boundaries are fixed calendar periods. They are descriptive labels, not
# an estimate of a policy effect. ECDFs and quantiles do not require normality.

# %%
stages = group_describe(total, ["stage"])
annual = group_describe(total, ["year"])
comparison = total[total.year.isin([2019, 2025])].copy()
monthly = group_describe(comparison, ["year", "month"])
weekly = group_describe(comparison, ["year", "weekday"])
month_weekday = group_describe(comparison, ["year", "month", "weekday"])
for frame, name in [(stages, "stage_summary.csv"), (annual, "annual_summary.csv"),
                    (monthly, "month_comparison_2019_2025.csv"), (weekly, "weekday_comparison_2019_2025.csv"),
                    (month_weekday, "month_weekday_comparison_2019_2025.csv")]:
    save_table(frame, name)
print("Stage statistics:\n", stages.round(2).to_string(index=False))
print("Annual statistics:\n", annual.round(2).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(total.date, total.visitor_arrivals / 1000, lw=.55, color="#4C7899")
ax.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2022-12-31"), color="#BBBBBB", alpha=.2)
ax.set(xlabel="Date", ylabel="Visitor arrivals (thousands/day)", title="Macao daily visitor arrivals, 2017-2025")
save_figure("01_daily_history.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ecdf_rows = []
shared_bins = np.arange(0, np.ceil(comparison.visitor_arrivals.max()/10000)*10+10, 10)
for year in [2019, 2025]:
    x = np.sort(comparison.loc[comparison.year.eq(year), "visitor_arrivals"].to_numpy())
    probability = np.arange(1, len(x)+1) / len(x)
    axes[0].step(x / 1000, probability, where="post", label=str(year), color=COLORS[year])
    axes[1].hist(x / 1000, bins=shared_bins, density=True, histtype="step", lw=1.6, label=str(year), color=COLORS[year])
    ecdf_rows.extend({"year": year, "arrivals": float(a), "ecdf": float(p)} for a,p in zip(x,probability))
axes[0].set(xlabel="Visitor arrivals (thousands/day)", ylabel="Fraction of days at or below x", title="Empirical CDF")
axes[1].set(xlabel="Visitor arrivals (thousands/day)", ylabel="Density per thousand arrivals", title="Same 10,000-arrival bins")
for ax in axes: ax.legend()
save_figure("02_ecdf_distribution_2019_2025.png")
save_table(pd.DataFrame(ecdf_rows), "ecdf_2019_2025.csv")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for year in [2019, 2025]:
    m = monthly[monthly.year.eq(year)]
    w = weekly[weekly.year.eq(year)]
    axes[0].plot(m.month, m["mean"] / 1000, "o-", label=str(year), color=COLORS[year])
    axes[1].plot(w.weekday, w["median"] / 1000, "o-", label=str(year), color=COLORS[year])
axes[0].set(xlabel="Month", ylabel="Mean daily arrivals (thousands)", title="Same-month descriptive comparison", xticks=range(1,13))
axes[1].set(xlabel="Day of week", ylabel="Median daily arrivals (thousands)", title="Within-year weekly pattern", xticks=range(7), xticklabels=WEEKDAYS)
for ax in axes: ax.legend()
save_figure("03_month_weekday_comparison.png")

fig, ax = plt.subplots(figsize=(10,4))
heat = total.pivot_table(index="year", columns="month", values="visitor_arrivals", aggfunc="mean") / 1000
im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="Blues")
ax.set(xticks=range(12), xticklabels=range(1,13), yticks=range(len(heat)), yticklabels=heat.index,
       xlabel="Month", ylabel="Year", title="Mean daily arrivals by year and month")
fig.colorbar(im, ax=ax, label="Thousands of arrivals/day")
save_figure("04_year_month_heatmap.png")

# %% [markdown]
# ## 3. Source composition and visitor types (overnight / same-day)
# Contribution means arithmetic change in group totals, not a causal attribution.
# Group shares have an explicit denominator. Same-day derivation and raw count
# discrepancies are preserved; annual ratios are ratio-of-sums, not mean shares.
# Contribution to net change divides each signed group change by the total net
# change. It may exceed 100% or be negative; it is not a composition share.

# %%
sources["year"] = sources.date.dt.year
source_annual = sources.groupby(["year", "source_code", "source_name_zh", "source_name_en"], as_index=False).visitor_arrivals.sum()
source_annual["share"] = source_annual.visitor_arrivals / source_annual.groupby("year").visitor_arrivals.transform("sum")
save_table(source_annual, "source_annual_shares.csv")
source_change = source_annual.pivot(index=["source_code", "source_name_zh", "source_name_en"], columns="year", values="visitor_arrivals")[[2019,2025]].reset_index()
source_change.columns = ["source_code", "source_name_zh", "source_name_en", "arrivals_2019", "arrivals_2025"]
source_change["change_arrivals"] = source_change.arrivals_2025 - source_change.arrivals_2019
source_change["change_pct_within_source"] = source_change.arrivals_2025 / source_change.arrivals_2019 - 1
source_change["contribution_to_net_change"] = source_change.change_arrivals / source_change.change_arrivals.sum()
for year in [2019,2025]:
    source_change[f"share_{year}"] = source_change[f"arrivals_{year}"] / source_change[f"arrivals_{year}"].sum()
source_change["share_change_pp"] = 100*(source_change.share_2025-source_change.share_2019)
source_change = source_change.sort_values("change_arrivals")
save_table(source_change, "source_change_2019_2025.csv")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(source_change.source_name_en, source_change.change_arrivals / 1e6,
        color=np.where(source_change.change_arrivals.ge(0), "#0072B2", "#D55E00"))
ax.axvline(0, color="black", lw=.6)
ax.set(xlabel="2025 minus 2019 arrivals (millions)", title="Which source groups account for the net change?")
save_figure("05_source_contributions.png")

types["year"] = types.date.dt.year
types["overnight_share"] = types.overnight / types.total.replace(0,np.nan)
types["same_day_share_analysis"] = types.same_day_analysis / types.total.replace(0,np.nan)
type_annual = types.groupby("year").agg(total=("total","sum"), overnight=("overnight","sum"),
    same_day_raw_sum=("same_day_raw","sum"), same_day_analysis_sum=("same_day_analysis","sum"),
    flagged_days=("type_anomaly_flag","sum"), raw_missing=("same_day_raw", lambda x:x.isna().sum())).reset_index()
type_annual["overnight_share"] = type_annual.overnight / type_annual.total
type_annual["same_day_share_analysis"] = type_annual.same_day_analysis_sum / type_annual.total
type_clean = types[~types.type_anomaly_flag].groupby("year").agg(total=("total","sum"), overnight=("overnight","sum")).reset_index()
type_clean["overnight_share_excluding_flagged_days"] = type_clean.overnight / type_clean.total
type_annual = type_annual.merge(type_clean[["year","overnight_share_excluding_flagged_days"]], on="year")
type_annual["sensitivity_difference_pp"] = 100*(type_annual.overnight_share_excluding_flagged_days-type_annual.overnight_share)
save_table(type_annual, "type_annual_shares_and_sensitivity.csv")
fig, ax = plt.subplots(figsize=(9,4))
ax.plot(type_annual.year, type_annual.overnight_share*100, "o-", label="Overnight / total")
ax.plot(type_annual.year, type_annual.same_day_share_analysis*100, "s-", label="Same-day / total (flagged derivation retained)")
ax.set(xlabel="Year", ylabel="Share of annual arrivals (%)", title="Overnight and same-day composition", ylim=(0,100))
ax.legend(fontsize=8)
save_figure("06_accommodation_shares.png")
print("Source changes:\n", source_change[["source_name_en","change_arrivals","share_change_pp"]].to_string(index=False))
print("Type shares:\n", type_annual.round(4).to_string(index=False))

# %% [markdown]
# ## 4. Predefined chronological validation
# Four validation folds cover 2024 Q1-Q4; each training window expands from
# 2023-01-01. All preprocessing is inside Pipeline and refit within each fold.
# The 2025 target is not used for selection. No holiday, lag or observed same-day
# composition feature is added after examining final test performance.

# %%
class WeekdayMean(RegressorMixin, BaseEstimator):
    def fit(self, X, y):
        self.means_ = pd.Series(np.asarray(y), index=np.asarray(X["weekday"])).groupby(level=0).mean()
        self.global_mean_ = float(np.mean(y))
        return self
    def predict(self, X):
        return X["weekday"].map(self.means_).fillna(self.global_mean_).to_numpy()

def calendar_pipeline(estimator, trend=False):
    transformers = [("calendar", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="error"), ["month", "weekday"])]
    if trend:
        transformers.append(("trend", StandardScaler(), ["day_trend"]))
    return Pipeline([("features", ColumnTransformer(transformers, remainder="drop")), ("model", estimator)])

model_data = total[total.year.ge(2023)].copy().reset_index(drop=True)
model_data["day_trend"] = (model_data.date-pd.Timestamp("2023-01-01")).dt.days
development = model_data[model_data.year.le(2024)].reset_index(drop=True)
test = model_data[model_data.year.eq(2025)].reset_index(drop=True)
assert len(development) == 731 and len(test) == 365
features = ["month", "weekday", "day_trend"]
Xdev, ydev = development[features], development.visitor_arrivals
Xtest, ytest = test[features], test.visitor_arrivals
folds, split_rows = [], []
for quarter, (start, end) in enumerate([("2024-01-01","2024-04-01"),("2024-04-01","2024-07-01"),
                                       ("2024-07-01","2024-10-01"),("2024-10-01","2025-01-01")], 1):
    tr = np.flatnonzero(development.date.lt(start))
    va = np.flatnonzero(development.date.ge(start)&development.date.lt(end))
    assert development.date.iloc[tr].max() < development.date.iloc[va].min()
    folds.append((tr, va))
    split_rows.append({"fold":quarter,"train_start":development.date.iloc[tr[0]],"train_end":development.date.iloc[tr[-1]],
                       "train_n":len(tr),"valid_start":development.date.iloc[va[0]],"valid_end":development.date.iloc[va[-1]],"valid_n":len(va)})
save_table(pd.DataFrame(split_rows), "chronological_folds.csv")
print("Chronological validation:\n", pd.DataFrame(split_rows).to_string(index=False))

alphas = [.01,.1,1,10,100,1000]
grid = GridSearchCV(calendar_pipeline(Ridge(), True), {"model__alpha":alphas}, scoring="neg_mean_squared_error", cv=folds, refit=True, n_jobs=1, return_train_score=True)
grid.fit(Xdev,ydev)
ridge_cv = pd.DataFrame({"alpha":[p["model__alpha"] for p in grid.cv_results_["params"]],
                        "mean_validation_mse":-grid.cv_results_["mean_test_score"],
                        "std_fold_mse":grid.cv_results_["std_test_score"]})
for i in range(4): ridge_cv[f"fold_{i+1}_mse"] = -grid.cv_results_[f"split{i}_test_score"]
save_table(ridge_cv,"ridge_alpha_validation.csv")
models = {"weekday_mean":WeekdayMean(), "ols_calendar":calendar_pipeline(LinearRegression()),
          "ols_calendar_trend":calendar_pipeline(LinearRegression(),True), "ridge_calendar_trend":clone(grid.best_estimator_)}
validation_rows = []
for name, estimator in models.items():
    result = cross_validate(estimator, Xdev,ydev,cv=folds,scoring="neg_mean_squared_error",n_jobs=1)
    errors = -result["test_score"]
    row = {"model":name,"validation_mse":float(errors.mean()),"validation_rmse":float(np.sqrt(errors.mean()))}
    row.update({f"fold_{i+1}_mse":float(v) for i,v in enumerate(errors)})
    validation_rows.append(row)
validation = pd.DataFrame(validation_rows).sort_values("validation_mse")
selected_name = validation.iloc[0].model
model_labels = {"weekday_mean": "Weekday mean", "ols_calendar": "OLS calendar",
                "ols_calendar_trend": "OLS calendar + trend", "ridge_calendar_trend": "Ridge calendar + trend"}
selected_label = model_labels[selected_name]
selection = {"selected_model":selected_name,"selected_ridge_alpha":float(grid.best_params_["model__alpha"]),
             "selection_rule":"Lowest unweighted mean of four chronological 2024-quarter validation MSEs; no 2025 information."}
# This selection file is deliberately written before evaluating any 2025 target.
(RESULTS/"model_selection_before_test.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
save_table(validation,"model_validation.csv")
print("Validation selection:\n",validation.round(2).to_string(index=False))
print(selection)

# %% [markdown]
# ## 5. Final 2025 historical holdout and error diagnostics
# Negative R-squared means worse than the test-period mean benchmark. All
# candidates are reported; a better test score cannot change the selection.
# Residual = observed minus estimated. The high-arrival threshold is learned
# from 2023-2024 only, then fixed for the 2025 diagnostic.

# %%
test_rows, prediction_rows, diagnostic_rows = [], [], []
high_threshold = float(ydev.quantile(.9))
for name, estimator in models.items():
    estimator.fit(Xdev,ydev)
    pred = estimator.predict(Xtest)
    mse = mean_squared_error(ytest,pred)
    test_rows.append({"model":name,"mse":mse,"rmse":np.sqrt(mse),"mae":mean_absolute_error(ytest,pred),
                      "r2":r2_score(ytest,pred),"bias_observed_minus_predicted":float(np.mean(ytest-pred)),
                      "negative_predictions":int((pred<0).sum()),"selected_from_validation":name==selected_name})
    part = test[["date","month","weekday","visitor_arrivals"]].copy()
    part["model"], part["prediction"], part["residual"] = name, pred, ytest-pred
    part["high_arrival_training_p90"] = part.visitor_arrivals.gt(high_threshold)
    prediction_rows.append(part)
    for key in ["month","weekday","high_arrival_training_p90"]:
        for value, g in part.groupby(key):
            diagnostic_rows.append({"model":name,"group_type":key,"group":str(value),"days":len(g),
                "rmse":float(np.sqrt(np.mean(g.residual**2))),"mae":float(np.mean(np.abs(g.residual))),
                "bias_observed_minus_predicted":float(g.residual.mean()),"mean_observed":float(g.visitor_arrivals.mean())})
scores = pd.DataFrame(test_rows).merge(validation[["model","validation_mse","validation_rmse"]],on="model")
predictions = pd.concat(prediction_rows,ignore_index=True)
diagnostics = pd.DataFrame(diagnostic_rows)
save_table(scores,"model_test_metrics.csv")
save_table(predictions,"test_predictions.csv")
save_table(diagnostics,"test_error_diagnostics.csv")
print("Final historical holdout:\n",scores.round(3).to_string(index=False))
print("Training P90 high-arrival threshold:",high_threshold)
chosen = predictions[predictions.model.eq(selected_name)].copy()

fig, axes = plt.subplots(2,1,figsize=(11,7),sharex=True)
axes[0].plot(chosen.date,chosen.visitor_arrivals/1000,lw=.8,label="Observed",color="#0072B2")
axes[0].plot(chosen.date,chosen.prediction/1000,lw=.8,label="Calendar estimate",color="#D55E00")
axes[0].set(ylabel="Arrivals (thousands/day)",title=f"2025 historical holdout: {selected_label}")
axes[0].legend()
axes[1].plot(chosen.date,chosen.residual/1000,lw=.7,color="#555555")
axes[1].axhline(0,color="black",lw=.7)
axes[1].set(xlabel="Date",ylabel="Observed - estimated (thousands)",title="Residuals preserve unexplained peaks")
save_figure("07_holdout_predictions_residuals.png")

fig, axes = plt.subplots(1,3,figsize=(13,4))
axes[0].scatter(chosen.prediction/1000,chosen.residual/1000,s=12,alpha=.5)
axes[0].axhline(0,color="black",lw=.7)
axes[0].set(xlabel="Estimated arrivals (thousands)",ylabel="Residual (thousands)",title="Residual vs estimate")
for ax,key,title in [(axes[1],"month","Error by month"),(axes[2],"weekday","Error by weekday")]:
    g=diagnostics[(diagnostics.model.eq(selected_name))&diagnostics.group_type.eq(key)].copy()
    g["order"]=g.group.astype(int);g=g.sort_values("order")
    ax.bar(g.order,g.rmse/1000,color="#0072B2")
    ax.set(xlabel="Month" if key=="month" else "Weekday",ylabel="RMSE (thousands)",title=title)
    ax.set_xticks(g.order, g.order if key=="month" else WEEKDAYS)
save_figure("08_holdout_error_diagnostics.png")

fig, ax=plt.subplots(figsize=(8,4))
positions=np.arange(len(scores));width=.36
ax.bar(positions-width/2,scores.validation_rmse/1000,width,label="2024 validation (root mean MSE)")
ax.bar(positions+width/2,scores.rmse/1000,width,label="2025 final holdout")
ax.set(xticks=positions,xticklabels=["Weekday mean","OLS calendar","OLS + trend","Ridge + trend"],ylabel="RMSE (thousands/day)",title="Validation and holdout errors: selection uses validation only")
ax.legend(fontsize=8)
save_figure("09_model_comparison.png")

high_diagnostic = diagnostics[(diagnostics.model.eq(selected_name)) &
                             diagnostics.group_type.eq("high_arrival_training_p90")].copy()
high_diagnostic = high_diagnostic.set_index("group").reindex(["False", "True"])
fig, ax = plt.subplots(figsize=(7,4))
bars = ax.bar(["At/below training P90", "Above training P90"], high_diagnostic.rmse/1000,
              color=["#0072B2", "#D55E00"])
ax.bar_label(bars, labels=[f"n={int(n)}, RMSE={r/1000:.1f}k" for n,r in
                          zip(high_diagnostic.days, high_diagnostic.rmse)], padding=4, fontsize=9)
ax.set(ylabel="RMSE (thousands/day)", ylim=(0, high_diagnostic.rmse.max()/1000*1.2),
       title=f"2025 error by observed arrivals; fixed training P90 = {high_threshold:,.0f}")
save_figure("10_high_arrival_error.png")
print("Selected-model high-arrival diagnostic:\n", high_diagnostic.round(2).to_string())

# %% [markdown]
# ## 6. Machine-readable conclusions and reproducibility manifest
# Limitations: possible serial dependence, moving holidays and policy changes; no causal
# identification; unknown publication latency. No future forecast is produced.

# %%
annual_index=annual.set_index("year")
year19,year25=annual_index.loc[2019],annual_index.loc[2025]
type_index=type_annual.set_index("year")
selected_score=scores.loc[scores.model.eq(selected_name)].iloc[0]
high_rows=diagnostics[(diagnostics.model.eq(selected_name))&diagnostics.group_type.eq("high_arrival_training_p90")]
summary={"audit":audit,"selection":selection,"stage_summary":stages.to_dict(orient="records"),
    "comparison_2019_2025":{"total_2019":int(year19.total),"total_2025":int(year25.total),
    "total_change":int(year25.total-year19.total),"total_growth_pct":float(100*(year25.total/year19.total-1)),
    "mean_2019":float(year19["mean"]),"mean_2025":float(year25["mean"]),
    "median_2019":float(year19["median"]),"median_2025":float(year25["median"]),
    "p90_2019":float(year19.p90),"p90_2025":float(year25.p90),
    "overnight_share_2019":float(type_index.loc[2019,"overnight_share"]),"overnight_share_2025":float(type_index.loc[2025,"overnight_share"])},
    "selected_model_test":selected_score.to_dict(),"all_models_test":scores.to_dict(orient="records"),
    "high_arrival_threshold_training_p90":high_threshold,"selected_model_high_arrival_diagnostics":high_rows.to_dict(orient="records"),
    "source_change":source_change.to_dict(orient="records"),
    "limitations":["Descriptive statistics cover the observed historical dates; daily records are not treated as iid random samples.",
       "No t-tests, ordinary bootstrap or causal attribution were performed.",
       "2025 is a historical holdout. DSEC publication latency is unknown; real-time availability is not established.",
       "Calendar features only: lunar holidays, weather, events, policy and travel-supply constraints are not encoded.",
       "Adjacent dates may be related and contexts change; residual autocorrelation was not formally tested. R-squared is not a causal effect or a guarantee of future performance.",
       "Nine visitor-type anomaly dates are retained and flagged; daily source-group totals have been verified."]}
(ROOT/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
text=(f"The data cover {len(total):,} dates; all 17 source groups sum to the daily total. Nine visitor-type dates are flagged (one raw missing value and eight +/-1 differences). High values are retained.\n"
      f"Total arrivals: {int(year19.total):,} in 2019 and {int(year25.total):,} in 2025; change {100*(year25.total/year19.total-1):.2f}%.\n"
      f"Daily median: {year19['median']:,.0f} to {year25['median']:,.0f}; P90: {year19.p90:,.1f} to {year25.p90:,.1f}.\n"
      f"Overnight share: {type_index.loc[2019,'overnight_share']:.2%} to {type_index.loc[2025,'overnight_share']:.2%}.\n"
      f"Four-fold 2024 validation MSE selects {selected_label}; selected Ridge alpha={selection['selected_ridge_alpha']}.\n"
      f"Selected model in the 2025 holdout: RMSE={selected_score.rmse:,.2f} arrivals/day, MAE={selected_score.mae:,.2f} arrivals/day, R-squared={selected_score.r2:.4f}.\n"
      f"Training P90 threshold={high_threshold:,.1f}; see results/test_error_diagnostics.csv for high-arrival and other-date errors.\n"
      +"\n".join(summary["limitations"]))
(ROOT/"summary.txt").write_text(text,encoding="utf-8")
versions={p:importlib.metadata.version(p) for p in ["numpy","pandas","scipy","scikit-learn","matplotlib","nbformat","nbclient","ipykernel"]}
(ROOT/"requirements.txt").write_text("\n".join(f"{p}=={v}" for p,v in versions.items())+"\n",encoding="utf-8")
manifest={"executed_at_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version,"packages":versions,
    "data_files":[{"file":str(f.relative_to(ROOT)),"bytes":f.stat().st_size,"sha256":hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(DATA.glob("*.csv"))],
    "protocol_sha256":hashlib.sha256((ROOT/"protocol.txt").read_bytes()).hexdigest(),
    "script_sha256":hashlib.sha256((ROOT/"analysis.py").read_bytes()).hexdigest(),
    "output_csv":[str(p.relative_to(ROOT)) for p in sorted(RESULTS.glob("*.csv"))],
    "output_png":[str(p.relative_to(ROOT)) for p in sorted(FIGURES.glob("*.png"))],"selection":selection}
(ROOT/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
print(text)
