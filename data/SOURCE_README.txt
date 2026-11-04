Data dictionary and measurement notes
Source: Statistics and Census Service (DSEC), Macao SAR. Snapshot: 28 October 2026.
Visitor database: https://www.dsec.gov.mo/TourismDBWeb/
Expenditure survey: https://www.dsec.gov.mo/en-US/Statistic?id=404
Database launch notice: https://www.gsef.gov.mo/zh/posts/1475
Expenditure CSVs retain official attachment URLs and source locations.

SIX INPUT FILES
1. macao_daily_total_2017_2025.csv
   3,287 rows, 2017-01-01 through 2025-12-31. Unique key: date.
   visitor_arrivals: total visits that day, not unique persons.
2. macao_daily_source_2017_2025.csv
   55,879 rows. Unique key: (date, source_code); source_code is categorical.
   source_name_zh/en: official Chinese/English source labels, preserved unchanged.
   visitor_arrivals: arrivals for that source and date. All 17 groups sum to the
   daily total. Source means document issuance place, not nationality or residence.
3. macao_daily_total_overnight_sameday_2017_2025.csv
   3,287 rows. Unique key: date. total: all arrivals; overnight: overnight arrivals.
   same_day_raw retains the original value, including one missing observation.
   same_day_analysis derives zero as total minus overnight only on 2018-09-16.
   same_day_status identifies observed/derived values.
   raw_additivity_residual = total - overnight - same_day_raw.
   Eight further dates have one-arrival differences. All nine flags predate 2023.
4. macao_quarterly_spending_2019_2023_2025.csv
   16 rows: all quarters of 2019, 2023, 2024 and 2025. Key: (year, quarter).
   period: YYYYQq; period_start/end: calendar-quarter boundaries.
   period_basis: individual_quarter; price_basis: nominal_current_prices.
   total/overnight/same_day_spending_million_mop: published group expenditure,
   in million MOP. Corresponding *_spending_mop fields multiply by 1,000,000;
   unit conversion adds no precision.
   *_per_capita_spending_mop: official spending per visit in MOP, not replaced by
   expenditure divided by administrative arrivals.
   component_sum_minus_reported_total_million_mop: type sum minus overall total.
   source_spending_url, source_per_capita_url, retrieved_date: original provenance.
5. spending_annual_controls.csv
   12 rows: four years and three visitor groups. published_annual_million_mop is
   annual published expenditure; quarterly_sum_million_mop sums four quarters.
   difference_million_mop is their difference; source_url/location identify origin.
   Never substitute an annual control for an individual-quarter observation.
6. spending_annual_per_capita_controls.csv
   12 rows: four years and three visitor groups. published_annual_per_capita_mop:
   official annual estimate. source_url/source_file/location identify the source.
   source_file names the official attachment; the attachment is not in this pack.

DEFINITIONS AND RETAINED DIFFERENCES
Arrivals count visits; a person may visit repeatedly. Overnight visitors are not
hotel guests or room nights. Arrivals are administrative counts; expenditure is a
quota-survey estimate. A quarter-level join supplies no individual records or causal
identification. All spending is nominal, excludes gaming, donations, assets and goods
for resale, and is not profit, tourism GDP or real purchasing power. No expenditure
observations were interpolated for 2020-2022.

Type sums minus overall expenditure are -1 million MOP in 2019Q1 and +1 million in
2024Q3. Retain these differences. Quarter sums versus annual controls are 64,078 vs
64,077 million MOP in 2019 and 75,357 vs 75,358 million in 2024; do not force agreement.
The 2019Q4 source workbook stores decimal per-visit values (total 1764.948161,
overnight 2883.361753 and same-day 758.585629 MOP), while its PDF rounds to integers.
The 2019 official annual per-visit value likewise retains 1626.0734524407242 MOP.
Use stored precision for calculations and round for display; extra stored digits
do not establish statistical precision. Annual per-visit spending
is not a simple mean of quarterly values. Keep official and derived per-visit values
separate.

Historical wording of the overnight definition differs between 2019 and 2023/2025;
wording alone does not establish a methodological break. No respondent microdata,
survey weights or usable variance estimates were obtained. Questionnaire counts in
the original survey are not this experiment's sample size. Published values may be
revised later; use this fixed snapshot to reproduce the results.
