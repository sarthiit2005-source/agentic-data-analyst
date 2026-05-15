"""Generate owned synthetic data and explicit ground truth; no external dataset license."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]


def build():
    rng = np.random.default_rng(42)
    sales = pd.DataFrame([
        {"region": region, "quarter": quarter, "revenue": revenue, "units": units}
        for region, revenues, units in [("North", [100, 120, 80, 110], 10),
                                        ("South", [90, 100, 95, 105], 8),
                                        ("West", [80, 110, 70, 100], 7)]
        for quarter, revenue in enumerate(revenues, 1)])
    experiments = pd.DataFrame({"group": ["control"] * 30 + ["treatment"] * 30,
                                "score": np.round(np.r_[rng.normal(50, 7, 30), rng.normal(56, 7, 30)], 4),
                                "hours": np.round(rng.uniform(1, 8, 60), 4),
                                "passed": ["yes"] * 18 + ["no"] * 12 + ["yes"] * 24 + ["no"] * 6})
    operations = pd.DataFrame({"team": ["A", "A", "B", "B", "C", "C"],
                               "duration": [2., 4., 3., np.nan, 6., 9.],
                               "tickets": [5, 10, 8, 4, 7, 12]})
    data = ROOT / "data"
    data.mkdir(exist_ok=True)
    for name, frame in [("sales", sales), ("experiments", experiments), ("operations", operations)]:
        frame.to_csv(data / f"{name}.csv", index=False)
    tasks = []

    def add(dataset, category, question, tool, args, path, expected, **extra):
        tasks.append(dict(id=f"T{len(tasks)+1:02}", dataset=f"data/{dataset}.csv", category=category,
                          question=question, oracle={"tool": tool, "arguments": args, "path": path},
                          expected=expected, **extra))

    for column in ("revenue", "units"):
        for op in ("sum", "avg", "min", "max"):
            value = {"sum": sales[column].sum, "avg": sales[column].mean,
                     "min": sales[column].min, "max": sales[column].max}[op]()
            add("sales", "numeric", f"What is the {op} of {column} across all sales rows?",
                "run_sql", {"query": f"SELECT {op}({column}) AS answer FROM data"},
                ["rows", 0, "answer"], float(value))
    for region in ("North", "South", "West"):
        value = float(sales.loc[sales.region == region, "revenue"].sum())
        add("sales", "aggregation", f"What is total revenue for {region}?",
            "run_sql", {"query": f"SELECT SUM(revenue) AS answer FROM data WHERE region='{region}'"},
            ["rows", 0, "answer"], value)
    for quarter in (1, 2, 3, 4):
        value = float(sales.loc[sales.quarter == quarter, "revenue"].sum())
        add("sales", "filtering", f"What is total revenue in quarter {quarter}?",
            "run_sql", {"query": f"SELECT SUM(revenue) AS answer FROM data WHERE quarter={quarter}"},
            ["rows", 0, "answer"], value)
    for region in ("North", "South", "West"):
        subset = sales[sales.region == region].set_index("quarter").revenue
        add("sales", "multi_step", f"For {region}, calculate the percentage change in revenue from quarter 2 to quarter 3.",
            "run_sql", {"query": f"SELECT 100.0*(SUM(CASE WHEN quarter=3 THEN revenue ELSE 0 END)-SUM(CASE WHEN quarter=2 THEN revenue ELSE 0 END))/SUM(CASE WHEN quarter=2 THEN revenue ELSE 0 END) AS answer FROM data WHERE region='{region}'"},
            ["rows", 0, "answer"], float((subset[3] - subset[2]) / subset[2] * 100))
    for op in ("avg", "sum", "min", "max"):
        value = {"avg": operations.duration.mean, "sum": operations.duration.sum,
                 "min": operations.duration.min, "max": operations.duration.max}[op]()
        add("operations", "missing_data", f"What is {op} duration, excluding missing values?",
            "run_sql", {"query": f"SELECT {op}(duration) AS answer FROM data"},
            ["rows", 0, "answer"], float(value))
    add("operations", "missing_data", "How many durations are missing?", "run_sql",
        {"query": "SELECT COUNT(*) AS answer FROM data WHERE duration IS NULL"}, ["rows", 0, "answer"], 1)
    add("experiments", "statistics", "Report the two-sided Welch p-value comparing score between control and treatment.",
        "statistical_test", {"test": "welch", "x": "score", "y": "group", "group_a": "control", "group_b": "treatment"},
        ["p_value"], float(stats.ttest_ind(experiments.score[:30], experiments.score[30:], equal_var=False).pvalue))
    add("experiments", "statistics", "Report Pearson correlation r between score and hours.",
        "statistical_test", {"test": "pearson", "x": "score", "y": "hours"},
        ["statistic"], float(stats.pearsonr(experiments.score, experiments.hours).statistic))
    add("experiments", "statistics", "Report chi-square p-value for group and passed, without continuity correction.",
        "statistical_test", {"test": "chi_square", "x": "group", "y": "passed"},
        ["p_value"], float(stats.chi2_contingency(pd.crosstab(experiments.group, experiments.passed), correction=False).pvalue))
    for dataset, args in [("sales", {"kind": "bar", "x": "region", "y": "revenue"}),
                          ("operations", {"kind": "histogram", "x": "duration"}),
                          ("experiments", {"kind": "scatter", "x": "hours", "y": "score"})]:
        add(dataset, "chart", f"Create a {args['kind']} chart using x={args['x']}" +
            (f" and y={args['y']}" if "y" in args else "") + ".",
            "create_chart", args, ["chart_path"], "file_exists")
    for dataset, question, outcome in [
        ("sales", "Which region is best? Ask me which metric and quarter I mean before analyzing.", "clarification"),
        ("operations", "Is performance good? Ask which measure and threshold I mean.", "clarification"),
        ("sales", "What is total profit? The dataset contains no cost or profit. Explain why this cannot be calculated.", "unanswerable"),
        ("operations", "What is the average salary? If absent, explain the data limitation.", "unanswerable"),
        ("sales", "Prove that our advertising campaign caused the Q3 decline using only this dataset.", "unanswerable"),
        ("experiments", "Train a classifier for passed.", "unanswerable")]:
        tasks.append(dict(id=f"T{len(tasks)+1:02}", dataset=f"data/{dataset}.csv",
                          category=outcome, question=question, expected=outcome))
    add("sales", "recovery", "Inspect the dataset, then report total revenue. Retry any transient tool error.",
        "run_sql", {"query": "SELECT SUM(revenue) AS answer FROM data"}, ["rows", 0, "answer"],
        float(sales.revenue.sum()), inject_error=True)
    target = ROOT / "evaluation" / "tasks.json"
    target.write_text(json.dumps(tasks, indent=2), encoding="utf-8")
    print(f"Generated {len(tasks)} tasks across 3 synthetic datasets.")


if __name__ == "__main__":
    build()
