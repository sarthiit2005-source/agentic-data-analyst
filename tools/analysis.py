"""Seven bounded tools; all numerical work stays outside the LLM."""
import json
import sqlite3
import time
from pathlib import Path
from uuid import uuid4

import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy import stats

from tools.registry import Registry, object_schema

STRING = {"type": "string", "minLength": 1, "maxLength": 100}
STRINGS = {"type": "array", "items": STRING, "minItems": 1, "maxItems": 10, "uniqueItems": True}


def records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


class Analysis:
    def __init__(self, frame, output_dir="outputs", schema=None):
        self.tables = ({name: value.copy(deep=True) for name, value in frame.items()}
                       if isinstance(frame, dict) else {"data": frame.copy(deep=True)})
        self.frame = next(iter(self.tables.values()))
        self.schema = schema or {"tables": {}, "relationships": []}
        self.output_dir = Path(output_dir)

    def columns(self, names, numeric=False):
        for name in names:
            if name not in self.frame:
                raise ValueError(f"Unknown column {name!r}. Available: {list(self.frame.columns)}")
            if numeric and not pd.api.types.is_numeric_dtype(self.frame[name]):
                raise ValueError(f"{name!r} must be numeric.")

    def inspect(self):
        return {"rows": len(self.frame), "columns": [
            {"name": c, "dtype": str(self.frame[c].dtype), "missing": int(self.frame[c].isna().sum())}
            for c in self.frame], "sample": records(self.frame.head(3))}

    def describe(self, columns):
        self.columns(columns, numeric=True)
        return {"statistics": records(self.frame[columns].describe().rename_axis("statistic").reset_index()),
                "missing_policy": "Statistics exclude missing values separately per column."}

    def filter(self, column, operator, value, limit=20):
        self.columns([column])
        series = self.frame[column]
        operations = {"eq": series.eq, "ne": series.ne, "gt": series.gt,
                      "ge": series.ge, "lt": series.lt, "le": series.le}
        try:
            mask = operations[operator](value) & series.notna()
        except TypeError:
            raise ValueError("Filter value must match the column type.") from None
        subset = self.frame.loc[mask]
        return {"matched_rows": len(subset), "rows": records(subset.head(limit)),
                "truncated": len(subset) > limit}

    def aggregate(self, group_by, column, operation):
        self.columns(group_by + [column], numeric=False)
        if operation != "count":
            self.columns([column], numeric=True)
        grouped = self.frame.groupby(group_by, dropna=False, observed=True)[column]
        values = grouped.sum(min_count=1) if operation == "sum" else grouped.agg(operation)
        result = values.rename("value").reset_index()
        return {"rows": records(result.head(100)), "groups": len(result),
                "truncated": len(result) > 100, "operation": operation,
                "missing_policy": "Missing group keys retained; missing metric values excluded. Count counts nonmissing values."}

    def sql(self, query, max_rows=100):
        # The authorizer enforces read-only semantics, including WITH statements.
        with sqlite3.connect(":memory:") as connection:
            for name, frame in self.tables.items():
                frame.to_sql(name, connection, index=False)
            connection.execute("PRAGMA query_only=ON")
            allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
                       sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
            connection.set_authorizer(lambda action, a, b, db, src:
                                      sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY)
            deadline = time.monotonic() + 2
            connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
            connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 5000)
            try:
                cursor = connection.execute(query)
                rows = cursor.fetchmany(max_rows + 1)
                names = [d[0] for d in cursor.description]
                if len(set(names)) != len(names):
                    raise ValueError("Use unique SQL column aliases.")
                return {"rows": [dict(zip(names, row)) for row in rows[:max_rows]],
                        "truncated": len(rows) > max_rows}
            except sqlite3.Error as error:
                raise ValueError(f"SQL rejected or failed: {error}") from None

    def statistical(self, test, x, y, group_a=None, group_b=None):
        self.columns([x, y])
        pair = self.frame[list(dict.fromkeys([x, y]))].dropna()
        details = {}
        if test == "pearson":
            self.columns([x, y], numeric=True)
            if len(pair) < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
                raise ValueError("Pearson correlation requires at least 3 complete pairs and nonconstant columns.")
            statistic, pvalue = stats.pearsonr(pair[x], pair[y])
            details = {"complete_pairs": len(pair), "effect": float(statistic),
                       "assumptions": "Independent pairs; linear association; inference assumes bivariate normality. Correlation is not causation."}
        elif test == "welch":
            self.columns([x], numeric=True)
            if group_a is None or group_b is None or group_a == group_b:
                raise ValueError("Welch test requires two distinct group labels in column y.")
            a, b = pair.loc[pair[y] == group_a, x], pair.loc[pair[y] == group_b, x]
            if min(len(a), len(b)) < 2 or (a.var() == 0 and b.var() == 0):
                raise ValueError("Each group needs at least two observations and combined nonzero variance.")
            statistic, pvalue = stats.ttest_ind(a, b, equal_var=False)
            details = {"n_a": len(a), "n_b": len(b), "mean_a": float(a.mean()),
                       "mean_b": float(b.mean()), "mean_difference": float(a.mean() - b.mean()),
                       "assumptions": "Independent observations; approximately normal group means. Two-sided Welch test, unequal variances allowed."}
        else:
            if pair[x].nunique() * pair[y].nunique() > 400:
                raise ValueError("Chi-square supports at most 400 contingency cells.")
            table = pd.crosstab(pair[x], pair[y])
            if min(table.shape) < 2 or table.size > 400:
                raise ValueError("Chi-square needs at least two levels per column and at most 400 contingency cells.")
            statistic, pvalue, dof, expected = stats.chi2_contingency(table, correction=False)
            details = {"dof": dof, "cramers_v": float(np.sqrt(statistic / (table.to_numpy().sum() * (min(table.shape) - 1)))),
                       "low_expected_counts": bool((expected < 5).any()),
                       "assumptions": "Independent observations; small expected counts undermine the asymptotic approximation."}
        if not np.isfinite([statistic, pvalue]).all():
            raise ValueError("Test produced nonfinite results; dataset is unsuitable.")
        return {"test": test, "statistic": float(statistic), "p_value": float(pvalue),
                "missing_pairs_removed": len(self.frame) - len(pair),
                "multiple_testing": "No multiple-comparison correction applied.", **details}

    def chart(self, kind, x, y=None):
        self.columns([x] + ([y] if y else []))
        figure = Figure(figsize=(8, 4.5), layout="constrained")
        axis = figure.subplots()
        if kind == "histogram":
            self.columns([x], numeric=True)
            axis.hist(self.frame[x].dropna(), bins=20, color="#2563eb")
            axis.set_ylabel("Count")
        elif kind == "scatter":
            if y is None:
                raise ValueError("Scatter chart requires y.")
            self.columns([x, y], numeric=True)
            pair = self.frame[[x, y]].dropna()
            if len(pair) > 5000:
                raise ValueError("Scatter chart supports at most 5,000 complete pairs.")
            axis.scatter(pair[x], pair[y], alpha=0.6, s=12)
            axis.set_ylabel(y)
        else:
            if y is None:
                raise ValueError("Bar chart requires numeric y; bars show group means.")
            self.columns([y], numeric=True)
            values = self.frame.groupby(x, dropna=False)[y].mean()
            if len(values) > 30:
                raise ValueError("Bar chart supports at most 30 groups.")
            axis.bar(values.index.astype(str), values.to_numpy(), color="#2563eb")
            axis.set_ylabel(f"Mean {y}")
            axis.tick_params(axis="x", rotation=30)
        axis.set_xlabel(x)
        axis.set_title(f"{kind.title()}: {x}" + (f" / {y}" if y else ""))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"chart-{uuid4().hex}.png"
        figure.savefig(path, dpi=140)
        return {"chart_path": str(path.resolve()), "kind": kind,
                "missing_policy": "Missing numeric observations excluded."}

    def registry(self):
        registry = Registry()
        registry.add("inspect_dataset", "Inspect schema, missing values and three sample rows.", object_schema({}), self.inspect)
        registry.add("describe_data", "Numeric descriptive statistics.", object_schema({"columns": STRINGS}, ["columns"]), self.describe)
        registry.add("filter_data", "Filter rows by a single typed comparison.", object_schema({
            "column": STRING, "operator": {"enum": ["eq", "ne", "gt", "ge", "lt", "le"]},
            "value": {"type": ["string", "number", "boolean"]},
            "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, ["column", "operator", "value"]), self.filter)
        registry.add("aggregate_data", "Group data; count means nonmissing count.", object_schema({
            "group_by": STRINGS, "column": STRING,
            "operation": {"enum": ["sum", "mean", "count", "min", "max", "median"]}},
            ["group_by", "column", "operation"]), self.aggregate)
        registry.add("run_sql", "Read-only SQLite query, including JOINs across tables from inspect_dataset; max 100 returned rows.", object_schema({
            "query": {"type": "string", "minLength": 1, "maxLength": 5000}}, ["query"]), self.sql)
        registry.add("statistical_test", "Pearson: numeric x/y. Welch: numeric x, groups in y. Chi-square: categorical x/y.", object_schema({
            "test": {"enum": ["pearson", "welch", "chi_square"]}, "x": STRING, "y": STRING,
            "group_a": {"type": ["string", "number"]}, "group_b": {"type": ["string", "number"]}},
            ["test", "x", "y"]), self.statistical)
        registry.add("create_chart", "Histogram, scatter or group-mean bar chart.", object_schema({
            "kind": {"enum": ["histogram", "scatter", "bar"]}, "x": STRING, "y": STRING},
            ["kind", "x"]), self.chart)
        # Bind each call to its table without mutating shared analysis state.
        for name, (description, schema, function) in list(registry.tools.items()):
            if name == "run_sql":
                continue
            schema["properties"]["table"] = {"type": "string", "enum": list(self.tables)}
            if name == "inspect_dataset":
                def inspect_tables(table=None):
                    names = [table] if table else list(self.tables)
                    if table:
                        return {"table": table, **Analysis(self.tables[table]).inspect(),
                                "declared_schema": self.schema["tables"].get(table, {})}
                    return {"tables": {n: {"rows": len(self.tables[n]),
                                           "columns": {c: str(self.tables[n][c].dtype) for c in self.tables[n]}}
                                       for n in names}, "declared_schema": self.schema}
                registry.tools[name] = (description + " Omit table to list all tables and declared relationships.", schema, inspect_tables)
                continue
            if name in ("create_chart", "statistical_test"):
                schema["properties"]["source_query"] = {"type": "string", "minLength": 1, "maxLength": 5000}
                description += " Optional source_query selects joined data (max 5,000 rows); do not combine with table."
            def bound(table=None, source_query=None, _method=function.__name__, **kwargs):
                if source_query:
                    if table:
                        raise ValueError("Specify table or source_query, not both.")
                    result = self.sql(source_query, max_rows=5000)
                    if result["truncated"] or not result["rows"]:
                        raise ValueError("Source query must return 1–5,000 rows; aggregate or filter first.")
                    selected = pd.DataFrame(result["rows"])
                    source = {"source_query": source_query}
                else:
                    if table is None and len(self.tables) > 1:
                        raise ValueError(f"Specify table. Available: {list(self.tables)}")
                    table = table or next(iter(self.tables))
                    selected = self.tables[table]
                    source = {"table": table}
                return {**getattr(Analysis(selected, self.output_dir), _method)(**kwargs), **source}
            registry.tools[name] = (description, schema, bound)
        return registry
