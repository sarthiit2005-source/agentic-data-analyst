import sqlite3

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tools.analysis import Analysis
from tools.dataset import load_csv


@pytest.fixture
def analysis(tmp_path):
    return Analysis(pd.DataFrame({"region": ["A", "A", "B", "B"],
                                 "sales": [10., 20., 30., 40.],
                                 "cost": [2., 4., 6., 8.]}), tmp_path)


def test_aggregate_and_filter(analysis):
    assert analysis.aggregate(["region"], "sales", "sum")["rows"] == [
        {"region": "A", "value": 30.}, {"region": "B", "value": 70.}]
    assert analysis.filter("sales", "gt", 20)["matched_rows"] == 2


@pytest.mark.parametrize("operation,expected", [("sum", 100), ("mean", 25), ("count", 4),
                                              ("min", 10), ("max", 40), ("median", 25)])
def test_aggregate_without_groups(analysis, operation, expected):
    result = analysis.registry().execute("aggregate_data", {
        "group_by": [], "column": "sales", "operation": operation})
    assert result["ok"]
    assert result["result"]["rows"] == [{"value": expected}]


@pytest.mark.parametrize("query", [
    "DROP TABLE data", "DELETE FROM data", "UPDATE data SET sales=0",
    "ATTACH DATABASE 'private.db' AS private", "PRAGMA table_info(data)",
    "SELECT load_extension('x')", "CREATE TABLE x (id)", "SELECT 1; SELECT 2"])
def test_sql_rejects_mutation_and_external_access(analysis, query):
    with pytest.raises(ValueError):
        analysis.sql(query)
    assert analysis.sql("SELECT SUM(sales) AS total FROM data")["rows"][0]["total"] == 100


def test_sql_cte_and_output_cap(analysis):
    assert analysis.sql("WITH s AS (SELECT sales FROM data) SELECT SUM(sales) AS total FROM s")["rows"] == [{"total": 100}]
    result = analysis.sql("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n WHERE x<150) SELECT x FROM n")
    assert len(result["rows"]) == 100 and result["truncated"]


def test_validation(analysis):
    registry = analysis.registry()
    assert not registry.execute("filter_data", {"column": "sales", "operator": "gt", "value": 1, "limit": -2})["ok"]
    assert not registry.execute("describe_data", '{"columns": ')["ok"]
    assert not registry.execute("describe_data", {"columns": ["unknown"]})["ok"]
    assert not registry.execute("inspect_dataset", {"extra": 1})["ok"]


def test_statistics(analysis):
    result = analysis.statistical("pearson", "sales", "cost")
    assert result["statistic"] == pytest.approx(1)
    result = analysis.statistical("welch", "sales", "region", "A", "B")
    oracle = stats.ttest_ind([10, 20], [30, 40], equal_var=False)
    assert result["p_value"] == pytest.approx(oracle.pvalue)


def test_pairwise_missingness_and_constant_rejection(tmp_path):
    a = Analysis(pd.DataFrame({"x": [1, 2, np.nan, 4], "y": [2, 4, 99, 8]}), tmp_path)
    assert a.statistical("pearson", "x", "y")["complete_pairs"] == 3
    with pytest.raises(ValueError):
        Analysis(pd.DataFrame({"x": [1, 1, 1], "y": [2, 3, 4]})).statistical("pearson", "x", "y")


def test_chart_is_real_png(analysis):
    from PIL import Image
    result = analysis.chart("bar", "region", "sales")
    with Image.open(result["chart_path"]) as image:
        assert image.format == "PNG" and image.width >= 1000


def test_csv_size_and_empty(tmp_path):
    import io
    with pytest.raises(ValueError):
        load_csv(io.BytesIO(b"x\n"))
    with pytest.raises(ValueError):
        load_csv(io.BytesIO(b"x" * (10 * 1024 * 1024 + 1)))


def test_all_missing_sum_is_not_zero(tmp_path):
    analysis = Analysis(pd.DataFrame({"g": ["a", "a"], "x": [np.nan, np.nan]}), tmp_path)
    assert analysis.aggregate(["g"], "x", "sum")["rows"][0]["value"] is None


def test_chi_square_cardinality_rejected_before_allocation():
    analysis = Analysis(pd.DataFrame({"x": range(1000), "y": range(1000)}))
    with pytest.raises(ValueError, match="400"):
        analysis.statistical("chi_square", "x", "y")


def test_sql_execution_deadline(analysis):
    with pytest.raises(ValueError, match="interrupted"):
        analysis.sql("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT SUM(x) FROM n")
