import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

MAX_BYTES = 10 * 1024 * 1024
MAX_ROWS = 100_000
MAX_COLUMNS = 100


def load_csv(source, dtypes=None):
    """Bound bytes before parsing, then bound dimensions and cell lengths."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported.")
        with path.open("rb") as handle:
            raw = handle.read(MAX_BYTES + 1)
    else:
        raw = source.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ValueError("CSV exceeds the 10 MiB upload limit.")
    frame = pd.read_csv(io.BytesIO(raw), nrows=MAX_ROWS + 1, dtype=dtypes)
    if frame.empty or len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise ValueError("CSV must contain 1–100,000 rows and at most 100 columns.")
    if any(len(str(c)) > 100 for c in frame.columns):
        raise ValueError("Column names must be at most 100 characters.")
    if any(frame[c].astype(str).str.len().max() > 2000 for c in frame.columns):
        raise ValueError("CSV cells must be at most 2,000 characters.")
    return frame.replace([np.inf, -np.inf], np.nan)


def table_name(filename):
    name = re.sub(r"[^a-z0-9_]", "_", Path(filename).stem.lower())
    if not name or name[0].isdigit():
        name = "table_" + name
    if name.startswith("sqlite_") or len(name) > 64:
        raise ValueError("Table names must be at most 64 characters and cannot start with sqlite_.")
    return name


def load_schema(raw=None):
    """Optional JSON metadata; never executable SQL or model instructions."""
    from jsonschema import validate, ValidationError
    if not raw:
        return {"tables": {}, "relationships": []}
    if len(raw) > 64_000:
        raise ValueError("Schema must be at most 64 KB.")
    text = {"type": "string", "maxLength": 1000}
    column = {"type": "object", "properties": {
        "type": {"enum": ["string", "integer", "number", "boolean"]},
        "description": text, "nullable": {"type": "boolean"}},
        "additionalProperties": False}
    table = {"type": "object", "properties": {
        "description": text, "columns": {"type": "object", "additionalProperties": column},
        "primary_key": {"type": "array", "items": {"type": "string"}, "minItems": 1, "uniqueItems": True}},
        "additionalProperties": False}
    endpoint = {"type": "object", "properties": {"table": {"type": "string"}, "column": {"type": "string"}},
                "required": ["table", "column"], "additionalProperties": False}
    schema = {"type": "object", "properties": {
        "tables": {"type": "object", "additionalProperties": table},
        "relationships": {"type": "array", "maxItems": 30, "items": {
            "type": "object", "properties": {"from": endpoint, "to": endpoint},
            "required": ["from", "to"], "additionalProperties": False}}},
        "additionalProperties": False}
    try:
        result = json.loads(raw)
        validate(result, schema)
    except (ValueError, ValidationError) as error:
        raise ValueError(f"Invalid schema JSON: {str(error)[:300]}") from None
    return {"tables": {}, "relationships": [], **result}


def load_tables(files, schema=None):
    """files is a list of (original filename, bytes); validation is atomic."""
    schema = schema or load_schema()
    if not 1 <= len(files) <= 10:
        raise ValueError("Select between 1 and 10 CSV files.")
    if sum(len(raw) for _, raw in files) > 20 * 1024 * 1024:
        raise ValueError("Combined CSV uploads exceed 20 MiB.")
    tables = {}
    for filename, raw in files:
        if Path(filename).suffix.lower() != ".csv":
            raise ValueError("Only CSV files are supported.")
        name = table_name(filename)
        if name in tables:
            raise ValueError(f"Duplicate table name {name!r}. Rename the CSV files.")
        definitions = schema["tables"].get(name, {}).get("columns", {})
        dtype_names = {"string": "string", "integer": "Int64", "number": "Float64", "boolean": "boolean"}
        dtypes = {c: dtype_names[d["type"]] for c, d in definitions.items() if "type" in d}
        tables[name] = load_csv(io.BytesIO(raw), dtypes=dtypes or None)
    if sum(len(frame) for frame in tables.values()) > MAX_ROWS:
        raise ValueError("Combined CSVs must contain at most 100,000 rows.")
    for name, definition in schema["tables"].items():
        if name not in tables:
            raise ValueError(f"Schema references missing table {name!r}.")
        frame = tables[name]
        for column, info in definition.get("columns", {}).items():
            if column not in frame:
                raise ValueError(f"Schema references missing column {name}.{column}.")
            if not info.get("nullable", True) and frame[column].isna().any():
                raise ValueError(f"{name}.{column} contains nulls but nullable is false.")
        keys = definition.get("primary_key", [])
        if keys:
            if any(c not in frame for c in keys):
                raise ValueError(f"Unknown primary-key column in {name}.")
            if frame[keys].isna().any().any() or frame.duplicated(keys).any():
                raise ValueError(f"Primary key in {name} contains null or duplicate values.")
    for relation in schema["relationships"]:
        for endpoint in (relation["from"], relation["to"]):
            if endpoint["table"] not in tables or endpoint["column"] not in tables[endpoint["table"]]:
                raise ValueError(f"Unknown relationship endpoint: {endpoint}.")
        source, target = relation["from"], relation["to"]
        a = tables[source["table"]][source["column"]].dropna()
        b = tables[target["table"]][target["column"]]
        if b.isna().any() or b.duplicated().any():
            raise ValueError("Relationship targets must be unique and non-null (many-to-one).")
        if not a.isin(b).all():
            raise ValueError("Relationship contains unmatched foreign-key values. Check keys and declared types.")
    return tables
