import numpy as np
import pandas as pd
import json
import urllib.request
import urllib.error


def profile_dataframe(df):
    profile = {
        "shape": list(df.shape),
        "columns": {},
    }

    for col in df.columns:
        col_info = _profile_column(df, col)
        col_info["_name"] = str(col)
        _infer_role(col_info, str(col))
        profile["columns"][col] = col_info

    return profile


def _profile_column(df, col):
    series = df[col]
    dtype = series.dtype
    null_count = int(series.isna().sum())
    null_pct = round(null_count / max(len(series), 1) * 100, 1)
    unique_count = int(series.nunique())
    unique_pct = round(unique_count / max(len(series), 1) * 100, 1)
    sample_values = series.dropna().head(5).tolist()

    info = {
        "dtype": str(dtype),
        "null_pct": null_pct,
        "unique_pct": unique_pct,
        "unique_count": unique_count,
        "sample_values": sample_values,
        "is_constant": unique_count <= 1,
    }

    if pd.api.types.is_numeric_dtype(dtype):
        info["type_category"] = "numeric"
        info["min"] = _safe_float(series.min())
        info["max"] = _safe_float(series.max())
        info["mean"] = _safe_float(series.mean())
        info["std"] = _safe_float(series.std())
        info["skew"] = _safe_float(series.skew())
        info["kurtosis"] = _safe_float(series.kurtosis())
        info["quantiles"] = {
            "25%": _safe_float(series.quantile(0.25)),
            "50%": _safe_float(series.quantile(0.5)),
            "75%": _safe_float(series.quantile(0.75)),
        }
    elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
        info["type_category"] = "categorical"
        value_counts = series.dropna().value_counts().head(10)
        info["top_values"] = {str(k): int(v) for k, v in value_counts.items()}
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        info["type_category"] = "datetime"
        info["min"] = str(series.min())
        info["max"] = str(series.max())
    else:
        info["type_category"] = "other"

    return info


def _safe_float(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def _infer_role(col_info, col_name=""):
    name_lower = col_name.lower()

    if col_info.get("is_constant"):
        col_info["suggested_role"] = "drop_constant"
        col_info["description"] = "Constant column — no predictive value"
        return

    date_keywords = ["date", "time", "day", "month", "year", "hour", "minute", "second", "timestamp"]
    if any(kw in name_lower for kw in date_keywords):
        col_info["suggested_role"] = "datetime"
        col_info["description"] = f"Date/time column — '{col_name}'"
        return

    if col_info.get("unique_pct", 0) > 95 and col_info.get("type_category") == "numeric":
        col_info["suggested_role"] = "id"
        col_info["description"] = "High cardinality numeric — likely an ID column"
        return

    if col_info.get("unique_pct", 0) > 90 and col_info.get("type_category") == "categorical":
        col_info["suggested_role"] = "id"
        col_info["description"] = "High cardinality categorical — likely an ID or text field"
        return

    if col_info.get("type_category") == "datetime":
        col_info["suggested_role"] = "datetime"
        col_info["description"] = "Datetime column — can be used for time-based features"
        return

    if col_info.get("type_category") == "numeric":
        skew = abs(col_info.get("skew", 0) or 0)
        if skew > 2:
            col_info["suggested_role"] = "feature"
            col_info["description"] = f"Highly skewed numeric (skew={col_info['skew']:.1f}) — may benefit from log/sqrt transformation"
        elif col_info.get("unique_count", 0) < 20:
            col_info["suggested_role"] = "categorical_numeric"
            col_info["description"] = f"Low-cardinality numeric ({col_info['unique_count']} values) — could be ordinal categorical"
        else:
            col_info["suggested_role"] = "feature"
            col_info["description"] = f"Numeric feature — range [{col_info.get('min', '?')}, {col_info.get('max', '?')}]"
    elif col_info.get("type_category") == "categorical":
        if col_info.get("unique_count", 0) == 2:
            col_info["suggested_role"] = "binary"
            col_info["description"] = "Binary categorical (2 unique values)"
        else:
            col_info["suggested_role"] = "feature"
            col_info["description"] = f"Categorical feature — {col_info.get('unique_count', 0)} unique values"
    elif col_info.get("type_category") == "other":
        col_info["suggested_role"] = "other"
        col_info["description"] = "Non-numeric, non-categorical column"


def enrich_with_llm(profile, df, ollama_url="http://localhost:11434/api/generate", model="qwen2.5:7b"):
    columns_data = []
    for col_name, col_info in profile["columns"].items():
        entry = {
            "name": col_name,
            "dtype": col_info["dtype"],
            "type_category": col_info["type_category"],
            "null_pct": col_info["null_pct"],
            "unique_count": col_info["unique_count"],
            "sample": col_info.get("sample_values", [])[:3],
        }
        if col_info["type_category"] == "numeric":
            entry["stats"] = {
                "mean": col_info.get("mean"),
                "std": col_info.get("std"),
                "min": col_info.get("min"),
                "max": col_info.get("max"),
                "skew": col_info.get("skew"),
            }
        elif col_info["type_category"] == "categorical":
            entry["top_values"] = list(col_info.get("top_values", {}).keys())[:5]
        columns_data.append(entry)

    prompt = f"""You are a data profiling assistant. Analyze this dataset with {len(columns_data)} columns.

For each column, return a JSON array of objects with:
- "name": column name
- "description": what this column likely represents (1 sentence in Russian)
- "suggested_role": "feature" | "target" | "id" | "datetime" | "drop"
- "suggested_transform": null or "log" | "sqrt" | "onehot" | "scale" | "binarize"

Columns:
{json.dumps(columns_data, ensure_ascii=False, indent=2)}

Respond ONLY with the JSON array, no other text."""

    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 4096}
    }).encode()

    try:
        req = urllib.request.Request(ollama_url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode())
        response_text = result.get("response", "")

        llm_data = _parse_llm_response(response_text, profile)
        for col_name, enrichment in llm_data.items():
            if col_name in profile["columns"]:
                profile["columns"][col_name].update(enrichment)
        profile["_llm_enriched"] = True
    except Exception as e:
        profile["_llm_enriched"] = False
        profile["_llm_error"] = str(e)

    return profile


def _parse_llm_response(text, profile):
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        text = text.strip()
        data = json.loads(text)
        if isinstance(data, list):
            return {item["name"]: {"llm_description": item.get("description", ""),
                                   "llm_role": item.get("suggested_role", ""),
                                   "llm_transform": item.get("suggested_transform")}
                    for item in data if "name" in item}
    except Exception:
        pass
    return {}
