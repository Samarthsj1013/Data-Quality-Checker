import pandas as pd
import numpy as np
import re

# ─────────────────────────────────────────────
# HIDDEN NULL VALUES
# ─────────────────────────────────────────────
NULL_LIKE = [
    "", " ", "  ", "none", "null", "na", "n/a", "nan",
    "not_available", "not available", "unknown", "undefined",
    "missing", "-", "--", "nil", "void", "N/A", "NA", "None"
]

def normalize_nulls(df):
    cleaned = df.copy()
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].replace(NULL_LIKE, np.nan)
        cleaned[col] = cleaned[col].apply(
            lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
        )
    return cleaned


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
def load_data(file):
    if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        df = pd.read_excel(file, dtype=str)
    else:
        df = pd.read_csv(file, dtype=str)
    df = normalize_nulls(df)
    return df


# ─────────────────────────────────────────────
# SMART TYPE INFERENCE
# ─────────────────────────────────────────────
def infer_column_types(df):
    result = {}
    for col in df.columns:
        series = df[col].dropna()
        coerced_num = pd.to_numeric(series, errors='coerce')
        num_success = coerced_num.notna().sum()
        num_rate = num_success / len(series) if len(series) > 0 else 0

        coerced_date = pd.to_datetime(series, errors='coerce')
        date_success = coerced_date.notna().sum()
        date_rate = date_success / len(series) if len(series) > 0 else 0

        actual = str(df[col].dtype)

        # Phone columns should never be inferred as numeric
        is_phone = any(k in col.lower() for k in ['phone', 'mobile', 'contact', 'tel'])

        if num_rate >= 0.6 and not is_phone:
            inferred = "numeric"
        elif date_rate >= 0.6:
            inferred = "datetime"
        else:
            inferred = "text"

        result[col] = {
            "actual": actual,
            "inferred": inferred,
            "coerced_numeric": pd.to_numeric(df[col], errors='coerce'),
            "coerced_datetime": pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True),
        }
    return result


# ─────────────────────────────────────────────
# MISSING VALUES
# ─────────────────────────────────────────────
def check_nulls(df):
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df) * 100).round(2)
    return pd.DataFrame({
        "Missing Count": null_counts,
        "Missing %": null_percent
    }).reset_index().rename(columns={"index": "Column"})


# ─────────────────────────────────────────────
# DUPLICATES — FIXED STRONG VERSION
# ─────────────────────────────────────────────
def check_duplicates(df):
    df_norm = df.copy()
    for col in df_norm.columns:
        df_norm[col] = df_norm[col].astype(str).str.strip().str.lower()
    df_norm = df_norm.replace("nan", np.nan)

    # Check duplicates ignoring id-like columns
    id_cols = [col for col in df_norm.columns 
               if col.lower() in ['id', 'index', 'row_id', 'record_id', 'sr', 'sr.no', 'sno']]
    check_cols = [col for col in df_norm.columns if col not in id_cols]

    return int(df_norm[check_cols].duplicated().sum())


# ─────────────────────────────────────────────
# DATA TYPES TABLE
# ─────────────────────────────────────────────
def check_data_types(df, type_info):
    rows = []
    for col in df.columns:
        info = type_info[col]
        rows.append({
            "Column": col,
            "Stored As": info["actual"],
            "Inferred Type": info["inferred"],
            "Type Mismatch": "⚠️ Yes" if info["actual"] in ["object", "str"] and info["inferred"] == "numeric" else "✅ No"
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# OUTLIERS
# ─────────────────────────────────────────────
def detect_outliers(df, type_info):
    results = []
    for col in df.columns:
        info = type_info[col]
        if info["inferred"] == "numeric":
            series = info["coerced_numeric"].dropna()
            if len(series) < 4:
                continue
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (info["coerced_numeric"] < Q1 - 1.5 * IQR) | \
                           (info["coerced_numeric"] > Q3 + 1.5 * IQR)
            outlier_count = int(outlier_mask.sum())
            results.append({
                "Column": col,
                "Outlier Count": outlier_count,
                "Min": round(float(series.min()), 2),
                "Max": round(float(series.max()), 2),
                "IQR Lower": round(float(Q1 - 1.5 * IQR), 2),
                "IQR Upper": round(float(Q3 + 1.5 * IQR), 2)
            })
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["Column", "Outlier Count", "Min", "Max", "IQR Lower", "IQR Upper"]
    )


# ─────────────────────────────────────────────
# EMAIL VALIDATION — SIMPLIFIED & FIXED
# ─────────────────────────────────────────────
EMAIL_REGEX = re.compile(r'^[\w\.\-]+@[\w\.\-]+\.\w{2,}$')

def validate_emails(df):
    results = []
    for col in df.columns:
        if "email" in col.lower() or "mail" in col.lower():
            valid_mask = df[col].notna() & df[col].apply(
                lambda x: bool(EMAIL_REGEX.match(str(x).strip()))
            )
            invalid_mask = df[col].notna() & ~valid_mask
            invalid_count = int(invalid_mask.sum())
            if invalid_count > 0:
                examples = df[col][invalid_mask].head(3).tolist()
                results.append({
                    "Column": col,
                    "Invalid Count": invalid_count,
                    "Invalid %": round(invalid_count / len(df) * 100, 2),
                    "Examples": ", ".join(str(e) for e in examples)
                })
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["Column", "Invalid Count", "Invalid %", "Examples"]
    )


# ─────────────────────────────────────────────
# DATE VALIDATION
# ─────────────────────────────────────────────
def validate_dates(df, type_info):
    results = []
    for col in df.columns:
        info = type_info[col]
        if info["inferred"] == "datetime" or "date" in col.lower():
            parsed = info["coerced_datetime"]
            invalid = df[col].notna() & parsed.isna()
            invalid_count = int(invalid.sum())
            if invalid_count > 0:
                examples = df[col][invalid].head(3).tolist()
                results.append({
                    "Column": col,
                    "Invalid Dates": invalid_count,
                    "Invalid %": round(invalid_count / len(df) * 100, 2),
                    "Examples": ", ".join(str(e) for e in examples)
                })
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["Column", "Invalid Dates", "Invalid %", "Examples"]
    )


# ─────────────────────────────────────────────
# PHONE VALIDATION
# ─────────────────────────────────────────────
PHONE_REGEX = re.compile(r'^\+?[\d\s\-\(\)]{7,15}$')

def validate_phones(df):
    results = []
    for col in df.columns:
        if any(k in col.lower() for k in ['phone', 'mobile', 'contact', 'tel']):
            valid = df[col].dropna().apply(
                lambda x: bool(PHONE_REGEX.match(str(x).strip()))
            )
            invalid_count = int((~valid).sum())
            if invalid_count > 0:
                results.append({
                    "Column": col,
                    "Invalid Count": invalid_count,
                    "Invalid %": round(invalid_count / len(df) * 100, 2)
                })
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["Column", "Invalid Count", "Invalid %"]
    )


# ─────────────────────────────────────────────
# VALUE CONSISTENCY
# ─────────────────────────────────────────────
def check_consistency(df):
    results = []
    for col in df.columns:
        if df[col].dtype == object or str(df[col].dtype) in ["str", "object"]:
            unique_vals = df[col].dropna().unique()
            if 2 <= len(unique_vals) <= 50:
                lower_map = {}
                for v in unique_vals:
                    key = str(v).strip().lower()
                    lower_map.setdefault(key, []).append(v)
                conflicts = {k: v for k, v in lower_map.items() if len(v) > 1}
                if conflicts:
                    examples = [f"{' / '.join(str(x) for x in vals)}" for vals in list(conflicts.values())[:3]]
                    results.append({
                        "Column": col,
                        "Issue": "Case/format inconsistency",
                        "Examples": ", ".join(examples)
                    })
    return pd.DataFrame(results) if results else pd.DataFrame(
        columns=["Column", "Issue", "Examples"]
    )


# ─────────────────────────────────────────────
# QUALITY SCORE — FIXED WEIGHTS
# ─────────────────────────────────────────────
def quality_score(df, null_df, duplicate_count, outlier_df,
                  email_df, date_df, phone_df, consistency_df, type_info):
    score = 100.0
    total_rows = len(df)

    # 1. Nulls (up to -30)
    avg_null = null_df["Missing %"].mean()
    score -= min(avg_null * 0.75, 30)

    # 2. Duplicates (up to -20)
    dup_pct = (duplicate_count / total_rows) * 100
    score -= min(dup_pct * 2, 20)

    # 3. Outliers (up to -20) — increased weight
    if not outlier_df.empty:
        total_outliers = outlier_df["Outlier Count"].sum()
        score -= min((total_outliers / total_rows) * 150, 20)

    # 4. Type mismatches (up to -10)
    mismatch_count = sum(
        1 for col in df.columns
        if type_info[col]["actual"] in ["object", "str"]
        and type_info[col]["inferred"] == "numeric"
    )
    score -= min(mismatch_count * 2, 10)

    # 5. Invalid emails (up to -10)
    if not email_df.empty:
        total_invalid_emails = email_df["Invalid Count"].sum()
        score -= min((total_invalid_emails / total_rows) * 100, 10)

    # 6. Invalid dates (up to -8)
    if not date_df.empty:
        total_invalid_dates = date_df["Invalid Dates"].sum()
        score -= min((total_invalid_dates / total_rows) * 100, 8)

    # 7. Invalid phones (up to -5)
    if not phone_df.empty:
        total_invalid_phones = phone_df["Invalid Count"].sum()
        score -= min((total_invalid_phones / total_rows) * 100, 5)

    # 8. Consistency issues (up to -5)
    if not consistency_df.empty:
        score -= min(len(consistency_df) * 1.5, 5)

    return round(max(score, 0), 1)


# ─────────────────────────────────────────────
# SUGGESTIONS — UPDATED
# ─────────────────────────────────────────────
def generate_suggestions(df, null_df, outlier_df, duplicate_count,
                          email_df, date_df, phone_df, consistency_df, type_info):
    suggestions = []

    for _, row in null_df.iterrows():
        col = row["Column"]
        missing_pct = row["Missing %"]
        if missing_pct == 0:
            continue
        elif missing_pct > 50:
            suggestions.append({
                "Column": col,
                "Issue": f"🔴 {missing_pct}% missing",
                "Suggestion": "Consider dropping this column - over half the data is missing"
            })
        elif type_info.get(col, {}).get("inferred") == "numeric":
            series = type_info[col]["coerced_numeric"]
            median_val = round(float(series.median()), 2)
            suggestions.append({
                "Column": col,
                "Issue": f"🟡 {missing_pct}% missing",
                "Suggestion": f"Fill nulls with median = {median_val} (numeric column)"
            })
        else:
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "N/A"
            suggestions.append({
                "Column": col,
                "Issue": f"🟡 {missing_pct}% missing",
                "Suggestion": f"Fill nulls with most frequent value: '{mode_val}'"
            })

    if duplicate_count > 0:
        suggestions.append({
            "Column": "Entire Dataset",
            "Issue": f"🟡 {duplicate_count} duplicate rows detected",
            "Suggestion": "Run df.drop_duplicates() after normalizing whitespace and case"
        })

    for _, row in outlier_df.iterrows():
        if row["Outlier Count"] > 0:
            suggestions.append({
                "Column": row["Column"],
                "Issue": f"🟠 {row['Outlier Count']} outliers (valid range: {row['IQR Lower']} to {row['IQR Upper']})",
                "Suggestion": "Cap values at IQR bounds or investigate - could be data entry errors"
            })

    for col in df.columns:
        info = type_info.get(col, {})
        if info.get("actual") in ["object", "str"] and info.get("inferred") == "numeric":
            suggestions.append({
                "Column": col,
                "Issue": "🔵 Stored as text but looks numeric",
                "Suggestion": f"Convert with pd.to_numeric(df['{col}'], errors='coerce')"
            })

    for _, row in email_df.iterrows():
        suggestions.append({
            "Column": row["Column"],
            "Issue": f"📧 {row['Invalid Count']} invalid emails ({row['Invalid %']}%)",
            "Suggestion": f"Flag/remove rows with bad emails. Examples: {row['Examples']}"
        })

    for _, row in date_df.iterrows():
        suggestions.append({
            "Column": row["Column"],
            "Issue": f"📅 {row['Invalid Dates']} invalid dates ({row['Invalid %']}%)",
            "Suggestion": f"Use pd.to_datetime(..., errors='coerce'). Examples: {row['Examples']}"
        })

    for _, row in phone_df.iterrows():
        suggestions.append({
            "Column": row["Column"],
            "Issue": f"📱 {row['Invalid Count']} invalid phone numbers",
            "Suggestion": "Standardize format - remove letters, check digit count"
        })

    for _, row in consistency_df.iterrows():
        suggestions.append({
            "Column": row["Column"],
            "Issue": f"⚠️ {row['Issue']}",
            "Suggestion": f"Standardize values: {row['Examples']}"
        })

    if not suggestions:
        suggestions.append({
            "Column": "All columns",
            "Issue": "✅ No major issues found",
            "Suggestion": "Your dataset looks clean!"
        })

    return pd.DataFrame(suggestions)


# ─────────────────────────────────────────────
# AUTO CLEAN — FULLY FIXED
# ─────────────────────────────────────────────
def auto_clean(df, type_info):
    cleaned = df.copy()
    changes = []

    # Normalize nulls
    before_nulls = cleaned.isnull().sum().sum()
    cleaned = normalize_nulls(cleaned)
    after_nulls = cleaned.isnull().sum().sum()
    if after_nulls > before_nulls:
        changes.append(f"✅ Converted {after_nulls - before_nulls} hidden null-like values to NaN")

    # Remove duplicates
    df_norm = cleaned.copy()
    for col in df_norm.columns:
        df_norm[col] = df_norm[col].astype(str).str.strip().str.lower()
    df_norm = df_norm.replace("nan", np.nan)
    dup_mask = df_norm.duplicated()
    removed = int(dup_mask.sum())
    if removed > 0:
        cleaned = cleaned[~dup_mask]
        changes.append(f"✅ Removed {removed} duplicate rows")

    # Fix nulls column by column
    for col in cleaned.columns:
        null_count = cleaned[col].isnull().sum()
        if null_count == 0:
            continue
        null_pct = (null_count / len(cleaned)) * 100
        info = type_info.get(col, {})
        is_phone = any(k in col.lower() for k in ['phone', 'mobile', 'contact', 'tel'])

        if null_pct > 50:
            cleaned = cleaned.drop(columns=[col])
            changes.append(f"✅ Dropped column '{col}' - {round(null_pct, 1)}% missing")
        elif is_phone:
            # Phone columns — use mode, never median
            if not cleaned[col].mode().empty:
                mode_val = cleaned[col].mode()[0]
                cleaned[col] = cleaned[col].fillna(mode_val)
                changes.append(f"✅ Filled '{col}' phone nulls with most frequent number ('{mode_val}')")
        elif is_phone:
            if not cleaned[col].mode().empty:
                mode_val = cleaned[col].mode()[0]
                cleaned[col] = cleaned[col].fillna(mode_val)
                changes.append(f"✅ Filled '{col}' phone nulls with most frequent number ('{mode_val}')")
        elif info.get("inferred") == "numeric":
            # Fix: coerce to numeric first, then fill median
            coerced = pd.to_numeric(cleaned[col], errors='coerce')
            median_val = round(float(coerced.median()), 2)
            cleaned[col] = coerced.fillna(median_val)
            changes.append(f"✅ Filled '{col}' nulls with median ({median_val})")
        else:
            if not cleaned[col].mode().empty:
                mode_val = cleaned[col].mode()[0]
                cleaned[col] = cleaned[col].fillna(mode_val)
                changes.append(f"✅ Filled '{col}' nulls with most frequent value ('{mode_val}')")

    if not changes:
        changes.append("✅ Dataset was already clean - no changes made!")

    return cleaned, changes


# ─────────────────────────────────────────────
# CORRELATION HEATMAP
# ─────────────────────────────────────────────
def correlation_heatmap(df, type_info):
    numeric_data = {}
    for col in df.columns:
        if type_info[col]["inferred"] == "numeric":
            numeric_data[col] = type_info[col]["coerced_numeric"]

    if len(numeric_data) < 2:
        return None

    numeric_df = pd.DataFrame(numeric_data)
    return numeric_df.corr().round(2)


# ─────────────────────────────────────────────
# DISTRIBUTION PLOTS
# ─────────────────────────────────────────────
def distribution_plots(df, type_info):
    return [col for col in df.columns if type_info[col]["inferred"] == "numeric"]


# ─────────────────────────────────────────────
# AI-STYLE INSIGHTS (rule-based but smart)
# ─────────────────────────────────────────────
def generate_ai_insights(df, null_df, duplicate_count, outlier_df,
                         email_df, date_df, phone_df, consistency_df, type_info):
    
    insights = []
    total_rows = len(df)

    # 🔴 Missing Data Insight
    avg_null = null_df["Missing %"].mean()
    if avg_null > 20:
        insights.append(f"🔴 High missing data ({round(avg_null,1)}%) detected — consider dropping or aggressively imputing columns.")
    elif avg_null > 5:
        insights.append(f"🟡 Moderate missing data ({round(avg_null,1)}%) — imputation recommended.")
    elif avg_null > 0:
        insights.append(f"🟢 Low missing data ({round(avg_null,1)}%) — manageable with simple fixes.")

    # 🟡 Duplicate Insight
    if duplicate_count > 0:
        dup_pct = round((duplicate_count / total_rows) * 100, 2)
        insights.append(f"🟡 {duplicate_count} duplicate rows ({dup_pct}%) detected — may indicate redundant data collection or merging issues.")

    # 🟠 Outlier Insight
    if not outlier_df.empty:
        total_outliers = outlier_df["Outlier Count"].sum()
        if total_outliers > 0:
            insights.append(f"🟠 {total_outliers} outliers detected — possible data entry errors or extreme cases affecting analysis.")

    # 🔵 Type Mismatch Insight
    mismatch_cols = [
        col for col in df.columns
        if type_info[col]["actual"] in ["object", "str"]
        and type_info[col]["inferred"] == "numeric"
    ]
    if mismatch_cols:
        insights.append(f"🔵 {len(mismatch_cols)} columns stored as text but appear numeric — may cause incorrect computations.")

    # 📧 Email Insight
    if not email_df.empty:
        invalid = email_df["Invalid Count"].sum()
        pct = round((invalid / total_rows) * 100, 2)
        insights.append(f"📧 {invalid} invalid emails ({pct}%) — may impact communication or user data integrity.")

    # 📅 Date Insight
    if not date_df.empty:
        invalid = date_df["Invalid Dates"].sum()
        pct = round((invalid / total_rows) * 100, 2)
        insights.append(f"📅 {invalid} invalid dates ({pct}%) — inconsistent formats may break time-based analysis.")

    # 📱 Phone Insight
    if not phone_df.empty:
        invalid = phone_df["Invalid Count"].sum()
        pct = round((invalid / total_rows) * 100, 2)
        insights.append(f"📱 {invalid} invalid phone numbers ({pct}%) — formatting issues detected.")

    # ⚠️ Consistency Insight
    if not consistency_df.empty:
        insights.append(f"⚠️ {len(consistency_df)} columns have inconsistent formatting (e.g., case or naming differences).")

    # 📊 Overall Summary Insight
    if len(insights) >= 5:
        insights.append("💡 Overall: Dataset has multiple quality issues — cleaning strongly recommended before analysis or ML usage.")
    elif len(insights) >= 2:
        insights.append("💡 Overall: Dataset is moderately clean but requires preprocessing.")
    else:
        insights.append("💡 Overall: Dataset is relatively clean and ready for analysis.")

    return insights