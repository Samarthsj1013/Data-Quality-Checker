import pandas as pd
import numpy as np

def load_data(file):
    if file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        return pd.read_excel(file)
    else:
        return pd.read_csv(file)

def check_nulls(df):
    null_counts = df.isnull().sum()
    null_percent = (null_counts / len(df) * 100).round(2)
    return pd.DataFrame({
        "Missing Count": null_counts,
        "Missing %": null_percent
    }).reset_index().rename(columns={"index": "Column"})

def check_duplicates(df):
    return df.duplicated().sum()

def check_data_types(df):
    return pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values.astype(str)
    })

def detect_outliers(df):
    results = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        outlier_count = ((df[col] < Q1 - 1.5 * IQR) |
                         (df[col] > Q3 + 1.5 * IQR)).sum()
        results.append({"Column": col, "Outlier Count": outlier_count})
    return pd.DataFrame(results)

def quality_score(df, null_df, duplicate_count, outlier_df):
    score = 100

    avg_null = null_df["Missing %"].mean()
    score -= min(avg_null, 40)

    dup_percent = (duplicate_count / len(df)) * 100
    score -= min(dup_percent, 30)

    if not outlier_df.empty:
        total_outliers = outlier_df["Outlier Count"].sum()
        outlier_penalty = min((total_outliers / len(df)) * 100, 30)
        score -= outlier_penalty

    return round(max(score, 0), 1)

def generate_suggestions(df, null_df, outlier_df, duplicate_count):
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
                "Suggestion": "Consider dropping this column - too much data missing"
            })
        elif df[col].dtype in ["float64", "int64"]:
            median_val = round(df[col].median(), 2)
            mean_val = round(df[col].mean(), 2)
            suggestions.append({
                "Column": col,
                "Issue": f"🟡 {missing_pct}% missing",
                "Suggestion": f"Fill nulls with median ({median_val}) - mean is ({mean_val})"
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
            "Issue": f"🟡 {duplicate_count} duplicate rows",
            "Suggestion": "Remove duplicate rows using drop_duplicates()"
        })

    for _, row in outlier_df.iterrows():
        col = row["Column"]
        count = row["Outlier Count"]
        if count > 0:
            suggestions.append({
                "Column": col,
                "Issue": f"🟠 {count} outliers detected",
                "Suggestion": "Cap outliers using IQR method or investigate extreme values"
            })

    if not suggestions:
        suggestions.append({
            "Column": "All columns",
            "Issue": "✅ No major issues",
            "Suggestion": "Your dataset looks clean!"
        })

    return pd.DataFrame(suggestions)

def auto_clean(df):
    cleaned = df.copy()
    changes = []

    before = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    removed = before - len(cleaned)
    if removed > 0:
        changes.append(f"✅ Removed {removed} duplicate rows")

    for col in cleaned.columns:
        null_count = cleaned[col].isnull().sum()
        if null_count == 0:
            continue

        null_pct = (null_count / len(cleaned)) * 100

        if null_pct > 50:
            cleaned = cleaned.drop(columns=[col])
            changes.append(f"✅ Dropped column '{col}' - {round(null_pct, 1)}% missing")
        elif cleaned[col].dtype in ["float64", "int64"]:
            median_val = cleaned[col].median()
            cleaned[col] = cleaned[col].fillna(median_val)
            changes.append(f"✅ Filled '{col}' nulls with median ({round(median_val, 2)})")
        else:
            if not cleaned[col].mode().empty:
                mode_val = cleaned[col].mode()[0]
                cleaned[col] = cleaned[col].fillna(mode_val)
                changes.append(f"✅ Filled '{col}' nulls with most frequent value ('{mode_val}')")

    if not changes:
        changes.append("✅ Dataset was already clean - no changes made!")

    return cleaned, changes

def correlation_heatmap(df):
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return None
    corr = numeric_df.corr().round(2)
    return corr

def distribution_plots(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols