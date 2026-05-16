import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from checker import generate_ai_insights, generate_cleaning_code
from checker import (
    load_data, check_nulls, check_duplicates, check_data_types,
    detect_outliers, quality_score, generate_suggestions, auto_clean,
    correlation_heatmap, distribution_plots, infer_column_types,
    validate_emails, validate_dates, validate_phones, check_consistency
)

st.set_page_config(page_title="Data Quality Checker", page_icon="✅", layout="wide")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/check-mark-button-emoji.png", width=60)
    st.title("Navigation")
    st.markdown("---")
    st.markdown("### 📌 Sections")
    st.markdown("- 📊 Data Health Report")
    st.markdown("- 🏥 Column Health Cards")
    st.markdown("- 📋 Dataset Summary")
    st.markdown("- 🏆 Quality Score")
    st.markdown("- 🤖 AI Insights")
    st.markdown("- 💡 Fix Suggestions")
    st.markdown("- 🔴 Missing Values")
    st.markdown("- 🟡 Duplicates & Types")
    st.markdown("- 🟠 Outliers")
    st.markdown("- 📧 Email / Date / Phone")
    st.markdown("- ⚠️ Consistency Check")
    st.markdown("- 🔗 Correlation Heatmap")
    st.markdown("- 📊 Distribution Plots")
    st.markdown("- 🔵 Scatter Plot")
    st.markdown("- 🧹 Auto Clean")
    st.markdown("- 🤖 AI Code Generator")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Upload any CSV or Excel file to instantly analyze data quality.")
    st.markdown("---")
    st.caption("Built with Python + Streamlit")

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("✅ Data Quality Checker")
st.markdown("Upload any CSV or Excel file and get a **full quality report** instantly — including email, date, phone, outlier, and consistency checks.")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = load_data(uploaded_file)
    type_info = infer_column_types(df)

    # ── Run all checks ────────────────────────────────────────────────────────
    null_df         = check_nulls(df)
    duplicate_count = check_duplicates(df)
    dtype_df        = check_data_types(df, type_info)
    outlier_df      = detect_outliers(df, type_info)
    email_df        = validate_emails(df)
    date_df         = validate_dates(df, type_info)
    phone_df        = validate_phones(df)
    consistency_df  = check_consistency(df)
    score           = quality_score(
                        df, null_df, duplicate_count, outlier_df,
                        email_df, date_df, phone_df, consistency_df, type_info
                      )
    suggestions_df  = generate_suggestions(
                        df, null_df, outlier_df, duplicate_count,
                        email_df, date_df, phone_df, consistency_df, type_info
                      )
    insights        = generate_ai_insights(
                        df, null_df, duplicate_count, outlier_df,
                        email_df, date_df, phone_df, consistency_df, type_info
                      )

    # ── Story Dashboard ───────────────────────────────────────────────────────
    critical = []
    warnings = []
    passed   = []

    for _, row in null_df.iterrows():
        if row["Missing %"] > 50:
            critical.append(f"🔴 '{row['Column']}' — {row['Missing %']}% missing (critical)")
        elif row["Missing %"] > 0:
            warnings.append(f"🟡 '{row['Column']}' — {row['Missing %']}% missing")
        else:
            passed.append(row["Column"])

    if duplicate_count > 0:
        dup_pct = round((duplicate_count / len(df)) * 100, 2)
        if dup_pct > 10:
            critical.append(f"🔴 {duplicate_count} duplicate rows ({dup_pct}%)")
        else:
            warnings.append(f"🟡 {duplicate_count} duplicate rows ({dup_pct}%)")

    if not outlier_df.empty:
        for _, row in outlier_df.iterrows():
            if row["Outlier Count"] > 0:
                pct = round((row["Outlier Count"] / len(df)) * 100, 1)
                if pct > 10:
                    critical.append(f"🔴 '{row['Column']}' — {row['Outlier Count']} outliers ({pct}%)")
                else:
                    warnings.append(f"🟡 '{row['Column']}' — {row['Outlier Count']} outliers ({pct}%)")

    mismatches = [col for col in df.columns
                  if type_info[col]["actual"] in ["object", "str"]
                  and type_info[col]["inferred"] == "numeric"]
    if mismatches:
        warnings.append(f"🟡 {len(mismatches)} columns stored as text but appear numeric")

    total_issues = len(critical) + len(warnings)
    readiness    = round(score, 1)
    bar_filled   = int(readiness / 5)
    bar_empty    = 20 - bar_filled
    progress_bar = "█" * bar_filled + "░" * bar_empty

    if score >= 80:
        score_color = "#00c853"
        score_label = "🟢 Good"
    elif score >= 50:
        score_color = "#ffa000"
        score_label = "🟡 Needs Cleaning"
    else:
        score_color = "#d50000"
        score_label = "🔴 Poor Quality"

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
                border-radius: 16px; padding: 28px; margin-bottom: 24px;
                border: 1px solid #3a3a5c;">
        <h2 style="margin:0 0 4px 0; color: #ffffff; font-size: 1.1em;
                   text-transform: uppercase; letter-spacing: 2px;">
            📊 Data Health Report
        </h2>
        <p style="color:#888; margin:0 0 20px 0; font-size:0.85em;">
            Generated for: <b style="color:#aaa">{uploaded_file.name}</b>
            &nbsp;|&nbsp; {df.shape[0]} rows × {df.shape[1]} columns
        </p>

        <div style="display:flex; gap:40px; margin-bottom:24px; flex-wrap:wrap;">
            <div>
                <div style="font-size:3em; font-weight:900; color:{score_color};
                            line-height:1">{score}</div>
                <div style="font-size:0.85em; color:#888; margin-top:4px">
                    Quality Score &nbsp; {score_label}
                </div>
            </div>
            <div style="flex:1; min-width:200px;">
                <div style="color:#aaa; font-size:0.8em; margin-bottom:8px">
                    DATA READINESS
                </div>
                <div style="font-family:monospace; font-size:1.1em;
                            color:{score_color}; letter-spacing:2px">
                    {progress_bar}
                </div>
                <div style="color:#888; font-size:0.78em; margin-top:6px">
                    {readiness}% ready for analysis
                </div>
            </div>
        </div>

        <div style="display:flex; gap:16px; flex-wrap:wrap;">
            <div style="background:#ff000022; border:1px solid #ff444466;
                        border-radius:10px; padding:12px 20px; min-width:120px;">
                <div style="font-size:1.8em; font-weight:800; color:#ff4444">
                    {len(critical)}
                </div>
                <div style="color:#ff8888; font-size:0.8em">Critical Issues</div>
            </div>
            <div style="background:#ffa00022; border:1px solid #ffa00066;
                        border-radius:10px; padding:12px 20px; min-width:120px;">
                <div style="font-size:1.8em; font-weight:800; color:#ffa000">
                    {len(warnings)}
                </div>
                <div style="color:#ffcc88; font-size:0.8em">Warnings</div>
            </div>
            <div style="background:#00c85322; border:1px solid #00c85366;
                        border-radius:10px; padding:12px 20px; min-width:120px;">
                <div style="font-size:1.8em; font-weight:800; color:#00c853">
                    {len(passed)}
                </div>
                <div style="color:#88ffaa; font-size:0.8em">Columns Passed</div>
            </div>
            <div style="background:#2979ff22; border:1px solid #2979ff66;
                        border-radius:10px; padding:12px 20px; min-width:120px;">
                <div style="font-size:1.8em; font-weight:800; color:#2979ff">
                    {total_issues}
                </div>
                <div style="color:#88aaff; font-size:0.8em">Total Issues</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if critical or warnings:
        with st.expander("🔍 View All Issues", expanded=False):
            if critical:
                st.markdown("**🔴 Critical Issues:**")
                for c in critical:
                    st.markdown(f"&nbsp;&nbsp;{c}")
            if warnings:
                st.markdown("**🟡 Warnings:**")
                for w in warnings:
                    st.markdown(f"&nbsp;&nbsp;{w}")

    # ── Column Health Cards ───────────────────────────────────────────────────
    st.markdown("### 🏥 Column Health Cards")
    cols_per_row = 4
    col_names = list(df.columns)
    for i in range(0, len(col_names), cols_per_row):
        row_cols = st.columns(cols_per_row)
        for j, col_name in enumerate(col_names[i:i + cols_per_row]):
            with row_cols[j]:
                null_row    = null_df[null_df["Column"] == col_name]
                missing_pct = float(null_row["Missing %"].values[0]) if not null_row.empty else 0.0
                col_type    = type_info[col_name]["inferred"]

                outlier_count = 0
                if not outlier_df.empty and col_name in outlier_df["Column"].values:
                    outlier_count = int(
                        outlier_df[outlier_df["Column"] == col_name]["Outlier Count"].values[0]
                    )

                if missing_pct > 50 or (len(df) > 0 and outlier_count / len(df) * 100 > 15):
                    card_color   = "#ff000033"
                    border_color = "#ff4444"
                    status       = "🔴 Critical"
                elif missing_pct > 5 or outlier_count > 0:
                    card_color   = "#ffa00022"
                    border_color = "#ffa000"
                    status       = "🟡 Warning"
                else:
                    card_color   = "#00c85322"
                    border_color = "#00c853"
                    status       = "✅ Healthy"

                fill     = int((100 - missing_pct) / 10)
                fill_bar = "█" * fill + "░" * (10 - fill)

                st.markdown(f"""
                <div style="background:{card_color}; border:1px solid {border_color};
                            border-radius:10px; padding:12px; margin-bottom:8px;
                            min-height:120px;">
                    <div style="font-weight:700; font-size:0.85em;
                                color:#fff; margin-bottom:4px;
                                white-space:nowrap; overflow:hidden;
                                text-overflow:ellipsis;" title="{col_name}">
                        {col_name}
                    </div>
                    <div style="font-size:0.7em; color:#aaa; margin-bottom:6px">
                        {col_type} &nbsp;|&nbsp; {status}
                    </div>
                    <div style="font-family:monospace; font-size:0.75em;
                                color:{border_color}">{fill_bar}</div>
                    <div style="font-size:0.7em; color:#888; margin-top:4px">
                        {round(100 - missing_pct, 1)}% complete
                        {f"&nbsp;|&nbsp; {outlier_count} outliers" if outlier_count > 0 else ""}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

# ── Industry Detector ─────────────────────────────────────────────────────
    from checker import detect_industry, run_industry_checks

    industry, confidence = detect_industry(df)

    industry_meta = {
        "sales":     ("🛒", "Sales Dataset",     "#00bcd4"),
        "hr":        ("👥", "HR Dataset",         "#9c27b0"),
        "medical":   ("🏥", "Medical Dataset",    "#f44336"),
        "transport": ("🚢", "Transport Dataset",  "#ff9800"),
        "finance":   ("💰", "Finance Dataset",    "#4caf50"),
        "ecommerce": ("🛍️", "E-Commerce Dataset", "#e91e63"),
        "generic":   ("📊", "Generic Dataset",    "#607d8b"),
    }

    icon, label, ind_color = industry_meta.get(industry, ("📊", "Generic Dataset", "#607d8b"))
    industry_issues = run_industry_checks(df, type_info, industry)
    issue_count = len([i for i in industry_issues if i.startswith("⚠️")])

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #0d1117, #1a1f2e);
                border-left: 5px solid {ind_color};
                border-radius: 12px; padding: 20px 24px; margin-bottom: 20px;">
        <div style="display:flex; align-items:center; gap:12px; margin-bottom:12px;">
            <span style="font-size:2em">{icon}</span>
            <div>
                <div style="font-size:1.15em; font-weight:800; color:#fff">
                    {label} Detected
                </div>
                <div style="font-size:0.78em; color:#888;">
                    Confidence: {confidence} keyword{"s" if confidence != 1 else ""} matched
                    &nbsp;|&nbsp; Running domain-specific checks...
                </div>
            </div>
            <div style="margin-left:auto; background:{ind_color}22;
                        border:1px solid {ind_color}88;
                        border-radius:8px; padding:8px 16px; text-align:center;">
                <div style="font-size:1.4em; font-weight:800; color:{ind_color}">
                    {issue_count}
                </div>
                <div style="font-size:0.7em; color:#aaa">Domain Issues</div>
            </div>
        </div>
        <div style="border-top:1px solid #ffffff11; padding-top:12px;">
            {"".join(f'<div style="font-size:0.82em; color:#ccc; padding:3px 0">{issue}</div>' for issue in industry_issues)}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Summary Card ─────────────────────────────────────────────────────────
    st.subheader("📋 Dataset Summary")
    # ── Summary Card ─────────────────────────────────────────────────────────
    st.subheader("📋 Dataset Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("📦 Total Rows", df.shape[0])
    with s2:
        st.metric("📊 Total Columns", df.shape[1])
    with s3:
        total_nulls = int(df.isnull().sum().sum())
        st.metric("🔴 Total Nulls", total_nulls)
    with s4:
        st.metric("🟡 Duplicates", duplicate_count)
    with s5:
        numeric_count = sum(1 for col in df.columns if type_info[col]["inferred"] == "numeric")
        st.metric("🔢 Numeric Cols", numeric_count)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(10))
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # ── Quality Score ─────────────────────────────────────────────────────────
    st.subheader("🏆 Overall Quality Score")
    color = "green" if score >= 80 else "orange" if score >= 50 else "red"
    label = "🟢 Good" if score >= 80 else "🟡 Needs Cleaning" if score >= 50 else "🔴 Poor"
    st.markdown(f"<h1 style='color:{color}'>{score} / 100 &nbsp;&nbsp; <span style='font-size:0.6em'>{label}</span></h1>",
                unsafe_allow_html=True)

    # ── AI Insights ───────────────────────────────────────────────────────────
    st.subheader("🤖 AI Insights")
    for insight in insights:
        st.markdown(f"- {insight}")

    # ── Score Breakdown ───────────────────────────────────────────────────────
    with st.expander("📊 See Score Breakdown"):
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            avg_null     = null_df["Missing %"].mean()
            null_penalty = min(avg_null * 0.75, 30)
            st.metric("Null Penalty", f"-{round(null_penalty, 1)}")
        with b2:
            dup_penalty = min((duplicate_count / len(df)) * 200, 20)
            st.metric("Duplicate Penalty", f"-{round(dup_penalty, 1)}")
        with b3:
            out_penalty = min((outlier_df["Outlier Count"].sum() / len(df)) * 100, 15) if not outlier_df.empty else 0
            st.metric("Outlier Penalty", f"-{round(out_penalty, 1)}")
        with b4:
            mismatch = sum(1 for col in df.columns
                           if type_info[col]["actual"] in ["object", "str"]
                           and type_info[col]["inferred"] == "numeric")
            st.metric("Type Mismatch Penalty", f"-{min(mismatch * 2, 10)}")

    # ── Suggestions ───────────────────────────────────────────────────────────
    st.subheader("💡 Auto Fix Suggestions")
    st.dataframe(suggestions_df, use_container_width=True)

    st.divider()

    # ── Main 3-column analysis ────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🔴 Missing Values")
        st.dataframe(null_df)
        fig = px.bar(null_df, x="Column", y="Missing %",
                     title="Missing % per Column", color="Missing %",
                     color_continuous_scale="Reds")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🟡 Duplicate Rows")
        st.metric("Total Duplicates (normalized)", duplicate_count)
        dup_percent = round((duplicate_count / len(df)) * 100, 2)
        st.metric("Duplicate %", f"{dup_percent}%")

        st.subheader("🔵 Data Types")
        st.dataframe(dtype_df)

    with col3:
        st.subheader("🟠 Outliers Detected")
        if not outlier_df.empty and outlier_df["Outlier Count"].sum() > 0:
            display_outlier = outlier_df[["Column", "Outlier Count", "IQR Lower", "IQR Upper"]]
            st.dataframe(display_outlier)
            fig2 = px.bar(outlier_df, x="Column", y="Outlier Count",
                          title="Outliers per Column",
                          color="Outlier Count",
                          color_continuous_scale="Oranges")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No outliers detected in numeric columns.")

    # ── Validation Section ────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔍 Data Validation")

    v1, v2, v3, v4 = st.columns(4)

    with v1:
        st.markdown("**📧 Email Validation**")
        if not email_df.empty:
            st.dataframe(email_df, use_container_width=True)
        else:
            st.success("No email columns found / all valid.")

    with v2:
        st.markdown("**📅 Date Validation**")
        if not date_df.empty:
            st.dataframe(date_df, use_container_width=True)
        else:
            st.success("No date columns found / all valid.")

    with v3:
        st.markdown("**📱 Phone Validation**")
        if not phone_df.empty:
            st.dataframe(phone_df, use_container_width=True)
        else:
            st.success("No phone columns found / all valid.")

    with v4:
        st.markdown("**⚠️ Consistency Check**")
        if not consistency_df.empty:
            st.dataframe(consistency_df, use_container_width=True)
        else:
            st.success("No consistency issues found.")

    # ── Correlation Heatmap ───────────────────────────────────────────────────
    st.divider()
    st.subheader("🔗 Correlation Heatmap")
    st.markdown("Shows relationships between numeric columns. Values close to **1 or -1** = strong relationship.")

    corr = correlation_heatmap(df, type_info)
    if corr is not None:
        fig3 = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                         zmin=-1, zmax=1, title="Correlation Matrix")
        fig3.update_layout(width=700, height=500)
        st.plotly_chart(fig3, use_container_width=True)

        strong = []
        cols   = corr.columns
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) >= 0.5:
                    direction = "positive 📈" if val > 0 else "negative 📉"
                    strong.append(f"**{cols[i]}** and **{cols[j]}** → {val} ({direction})")
        if strong:
            st.markdown("**🔍 Strong Correlations Found:**")
            for s in strong:
                st.markdown(f"- {s}")
        else:
            st.markdown("No strong correlations found.")
    else:
        st.info("Need at least 2 numeric columns for correlation heatmap.")

    # ── Distribution Plots ────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Column Distribution Plots")
    st.markdown("See how data is distributed across each numeric column.")

    numeric_cols = distribution_plots(df, type_info)
    if numeric_cols:
        selected_col = st.selectbox("Select a column to explore:", numeric_cols)
        plot_series  = type_info[selected_col]["coerced_numeric"]
        fig4 = px.histogram(x=plot_series, nbins=30,
                            title=f"Distribution of {selected_col}",
                            color_discrete_sequence=["#636EFA"],
                            marginal="box",
                            labels={"x": selected_col})
        fig4.update_layout(bargap=0.1)
        st.plotly_chart(fig4, use_container_width=True)

        col_x, col_y, col_z, col_w = st.columns(4)
        with col_x:
            st.metric("Mean", round(float(plot_series.mean()), 2))
        with col_y:
            st.metric("Median", round(float(plot_series.median()), 2))
        with col_z:
            st.metric("Std Dev", round(float(plot_series.std()), 2))
        with col_w:
            st.metric("Skewness", round(float(plot_series.skew()), 2))
    else:
        st.info("No numeric columns detected for distribution plots.")

    # ── Scatter Plot ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🔵 Column vs Column Scatter Plot")
    st.markdown("Pick any 2 numeric columns to see their relationship.")

    if len(numeric_cols) >= 2:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            x_col = st.selectbox("Select X axis:", numeric_cols, key="x_axis")
        with col_s2:
            y_col = st.selectbox("Select Y axis:", numeric_cols, index=1, key="y_axis")

        scatter_df = pd.DataFrame({
            x_col: type_info[x_col]["coerced_numeric"],
            y_col: type_info[y_col]["coerced_numeric"]
        }).dropna()

        fig5 = px.scatter(scatter_df, x=x_col, y=y_col,
                          title=f"{x_col} vs {y_col}",
                          trendline="ols",
                          color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for scatter plot.")

    # ── Auto Clean ────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🧹 Auto Clean Dataset")
    st.markdown("Automatically fix all detected issues — remove duplicates, fill nulls, drop bad columns.")

    if st.button("✨ Clean My Dataset"):
        cleaned_df, changes = auto_clean(df, type_info)
        st.success("Dataset cleaned successfully!")

        st.markdown("**Changes made:**")
        for change in changes:
            st.markdown(f"- {change}")

        new_type_info   = infer_column_types(cleaned_df)
        new_null_df     = check_nulls(cleaned_df)
        new_dup         = check_duplicates(cleaned_df)
        new_outlier_df  = detect_outliers(cleaned_df, new_type_info)
        new_email_df    = validate_emails(cleaned_df)
        new_date_df     = validate_dates(cleaned_df, new_type_info)
        new_phone_df    = validate_phones(cleaned_df)
        new_consistency = check_consistency(cleaned_df)
        new_score       = quality_score(cleaned_df, new_null_df, new_dup, new_outlier_df,
                                        new_email_df, new_date_df, new_phone_df,
                                        new_consistency, new_type_info)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Old Quality Score", f"{score} / 100")
        with col_b:
            st.metric("New Quality Score", f"{new_score} / 100",
                      delta=f"+{round(new_score - score, 1)}")

        clean_csv = cleaned_df.to_csv(index=False)
        st.download_button("📥 Download Cleaned Dataset",
                           data=clean_csv,
                           file_name="cleaned_dataset.csv",
                           mime="text/csv")

    st.divider()

    # ── AI Code Generator ─────────────────────────────────────────────────────
    st.subheader("🤖 AI Cleaning Code Generator")
    st.markdown("Click below and AI will write the **exact Python code** to clean your specific dataset — ready to copy and run.")

    if st.button("⚡ Generate Cleaning Code"):
        code = generate_cleaning_code(df, null_df, duplicate_count, outlier_df, type_info)
        st.code(code, language="python")
        st.download_button(
            "📥 Download Cleaning Script",
            data=code,
            file_name="clean_dataset.py",
            mime="text/plain"
        )
        st.success("✅ Copy the code above or download it as a .py file!")

    st.divider()

    # ── Download Report ───────────────────────────────────────────────────────
    st.subheader("📥 Download Report")
    report = null_df.copy()
    report["Duplicate Rows"] = duplicate_count
    report["Quality Score"]  = score
    csv = report.to_csv(index=False)
    st.download_button("📊 Download CSV Report",
                       data=csv,
                       file_name="quality_report.csv",
                       mime="text/csv")

else:
    st.info("👆 Upload a CSV file to get started!")