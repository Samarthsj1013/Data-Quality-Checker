import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from checker import (load_data, check_nulls, check_duplicates,
                     check_data_types, detect_outliers, quality_score,
                     generate_suggestions, auto_clean, correlation_heatmap,
                     distribution_plots)

st.set_page_config(page_title="Data Quality Checker", page_icon="✅", layout="wide")

st.title("✅ Data Quality Checker")
st.markdown("Upload any CSV or Excel file and get a full quality report instantly.")

with st.sidebar:
    st.image("https://img.icons8.com/emoji/96/check-mark-button-emoji.png", width=60)
    st.title("Navigation")
    st.markdown("---")
    st.markdown("### 📌 Sections")
    st.markdown("- 📋 Dataset Summary")
    st.markdown("- 🏆 Quality Score")
    st.markdown("- 💡 Fix Suggestions")
    st.markdown("- 🔴 Missing Values")
    st.markdown("- 🟠 Outliers")
    st.markdown("- 🔗 Correlation Heatmap")
    st.markdown("- 📊 Distribution Plots")
    st.markdown("- 🔵 Scatter Plot")
    st.markdown("- 🧹 Auto Clean")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("Upload any CSV or Excel file to instantly analyze data quality.")
    st.markdown("---")
    st.caption("Built with Python + Streamlit")

uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file:
    df = load_data(uploaded_file)

    # Summary Card
    st.subheader("📋 Dataset Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    with s1:
        st.metric("📦 Total Rows", df.shape[0])
    with s2:
        st.metric("📊 Total Columns", df.shape[1])
    with s3:
        total_nulls = df.isnull().sum().sum()
        st.metric("🔴 Total Nulls", total_nulls)
    with s4:
        st.metric("🟡 Duplicates", df.duplicated().sum())
    with s5:
        numeric_count = df.select_dtypes(include=[np.number]).shape[1]
        st.metric("🔢 Numeric Cols", numeric_count)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(10))
    st.markdown(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")

    # Run checks
    null_df         = check_nulls(df)
    duplicate_count = check_duplicates(df)
    dtype_df        = check_data_types(df)
    outlier_df      = detect_outliers(df)
    score           = quality_score(df, null_df, duplicate_count, outlier_df)
    suggestions_df  = generate_suggestions(df, null_df, outlier_df, duplicate_count)

    # Quality Score
    st.subheader("🏆 Overall Quality Score")
    color = "green" if score >= 80 else "orange" if score >= 50 else "red"
    st.markdown(f"<h1 style='color:{color}'>{score} / 100</h1>",
                unsafe_allow_html=True)

    # Auto Fix Suggestions
    st.subheader("💡 Auto Fix Suggestions")
    st.dataframe(suggestions_df, use_container_width=True)

    st.divider()
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
        st.metric("Total Duplicates", duplicate_count)
        dup_percent = round((duplicate_count / len(df)) * 100, 2)
        st.metric("Duplicate %", f"{dup_percent}%")

        st.subheader("🔵 Data Types")
        st.dataframe(dtype_df)

    with col3:
        st.subheader("🟠 Outliers Detected")
        st.dataframe(outlier_df)
        if not outlier_df.empty:
            fig2 = px.bar(outlier_df, x="Column", y="Outlier Count",
                          title="Outliers per Column",
                          color="Outlier Count",
                          color_continuous_scale="Oranges")
            st.plotly_chart(fig2, use_container_width=True)

    # Correlation Heatmap
    st.divider()
    st.subheader("🔗 Correlation Heatmap")
    st.markdown("Shows how strongly numeric columns are related to each other. Values close to **1 or -1** mean strong relationship, close to **0** means no relationship.")

    corr = correlation_heatmap(df)
    if corr is not None:
        fig3 = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation Matrix"
        )
        fig3.update_layout(width=700, height=500)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**🔍 Strong Correlations Found:**")
        strong = []
        cols = corr.columns
        for i in range(len(cols)):
            for j in range(i+1, len(cols)):
                val = corr.iloc[i, j]
                if abs(val) >= 0.5:
                    direction = "positive 📈" if val > 0 else "negative 📉"
                    strong.append(f"**{cols[i]}** and **{cols[j]}** → {val} ({direction})")
        if strong:
            for s in strong:
                st.markdown(f"- {s}")
        else:
            st.markdown("No strong correlations found.")
    else:
        st.info("Need at least 2 numeric columns for correlation heatmap.")

    # Distribution Plots
    st.divider()
    st.subheader("📊 Column Distribution Plots")
    st.markdown("See how data is distributed across each numeric column.")

    numeric_cols = distribution_plots(df)
    if numeric_cols:
        selected_col = st.selectbox("Select a column to explore:", numeric_cols)
        fig4 = px.histogram(
            df, x=selected_col,
            title=f"Distribution of {selected_col}",
            color_discrete_sequence=["#636EFA"],
            marginal="box"
        )
        fig4.update_layout(bargap=0.1)
        st.plotly_chart(fig4, use_container_width=True)

        col_x, col_y, col_z, col_w = st.columns(4)
        with col_x:
            st.metric("Mean", round(df[selected_col].mean(), 2))
        with col_y:
            st.metric("Median", round(df[selected_col].median(), 2))
        with col_z:
            st.metric("Std Dev", round(df[selected_col].std(), 2))
        with col_w:
            st.metric("Skewness", round(df[selected_col].skew(), 2))
    else:
        st.info("No numeric columns found for distribution plots.")

    # Scatter Plot
    st.divider()
    st.subheader("🔵 Column vs Column Scatter Plot")
    st.markdown("Pick any 2 numeric columns to see their relationship.")

    if len(numeric_cols) >= 2:
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            x_col = st.selectbox("Select X axis:", numeric_cols, key="x_axis")
        with col_s2:
            y_col = st.selectbox("Select Y axis:", numeric_cols, index=1, key="y_axis")

        fig5 = px.scatter(df, x=x_col, y=y_col,
                          title=f"{x_col} vs {y_col}",
                          trendline="ols",
                          color_discrete_sequence=["#636EFA"])
        st.plotly_chart(fig5, use_container_width=True)
    else:
        st.info("Need at least 2 numeric columns for scatter plot.")

    st.divider()

    # Auto Clean Section
    st.subheader("🧹 Auto Clean Dataset")
    st.markdown("Automatically fix all issues - remove duplicates, fill nulls, drop bad columns.")

    if st.button("✨ Clean My Dataset"):
        cleaned_df, changes = auto_clean(df)
        st.success("Dataset cleaned successfully!")

        st.markdown("**Changes made:**")
        for change in changes:
            st.markdown(f"- {change}")

        new_null_df    = check_nulls(cleaned_df)
        new_dup        = check_duplicates(cleaned_df)
        new_outlier_df = detect_outliers(cleaned_df)
        new_score      = quality_score(cleaned_df, new_null_df, new_dup, new_outlier_df)

        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Old Quality Score", f"{score} / 100")
        with col_b:
            st.metric("New Quality Score", f"{new_score} / 100",
                      delta=f"+{round(new_score - score, 1)}")

        clean_csv = cleaned_df.to_csv(index=False)
        st.download_button(
            "📥 Download Cleaned Dataset",
            data=clean_csv,
            file_name="cleaned_dataset.csv",
            mime="text/csv"
        )

    st.divider()

    # Download Report
    st.subheader("📥 Download Report")
    report = null_df.copy()
    report["Duplicate Rows"] = duplicate_count
    report["Quality Score"] = score
    csv = report.to_csv(index=False)
    st.download_button("📊 Download CSV Report",
                       data=csv,
                       file_name="quality_report.csv",
                       mime="text/csv")

else:
    st.info("👆 Upload a CSV file to get started!")