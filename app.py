import streamlit as st
import pandas as pd
import plotly.express as px

from src.data_loader import load_csv
from src.profiling import (
    profile_dataset,
    get_column_profile,
    detect_outliers_iqr,
    calculate_data_quality_score,
    generate_data_quality_insights
)

# Stage 3: Natural Language Understanding
from src.nlu.query_understanding import understand_query

# Stage 5: Safe natural-language analysis execution
from src.analysis_pipeline import run_analysis


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Data Analyst AI Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analyst AI Agent")

st.write(
    "Upload a CSV file to explore data, identify missing values, "
    "view statistics, detect data quality issues, and create "
    "interactive visualizations."
)


# ==========================================================
# 2. CSV UPLOAD
# ==========================================================

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is None:

    st.info(
        "Upload a CSV file from the sidebar to get started."
    )

    sample = pd.DataFrame({
        "Department": ["IT", "HR", "Sales", "IT", "Sales"],
        "Employee": ["A", "B", "C", "D", "E"],
        "Salary": [45000, 35000, 40000, 52000, 42000],
        "Experience": [2, 1, 3, 5, 4]
    })

    st.subheader("Example dataset")

    st.dataframe(
        sample,
        use_container_width=True
    )

else:

    try:

        df = load_csv(uploaded_file)

        if df.empty or len(df.columns) == 0:

            st.error(
                "The uploaded CSV contains no data."
            )

            st.stop()

        st.success(
            "Dataset uploaded successfully!"
        )

        # ==================================================
        # 3. GENERATE DATA PROFILES
        # ==================================================

        profile = profile_dataset(df)

        column_profile = get_column_profile(df)

        outlier_profile = detect_outliers_iqr(df)

        quality_report = calculate_data_quality_score(df)

        quality_insights = generate_data_quality_insights(df)

        # ==================================================
        # 4. APPLICATION TABS
        # ==================================================

        overview, preview, missing, quality, charts, assistant = st.tabs([
            "Overview",
            "Data Preview",
            "Missing Values",
            "Data Quality",
            "Visualizations",
            "AI Assistant"
        ])

        # ==================================================
        # 5. OVERVIEW TAB
        # ==================================================

        with overview:

            st.subheader("Dataset Overview")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Rows",
                f"{df.shape[0]:,}"
            )

            col2.metric(
                "Columns",
                df.shape[1]
            )

            col3.metric(
                "Missing Cells",
                f"{profile['missing_cells']:,}"
            )

            st.divider()

            st.subheader("Statistical Summary")

            st.dataframe(
                df.describe(include="all").T,
                use_container_width=True
            )

            st.subheader("Column Information")

            info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values,
                "Unique Values": df.nunique().values
            })

            st.dataframe(
                info,
                use_container_width=True
            )

        # ==================================================
        # 6. DATA PREVIEW TAB
        # ==================================================

        with preview:

            st.subheader("Explore Your Data")

            st.dataframe(
                df,
                use_container_width=True
            )

            st.download_button(
                "Download dataset",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="analysed_dataset.csv",
                mime="text/csv"
            )

        # ==================================================
        # 7. MISSING VALUES TAB
        # ==================================================

        with missing:

            st.subheader("Missing Value Analysis")

            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing Values": df.isna().sum().values,
                "Missing Percentage": (
                    df.isna().mean().values * 100
                ).round(2)
            })

            st.dataframe(
                missing_df,
                use_container_width=True
            )

            missing_only = missing_df[
                missing_df["Missing Values"] > 0
            ].sort_values(
                by="Missing Percentage",
                ascending=False
            )

            if not missing_only.empty:

                fig = px.bar(
                    missing_only,
                    x="Column",
                    y="Missing Percentage",
                    hover_data=["Missing Values"],
                    title="Missing Values by Column (%)"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

            else:

                st.success(
                    "No missing values found!"
                )

        # ==================================================
        # 8. DATA QUALITY TAB
        # ==================================================

        with quality:

            st.subheader("Automated Data Profiling")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Total Rows",
                f"{profile['total_rows']:,}"
            )

            c2.metric(
                "Total Columns",
                profile["total_columns"]
            )

            c3.metric(
                "Duplicate Rows",
                profile["duplicate_rows"]
            )

            c4.metric(
                "Data Completeness",
                f"{100 - profile['missing_percentage']:.2f}%"
            )

            st.divider()

            st.subheader("Overall Data Quality Score")

            score = quality_report["overall_score"]
            grade = quality_report["overall_grade"]

            c1, c2, c3 = st.columns(3)

            c1.metric("Quality Score", f"{score}/100")
            c2.metric("Quality Grade", grade)

            c3.metric(
                "Total Quality Issues",
                quality_insights["total_insights"]
            )

            st.progress(score / 100)

            st.info(quality_insights["summary"])

            st.divider()

            st.subheader("Column-wise Quality Scores")

            st.dataframe(
                quality_report["column_scores"],
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader("Automated Data Quality Insights")

            if quality_insights["insights"]:

                for insight in quality_insights["insights"]:

                    with st.expander(
                        f"{insight['severity']} | "
                        f"{insight['category']} | "
                        f"{insight['column'] or 'Dataset'}"
                    ):

                        st.write(insight["message"])

            else:

                st.success("No data quality issues detected!")

            st.subheader("Recommended Actions")

            if quality_insights["recommendations"]:

                for recommendation in quality_insights["recommendations"]:

                    st.write(f"• {recommendation}")

            else:

                st.success("No corrective actions required.")

            st.subheader("Dataset Strengths")

            for strength in quality_insights["strengths"]:

                st.success(strength)

            st.divider()

            st.subheader("Column Classification")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Numeric Columns",
                len(profile["numeric_columns"])
            )

            col2.metric(
                "Categorical Columns",
                len(profile["categorical_columns"])
            )

            col3.metric(
                "Datetime Columns",
                len(profile["datetime_columns"])
            )

            col4.metric(
                "Time Columns",
                len(profile["time_columns"])
            )

            with st.expander("View classified columns"):

                st.write(
                    "Numeric:",
                    profile["numeric_columns"]
                )

                st.write(
                    "Categorical:",
                    profile["categorical_columns"]
                )

                st.write(
                    "Datetime:",
                    profile["datetime_columns"]
                )

                st.write(
                    "Time:",
                    profile["time_columns"]
                )

            st.divider()

            st.subheader("Advanced Column Analysis")

            st.caption(
                "Detailed statistics for every dataset column."
            )

            st.dataframe(
                column_profile,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader("Outlier Detection (IQR Method)")

            if not outlier_profile.empty:

                st.dataframe(
                    outlier_profile,
                    use_container_width=True,
                    hide_index=True
                )

                total_outliers = int(
                    outlier_profile["outlier_count"].sum()
                )

                st.info(
                    f"Potential numerical outliers detected: "
                    f"{total_outliers:,}"
                )

                fig = px.bar(
                    outlier_profile.sort_values(
                        "outlier_count",
                        ascending=False
                    ),
                    x="column",
                    y="outlier_count",
                    title="Potential Outliers by Column"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.caption(
                    "Outliers are statistical observations, not "
                    "automatically incorrect records."
                )

            else:

                st.success(
                    "No suitable numerical columns available "
                    "for outlier analysis."
                )

            st.divider()

            st.subheader("Data Quality Observations")

            if profile["missing_cells"] > 0:

                st.warning(
                    f"{profile['missing_cells']:,} missing cells "
                    f"found ({profile['missing_percentage']}%)."
                )

            else:

                st.success("No missing values detected.")

            if profile["duplicate_rows"] > 0:

                st.warning(
                    f"{profile['duplicate_rows']:,} duplicate "
                    "rows found. Review before removing."
                )

            else:

                st.success("No duplicate rows detected.")

            if not outlier_profile.empty:

                if total_outliers > 0:

                    st.warning(
                        "Potential outliers require further investigation."
                    )

                else:

                    st.success("No potential outliers detected.")

            if profile["datetime_columns"]:

                st.success(
                    "Datetime columns identified: "
                    + ", ".join(profile["datetime_columns"])
                )

            if profile["time_columns"]:

                st.success(
                    "Time columns identified: "
                    + ", ".join(profile["time_columns"])
                )

        # ==================================================
        # 9. VISUALIZATIONS TAB
        # ==================================================

        with charts:

            st.subheader("Create a Visualization")

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            chart_type = st.selectbox(
                "Choose chart type",
                [
                    "Histogram",
                    "Bar Chart",
                    "Scatter Plot",
                    "Line Chart"
                ]
            )

            if chart_type == "Histogram":

                column = st.selectbox(
                    "Select a numeric column",
                    numeric_columns
                ) if numeric_columns else None

                if column:

                    fig = px.histogram(
                        df,
                        x=column,
                        title=f"Distribution of {column}"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "No numeric columns are available."
                    )

            elif chart_type == "Bar Chart":

                x_col = st.selectbox(
                    "Category column",
                    df.columns.tolist()
                )

                y_col = st.selectbox(
                    "Numeric value column",
                    numeric_columns
                ) if numeric_columns else None

                if y_col:

                    grouped = df.groupby(
                        x_col,
                        dropna=False
                    )[y_col].mean().reset_index()

                    fig = px.bar(
                        grouped,
                        x=x_col,
                        y=y_col,
                        title=f"Average {y_col} by {x_col}"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "A numeric column is required."
                    )

            else:

                if len(numeric_columns) >= 2:

                    x_col = st.selectbox(
                        "X-axis",
                        numeric_columns,
                        index=0
                    )

                    y_col = st.selectbox(
                        "Y-axis",
                        numeric_columns,
                        index=1
                    )

                    if chart_type == "Scatter Plot":

                        fig = px.scatter(
                            df,
                            x=x_col,
                            y=y_col,
                            title=f"{y_col} vs {x_col}"
                        )

                    else:

                        fig = px.line(
                            df,
                            x=x_col,
                            y=y_col,
                            title=f"{y_col} over {x_col}"
                        )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:

                    st.warning(
                        "At least two numeric columns are required."
                    )

        # ==================================================
        # 10. STAGE 3 + STAGE 5: AI DATA ANALYST ASSISTANT
        # ==================================================

        with assistant:

            st.subheader("AI Data Analyst Assistant")

            st.write(
                "Ask questions about your uploaded dataset "
                "using natural language."
            )

            st.caption(
                "The assistant interprets your question, selects "
                "an approved analysis tool, and executes it safely."
            )

            # Initialize session state
            if "nlu_result" not in st.session_state:
                st.session_state["nlu_result"] = None

            if "nlu_question" not in st.session_state:
                st.session_state["nlu_question"] = ""

            if "analysis_result" not in st.session_state:
                st.session_state["analysis_result"] = None

            # Accept a natural-language question
            user_query = st.chat_input(
                "Ask something about your dataset...",
                key="nlu_chat_input"
            )

            # Process a new question
            if user_query:

                try:

                    # Stage 3: Understand the question
                    nlu_result = understand_query(
                        user_query,
                        df
                    )

                    st.session_state["nlu_result"] = nlu_result
                    st.session_state["nlu_question"] = user_query
                    st.session_state["analysis_result"] = None

                    # Stage 5: Execute the analysis
                    if not nlu_result.get(
                        "needs_clarification",
                        False
                    ):

                        analysis_result = run_analysis(
                            user_query,
                            df
                        )

                        st.session_state["analysis_result"] = (
                            analysis_result
                        )

                except Exception as error:

                    st.error(
                        f"Analysis processing error: {error}"
                    )

                    st.session_state["nlu_result"] = None
                    st.session_state["analysis_result"] = None

            # Retrieve stored results
            result = st.session_state.get("nlu_result")

            analysis_result = st.session_state.get(
                "analysis_result"
            )

            # Display NLU results
            if result:

                st.markdown("#### Your question")

                st.write(
                    st.session_state.get("nlu_question", "")
                )

                metric_col, intent_col = st.columns(2)

                metric_col.metric(
                    "Confidence Score",
                    f"{result.get('confidence', 0):.2f}"
                )

                intent_col.metric(
                    "Detected Intent",
                    result.get("intent", "UNKNOWN")
                )

                if result.get("needs_clarification", False):

                    st.warning(
                        result.get(
                            "clarification_question",
                            "I could not confidently understand "
                            "your question. Please rephrase it."
                        )
                    )

                else:

                    st.success(
                        "Your question was successfully interpreted!"
                    )

                st.markdown("#### Extracted Entities")

                st.json(
                    result.get("entities", {})
                )

                with st.expander("View Complete NLU Output"):

                    st.json(result)

                # ==========================================
                # STAGE 5: DISPLAY EXECUTION RESULTS
                # ==========================================

                if analysis_result:

                    st.divider()

                    st.markdown("### Analysis Execution")

                    execution_status = analysis_result.get(
                        "status"
                    )

                    if execution_status == "success":

                        st.success(
                            "Analysis executed successfully!"
                        )

                        execution = (
                            analysis_result.get("execution") or {}
                        )

                        calculated_result = execution.get("result")

                        # Display DataFrame results
                        if isinstance(
                            calculated_result,
                            pd.DataFrame
                        ):

                            st.dataframe(
                                calculated_result,
                                use_container_width=True
                            )

                        # Display numerical results
                        elif isinstance(
                            calculated_result,
                            (int, float)
                        ) and not isinstance(
                            calculated_result,
                            bool
                        ):

                            st.metric(
                                "Calculated Result",
                                f"{calculated_result:,.2f}"
                            )

                        # Display dictionary results
                        elif isinstance(calculated_result, dict):

                            st.json(calculated_result)

                        # Display other results
                        else:

                            st.write(calculated_result)

                        st.caption(
                            "Executed tool: "
                            f"{execution.get('tool_name', 'Unknown')}"
                        )

                    elif execution_status == "needs_parameters":

                        st.warning(
                            "This question needs additional information "
                            "before it can be executed. Try naming the "
                            "column you want to analyse."
                        )

                    elif execution_status == "unsupported":

                        st.warning(
                            "This type of analysis is not supported yet. "
                            "Try asking for a dataset overview, missing "
                            "values, a numeric summary, a correlation, "
                            "top records, or a column aggregation."
                        )

                    else:

                        st.error(
                            analysis_result.get(
                                "message",
                                "Analysis execution failed."
                            )
                        )

            else:

                st.info(
                    "Enter a question above to see its detected "
                    "intent and extracted entities."
                )

    except Exception as error:

        st.error(
            f"Unable to read or analyse this CSV: {error}"
        )