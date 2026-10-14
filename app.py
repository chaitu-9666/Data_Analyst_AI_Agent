
import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_csv

st.set_page_config(
    page_title="Data Analyst AI Agent",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analyst AI Agent")
st.write(
    "Upload a CSV file to explore data, identify missing values, "
    "view statistics, and create interactive visualizations."
)

uploaded_file = st.sidebar.file_uploader(
    "Upload your CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a CSV file from the sidebar to get started.")

    sample = pd.DataFrame({
        "Department": ["IT", "HR", "Sales", "IT", "Sales"],
        "Employee": ["A", "B", "C", "D", "E"],
        "Salary": [45000, 35000, 40000, 52000, 42000],
        "Experience": [2, 1, 3, 5, 4]
    })

    st.subheader("Example dataset")
    st.dataframe(sample, use_container_width=True)

else:
    try:
        df = load_csv(uploaded_file)

        if df.empty or len(df.columns) == 0:
            st.error("The uploaded CSV contains no data.")
            st.stop()

        st.success("Dataset uploaded successfully!")

        overview, preview, missing, charts = st.tabs([
            "Overview",
            "Data Preview",
            "Missing Values",
            "Visualizations"
        ])

        with overview:
            st.subheader("Dataset Overview")

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows", df.shape[0])
            col2.metric("Columns", df.shape[1])
            col3.metric(
                "Missing Cells",
                int(df.isna().sum().sum())
            )

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
            st.dataframe(info, use_container_width=True)

        with preview:
            st.subheader("Explore Your Data")
            st.dataframe(df, use_container_width=True)

            st.download_button(
                "Download dataset",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="analysed_dataset.csv",
                mime="text/csv"
            )

        with missing:
            st.subheader("Missing Value Analysis")

            missing_df = pd.DataFrame({
                "Column": df.columns,
                "Missing Values": df.isna().sum().values,
                "Missing Percentage": (
                    df.isna().mean().values * 100
                ).round(2)
            })

            st.dataframe(missing_df, use_container_width=True)

            missing_only = missing_df[
                missing_df["Missing Values"] > 0
            ]

            if not missing_only.empty:
                fig = px.bar(
                    missing_only,
                    x="Column",
                    y="Missing Values",
                    title="Missing Values by Column"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values found!")

        with charts:
            st.subheader("Create a Visualization")

            numeric_columns = df.select_dtypes(
                include="number"
            ).columns.tolist()

            chart_type = st.selectbox(
                "Choose chart type",
                ["Histogram", "Bar Chart", "Scatter Plot", "Line Chart"]
            )

            if chart_type == "Histogram":
                column = st.selectbox(
                    "Select a numeric column",
                    numeric_columns
                ) if numeric_columns else None

                if column:
                    fig = px.histogram(
                        df, x=column, title=f"Distribution of {column}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("No numeric columns are available.")

            elif chart_type == "Bar Chart":
                x_col = st.selectbox("Category column", df.columns.tolist())
                y_col = st.selectbox(
                    "Numeric value column",
                    numeric_columns
                ) if numeric_columns else None

                if y_col:
                    grouped = df.groupby(
                        x_col, dropna=False
                    )[y_col].mean().reset_index()

                    fig = px.bar(
                        grouped,
                        x=x_col,
                        y=y_col,
                        title=f"Average {y_col} by {x_col}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("A numeric column is required.")

            else:
                if len(numeric_columns) >= 2:
                    x_col = st.selectbox(
                        "X-axis", numeric_columns, index=0
                    )
                    y_col = st.selectbox(
                        "Y-axis",
                        numeric_columns,
                        index=1 if len(numeric_columns) > 1 else 0
                    )

                    if chart_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_col, y=y_col)
                    else:
                        fig = px.line(df, x=x_col, y=y_col)

                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning(
                        "At least two numeric columns are required."
                    )

    except Exception as error:
        st.error(f"Unable to read or analyse this CSV: {error}")

