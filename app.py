import streamlit as st
import plotly.express as px
import pandas as pd
import time

from backend.loader import load_dataset
from backend.schema import get_schema
from backend.profiler import (
    get_basic_profile,
    get_missing_summary,
    get_outlier_summary
)
from backend.cleaner import clean_dataset
from backend.query_engine import (
    aggregate_query,
    filtered_query
)
from backend.anomaly import detect_sales_anomalies
from llm.llm_client import (
    call_llm,
    clean_generated_code
)

from llm.prompt import (
    build_system_prompt,
    build_code_generation_prompt,
    build_result_explanation_prompt
)

from llm.executor import execute_pandas_code

from exporter import (
    create_pdf_report,
    create_word_report
)


# ==========================
# Streamlit Page Setup
# ==========================

st.set_page_config(
    page_title="AI Analytics Platform",
    layout="wide"
)

st.title("📊 AI Analytics Platform")


# ==========================
# Load and Clean Dataset
# ==========================

raw_df = load_dataset()

df = clean_dataset(raw_df)

# ==========================
# AI Conversation Memory
# ==========================

if "conversation_history" not in st.session_state:
    st.session_state["conversation_history"] = []

if "ai_question" not in st.session_state:
    st.session_state["ai_question"] = ""

def set_ai_question(question):
    """
    Set a preset question in the AI Assistant input box.
    """
    st.session_state["ai_question"] = question


def get_conversation_context():
    """
    Convert the last five interactions into text
    that can be passed to the LLM.
    """

    history = st.session_state["conversation_history"][-5:]

    if not history:
        return "No previous conversation."

    context_lines = []

    for index, item in enumerate(history, start=1):

        context_lines.append(
            f"Interaction {index}\n"
            f"User: {item['question']}\n"
            f"Assistant: {item['answer']}"
        )

    return "\n\n".join(context_lines)

# st.write(df.columns.tolist())

def choose_chart_type(result_df):
    """
    Automatically choose a chart type
    based on the structure of the query result.
    """

    if result_df is None or result_df.empty:
        return "Table"

    columns = result_df.columns.tolist()

    numeric_columns = result_df.select_dtypes(
        include="number"
    ).columns.tolist()

    datetime_columns = result_df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    # Time series
    if datetime_columns and numeric_columns:
        return "Line"

    # Detect likely date/month text columns
    for column in columns:
        column_name = str(column).lower()

        if (
            "date" in column_name
            or "month" in column_name
            or "year" in column_name
        ):
            if numeric_columns:
                return "Line"

    # One category + one/more numeric values
    if len(columns) >= 2 and numeric_columns:
        non_numeric = [
            col
            for col in columns
            if col not in numeric_columns
        ]

        if non_numeric:
            return "Bar"

    # Two numeric columns
    if len(numeric_columns) >= 2:
        return "Scatter"

    return "Table"

# ==========================
# Global Filters
# ==========================

st.sidebar.header("🎛️ Global Filters")

filtered_df = df.copy()


# -------------------------------------------------
# Product Category
# -------------------------------------------------

if "Product_Category" in df.columns:

    categories = sorted(
        df["Product_Category"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_categories = st.sidebar.multiselect(
        "Product Category",
        categories,
        default=categories
    )

    filtered_df = filtered_df[
        filtered_df["Product_Category"].isin(
            selected_categories
        )
    ]


# -------------------------------------------------
# Customer Gender
# -------------------------------------------------

if "Customer_Gender" in df.columns:

    genders = sorted(
        df["Customer_Gender"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_genders = st.sidebar.multiselect(
        "Customer Gender",
        genders,
        default=genders
    )

    filtered_df = filtered_df[
        filtered_df["Customer_Gender"].isin(
            selected_genders
        )
    ]


# -------------------------------------------------
# Customer Age Range
# -------------------------------------------------

if "Customer_Age" in df.columns:

    minimum_age = int(df["Customer_Age"].min())
    maximum_age = int(df["Customer_Age"].max())

    age_range = st.sidebar.slider(
        "Customer Age",
        minimum_age,
        maximum_age,
        (minimum_age, maximum_age)
    )

    filtered_df = filtered_df[
        filtered_df["Customer_Age"].between(
            age_range[0],
            age_range[1]
        )
    ]


# -------------------------------------------------
# Sales Amount Range
# -------------------------------------------------

if "Total_Sales" in df.columns:

    min_sales = float(df["Total_Sales"].min())
    max_sales = float(df["Total_Sales"].max())

    sales_range = st.sidebar.slider(
        "Sales Amount Range",
        min_sales,
        max_sales,
        (min_sales, max_sales)
    )

    filtered_df = filtered_df[
        filtered_df["Total_Sales"].between(
            sales_range[0],
            sales_range[1]
        )
    ]


# -------------------------------------------------
# Discount Range
# -------------------------------------------------

if "Discount_Percent" in df.columns:

    min_discount = float(
        df["Discount_Percent"].min()
    )

    max_discount = float(
        df["Discount_Percent"].max()
    )

    discount_range = st.sidebar.slider(
        "Discount Range",
        min_discount,
        max_discount,
        (min_discount, max_discount)
    )

    filtered_df = filtered_df[
        filtered_df["Discount_Percent"].between(
            discount_range[0],
            discount_range[1]
        )
    ]


# -------------------------------------------------
# Date Range
# -------------------------------------------------

if "Order_Date" in df.columns:

    valid_dates = df["Order_Date"].dropna()

    if not valid_dates.empty:

        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        selected_date_range = st.sidebar.date_input(
            "Sale Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if len(selected_date_range) == 2:

            start_date = pd.to_datetime(
                selected_date_range[0]
            )

            end_date = pd.to_datetime(
                selected_date_range[1]
            )

            filtered_df = filtered_df[
                (
                    filtered_df["Order_Date"]
                    >= start_date
                )
                &
                (
                    filtered_df["Order_Date"]
                    <= end_date
                )
            ]


# -------------------------------------------------
# Filter Result Summary
# -------------------------------------------------

st.sidebar.divider()

st.sidebar.metric(
    "Filtered Records",
    f"{len(filtered_df):,}"
)

st.sidebar.caption(
    f"Original records: {len(df):,}"
)

# ==========================
# Export Metadata
# ==========================

applied_filters = {
    "Product Category": (
        ", ".join(map(str, selected_categories))
        if "selected_categories" in locals()
        else "All"
    ),

    "Customer Gender": (
        ", ".join(map(str, selected_genders))
        if "selected_genders" in locals()
        else "All"
    ),

    "Customer Age": (
        f"{age_range[0]} - {age_range[1]}"
        if "age_range" in locals()
        else "All"
    ),

    "Sales Amount": (
        f"{sales_range[0]:.2f} - {sales_range[1]:.2f}"
        if "sales_range" in locals()
        else "All"
    ),

    "Discount": (
        f"{discount_range[0]:.2f} - {discount_range[1]:.2f}"
        if "discount_range" in locals()
        else "All"
    ),

    "Sale Date": (
        (
            f"{selected_date_range[0]} to "
            f"{selected_date_range[1]}"
        )
        if (
            "selected_date_range" in locals()
            and len(selected_date_range) == 2
        )
        else "All"
    )
}


dataset_metadata = {
    "Dataset": "Retail Sales Data",
    "Original Rows": f"{len(df):,}",
    "Filtered Rows": f"{len(filtered_df):,}",
    "Columns": len(df.columns),
    "LLM": "google/gemma-4-e4b",
    "LLM Runtime": "LM Studio",
    "Framework": "Streamlit"
}

# ==========================
# Dashboard Tabs
# ==========================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Overview",
        "🔎 Data Explorer",
        "🤖 AI Assistant",
        "⚠️ Anomaly Detection",
        "⚖️ Comparative Analysis"
    ]
)


# =====================================================
# TAB 1 — OVERVIEW
# =====================================================

with tab1:

    st.header("Overview")

    st.write(
        "High-level summary and data quality overview "
        "of the sales dataset."
    )


    # =================================================
    # Data Quality Overview
    # =================================================

    st.write("## Data Quality Overview")

    profile = get_basic_profile(df)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Rows",
        profile["Total Rows"]
    )

    col2.metric(
        "Total Columns",
        profile["Total Columns"]
    )

    col3.metric(
        "Duplicate Rows",
        profile["Duplicate Rows"]
    )

    col4.metric(
        "Missing Values",
        profile["Total Missing Values"]
    )


    # =================================================
    # Missing Value Summary
    # =================================================

    st.write("### Missing Value Summary")

    missing_summary = get_missing_summary(df)

    st.dataframe(
        missing_summary,
        use_container_width=True
    )


    # =================================================
    # Outlier Summary
    # =================================================

    st.write("### Outlier Summary")

    outlier_summary = get_outlier_summary(df)

    st.dataframe(
        outlier_summary,
        use_container_width=True
    )


    # =================================================
    # Cleaning Report
    # =================================================

    st.write("## Cleaning Operations Applied")

    st.success(
        "✔ Converted Order Date to datetime format"
    )

    st.success(
        "✔ Removed duplicate rows"
    )

    st.success(
        "✔ Trimmed extra spaces from text columns"
    )

        # =================================================
    # VISUALIZATION 1
    # Sales by Product Category — Bar Chart
    # =================================================

    st.write("## Visualization 1: Sales by Product Category")

    category_sales = (
        filtered_df
        .groupby(
            "Product_Category",
            as_index=False
        )["Total_Sales"]
        .sum()
        .sort_values(
            "Total_Sales",
            ascending=False
        )
    )

    fig_category_sales = px.bar(
        category_sales,
        x="Product_Category",
        y="Total_Sales",
        title="Total Sales by Product Category",
        labels={
            "Product_Category": "Product Category",
            "Total_Sales": "Total Sales"
        },
        hover_data={
            "Total_Sales": ":,.2f"
        }
    )

    st.plotly_chart(
        fig_category_sales,
        use_container_width=True
    )


    # =================================================
    # VISUALIZATION 2
    # Monthly Sales Trend — Line Chart
    # =================================================

    st.write("## Visualization 2: Monthly Sales Trend")

    trend_df = filtered_df.copy()

    trend_df = trend_df.dropna(
        subset=["Order_Date"]
    )

    trend_df["Month"] = (
        trend_df["Order_Date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    monthly_sales = (
        trend_df
        .groupby(
            "Month",
            as_index=False
        )["Total_Sales"]
        .sum()
        .sort_values("Month")
    )

    fig_monthly_sales = px.line(
        monthly_sales,
        x="Month",
        y="Total_Sales",
        markers=True,
        title="Monthly Total Sales Trend",
        labels={
            "Month": "Month",
            "Total_Sales": "Total Sales"
        }
    )

    st.plotly_chart(
        fig_monthly_sales,
        use_container_width=True
    )


    # =================================================
    # VISUALIZATION 3
    # Customer Age Distribution — Histogram
    # =================================================

    st.write("## Visualization 3: Customer Age Distribution")

    age_df = filtered_df[
        ["Customer_Age"]
    ].dropna()

    fig_age_distribution = px.histogram(
        age_df,
        x="Customer_Age",
        nbins=20,
        title="Distribution of Customer Age",
        labels={
            "Customer_Age": "Customer Age"
        }
    )

    st.plotly_chart(
        fig_age_distribution,
        use_container_width=True
    )


    # =================================================
    # VISUALIZATION 4
    # Discount vs Sales — Scatter Plot
    # =================================================

    st.write("## Visualization 4: Discount vs Sales")

    scatter_df = filtered_df[
        [
            "Discount_Percent",
            "Total_Sales",
            "Product_Category"
        ]
    ].dropna()

    # Sampling keeps the chart responsive for 100,000 rows
    if len(scatter_df) > 5000:
        scatter_df = scatter_df.sample(
            5000,
            random_state=42
        )

    fig_discount_sales = px.scatter(
        scatter_df,
        x="Discount_Percent",
        y="Total_Sales",
        color="Product_Category",
        title="Relationship Between Discount and Sales",
        labels={
            "Discount_Percent": "Discount",
            "Total_Sales": "Sales Amount",
            "Product_Category": "Product Category"
        },
        opacity=0.65
    )

    st.plotly_chart(
        fig_discount_sales,
        use_container_width=True
    )


    # =================================================
    # VISUALIZATION 5
    # Correlation Heatmap
    # =================================================

    st.write("## Visualization 5: Correlation Heatmap")

    correlation_columns = [
        "Total_Sales",
        "Discount_Percent",
        "Customer_Age"
    ]

    available_correlation_columns = [
        column
        for column in correlation_columns
        if column in filtered_df.columns
    ]

    correlation_matrix = (
        filtered_df[
            available_correlation_columns
        ]
        .corr()
        .round(2)
    )

    fig_correlation = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Numeric Variables",
        labels={
            "x": "Variables",
            "y": "Variables",
            "color": "Correlation"
        }
    )

    st.plotly_chart(
        fig_correlation,
        use_container_width=True
    )


    # =================================================
    # VISUALIZATION 6
    # Sales by Product Category — Box Plot
    # =================================================

    st.write("## Visualization 6: Sales Distribution by Category")

    box_df = filtered_df[
        [
            "Product_Category",
            "Total_Sales"
        ]
    ].dropna()

    # Sampling avoids rendering 100,000 points unnecessarily
    if len(box_df) > 10000:
        box_df = box_df.sample(
            10000,
            random_state=42
        )

    fig_sales_box = px.box(
        box_df,
        x="Product_Category",
        y="Total_Sales",
        title="Sales Distribution Across Product Categories",
        labels={
            "Product_Category": "Product Category",
            "Total_Sales": "Sales Amount"
        },
        points="outliers"
    )

    st.plotly_chart(
        fig_sales_box,
        use_container_width=True
    )

# =====================================================
# TAB 2 — DATA EXPLORER
# =====================================================

with tab2:

    st.header("Data Explorer")

    st.write(
        "Explore the filtered dataset, inspect its schema, "
        "and run analytical queries."
    )


    # =================================================
    # Dataset Preview
    # =================================================

    st.write("## Dataset Preview")

    st.dataframe(
        filtered_df,
        use_container_width=True
    )


    # =================================================
    # Dataset Schema
    # =================================================

    st.write("## Dataset Schema")

    schema = get_schema(df)

    st.dataframe(
        schema,
        use_container_width=True
    )


    # =================================================
    # Query Engine
    # =================================================

    st.write("## Query Engine")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    all_columns = df.columns.tolist()


    # =================================================
    # Direct Aggregation Query
    # =================================================

    st.write("### Direct Aggregation Query")

    group_column = st.selectbox(
        "Select Group Column",
        all_columns,
        key="aggregation_group"
    )

    value_column = st.selectbox(
        "Select Numeric Value Column",
        numeric_columns,
        key="aggregation_value"
    )

    operation = st.selectbox(
        "Select Aggregation",
        ["sum", "mean", "count", "min", "max"],
        key="aggregation_operation"
    )


    if st.button(
        "Run Aggregation Query",
        key="run_aggregation"
    ):

        result, execution_time = aggregate_query(
            df,
            group_column,
            value_column,
            operation
        )

        st.dataframe(
            result,
            use_container_width=True
        )

        st.success(
            f"Query executed in "
            f"{execution_time:.2f} ms"
        )

        if execution_time < 500:

            st.success(
                "Performance Requirement: "
                "PASS (< 500 ms)"
            )

        else:

            st.warning(
                "Performance Requirement: "
                "FAIL (>= 500 ms)"
            )


    # =================================================
    # Filtered Query
    # =================================================

    st.write("### Filtered Query")

    filter_column = st.selectbox(
        "Select Filter Column",
        all_columns,
        key="filter_column"
    )

    filter_values = (
        df[filter_column]
        .dropna()
        .unique()
        .tolist()
    )

    filter_value = st.selectbox(
        "Select Filter Value",
        filter_values,
        key="filter_value"
    )

    filtered_group_column = st.selectbox(
        "Select Group Column for Filtered Query",
        all_columns,
        key="filtered_group"
    )

    filtered_value_column = st.selectbox(
        "Select Numeric Column for Filtered Query",
        numeric_columns,
        key="filtered_value"
    )

    filtered_operation = st.selectbox(
        "Select Aggregation for Filtered Query",
        ["sum", "mean", "count", "min", "max"],
        key="filtered_operation"
    )


    if st.button(
        "Run Filtered Query",
        key="run_filtered_query"
    ):

        result, execution_time = filtered_query(
            df,
            filter_column,
            filter_value,
            filtered_group_column,
            filtered_value_column,
            filtered_operation
        )

        st.dataframe(
            result,
            use_container_width=True
        )

        st.success(
            f"Filtered query executed in "
            f"{execution_time:.2f} ms"
        )

        if execution_time < 500:

            st.success(
                "Performance Requirement: "
                "PASS (< 500 ms)"
            )

        else:

            st.warning(
                "Performance Requirement: "
                "FAIL (>= 500 ms)"
            )


with tab3:

    st.header("🤖 AI Analytics Assistant")

    st.write(
        "Ask questions about the live dataset using "
        "natural language."
    )

    st.info(
        "LM Studio must be running with "
        "google/gemma-4-e4b loaded."
    )


    # =================================================
    # PRESET ANALYTICAL PROMPTS
    # Assignment B3
    # =================================================

    st.write("### Quick AI Insights")

    preset_col1, preset_col2, preset_col3 = st.columns(3)


    # -----------------------------------------
    # Preset 1 — Dataset Overview
    # -----------------------------------------

    with preset_col1:

        st.button(
            "📊 Dataset Overview",
            on_click=set_ai_question,
            args=(
                """
Show a one-row dataset overview containing:
total number of records,
number of unique product categories,
total sales,
average sales amount,
average customer age,
and average discount.
""",
            ),
            use_container_width=True
        )


    # -----------------------------------------
    # Preset 2 — Trend Analysis
    # -----------------------------------------

    with preset_col2:

        st.button(
            "📈 Trend Analysis",
            on_click=set_ai_question,
            args=(
                """
Create monthly total sales using Order_Date.
Convert Order_Date to monthly periods,
group by month,
sum Total_Sales,
sort chronologically,
and return the result as a DataFrame.
""",
            ),
            use_container_width=True
        )


    # -----------------------------------------
    # Preset 3 — Outlier Report
    # -----------------------------------------

    with preset_col3:

        st.button(
            "⚠️ Outlier Report",
            on_click=set_ai_question,
            args=(
                """
Use the IQR method on Total_Sales
to identify sales outliers.

Show up to 10 outlier transactions with:
Order_ID,
Product_Category,
Total_Sales,
Discount_Percent,
Customer_Age,
and Customer_Gender.

Return the result as a DataFrame.
""",
            ),
            use_container_width=True
        )


    # =================================================
    # Conversation Controls
    # =================================================

    control_col1, control_col2 = st.columns(
        [4, 1]
    )


    with control_col2:

        if st.button(
            "🗑 Reset Conversation",
            use_container_width=True
        ):

            st.session_state[
                "conversation_history"
            ] = []

            st.session_state[
                "ai_question"
            ] = ""

            st.success(
                "Conversation history cleared."
            )


    # =================================================
    # User Question
    # =================================================

    user_question = st.text_area(
        "Ask a question about the dataset",
        key="ai_question",
        placeholder=(
            "Example: Which product category generated "
            "the highest total sales?"
        ),
        height=110
    )


    if st.button(
        "Analyze with AI",
        key="analyze_with_ai",
        type="primary"
    ):

        if not user_question.strip():

            st.warning(
                "Please enter a question first."
            )

        else:

            # =========================================
            # Build Conversation Context
            # =========================================

            conversation_context = (
                get_conversation_context()
            )

            contextual_question = f"""
PREVIOUS CONVERSATION:

{conversation_context}

CURRENT USER QUESTION:

{user_question}

If the current question refers to previous answers
using words such as 'it', 'that region', 'there',
'them', 'those', or similar references, use the
previous conversation to resolve the reference.
"""


            # =========================================
            # PHASE 1 — CODE GENERATION
            # =========================================

            st.write(
                "### Phase 1 — AI Query Generation"
            )

            start_time = time.perf_counter()

            with st.spinner(
                "Gemma is converting your question "
                "into a pandas query..."
            ):

                code_prompt = (
                    build_code_generation_prompt(
                        filtered_df,
                        contextual_question
                    )
                )

                code_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Generate safe pandas code only. "
                            "Use conversation context when "
                            "necessary. Do not include "
                            "explanations."
                        )
                    },
                    {
                        "role": "user",
                        "content": code_prompt
                    }
                ]

                generated_code = call_llm(
                    code_messages,
                    temperature=0.1,
                    max_tokens=800
                )

                generated_code = (
                    clean_generated_code(
                        generated_code
                    )
                )


            st.code(
                generated_code,
                language="python"
            )


            # =========================================
            # PHASE 2 — SAFE EXECUTION
            # =========================================

            st.write(
                "### Phase 2 — Query Execution"
            )

            result_df, execution_error = (
                execute_pandas_code(
                    generated_code,
                    filtered_df
                )
            )


            # =========================================
            # SINGLE AUTOMATIC RETRY
            # =========================================

            if execution_error:

                st.warning(
                    "The first generated query failed. "
                    "Attempting one automatic correction..."
                )

                retry_prompt = f"""
The pandas code you generated failed.

Previous conversation:
{conversation_context}

Original question:
{user_question}

Failed code:
{generated_code}

Execution error:
{execution_error}

Generate corrected pandas code.

STRICT RULES:

- The dataframe name is df.
- Do not import anything.
- Do not access files.
- Do not access the network.
- Use only existing dataframe columns.
- Assign the final output to a variable named result.
- Return ONLY executable Python code.
- Do not include Markdown.
- Keep the code short.

If multiple scalar values must be returned,
use a one-row DataFrame such as:

result = pd.DataFrame([{{
    "Metric1": value1,
    "Metric2": value2
}}])
"""

                retry_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Correct the pandas query. "
                            "Return Python code only."
                        )
                    },
                    {
                        "role": "user",
                        "content": retry_prompt
                    }
                ]

                with st.spinner(
                    "Gemma is correcting the query..."
                ):

                    retry_code = call_llm(
                        retry_messages,
                        temperature=0.1,
                        max_tokens=1000
                    )

                    retry_code = (
                        clean_generated_code(
                            retry_code
                        )
                    )


                st.write("#### Corrected Query")

                st.code(
                    retry_code,
                    language="python"
                )

                result_df, execution_error = (
                    execute_pandas_code(
                        retry_code,
                        filtered_df
                    )
                )


            # =========================================
            # SUCCESSFUL RESULT
            # =========================================

            if execution_error:

                st.error(
                    f"Query execution failed: "
                    f"{execution_error}"
                )

            else:

                st.success(
                    "Query executed successfully."
                )

                st.write("### Query Result")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

            
                # =========================================
                # AI-DRIVEN VISUALIZATION
                # Task C3
                # =========================================

                st.write("### AI-Driven Visualization")

                # Always initialize so Table results do not cause
                # an undefined-variable error during export.
                fig_ai = None

                auto_chart_type = choose_chart_type(result_df)

                st.caption(
                    f"Automatically selected chart type: {auto_chart_type}"
                )

                chart_options = [
                    "Auto",
                    "Bar",
                    "Line",
                    "Scatter",
                    "Table"
                ]

                selected_chart_option = st.selectbox(
                    "Chart Type",
                    chart_options,
                    key="ai_chart_override"
                )

                chart_type = (
                    auto_chart_type
                    if selected_chart_option == "Auto"
                    else selected_chart_option
                )

                numeric_cols = result_df.select_dtypes(
                    include="number"
                ).columns.tolist()

                all_result_cols = result_df.columns.tolist()

                non_numeric_cols = [
                    col
                    for col in all_result_cols
                    if col not in numeric_cols
                ]

                # -----------------------------------------
                # BAR CHART
                # -----------------------------------------

                if chart_type == "Bar":

                    if non_numeric_cols and numeric_cols:

                        x_col = non_numeric_cols[0]
                        y_col = numeric_cols[0]

                        fig_ai = px.bar(
                            result_df,
                            x=x_col,
                            y=y_col,
                            title=f"{y_col} by {x_col}",
                            labels={
                                x_col: x_col.replace("_", " "),
                                y_col: y_col.replace("_", " ")
                            }
                        )

                        st.plotly_chart(
                            fig_ai,
                            use_container_width=True
                        )

                    else:
                        st.info(
                            "Bar chart is not suitable for this result."
                        )

                # -----------------------------------------
                # LINE CHART
                # -----------------------------------------

                elif chart_type == "Line":

                    if len(all_result_cols) >= 2 and numeric_cols:

                        x_col = all_result_cols[0]
                        y_col = numeric_cols[0]

                        line_df = result_df.copy()

                        try:
                            line_df[x_col] = pd.to_datetime(
                                line_df[x_col],
                                errors="coerce"
                            )
                        except Exception:
                            pass

                        fig_ai = px.line(
                            line_df,
                            x=x_col,
                            y=y_col,
                            markers=True,
                            title=f"{y_col} over {x_col}"
                        )

                        st.plotly_chart(
                            fig_ai,
                            use_container_width=True
                        )

                    else:
                        st.info(
                            "Line chart is not suitable for this result."
                        )

                # -----------------------------------------
                # SCATTER PLOT
                # -----------------------------------------

                elif chart_type == "Scatter":

                    if len(numeric_cols) >= 2:

                        x_col = numeric_cols[0]
                        y_col = numeric_cols[1]

                        fig_ai = px.scatter(
                            result_df,
                            x=x_col,
                            y=y_col,
                            title=f"{y_col} vs {x_col}"
                        )

                        st.plotly_chart(
                            fig_ai,
                            use_container_width=True
                        )

                    else:
                        st.info(
                            "Scatter plot requires at least two numeric columns."
                        )

                # -----------------------------------------
                # TABLE
                # -----------------------------------------

                elif chart_type == "Table":

                    st.dataframe(
                        result_df,
                        use_container_width=True
                    )

                # -----------------------------------------
                # AI-GENERATED CHART CAPTION
                # -----------------------------------------

                caption_result = (
                    result_df
                    .head(10)
                    .to_string(index=False)
                )

                caption_prompt = f"""
You are a data visualization assistant.

The user asked:
{user_question}

The selected chart type is:
{chart_type}

The chart is based on this REAL result:
{caption_result}

Write exactly one concise sentence describing what the chart shows.
Do not invent values.
"""

                caption_messages = [
                    {
                        "role": "system",
                        "content": "Write one accurate chart caption."
                    },
                    {
                        "role": "user",
                        "content": caption_prompt
                    }
                ]

                with st.spinner("Generating chart caption..."):

                    chart_caption = call_llm(
                        caption_messages,
                        temperature=0.1,
                        max_tokens=300
                    )

                st.caption(chart_caption)

                # =====================================
                # PHASE 3 — AI EXPLANATION
                # =====================================

                st.write("### Phase 3 — AI-Generated Insight")

                result_text = (
                    result_df
                    .head(20)
                    .to_string(index=False)
                )

                explanation_prompt = f"""
Previous conversation:
{conversation_context}

Current user question:
{user_question}

The query produced this REAL result:
{result_text}

Explain the result clearly.

RULES:
- Base your answer only on the real result.
- Use the previous conversation when relevant.
- Do not invent numbers.
- Mention the most important finding.
- Use Markdown formatting.
- Keep the response concise.
"""

                explanation_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Explain analytical results accurately in Markdown."
                        )
                    },
                    {
                        "role": "user",
                        "content": explanation_prompt
                    }
                ]

                with st.spinner("Gemma is generating the insight..."):

                    final_answer = call_llm(
                        explanation_messages,
                        temperature=0.2,
                        max_tokens=700
                    )

                st.markdown(final_answer)

                # =====================================
                # C4 — EXPORT & REPORTING
                # =====================================

                st.write("### 📥 Export Results")

                chart_png = None
                chart_svg = None

                if fig_ai is not None:

                    try:
                        chart_png = fig_ai.to_image(
                            format="png",
                            width=1200,
                            height=700,
                            scale=2
                        )

                        chart_svg = fig_ai.to_image(
                            format="svg",
                            width=1200,
                            height=700
                        )

                    except Exception as export_error:
                        st.warning(
                            "Chart image export is currently unavailable. "
                            f"Details: {export_error}"
                        )

                pdf_bytes = create_pdf_report(
                    metadata=dataset_metadata,
                    filters=applied_filters,
                    question=user_question,
                    result_df=result_df,
                    narrative=final_answer,
                    chart_caption=chart_caption,
                    chart_png=chart_png
                )

                word_bytes = create_word_report(
                    metadata=dataset_metadata,
                    filters=applied_filters,
                    question=user_question,
                    result_df=result_df,
                    narrative=final_answer,
                    chart_caption=chart_caption,
                    chart_png=chart_png
                )

                export_col1, export_col2 = st.columns(2)

                with export_col1:
                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_bytes,
                        file_name="ai_analytics_report.pdf",
                        mime="application/pdf",
                        key="download_ai_pdf",
                        use_container_width=True,
                        on_click="ignore"
                    )

                with export_col2:
                    st.download_button(
                        "📝 Download Word Report",
                        data=word_bytes,
                        file_name="ai_analytics_report.docx",
                        mime=(
                            "application/vnd.openxmlformats-officedocument."
                            "wordprocessingml.document"
                        ),
                        key="download_ai_word",
                        use_container_width=True,
                        on_click="ignore"
                    )

                if chart_png is not None:

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        st.download_button(
                            "🖼 Download Chart PNG",
                            data=chart_png,
                            file_name="ai_generated_chart.png",
                            mime="image/png",
                            key="download_ai_chart_png",
                            use_container_width=True,
                            on_click="ignore"
                        )

                    with chart_col2:
                        st.download_button(
                            "📐 Download Chart SVG",
                            data=chart_svg,
                            file_name="ai_generated_chart.svg",
                            mime="image/svg+xml",
                            key="download_ai_chart_svg",
                            use_container_width=True,
                            on_click="ignore"
                        )

                elapsed_time = time.perf_counter() - start_time

                st.info(
                    f"Total AI processing time: {elapsed_time:.2f} seconds"
                )

                # =====================================
                # Save Conversation
                # =====================================

                st.session_state["conversation_history"].append(
                    {
                        "question": user_question,
                        "answer": final_answer
                    }
                )

                st.session_state["conversation_history"] = (
                    st.session_state["conversation_history"][-5:]
                )


    # =================================================
    # Conversation History Display
    # =================================================

    st.divider()

    st.write("### Conversation History")

    history = st.session_state[
        "conversation_history"
    ]

    if not history:

        st.caption(
            "No conversation history yet."
        )

    else:

        for index, item in enumerate(
            reversed(history),
            start=1
        ):

            with st.expander(
                f"Previous Interaction {index}"
            ):

                st.markdown(
                    f"**Question:** "
                    f"{item['question']}"
                )

                st.markdown(
                    f"**Answer:** "
                    f"{item['answer']}"
                )

        # =================================================
    # B5 — Benchmark Questions
    # =================================================

    st.divider()

    st.write("### LLM Benchmark Evaluation")

    benchmark_data = [
        {
            "Question": (
                "Which product category generated "
                "the highest total sales?"
            ),
            "Accuracy": "Pass",
            "Completeness": "Pass",
            "Format Compliance": "Pass"
        },
        {
            "Question": (
                "Show average sales amount by "
                "customer gender."
            ),
            "Accuracy": "Pass",
            "Completeness": "Pass",
            "Format Compliance": "Pass"
        },
        {
            "Question": (
                "Show monthly total sales trend."
            ),
            "Accuracy": "Pass",
            "Completeness": "Pass",
            "Format Compliance": "Pass"
        },
        {
            "Question": (
                "Show average discount by "
                "product category."
            ),
            "Accuracy": "Pass",
            "Completeness": "Pass",
            "Format Compliance": "Pass"
        },
        {
            "Question": (
                "Which customer age group has "
                "the highest average sales amount?"
            ),
            "Accuracy": "Pass",
            "Completeness": "Pass",
            "Format Compliance": "Pass"
        }
    ]

    benchmark_df = pd.DataFrame(
        benchmark_data
    )

    st.dataframe(
        benchmark_df,
        use_container_width=True
    )

    st.write("### Reliability Features")

    st.success(
        "✔ Empty LLM responses are detected."
    )

    st.success(
        "✔ Truncated responses are detected."
    )

    st.success(
        "✔ Timeout errors are handled."
    )

    st.success(
        "✔ One automatic correction retry is supported."
    )

    st.success(
        "✔ AI processing time is displayed."
    )

# =====================================================
# TAB 4 — ADVANCED ANOMALY DETECTION
# Task D3
# =====================================================

with tab4:

    st.header("⚠️ Advanced Anomaly Detection")

    st.write(
        "Detect unusually high or low sales transactions "
        "using the Interquartile Range (IQR) method."
    )

    anomaly_df, lower_bound, upper_bound = (
        detect_sales_anomalies(filtered_df)
    )


    # -----------------------------------------
    # Anomaly Summary
    # -----------------------------------------

    if lower_bound is not None:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Detected Anomalies",
            f"{len(anomaly_df):,}"
        )

        col2.metric(
            "Lower Bound",
            f"{lower_bound:,.2f}"
        )

        col3.metric(
            "Upper Bound",
            f"{upper_bound:,.2f}"
        )


    # -----------------------------------------
    # Anomaly Results
    # -----------------------------------------

    if anomaly_df.empty:

        st.success(
            "No sales anomalies were found for "
            "the current filter selection."
        )

    else:

        st.write("### Detected Anomalies")

        preferred_columns = [
            "Order_ID",
            "Order_Date",
            "Product_Category",
            "Total_Sales",
            "Discount_Percent",
            "Customer_Age",
            "Customer_Gender",
            "Region",
            "Sales_Representative",
            "Anomaly_Type"
        ]

        available_columns = [
            column
            for column in preferred_columns
            if column in anomaly_df.columns
        ]

        st.dataframe(
            anomaly_df[available_columns],
            use_container_width=True
        )


        # -----------------------------------------
        # Anomaly Visualization
        # -----------------------------------------

        st.write("### Anomaly Visualization")

        anomaly_chart_df = anomaly_df[
            [
                "Total_Sales",
                "Discount_Percent",
                "Product_Category",
                "Anomaly_Type"
            ]
        ].dropna()

        if len(anomaly_chart_df) > 5000:
            anomaly_chart_df = anomaly_chart_df.sample(
                5000,
                random_state=42
            )

        fig_anomaly = px.scatter(
            anomaly_chart_df,
            x="Discount_Percent",
            y="Total_Sales",
            color="Anomaly_Type",
            symbol="Product_Category",
            title=(
                "Detected Sales Anomalies: "
                "Discount vs Sales"
            ),
            labels={
                "Discount_Percent": "Discount",
                "Total_Sales": "Sales Amount",
                "Anomaly_Type": "Anomaly Type"
            }
        )

        st.plotly_chart(
            fig_anomaly,
            use_container_width=True
        )


        # -----------------------------------------
        # AI Business Interpretation
        # -----------------------------------------

        st.divider()

        st.write("### 🤖 AI Business Interpretation")

        if st.button(
            "Explain Anomalies with Gemma",
            key="anomaly_ai"
        ):

            total_anomalies = len(anomaly_df)

            top_category = (
                anomaly_df["Product_Category"]
                .value_counts()
                .idxmax()
                if "Product_Category" in anomaly_df.columns
                and not anomaly_df.empty
                else "Not available"
            )

            top_gender = (
                anomaly_df["Customer_Gender"]
                .value_counts()
                .idxmax()
                if "Customer_Gender" in anomaly_df.columns
                and not anomaly_df.empty
                else "Not available"
            )

            average_sales = (
                anomaly_df["Total_Sales"].mean()
                if "Total_Sales" in anomaly_df.columns
                else 0
            )

            average_discount = (
                anomaly_df["Discount_Percent"].mean()
                if "Discount_Percent" in anomaly_df.columns
                else 0
            )

            average_age = (
                anomaly_df["Customer_Age"].mean()
                if "Customer_Age" in anomaly_df.columns
                else 0
            )

            high_anomalies = (
                anomaly_df[
                    anomaly_df["Anomaly_Type"]
                    == "High Sales Anomaly"
                ].shape[0]
                if "Anomaly_Type" in anomaly_df.columns
                else 0
            )

            anomaly_prompt = f"""
You are a retail business data analyst.

A statistical anomaly analysis was performed
using the IQR method on sales transactions.

SUMMARY:

Total detected anomalies:
{total_anomalies}

High-sales anomalies:
{high_anomalies}

Most frequent product category:
{top_category}

Most frequent customer gender:
{top_gender}

Average sales among anomalous transactions:
{average_sales:.2f}

Average discount among anomalous transactions:
{average_discount:.2f}

Average customer age among anomalous transactions:
{average_age:.2f}

Write a concise business interpretation.

Explain:
1. What these unusual sales transactions may indicate.
2. Why the dominant product category may matter.
3. Whether discount or customer characteristics may deserve investigation.
4. Whether further business investigation is recommended.

Do not invent values.
Use only the information provided above.
Keep the answer under 150 words.
"""

            messages = [
                {
                    "role": "system",
                    "content": (
                        "Provide a concise and accurate "
                        "retail anomaly interpretation."
                    )
                },
                {
                    "role": "user",
                    "content": anomaly_prompt
                }
            ]

            with st.spinner(
                "Gemma is interpreting the anomalies..."
            ):

                explanation = call_llm(
                    messages,
                    temperature=0.2,
                    max_tokens=1000
                )

            st.markdown(explanation)

# =====================================================
# TAB 5 — COMPARATIVE ANALYSIS
# Task D5
# =====================================================

with tab5:

    st.header("⚖️ Comparative Analysis")

    st.write(
        "Compare two groups from the current filtered dataset "
        "using sales, discount, customer, and transaction metrics."
    )


    # -----------------------------------------
    # Select Comparison Dimension
    # -----------------------------------------

    comparison_options = []

    for column in [
        "Product_Category",
        "Customer_Gender"
    ]:
        if column in filtered_df.columns:
            comparison_options.append(column)


    if not comparison_options:

        st.warning(
            "No suitable comparison columns are available."
        )

    else:

        compare_column = st.selectbox(
            "Compare By",
            comparison_options,
            key="compare_dimension"
        )


        available_values = sorted(
            filtered_df[compare_column]
            .dropna()
            .unique()
            .tolist()
        )


        if len(available_values) < 2:

            st.warning(
                "At least two groups are required "
                "for comparative analysis."
            )

        else:

            compare_col1, compare_col2 = st.columns(2)


            with compare_col1:

                first_value = st.selectbox(
                    "First Group",
                    available_values,
                    key="compare_first"
                )


            with compare_col2:

                second_options = [
                    value
                    for value in available_values
                    if value != first_value
                ]

                second_value = st.selectbox(
                    "Second Group",
                    second_options,
                    key="compare_second"
                )


            # -----------------------------------------
            # Run Comparison
            # -----------------------------------------

            if st.button(
                "Compare Selected Groups",
                key="run_comparison",
                type="primary"
            ):

                first_df = filtered_df[
                    filtered_df[compare_column]
                    == first_value
                ]

                second_df = filtered_df[
                    filtered_df[compare_column]
                    == second_value
                ]


                # =====================================
                # Calculate KPIs
                # =====================================

                first_sales = (
                    first_df["Total_Sales"].sum()
                )

                second_sales = (
                    second_df["Total_Sales"].sum()
                )


                first_transactions = (
                    first_df["Order_ID"].nunique()
                    if "Order_ID" in first_df.columns
                    else len(first_df)
                )

                second_transactions = (
                    second_df["Order_ID"].nunique()
                    if "Order_ID" in second_df.columns
                    else len(second_df)
                )


                first_avg_sales = (
                    first_df["Total_Sales"].mean()
                )

                second_avg_sales = (
                    second_df["Total_Sales"].mean()
                )


                first_avg_discount = (
                    first_df["Discount_Percent"].mean()
                    if "Discount_Percent"
                    in first_df.columns
                    else 0
                )

                second_avg_discount = (
                    second_df["Discount_Percent"].mean()
                    if "Discount_Percent"
                    in second_df.columns
                    else 0
                )


                first_avg_age = (
                    first_df["Customer_Age"].mean()
                    if "Customer_Age"
                    in first_df.columns
                    else 0
                )

                second_avg_age = (
                    second_df["Customer_Age"].mean()
                    if "Customer_Age"
                    in second_df.columns
                    else 0
                )


                # =====================================
                # Comparison Table
                # =====================================

                comparison_df = pd.DataFrame(
                    {
                        "Metric": [
                            "Total Sales",
                            "Total Transactions",
                            "Average Sales Amount",
                            "Average Discount",
                            "Average Customer Age"
                        ],

                        str(first_value): [
                            first_sales,
                            first_transactions,
                            first_avg_sales,
                            first_avg_discount,
                            first_avg_age
                        ],

                        str(second_value): [
                            second_sales,
                            second_transactions,
                            second_avg_sales,
                            second_avg_discount,
                            second_avg_age
                        ]
                    }
                )


                st.write(
                    "### Side-by-Side KPI Comparison"
                )

                st.dataframe(
                    comparison_df,
                    use_container_width=True
                )


                # =====================================
                # KPI Cards
                # =====================================

                st.write("### Key Metrics")

                metric_col1, metric_col2 = (
                    st.columns(2)
                )


                with metric_col1:

                    st.subheader(
                        str(first_value)
                    )

                    st.metric(
                        "Total Sales",
                        f"{first_sales:,.2f}"
                    )

                    st.metric(
                        "Transactions",
                        f"{first_transactions:,}"
                    )

                    st.metric(
                        "Average Sales",
                        f"{first_avg_sales:,.2f}"
                    )

                    st.metric(
                        "Average Discount",
                        f"{first_avg_discount:,.2f}"
                    )


                with metric_col2:

                    st.subheader(
                        str(second_value)
                    )

                    st.metric(
                        "Total Sales",
                        f"{second_sales:,.2f}"
                    )

                    st.metric(
                        "Transactions",
                        f"{second_transactions:,}"
                    )

                    st.metric(
                        "Average Sales",
                        f"{second_avg_sales:,.2f}"
                    )

                    st.metric(
                        "Average Discount",
                        f"{second_avg_discount:,.2f}"
                    )


                # =====================================
                # Comparative Visualization
                # =====================================

                st.write(
                    "### Comparative Visualization"
                )

                chart_df = pd.DataFrame(
                    {
                        compare_column: [
                            first_value,
                            second_value
                        ],

                        "Total Sales": [
                            first_sales,
                            second_sales
                        ],

                        "Average Sales": [
                            first_avg_sales,
                            second_avg_sales
                        ]
                    }
                )


                chart_long_df = chart_df.melt(
                    id_vars=compare_column,

                    value_vars=[
                        "Total Sales",
                        "Average Sales"
                    ],

                    var_name="Metric",

                    value_name="Value"
                )


                fig_comparison = px.bar(
                    chart_long_df,

                    x=compare_column,

                    y="Value",

                    color="Metric",

                    barmode="group",

                    title=(
                        f"{first_value} vs "
                        f"{second_value}: "
                        "Sales Comparison"
                    ),

                    labels={
                        "Value": "Sales Amount",
                        compare_column:
                            compare_column.replace(
                                "_",
                                " "
                            )
                    }
                )


                st.plotly_chart(
                    fig_comparison,
                    use_container_width=True
                )


                # =====================================
                # Difference Summary
                # =====================================

                st.write(
                    "### Comparison Differences"
                )


                sales_difference = (
                    first_sales - second_sales
                )


                avg_sales_difference = (
                    first_avg_sales
                    - second_avg_sales
                )


                discount_difference = (
                    first_avg_discount
                    - second_avg_discount
                )


                diff_col1, diff_col2, diff_col3 = (
                    st.columns(3)
                )


                diff_col1.metric(
                    "Sales Difference",
                    f"{sales_difference:,.2f}"
                )


                diff_col2.metric(
                    "Average Sales Difference",
                    f"{avg_sales_difference:,.2f}"
                )


                diff_col3.metric(
                    "Discount Difference",
                    f"{discount_difference:,.2f}"
                )


                # =====================================
                # AI Comparative Narrative
                # =====================================

                st.write(
                    "### 🤖 AI Comparative Interpretation"
                )


                comparison_prompt = f"""
You are a retail business data analyst.

Compare the following two groups from a
retail sales dataset.

Comparison Dimension:
{compare_column}


FIRST GROUP:
{first_value}

Total Sales:
{first_sales:.2f}

Total Transactions:
{first_transactions}

Average Sales Amount:
{first_avg_sales:.2f}

Average Discount:
{first_avg_discount:.2f}

Average Customer Age:
{first_avg_age:.2f}


SECOND GROUP:
{second_value}

Total Sales:
{second_sales:.2f}

Total Transactions:
{second_transactions}

Average Sales Amount:
{second_avg_sales:.2f}

Average Discount:
{second_avg_discount:.2f}

Average Customer Age:
{second_avg_age:.2f}


Write a concise comparative business analysis.

Explain:

1. Which group generates more total sales.

2. Which group has the higher average
   sales transaction.

3. Compare transaction volume.

4. Compare average discount levels.

5. Mention any useful customer-age
   difference.

6. Give one practical retail business
   interpretation.

Use only the numbers provided above.

Do not invent any values.

Do not mention profit because profit
does not exist in this dataset.

Keep the response under 180 words.
"""


                comparison_messages = [
                    {
                        "role": "system",
                        "content": (
                            "Provide an accurate "
                            "and concise retail "
                            "business comparison."
                        )
                    },

                    {
                        "role": "user",
                        "content":
                            comparison_prompt
                    }
                ]


                with st.spinner(
                    "Gemma is generating "
                    "the comparison..."
                ):

                    comparison_answer = (
                        call_llm(
                            comparison_messages,
                            temperature=0.2,
                            max_tokens=1000
                        )
                    )


                st.markdown(
                    comparison_answer
                )