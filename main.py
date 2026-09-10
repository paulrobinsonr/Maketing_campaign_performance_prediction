import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Marketing Campaign Revenue Prediction",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# APP TITLE
# ============================================================

st.title("📊 Marketing Campaign Revenue Prediction")

st.markdown(
    "Enter campaign information to predict the expected revenue."
)

st.divider()


# ============================================================
# LOAD DATASET FILES
# ============================================================

@st.cache_data
def get_csv_files():

    data_folder = Path("data")

    if not data_folder.exists():
        return []

    csv_files = list(data_folder.glob("*.csv"))

    return csv_files


# ============================================================
# LOAD SELECTED DATASET
# ============================================================

@st.cache_data
def load_data(file_path):

    df = pd.read_csv(file_path)

    # Clean column names
    df.columns = df.columns.str.strip()

    return df


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    possible_models = [
        "lightgbm_revenue_model.pkl",
        "lgbm_revenue_model.pkl",
        "extra_trees_revenue_model.pkl"
    ]

    for model_file in possible_models:

        if Path(model_file).exists():

            model = joblib.load(model_file)

            return model, model_file

    return None, None


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

@st.cache_resource
def load_preprocessor():

    possible_preprocessors = [
        "revenue_preprocessor.pkl",
        "preprocessor.pkl"
    ]

    for preprocessor_file in possible_preprocessors:

        if Path(preprocessor_file).exists():

            return joblib.load(preprocessor_file)

    return None


# ============================================================
# GET CSV FILES
# ============================================================

csv_files = get_csv_files()


if len(csv_files) == 0:

    st.error(
        "❌ No CSV dataset found inside the data folder."
    )

    st.stop()


# ============================================================
# DATASET SELECTION
# ============================================================

csv_names = [
    file.name
    for file in csv_files
]


if len(csv_names) > 1:

    selected_csv_name = st.sidebar.selectbox(
        "📂 Select Dataset",
        csv_names
    )

else:

    selected_csv_name = csv_names[0]


# Find selected file

selected_file = None

for file in csv_files:

    if file.name == selected_csv_name:

        selected_file = file

        break


# ============================================================
# LOAD DATA
# ============================================================

df = load_data(
    str(selected_file)
)


# ============================================================
# LOAD MODEL
# ============================================================

model, model_file = load_model()


if model is None:

    st.error(
        "❌ Model file not found."
    )

    st.info(
        "Please make sure lightgbm_revenue_model.pkl exists."
    )

    st.stop()


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

preprocessor = load_preprocessor()


if preprocessor is None:

    st.error(
        "❌ Preprocessor file not found."
    )

    st.info(
        "Please make sure revenue_preprocessor.pkl exists."
    )

    st.stop()


# ============================================================
# DETERMINE MODEL NAME
# ============================================================

if "lightgbm" in model_file.lower():

    model_name = "LightGBM Regressor"

elif "extra" in model_file.lower():

    model_name = "Extra Trees Regressor"

else:

    model_name = "Revenue Prediction Model"


# ============================================================
# SIDEBAR DATASET INFORMATION
# ============================================================

with st.sidebar:

    st.header("📊 Dataset Information")

    st.write(
        f"**Dataset:** {selected_file.name}"
    )

    st.write(
        f"**Rows:** {df.shape[0]:,}"
    )

    st.write(
        f"**Columns:** {df.shape[1]}"
    )

    st.write(
        f"**Model:** {model_name}"
    )


# ============================================================
# SAFE UNIQUE VALUE FUNCTION
# ============================================================

def get_unique_values(column, default):

    if column not in df.columns:

        return default

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    values = values[
        values != ""
    ]

    unique_values = sorted(
        values.unique().tolist()
    )

    if len(unique_values) == 0:

        return default

    return unique_values


# ============================================================
# NUMERIC DEFAULT FUNCTION
# ============================================================

def get_numeric_default(column, default_value):

    if column not in df.columns:

        return default_value

    values = pd.to_numeric(
        df[column],
        errors="coerce"
    )

    median_value = values.median()

    if pd.isna(median_value):

        return default_value

    return float(median_value)


# ============================================================
# GET DROPDOWN OPTIONS
# ============================================================

campaign_types = get_unique_values(
    "Campaign_Type",
    ["Social Media"]
)


target_audiences = get_unique_values(
    "Target_Audience",
    ["College Students"]
)


channels = get_unique_values(
    "Channel_Used",
    ["WhatsApp, YouTube"]
)


languages = get_unique_values(
    "Language",
    ["English"]
)


customer_segments = get_unique_values(
    "Customer_Segment",
    ["College Students"]
)


brands = get_unique_values(
    "Brand",
    ["Nykaa"]
)


# ============================================================
# DISPLAY AVAILABLE BRANDS
# ============================================================

with st.sidebar:

    st.divider()

    st.subheader("🏷️ Brands Found")

    st.write(
        f"Total Brands: {len(brands)}"
    )

    for available_brand in brands:

        st.caption(
            f"• {available_brand}"
        )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.info(
    f"🤖 Active Model: **{model_name}**"
)


# ============================================================
# CAMPAIGN INFORMATION
# ============================================================

st.subheader(
    "📋 Campaign Information"
)


col1, col2, col3 = st.columns(3)


with col1:

    campaign_type = st.selectbox(
        "Campaign Type",
        campaign_types
    )


with col2:

    target_audience = st.selectbox(
        "Target Audience",
        target_audiences
    )


with col3:

    duration = st.number_input(
        "Duration (Days)",
        min_value=1,
        value=int(
            get_numeric_default(
                "Duration",
                30
            )
        )
    )


# ------------------------------------------------------------


col1, col2 = st.columns(2)


with col1:

    channel_used = st.selectbox(
        "Channel Used",
        channels
    )


with col2:

    campaign_date = st.date_input(
        "Campaign Date"
    )


# ============================================================
# PERFORMANCE METRICS
# ============================================================

st.divider()

st.subheader(
    "📈 Campaign Performance Metrics"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    impressions = st.number_input(
        "Impressions",
        min_value=0,
        value=int(
            get_numeric_default(
                "Impressions",
                5000
            )
        )
    )


with col2:

    clicks = st.number_input(
        "Clicks",
        min_value=0,
        value=int(
            get_numeric_default(
                "Clicks",
                1000
            )
        )
    )


with col3:

    leads = st.number_input(
        "Leads",
        min_value=0,
        value=int(
            get_numeric_default(
                "Leads",
                500
            )
        )
    )


with col4:

    conversions = st.number_input(
        "Conversions",
        min_value=0,
        value=int(
            get_numeric_default(
                "Conversions",
                100
            )
        )
    )


# ============================================================
# FINANCIAL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "💰 Financial Information"
)


col1, col2, col3 = st.columns(3)


with col1:

    acquisition_cost = st.number_input(
        "Acquisition Cost",
        min_value=0.0,
        value=float(
            get_numeric_default(
                "Acquisition_Cost",
                1000.0
            )
        )
    )


with col2:

    roi = st.number_input(
        "ROI",
        value=float(
            get_numeric_default(
                "ROI",
                1.0
            )
        )
    )


with col3:

    engagement_score = st.number_input(
        "Engagement Score",
        min_value=0.0,
        value=float(
            get_numeric_default(
                "Engagement_Score",
                10.0
            )
        )
    )


# ============================================================
# ADDITIONAL INFORMATION
# ============================================================

st.divider()

st.subheader(
    "📑 Additional Campaign Information"
)


col1, col2, col3 = st.columns(3)


with col1:

    language = st.selectbox(
        "Language",
        languages
    )


with col2:

    customer_segment = st.selectbox(
        "Customer Segment",
        customer_segments
    )


with col3:

    brand = st.selectbox(
        "Brand",
        brands
    )


# ============================================================
# DATE FEATURE ENGINEERING
# ============================================================

campaign_timestamp = pd.Timestamp(
    campaign_date
)


year = campaign_timestamp.year

month = campaign_timestamp.month

day = campaign_timestamp.day

day_of_week = campaign_timestamp.dayofweek


# ============================================================
# DATE MISSING FLAG
# ============================================================

date_missing = 0


# ============================================================
# CALCULATED FEATURES
# ============================================================

conversion_rate = (
    conversions / impressions
    if impressions > 0
    else 0
)


lead_conversion_rate = (
    conversions / leads
    if leads > 0
    else 0
)


click_through_rate = (
    clicks / impressions
    if impressions > 0
    else 0
)


lead_rate = (
    leads / clicks
    if clicks > 0
    else 0
)


cost_per_click = (
    acquisition_cost / clicks
    if clicks > 0
    else 0
)


cost_per_conversion = (
    acquisition_cost / conversions
    if conversions > 0
    else 0
)


# ============================================================
# EXACT 25 TRAINING FEATURES
# ============================================================

expected_features = [

    "Campaign_Type",
    "Target_Audience",
    "Duration",
    "Channel_Used",
    "Impressions",
    "Clicks",
    "Leads",
    "Conversions",
    "Acquisition_Cost",
    "ROI",
    "Language",
    "Engagement_Score",
    "Customer_Segment",
    "Brand",
    "Year",
    "Month",
    "Day",
    "Day_of_Week",
    "Date_Missing",
    "Conversion_Rate",
    "Lead_Conversion_Rate",
    "Click_Through_Rate",
    "Lead_Rate",
    "Cost_Per_Click",
    "Cost_Per_Conversion"

]


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()


if st.button(
    "🔮 Predict Revenue",
    use_container_width=True
):

    try:

        # ====================================================
        # CREATE INPUT DATA
        # ====================================================

        input_data = pd.DataFrame({

            "Campaign_Type": [
                campaign_type
            ],

            "Target_Audience": [
                target_audience
            ],

            "Duration": [
                duration
            ],

            "Channel_Used": [
                channel_used
            ],

            "Impressions": [
                impressions
            ],

            "Clicks": [
                clicks
            ],

            "Leads": [
                leads
            ],

            "Conversions": [
                conversions
            ],

            "Acquisition_Cost": [
                acquisition_cost
            ],

            "ROI": [
                roi
            ],

            "Language": [
                language
            ],

            "Engagement_Score": [
                engagement_score
            ],

            "Customer_Segment": [
                customer_segment
            ],

            "Brand": [
                brand
            ],

            "Year": [
                year
            ],

            "Month": [
                month
            ],

            "Day": [
                day
            ],

            "Day_of_Week": [
                day_of_week
            ],

            "Date_Missing": [
                date_missing
            ],

            "Conversion_Rate": [
                conversion_rate
            ],

            "Lead_Conversion_Rate": [
                lead_conversion_rate
            ],

            "Click_Through_Rate": [
                click_through_rate
            ],

            "Lead_Rate": [
                lead_rate
            ],

            "Cost_Per_Click": [
                cost_per_click
            ],

            "Cost_Per_Conversion": [
                cost_per_conversion
            ]

        })


        # ====================================================
        # ENSURE EXACT TRAINING COLUMN ORDER
        # ====================================================

        input_data = input_data[
            expected_features
        ]


        # ====================================================
        # PREPROCESS INPUT
        # ====================================================

        processed_data = preprocessor.transform(
            input_data
        )


        # ====================================================
        # PREDICT REVENUE
        # ====================================================

        prediction = model.predict(
            processed_data
        )[0]


        # ====================================================
        # SUCCESS
        # ====================================================

        st.success(
            "Prediction completed successfully!"
        )


        # ====================================================
        # RESULT
        # ====================================================

        result_col1, result_col2 = st.columns(2)


        with result_col1:

            st.metric(
                "💰 Predicted Revenue",
                f"₹ {prediction:,.2f}"
            )


        with result_col2:

            st.metric(
                "🤖 Model",
                model_name
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Campaign Input Summary"
        )


        st.dataframe(
            input_data,
            use_container_width=True
        )


        # ====================================================
        # CALCULATED METRICS
        # ====================================================

        st.subheader(
            "📊 Calculated Campaign Metrics"
        )


        metric_col1, metric_col2, metric_col3 = st.columns(3)


        with metric_col1:

            st.metric(
                "Click Through Rate",
                f"{click_through_rate:.4f}"
            )


        with metric_col2:

            st.metric(
                "Conversion Rate",
                f"{conversion_rate:.4f}"
            )


        with metric_col3:

            st.metric(
                "Cost Per Conversion",
                f"₹ {cost_per_conversion:,.2f}"
            )


    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Marketing Campaign Performance Prediction | Machine Learning Project"
)