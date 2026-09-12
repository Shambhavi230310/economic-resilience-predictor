import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import plotly.graph_objects as go
from groq import Groq

st.set_page_config(
    page_title="Economic Resilience Predictor",
    page_icon="🏝️",
    layout="wide",
)

st.markdown("""
    <style>
    .stApp {
        background-color: #faf6f0;
        color: #1a1a1a;
    }
        [data-testid="stSidebar"] {
        background-color: #4a5d3a;
    }
    [data-testid="stSidebar"] * {
        color: #1a1a1a;
        font-weight: 600;
    }
    .stMetric {
        background-color: #b8946a;
        border: 1px solid #a07f56;
        border-radius: 10px;
        padding: 15px;
    }
    [data-testid="stMetricLabel"] {
        color: #ffffff !important;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    h1 {
        color: #1a1a1a;
        font-weight: 700;
    }
        p, li, span {
        color: #1a1a1a;
    }
    .stChatInput textarea {
        color: #ffffff;
        min-height: 40px !important;
        padding: 8px 12px !important;
    }
    div[data-testid="stChatInput"] {
        max-width: 700px !important;
        margin: 0 auto !important;
    }
        .stButton button {
        color: #ffffff !important;
    }
    .stButton button p {
        color: #ffffff !important;
    }
        [data-testid="stHeader"] {
        background-color: #4a5d3a;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🏝️ About This Project")
    st.write(
        "This tool predicts five-year average GDP growth for tourism-dependent "
        "island economies using a machine learning model trained on structural "
        "and economic indicators."
    )
    st.markdown("---")
    st.write(
        "**Capstone Project**  \n"
        "Building Economic Resilience in Tourism-Dependent Island Economies"
    )
    st.markdown("---")
    st.caption(
        "⚠️ This analysis uses a synthetic prototype dataset for "
        "methodological demonstration. Findings reflect the analytical "
        "framework rather than empirical evidence about real economies."
    )

# --- Load the trained model, scaler, and feature list ---
# @st.cache_resource means: load these files ONCE when the app starts,
# not every single time someone clicks something. This keeps the app fast.
@st.cache_resource
def load_model_files():
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    with open("features.json") as f:
        features = json.load(f)
    return model, scaler, features

model, scaler, features = load_model_files()
explainer = shap.TreeExplainer(model)

st.title("🏝️ Economic Resilience Predictor")
st.markdown('<p style="color: #7a6d5d; font-size: 1.1rem;">Predicting medium-term GDP growth for tourism-dependent island economies</p>', unsafe_allow_html=True)
st.write("Model and supporting files loaded successfully.")

# --- Load the dataset for country dropdown & auto-fill ---
@st.cache_data
def load_dataset():
    df = pd.read_excel("Island_Economy_Capstone_Dataset.xlsx", sheet_name="Panel_Data")
    return df

df = load_dataset()

# --- Country selector ---
core_islands = sorted(df[df["Is_Core_Island_Sample"] == 1]["Country"].unique())

show_all = st.checkbox("Show all 216 countries (exploratory)")

if show_all:
    country_list = sorted(df["Country"].unique())
else:
    country_list = core_islands

default_index = country_list.index("Maldives") if "Maldives" in country_list else 0

selected_country = st.selectbox(
    "Select a country",
    country_list,
    index=default_index,
)

st.write(f"You selected: **{selected_country}**")
if "last_country" not in st.session_state:
    st.session_state.last_country = selected_country

if st.session_state.last_country != selected_country:
    st.session_state.chat_history = []
    st.session_state.last_country = selected_country

# --- Load pre-cleaned country profiles (matches notebook's imputation exactly) ---
@st.cache_data
def load_country_profiles():
    with open("country_profiles.json") as f:
        return json.load(f)

country_profiles = load_country_profiles()

if selected_country in country_profiles:
    feature_values = country_profiles[selected_country]
else:
    # Fallback for countries outside the 15 core islands (only relevant if "Show all" is ticked)
    country_rows = df[df["Country"] == selected_country].sort_values("Year")

    if country_rows.empty:
        st.error(f"No data available for {selected_country}.")
        st.stop()

    country_row = country_rows.tail(1)
    feature_values = {}
    for feat in features:
        value = country_row[feat].values[0]
        if pd.isna(value):
            value = df[feat].mean()
            st.caption(f"⚠️ {feat} was missing for {selected_country} — using dataset average instead.")
        feature_values[feat] = value
        
st.divider()
st.subheader("Scenario Analysis")
st.write("Compare predicted growth under three diversification scenarios.")

def build_scenario(base_values, div_mult, tour_mult):
    """Adjust structural features by the given multipliers and predict."""
    s = base_values.copy()
    s["Diversification_Index"] *= div_mult
    s["Diversification_Index_lag1"] *= div_mult
    s["Tourism_Receipts_pct_GDP"] *= tour_mult
    s["Services_pct_GDP"] *= (1 - 0.05 * (div_mult - 1) / 0.50)
    s["Industry_pct_GDP"] *= (1 + 0.12 * (div_mult - 1) / 0.50)

    input_df = pd.DataFrame([s])[features]
    input_scaled = scaler.transform(input_df)
    return model.predict(input_scaled)[0]

scenario_specs = {
    "A. Current structure": (1.00, 1.00),
    "B. Moderate diversification": (1.25, 0.85),
    "C. Strong diversification": (1.50, 0.70),
}

scenario_results = {
    name: build_scenario(feature_values, div_mult, tour_mult)
    for name, (div_mult, tour_mult) in scenario_specs.items()
}

st.subheader("Scenario Comparison")

col1, col2, col3 = st.columns(3)
for col, (name, value) in zip([col1, col2, col3], scenario_results.items()):
    with col:
        st.metric(label=name, value=f"{value:.2f}%")

# --- Comparison bar chart ---
colors = ["#c96f4a", "#e0a83e", "#7d9d6f"]

fig = go.Figure(data=[
    go.Bar(
        x=list(scenario_results.keys()),
        y=list(scenario_results.values()),
        marker_color=colors,
        text=[f"{v:.2f}%" for v in scenario_results.values()],
        textposition="outside",
    )
])

fig.update_layout(
    title=dict(
        text=f"{selected_country}: Predicted 5-Year Average GDP Growth by Scenario",
        font=dict(color="#000000", size=18),
    ),
    yaxis_title="Predicted GDP Growth (%)",
    plot_bgcolor="#faf6f0",
    paper_bgcolor="#faf6f0",
    font=dict(color="#000000"),
    height=400,
    xaxis=dict(color="#000000"),
    yaxis=dict(color="#000000"),
)
fig.add_hline(y=0, line_color="#3d3229", line_width=1)

st.plotly_chart(fig, use_container_width=True, theme=None)

st.divider()

with st.expander("🔧 Explore manually — adjust key indicators"):
    st.write("Adjust any value below to see how it changes the prediction.")

    manual_diversification = st.slider(
        "Diversification Index",
        min_value=0.0, max_value=1.0,
        value=float(feature_values["Diversification_Index"]),
        step=0.01,
        help="Higher = more diversified economy",
    )
    manual_tourism = st.slider(
        "Tourism Receipts (% of GDP)",
        min_value=0.0, max_value=50.0,
        value=float(feature_values["Tourism_Receipts_pct_GDP"]),
        step=0.5,
    )
    manual_gdp_per_capita = st.slider(
        "GDP per Capita (USD)",
        min_value=0.0, max_value=100000.0,
        value=float(feature_values["GDP_per_Capita_USD"]),
        step=500.0,
    )
    manual_unemployment = st.slider(
        "Unemployment (%)",
        min_value=0.0, max_value=40.0,
        value=float(feature_values["Unemployment_pct"]),
        step=0.5,
    )
    manual_global_shock = st.selectbox(
        "Simulate a global shock year (e.g. 2008, 2020)?",
        options=[0, 1],
        index=int(feature_values["Global_Shock_Year"]),
        format_func=lambda x: "Yes" if x == 1 else "No",
    )

    st.markdown("---")

    # Take the country's full data, override these 5 values, then predict
    custom_values = feature_values.copy()
    custom_values["Diversification_Index"] = manual_diversification
    custom_values["Tourism_Receipts_pct_GDP"] = manual_tourism
    custom_values["GDP_per_Capita_USD"] = manual_gdp_per_capita
    custom_values["Unemployment_pct"] = manual_unemployment
    custom_values["Global_Shock_Year"] = manual_global_shock

    custom_input_df = pd.DataFrame([custom_values])[features]
    custom_input_scaled = scaler.transform(custom_input_df)
    custom_prediction = model.predict(custom_input_scaled)[0]

    st.metric(
        label="Predicted growth with your adjustments",
        value=f"{custom_prediction:.2f}%",
    )
    st.markdown("---")
    st.write("**Why this custom prediction?**")

    custom_explainer_values = explainer.shap_values(custom_input_scaled)

    custom_shap_df = pd.DataFrame({
        "Feature": features,
        "Impact": custom_explainer_values[0]
    }).sort_values("Impact", ascending=True)

    custom_shap_colors = ["#7a3b2e" if val < 0 else "#4a5d3a" for val in custom_shap_df["Impact"]]

    custom_shap_fig = go.Figure(go.Bar(
        x=custom_shap_df["Impact"],
        y=custom_shap_df["Feature"],
        orientation="h",
        marker_color=custom_shap_colors,
    ))

    custom_shap_fig.update_layout(
        plot_bgcolor="#faf6f0",
        paper_bgcolor="#faf6f0",
        font=dict(color="#000000"),
        height=450,
        margin=dict(l=180, r=40, t=20, b=40),
        xaxis=dict(color="#000000", title="Impact on predicted growth (pp)"),
        yaxis=dict(color="#000000"),
    )
    custom_shap_fig.add_vline(x=0, line_color="#3d3229", line_width=1)

    st.plotly_chart(custom_shap_fig, use_container_width=True, theme=None)

st.divider()
st.subheader("📊 Why does the model predict this?")
st.write(
   "Ranked by how much each factor influences the model's predictions, "
    "based on patterns learned across all 15 island economies."
)

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=True)

importance_fig = go.Figure(go.Bar(
    x=importance_df["Importance"],
    y=importance_df["Feature"],
    orientation="h",
    marker_color="#8b6f47",
))

importance_fig.update_layout(
    title=dict(
        text="Feature Importance in the Prediction Model",
        font=dict(color="#000000", size=18),
    ),
    xaxis_title="Importance",
    plot_bgcolor="#faf6f0",
    paper_bgcolor="#faf6f0",
    font=dict(color="#000000"),
    height=550,
    margin=dict(l=180, r=40, t=50, b=40),
    xaxis=dict(color="#000000"),
    yaxis=dict(color="#000000"),
)

st.plotly_chart(importance_fig, use_container_width=True, theme=None)

st.divider()
st.caption(
    "Built with Streamlit · Random Forest model trained on a synthetic "
    "island economies dataset · Capstone Project 2026"
)

st.divider()
st.subheader("💬 Ask About This Prediction")

groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.write("Try asking:")
btn_col1, btn_col2, btn_col3 = st.columns(3)
with btn_col1:
    if st.button("Why this prediction?"):
        st.session_state.pending_question = f"Why is the prediction for {selected_country} what it is?"
with btn_col2:
    if st.button("What matters most?"):
        st.session_state.pending_question = "What features matter most to this model?"
with btn_col3:
    if st.button("Explain diversification"):
        st.session_state.pending_question = "What does the Diversification Index mean, and how does it relate to growth?"

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

typed_question = st.text_input("Or ask your own question...", key="chat_text_input")

if "pending_question" in st.session_state:
    user_question = st.session_state.pending_question
    del st.session_state.pending_question
elif typed_question and typed_question != st.session_state.get("last_typed_question", ""):
    user_question = typed_question
    st.session_state.last_typed_question = typed_question
else:
    user_question = None

if user_question:
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    with st.chat_message("user"):
        st.write(user_question)

    system_context = f"""You are an assistant explaining a machine learning model's predictions for a capstone project on economic resilience in tourism-dependent island economies.

Current context:
- Selected country: {selected_country}
- Current structure prediction: {scenario_results['A. Current structure']:.2f}% (5-year avg GDP growth)
- Moderate diversification prediction: {scenario_results['B. Moderate diversification']:.2f}%
- Strong diversification prediction: {scenario_results['C. Strong diversification']:.2f}%
- Model: Random Forest, R² = 0.823 on held-out test data
- Key features (highest importance): Diversification_Index, Agriculture_pct_GDP, Services_pct_GDP
- Diversification rank: {selected_country} is one of 15 core island economies studied; Maldives specifically ranks 11th of 15 on diversification, below the peer median
- Econometric finding: Diversification is positively and significantly associated with next-year GDP growth across all model specifications tested (Pooled OLS, Fixed Effects, Random Effects)
- Dataset: synthetic prototype, for methodological demonstration only

Only state specific numbers or facts explicitly given in this context above. If asked about something not covered here (e.g. specific SHAP values, econometric model results, or peer country comparisons), say you don't have that specific detail available in this view, rather than guessing.


Answer only questions related to this model, its predictions, its features, or economic resilience/diversification concepts. Keep answers concise (2-4 sentences). If asked something unrelated, politely redirect to the topic of this app."""

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": system_context},
            *st.session_state.chat_history
        ],
    )
    answer = response.choices[0].message.content

    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.write(answer)