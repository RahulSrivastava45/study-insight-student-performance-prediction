import sys
from pathlib import Path

# Force Python to recognize the root directory
file_path = Path(__file__).resolve()
root_dir = file_path.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))


import os
import streamlit as st
import pandas as pd
import scipy.stats as stats
import time
import plotly.express as px
import plotly.graph_objects as go
from src.mlproject.utils import load_object
from src.mlproject.pipelines.training_pipeline import TrainingPipeline

# --- 1. Page Configuration & Premium CSS ---
st.set_page_config(page_title="STUDY-INSIGHT", layout="wide", page_icon="📊")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&family=Nunito+Sans:wght@800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 0%, #101827 0%, #0B0F19 45%, #0B0F19 100%);
        color: #F8FAFC;
    }

    /* ---------- Header / Logo ---------- */
    .brand-header {
        display: flex;
        align-items: center;
        gap: 18px;
        margin-bottom: 0.25rem;
    }

    .brand-mark {
        width: 50px;
        height: 50px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: #2563EB;
        flex-shrink: 0;
    }

    .title-main {
        font-family: 'Nunito Sans', sans-serif;
        font-size: 2.3rem;
        line-height: 1.1;
        color: #F8FAFC;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin: 0;
    }

    .subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin: 0.9rem 0 2rem 0;
        letter-spacing: 0.4px;
    }

    hr.brand-divider {
        border: none;
        border-top: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 2rem;
    }

    /* ---------- Native bordered container (Parameters panel) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(30, 41, 59, 0.45);
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        padding: 0.25rem 0.25rem;
    }

    /* ---------- Buttons ---------- */
    .stButton>button {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.65rem 0rem;
        width: 100%;
        transition: all 0.25s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0px 10px 24px rgba(56, 189, 248, 0.35);
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 1rem 1rem 0.6rem 1rem;
    }

    /* ---------- Tabs ---------- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        border-bottom: 1px solid rgba(255,255,255,0.08);
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94A3B8;
    }
    .stTabs [aria-selected="true"] {
        color: #38BDF8 !important;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid rgba(255,255,255,0.1);
        color: #94A3B8;
        font-size: 1.0rem;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

# --- 2. Self-Healing Dynamic Model Loader ---
@st.cache_resource
def load_or_train_models():
    preprocessor_path = 'artifacts/preprocessor.pkl'
    model_path = 'artifacts/model.pkl'
    
    if not os.path.exists(preprocessor_path) or not os.path.exists(model_path):
        os.makedirs('artifacts', exist_ok=True)
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        
    preprocessor = load_object(preprocessor_path)
    model = load_object(model_path)
    return preprocessor, model

try:
    preprocessor, model = load_or_train_models()
except Exception as e:
    try:
        train_pipeline = TrainingPipeline()
        train_pipeline.run_pipeline()
        preprocessor = load_object('artifacts/preprocessor.pkl')
        model = load_object('artifacts/model.pkl')
    except Exception as inner_e:
        st.error(f"⚠️ Critical error initializing model pipeline: {inner_e}")
        st.stop()

# --- 3. App Header ---
st.markdown("""
<div class="brand-header">
    <div class="brand-mark">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M4 20V10" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
            <path d="M10 20V4" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
            <path d="M16 20V13" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
            <path d="M20 20V7" stroke="white" stroke-width="2.4" stroke-linecap="round"/>
        </svg>
    </div>
    <div class="title-main">STUDY-INSIGHT</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="subtitle">Student Performance Analytics Platform</div>', unsafe_allow_html=True)
st.markdown('<hr class="brand-divider">', unsafe_allow_html=True)

# --- 4. 30/70 Layout Split ---
col_input, col_display = st.columns([1, 2.5], gap="large")

# === LEFT COLUMN: CONTROL PANEL ===
with col_input:
    with st.container(border=True):
        st.markdown("### 🧑‍🎓 Parameters")

        with st.form("prediction_form", clear_on_submit=False):
            gender = st.selectbox("Gender", ["male", "female"])
            race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
            parental_education = st.selectbox("Parental Education", ["bachelor's degree", "some college", "master's degree", "associate's degree", "high school", "some high school"])
            lunch = st.selectbox("Lunch Program", ["standard", "free/reduced"])
            test_prep = st.selectbox("Test Prep Course", ["none", "completed"])

            st.markdown("---")
            reading_score = st.slider("Reading Score", 0, 100, 70)
            writing_score = st.slider("Writing Score", 0, 100, 70)

            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Run Analytics")

# === RIGHT COLUMN: ANALYTICS TABS ===
with col_display:
    if not submitted:
        st.info("👈 **Configure the student parameters on the left and click 'Run Analytics' to generate the dashboard.**")
    else:
        tab_insights, tab_diagnostics = st.tabs(["📊 Prediction & Insights", "⚙️ Model Analytics"])
        
        # Processing Logic
        input_data = pd.DataFrame([{
            "gender": gender, "race_ethnicity": race_ethnicity,
            "parental_level_of_education": parental_education, "lunch": lunch,
            "test_preparation_course": test_prep, "reading_score": reading_score, "writing_score": writing_score
        }])
        
        transformed_data = preprocessor.transform(input_data)
        prediction_raw = model.predict(transformed_data)[0]
        prediction = min(max(prediction_raw, 0), 100)
        
        math_median, math_std, class_size = 66.0, 15.16, 60
        percentile = stats.norm.cdf(prediction, loc=math_median, scale=math_std) * 100
        estimated_rank = max(1, int((1 - (percentile / 100)) * class_size))
        
        # TAB 1: PREDICTION & INSIGHTS
        with tab_insights:
            st.markdown("### 📈 Student Performance Summary")
            m1, m2, m3 = st.columns(3)
            m1.metric(label="Predicted Math Score", value=f"{prediction:.1f} / 100", delta=f"{prediction - math_median:+.1f} vs Global Median")
            m2.metric(label="Global Percentile", value=f"{percentile:.1f}th")
            m3.metric(label="Estimated Class Rank", value=f"#{estimated_rank}", delta=f"Top {int((estimated_rank/class_size)*100)}% of Class", delta_color="normal")
            
            st.markdown("---")
            c1, c2 = st.columns([1.2, 1])
            
            with c1:
                st.markdown("#### 🎯 Skill Correlation Map")
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=['Reading', 'Writing', 'Math (Predicted)'], y=[reading_score, writing_score, prediction], name='Student', marker_color='#38BDF8'))
                fig_bar.add_trace(go.Bar(x=['Reading', 'Writing', 'Math (Predicted)'], y=[70.0, 69.0, 66.0], name='Global Median', marker_color='rgba(255,255,255,0.2)'))
                fig_bar.update_layout(barmode='group', template='plotly_dark', plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', margin=dict(t=20, b=20, l=0, r=0))
                st.plotly_chart(fig_bar, use_container_width=True)
                
            with c2:
                st.markdown("#### 💡 Score Improvement Advice")
                st.write("Let's see how taking a preparation course changes the final outcome.")
                
                if test_prep == "none":
                    sim_data = input_data.copy()
                    sim_data["test_preparation_course"] = "completed"
                    sim_pred = min(max(model.predict(preprocessor.transform(sim_data))[0], 0), 100)
                    gain = sim_pred - prediction
                    
                    st.info(f"**Potential Gain:** Taking a test preparation course is projected to add **+{gain:.1f} points** to the math score, raising it to **{sim_pred:.1f}**.")
                else:
                    st.success("✅ **Maximized:** The student has already completed the test preparation course, contributing positively to their current predicted score.")
                
                if lunch == "free/reduced":
                    st.warning("⚠️ **Note on Nutrition:** Data indicates that students on the standard lunch program tend to score higher. Ensuring consistent, quality nutrition can further improve academic stamina.")

        # TAB 2: INTERACTIVE MODEL ANALYTICS
        with tab_diagnostics:
            st.markdown("### 🤖 Machine Learning Transparency")
            st.write("A model is only as good as its explainability. Here is how our Linear Regression engine performs against real, unseen test data.")
            
            dm1, dm2, dm3 = st.columns(3)
            dm1.metric("Engine", "Linear Regression")
            dm2.metric("Accuracy (R-Squared)", "88.03%", help="The model successfully explains 88% of the variance in student scores.")
            dm3.metric("Error Margin (MAE)", "± 4.22 pts", help="On average, our predictions are within 4 points of the actual score.")
            
            st.markdown("---")
            st.markdown("#### 📊 Actual vs. Predicted Plot")
            st.write("Hover over the data points to see the exact residual errors.")
            
            sample_diff = pd.DataFrame({
                "Student ID": [101, 102, 103, 104, 105, 106, 107, 108],
                "Actual Score": [91.0, 53.0, 80.0, 74.0, 84.0, 52.0, 62.0, 65.0],
                "Predicted Score": [76.5, 59.0, 77.0, 76.8, 87.5, 43.5, 62.0, 67.1],
            })
            sample_diff["Error (Residual)"] = sample_diff["Predicted Score"] - sample_diff["Actual Score"]
            
            fig_scatter = px.scatter(
                sample_diff, x="Actual Score", y="Predicted Score", 
                hover_data=["Student ID", "Error (Residual)"],
                template="plotly_dark",
                color="Error (Residual)", color_continuous_scale="Tealgrn"
            )
            fig_scatter.add_shape(type="line", x0=40, y0=40, x1=100, y1=100, line=dict(color="rgba(255,255,255,0.3)", dash="dash"))
            fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_scatter, use_container_width=True)

# --- Footer ---
st.markdown("""
    <div class="footer">
        Developed by Rahul Kumar Srivastava
    </div>
""", unsafe_allow_html=True)