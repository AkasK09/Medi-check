import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ========== Page Configuration ==========
st.set_page_config(
    page_title="MediCheck - Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========== Mobile-Friendly CSS Styling ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --accent-gradient: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        --success-gradient: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        --warning-gradient: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
    }
    
    /* Base responsive font sizing */
    html {
        font-size: 16px;
    }
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        font-family: 'Inter', sans-serif;
        padding: 0 !important;
    }
    
    /* Mobile-first approach */
    .main-header {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .brand-title {
        font-size: 2rem;
        font-weight: 700;
        background: var(--primary-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
        text-align: center;
        line-height: 1.2;
    }
    
    .brand-subtitle {
        font-size: 1rem;
        color: #4a5568;
        font-weight: 500;
        opacity: 0.8;
        line-height: 1.4;
    }
    
    .section-card {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 1rem;
        margin: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(20px);
    }
    
    .feature-icon {
        width: 60px;
        height: 60px;
        background: var(--accent-gradient);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        color: white;
        font-size: 1.5rem;
    }
    
    .prediction-result {
        background: var(--success-gradient);
        color: white;
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1rem;
        font-weight: 600;
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    .precaution-item {
        background: rgba(67, 233, 123, 0.1);
        border-left: 4px solid #43e97b;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.9rem;
        line-height: 1.4;
    }
    
    /* Form elements mobile optimization */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        min-height: 44px; /* Touch target size */
    }
    
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.9);
        border: 2px solid rgba(102, 126, 234, 0.2);
        border-radius: 12px;
        min-height: 44px; /* Touch target size */
    }
    
    .stButton > button {
        background: var(--accent-gradient);
        color: white;
        border: none;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 1rem;
        min-height: 44px; /* Touch target size */
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
    }
    
    /* Hide floating shapes on mobile for performance */
    .floating-shapes {
        display: none;
    }
    
    /* Responsive text sizing */
    h1 { font-size: 1.8rem; }
    h2 { font-size: 1.4rem; }
    h3 { font-size: 1.2rem; }
    p { font-size: 0.9rem; line-height: 1.5; }
    
    /* Chart container mobile optimization */
    .stPlotlyChart {
        height: 300px !important;
    }
    
    /* Tab styling for mobile */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        overflow-x: auto;
        white-space: nowrap;
        padding-bottom: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        min-width: 120px;
        font-size: 0.9rem;
        padding: 0.5rem;
        flex-shrink: 0;
    }
    
    /* DataFrame mobile styling */
    .stDataFrame {
        font-size: 0.8rem;
    }
    
    .stDataFrame table {
        width: 100%;
        border-collapse: collapse;
    }
    
    .stDataFrame th,
    .stDataFrame td {
        padding: 0.5rem 0.25rem;
        text-align: left;
    }
    
    /* Info/warning boxes mobile */
    .stAlert {
        margin: 0.5rem 0;
        padding: 0.75rem;
        border-radius: 8px;
        font-size: 0.9rem;
    }
    
    /* Sidebar mobile adjustments */
    .css-1d391kg {
        padding-top: 1rem;
    }
    
    /* Container padding adjustments */
    .main > div {
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        max-width: 100% !important;
    }
    
    /* Column gap reduction for mobile */
    .row-widget.stHorizontal {
        gap: 0.5rem;
    }
    
    /* Tablet and larger screens */
    @media (min-width: 768px) {
        html { font-size: 18px; }
        
        .main-header {
            padding: 2rem;
            margin: 1rem;
            border-radius: 24px;
        }
        
        .brand-title {
            font-size: 3rem;
        }
        
        .brand-subtitle {
            font-size: 1.2rem;
        }
        
        .section-card {
            padding: 2rem;
            margin: 1rem;
            border-radius: 20px;
        }
        
        .feature-icon {
            width: 80px;
            height: 80px;
            font-size: 2rem;
        }
        
        .prediction-result {
            padding: 1.5rem;
            font-size: 1.2rem;
        }
        
        .floating-shapes {
            display: block;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: -1;
        }
        
        .shape {
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.1);
            animation: float 6s ease-in-out infinite;
        }
        
        .shape-1 {
            width: 80px;
            height: 80px;
            top: 20%;
            left: 10%;
            animation-delay: 0s;
        }
        
        .shape-2 {
            width: 120px;
            height: 120px;
            top: 60%;
            right: 15%;
            animation-delay: -2s;
        }
        
        .shape-3 {
            width: 60px;
            height: 60px;
            top: 30%;
            right: 30%;
            animation-delay: -4s;
        }
        
        .stPlotlyChart {
            height: 400px !important;
        }
        
        h1 { font-size: 2.5rem; }
        h2 { font-size: 2rem; }
        h3 { font-size: 1.5rem; }
        p { font-size: 1rem; }
    }
    
    /* Large screens */
    @media (min-width: 1200px) {
        .brand-title {
            font-size: 3.5rem;
        }
        
        .brand-subtitle {
            font-size: 1.3rem;
        }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    /* Touch optimization */
    button, select, input {
        -webkit-tap-highlight-color: rgba(0,0,0,0);
        touch-action: manipulation;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Prevent horizontal scroll */
    .stApp, .main, .block-container {
        overflow-x: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ========== Floating Background Shapes (Tablet+ only) ==========
st.markdown("""
<div class="floating-shapes">
    <div class="shape shape-1"></div>
    <div class="shape shape-2"></div>
    <div class="shape shape-3"></div>
</div>
""", unsafe_allow_html=True)

# ========== Load Models & Data ==========
@st.cache_resource
def load_models():
    try:
        clf = joblib.load("model/disease_model.pkl")
        mlb = joblib.load("model/symptom_encoder.pkl")
        le = joblib.load("model/label_encoder.pkl")
        return clf, mlb, le
    except:
        st.error("❌ Model files not found. Please ensure the model files are in the correct directory.")
        st.stop()

@st.cache_data
def load_precautions():
    try:
        precaution_df = pd.read_csv("Disease precaution.csv")
        precaution_map = {}
        for _, row in precaution_df.iterrows():
            disease = row["Disease"]
            precs = [str(row[f"Precaution_{i}"]) for i in range(1, 5) if pd.notna(row[f"Precaution_{i}"])]
            precaution_map[disease] = precs
        return precaution_map
    except:
        st.warning("⚠️ Precautions data not found. Predictions will work without precautions.")
        return {}

# Load models and data
clf, mlb, le = load_models()
precaution_map = load_precautions()

# ========== Main Header ==========
st.markdown("""
<div class="main-header">
    <h1 class="brand-title">🩺 MEDI-CHECK</h1>
    <p class="brand-subtitle">AI-Powered Disease Prediction</p>
    <p style="color: #718096; margin-top: 0.5rem; font-size: 0.9rem;">Enter symptoms for intelligent health analysis</p>
</div>
""", unsafe_allow_html=True)

# ========== Mobile-First Layout ==========
# On mobile: single column layout
# On tablet+: two column layout

# Check if we should use mobile layout (this is a simple approach)
# In practice, you might want to use JavaScript to detect screen size

# Symptom Input Section
st.markdown("""
<div class="section-card">
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div class="feature-icon">🔍</div>
        <h2 style="color: #2d3748; margin-bottom: 0.5rem;">Symptom Analysis</h2>
        <p style="color: #4a5568; font-size: 0.9rem;">Select your symptoms for AI analysis</p>
    </div>
""", unsafe_allow_html=True)

# Symptom Selection
all_symptoms = sorted(mlb.classes_)
selected_symptoms = st.multiselect(
    "**Choose your symptoms:**",
    options=all_symptoms,
    help="Select one or more symptoms you're experiencing"
)

# Show selected symptoms count
if selected_symptoms:
    st.info(f"✅ Selected {len(selected_symptoms)} symptom(s)")

# Prediction Button
predict_button = st.button("🔮 Analyze Symptoms", type="primary")

st.markdown("</div>", unsafe_allow_html=True)

# Results Section
st.markdown("""
<div class="section-card">
    <div style="text-align: center; margin-bottom: 1.5rem;">
        <div class="feature-icon">📊</div>
        <h2 style="color: #2d3748; margin-bottom: 0.5rem;">Results</h2>
        <p style="color: #4a5568; font-size: 0.9rem;">AI health insights and recommendations</p>
    </div>
""", unsafe_allow_html=True)

if predict_button:
    if not selected_symptoms:
        st.warning("⚠️ Please select at least one symptom to continue.")
    else:
        with st.spinner("🤖 AI is analyzing your symptoms..."):
            # Vectorize input
            input_vec = mlb.transform([selected_symptoms])
            
            # Get prediction probabilities
            pred_proba = clf.predict_proba(input_vec)[0]
            pred_class = clf.predict(input_vec)[0]
            
            # Get disease name
            disease = le.inverse_transform([pred_class])[0]
            confidence = pred_proba[pred_class] * 100
            
            # Display main prediction
            st.markdown(f"""
            <div class="prediction-result">
                <h3>🎯 Primary Prediction</h3>
                <h2 style="margin: 0.5rem 0;">{disease}</h2>
                <p style="margin: 0;">Confidence: {confidence:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Get top 5 predictions for visualization
            top_5_indices = np.argsort(pred_proba)[-5:][::-1]
            top_5_diseases = le.inverse_transform(top_5_indices)
            top_5_proba = pred_proba[top_5_indices] * 100

st.markdown("</div>", unsafe_allow_html=True)

# ========== Visualization Section ==========
if predict_button and selected_symptoms:
    st.markdown("""
    <div class="section-card">
        <h2 style="color: #2d3748; text-align: center; margin-bottom: 1rem;">📈 Analysis</h2>
    """, unsafe_allow_html=True)
    
    # Mobile-friendly tabs
    tab1, tab2, tab3 = st.tabs(["📊 Top 5", "🎯 Radar", "📋 Details"])
    
    with tab1:
        # Mobile-optimized bar chart
        fig_bar = px.bar(
            x=top_5_proba,
            y=top_5_diseases,
            orientation='h',
            color=top_5_proba,
            color_continuous_scale='Blues',
            title="Top 5 Predictions",
            labels={'x': 'Probability (%)', 'y': 'Disease'}
        )
        fig_bar.update_layout(
            height=300,
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(size=10),
            title_font_size=14
        )
        fig_bar.update_yaxes(tickfont=dict(size=9))
        fig_bar.update_xaxes(tickfont=dict(size=9))
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        # Mobile-optimized radar chart
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=top_5_proba,
            theta=top_5_diseases,
            fill='toself',
            name='Probability',
            line=dict(color='rgb(102, 126, 234)')
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title="Confidence Radar",
            height=350,
            margin=dict(l=10, r=10, t=40, b=10),
            font=dict(size=10),
            title_font_size=14
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with tab3:
        # Mobile-optimized analysis table
        analysis_df = pd.DataFrame({
            'Disease': top_5_diseases,
            'Probability': [f"{prob:.1f}%" for prob in top_5_proba],
            'Risk': [
                '🔴 High' if prob >= 70 else
                '🟡 Medium' if prob >= 40 else
                '🟢 Low' for prob in top_5_proba
            ]
        })
        
        st.dataframe(
            analysis_df,
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== Mobile-Friendly Precautions Section ==========
    if disease in precaution_map and precaution_map[disease]:
        st.markdown("""
        <div class="section-card">
            <h2 style="color: #2d3748; text-align: center; margin-bottom: 1rem;">💡 Recommendations</h2>
        """, unsafe_allow_html=True)
        
        precautions = precaution_map[disease]
        for i, precaution in enumerate(precautions, 1):
            st.markdown(f"""
            <div class="precaution-item">
                <strong>{i}. {precaution}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ========== Mobile-Friendly Footer Section ==========
st.markdown("""
<div class="section-card" style="text-align: center; margin-top: 2rem;">
    <h3 style="color: #2d3748; margin-bottom: 1rem; font-size: 1.1rem;">⚠️ Medical Disclaimer</h3>
    <p style="color: #4a5568; line-height: 1.5; font-size: 0.85rem;">
        MediCheck is for informational purposes only. 
        Always consult qualified healthcare providers for medical concerns.
    </p>
    <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid rgba(0,0,0,0.1);">
        <p style="color: #718096; font-size: 0.8rem;">
            Made with ❤️ using AI | © 2024 MediCheck
        </p>
    </div>
</div>
""", unsafe_allow_html=True)

# ========== Mobile-Optimized Sidebar ==========
with st.sidebar:
    st.markdown("""
    ### 🩺 About MediCheck
    
    **Features:**
    - 🤖 AI symptom analysis
    - 📊 Visual insights
    - 💡 Health tips
    - ⚡ Fast predictions
    
    **Quick Guide:**
    1. Pick symptoms
    2. Get AI analysis
    3. Review results
    4. Consult doctors
    
    ---
    
    **Support:**
    - 📧 Help center
    - 📖 User guide
    - 🏥 Find clinics
    """)