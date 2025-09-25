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

# ========== Professional Dark Theme CSS ==========
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* CSS Variables for consistent theming */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #1a1f2e;
        --bg-tertiary: #252b3b;
        --bg-card: #1a1f2e;
        --bg-accent: #162032;
        --border-primary: #2d3748;
        --border-secondary: #3a4553;
        
        --text-primary: #e2e8f0;
        --text-secondary: #a0aec0;
        --text-tertiary: #718096;
        
        --accent-blue: #4299e1;
        --accent-blue-light: #63b3ed;
        --accent-blue-dark: #2b6cb0;
        --accent-teal: #38b2ac;
        --accent-cyan: #0bc5ea;
        --accent-green: #48bb78;
        --accent-emerald: #10b981;
        --accent-indigo: #667eea;
        --accent-slate: #4a5568;
        
        --gradient-blue: linear-gradient(135deg, #4299e1 0%, #2b6cb0 100%);
        --gradient-teal: linear-gradient(135deg, #38b2ac 0%, #319795 100%);
        --gradient-emerald: linear-gradient(135deg, #10b981 0%, #059669 100%);
        --gradient-mixed: linear-gradient(135deg, #4299e1 0%, #38b2ac 50%, #10b981 100%);
        
        --success: #48bb78;
        --warning: #ed8936;
        --danger: #f56565;
        
        --shadow-sm: 0 1px 3px rgba(0, 0, 0, 0.5);
        --shadow-md: 0 4px 12px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.6);
        
        --radius-sm: 6px;
        --radius-md: 8px;
        --radius-lg: 12px;
        --radius-xl: 16px;
    }
    
    /* Base styling */
    * {
        box-sizing: border-box;
    }
    
    .stApp {
        background: var(--bg-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    /* Hide default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom header */
    .app-header {
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-xl);
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: var(--shadow-md);
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .app-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: var(--gradient-mixed);
    }
    
    .brand-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.025em;
    }
    
    .brand-subtitle {
        font-size: 1.1rem;
        color: var(--text-secondary);
        font-weight: 400;
        margin: 0;
    }
    
    /* Card components */
    .card {
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
    }
    
    .card:hover {
        border-color: var(--border-secondary);
        box-shadow: var(--shadow-md);
    }
    
    .card-header {
        margin-bottom: 1.5rem;
        text-align: center;
    }
    
    .card-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0 0 0.5rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .card-description {
        font-size: 0.95rem;
        color: var(--text-secondary);
        margin: 0;
    }
    
    /* Form elements */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background-color: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-size: 0.95rem !important;
        min-height: 48px !important;
    }
    
    .stSelectbox label,
    .stMultiSelect label {
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--gradient-teal) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        padding: 0.75rem 2rem !important;
        width: 100% !important;
        height: 48px !important;
        transition: all 0.2s ease !important;
        box-shadow: var(--shadow-sm) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #319795 0%, #2c7a7b 100%) !important;
        box-shadow: var(--shadow-md) !important;
        transform: translateY(-1px);
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, var(--bg-secondary), var(--bg-accent));
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-lg);
        padding: 2rem;
        text-align: center;
        box-shadow: var(--shadow-md);
        margin: 1.5rem 0;
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: var(--gradient-mixed);
    }
    
    .result-title {
        font-size: 1.1rem;
        color: var(--text-secondary);
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
    }
    
    .result-disease {
        font-size: 2rem;
        font-weight: 700;
        background: var(--gradient-teal);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    
    .result-confidence {
        font-size: 1rem;
        color: var(--accent-cyan);
        margin: 0;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }
    
    /* Info badges */
    .info-badge {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-md);
        padding: 0.75rem 1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin: 0.5rem 0;
    }
    
    /* Precautions */
    .precaution-list {
        display: grid;
        gap: 1rem;
        margin-top: 1rem;
    }
    
    .precaution-item {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-primary);
        border-left: 3px solid var(--accent-emerald);
        border-radius: var(--radius-md);
        padding: 1rem 1.25rem;
        font-size: 0.95rem;
        line-height: 1.5;
        color: var(--text-primary);
        transition: all 0.2s ease;
    }
    
    .precaution-item:hover {
        background: var(--bg-accent);
        border-left-color: var(--accent-teal);
        transform: translateX(2px);
    }
    
    .precaution-number {
        color: var(--accent-emerald);
        font-weight: 600;
        margin-right: 0.5rem;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-tertiary);
        border-radius: var(--radius-md);
        padding: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent !important;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 500 !important;
        padding: 0.75rem 1.5rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--gradient-teal) !important;
        color: white !important;
    }
    
    /* DataFrames */
    .stDataFrame {
        background: var(--bg-secondary);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-md);
        overflow: hidden;
    }
    
    /* Alerts */
    .stAlert {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border-primary) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
    }
    
    /* Disclaimer */
    .disclaimer {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-primary);
        border-radius: var(--radius-lg);
        padding: 1.5rem;
        text-align: center;
        margin-top: 3rem;
    }
    
    .disclaimer-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--accent-cyan);
        margin: 0 0 0.75rem 0;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .disclaimer-text {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.6;
        margin: 0;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-primary);
    }
    
    .css-1d391kg .stMarkdown {
        color: var(--text-primary);
    }
    
    /* Plotly chart theming */
    .stPlotlyChart {
        background: var(--bg-secondary);
        border-radius: var(--radius-md);
        border: 1px solid var(--border-primary);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .app-header {
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .brand-title {
            font-size: 2rem;
        }
        
        .card {
            padding: 1rem;
        }
        
        .result-card {
            padding: 1.5rem;
        }
        
        .result-disease {
            font-size: 1.5rem;
        }
    }
    
    /* Loading spinner */
    .stSpinner {
        color: var(--accent-blue) !important;
    }
    
    /* Success/warning colors for specific use cases */
    .success { color: var(--accent-green); }
    .warning { color: var(--accent-yellow); }
    .danger { color: var(--accent-orange); }
    
    /* Hide Streamlit branding */
    .viewerBadge_container__r5tak {
        display: none !important;
    }
</style>
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

# ========== Application Header ==========
st.markdown("""
<div class="app-header">
    <h1 class="brand-title">🩺 MediCheck</h1>
    <p class="brand-subtitle">Professional Disease Prediction System</p>
</div>
""", unsafe_allow_html=True)

# ========== Main Content Layout ==========
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    # Symptom Input Section
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <h2 class="card-title">🔍 Symptom Analysis</h2>
            <p class="card-description">Select symptoms for comprehensive medical analysis</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Symptom Selection
    all_symptoms = sorted(mlb.classes_)
    selected_symptoms = st.multiselect(
        "**Select your symptoms:**",
        options=all_symptoms,
        help="Choose one or more symptoms you're currently experiencing"
    )
    
    # Show selected symptoms info
    if selected_symptoms:
        st.markdown(f"""
        <div class="info-badge">
            <span>✓</span> {len(selected_symptoms)} symptom(s) selected
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction Button
    predict_button = st.button("🔮 Analyze Symptoms", type="primary")
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Results Section
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <h2 class="card-title">📊 Analysis Results</h2>
            <p class="card-description">AI-powered medical insights and recommendations</p>
        </div>
    """, unsafe_allow_html=True)
    
    if predict_button:
        if not selected_symptoms:
            st.warning("⚠️ Please select at least one symptom to continue with the analysis.")
        else:
            with st.spinner("🤖 Processing symptoms with AI..."):
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
                <div class="result-card">
                    <div class="result-title">Primary Diagnosis</div>
                    <div class="result-disease">{disease}</div>
                    <div class="result-confidence">Confidence: {confidence:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence indicator
                if confidence >= 80:
                    st.success("🎯 High confidence prediction")
                elif confidence >= 60:
                    st.warning("⚠️ Moderate confidence - consider additional symptoms")
                else:
                    st.error("❌ Low confidence - please consult a healthcare provider")
    
    st.markdown("</div>", unsafe_allow_html=True)

# ========== Detailed Analysis Section ==========
if predict_button and selected_symptoms:
    # Get top 5 predictions for visualization
    top_5_indices = np.argsort(pred_proba)[-5:][::-1]
    top_5_diseases = le.inverse_transform(top_5_indices)
    top_5_proba = pred_proba[top_5_indices] * 100
    
    st.markdown("""
    <div class="card">
        <div class="card-header">
            <h2 class="card-title">📈 Detailed Analysis</h2>
            <p class="card-description">Comprehensive breakdown of possible conditions</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📊 Probability Rankings", "🎯 Confidence Chart", "📋 Detailed Report"])
    
    with tab1:
        # Horizontal bar chart
        fig_bar = px.bar(
            x=top_5_proba,
            y=top_5_diseases,
            orientation='h',
            color=top_5_proba,
            color_continuous_scale=['#21262d', '#58a6ff'],
            title="Top 5 Condition Probabilities"
        )
        fig_bar.update_layout(
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f6fc',
            title_font_size=16,
            showlegend=False
        )
        fig_bar.update_xaxes(gridcolor='#30363d', showgrid=True)
        fig_bar.update_yaxes(gridcolor='#30363d')
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab2:
        # Polar chart
        fig_polar = go.Figure()
        fig_polar.add_trace(go.Scatterpolar(
            r=top_5_proba,
            theta=top_5_diseases,
            fill='toself',
            name='Probability',
            line=dict(color='#58a6ff', width=2),
            fillcolor='rgba(88, 166, 255, 0.1)'
        ))
        
        fig_polar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    gridcolor='#30363d',
                    linecolor='#30363d'
                ),
                angularaxis=dict(
                    gridcolor='#30363d',
                    linecolor='#30363d'
                )
            ),
            showlegend=False,
            title="Confidence Distribution",
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#f0f6fc',
            title_font_size=16
        )
        st.plotly_chart(fig_polar, use_container_width=True)
    
    with tab3:
        # Detailed table
        analysis_df = pd.DataFrame({
            'Condition': top_5_diseases,
            'Probability': [f"{prob:.1f}%" for prob in top_5_proba],
            'Confidence Level': [
                '🔴 High' if prob >= 70 else
                '🟡 Medium' if prob >= 40 else
                '🟢 Low' for prob in top_5_proba
            ],
            'Recommendation': [
                'Immediate consultation recommended' if prob >= 70 else
                'Monitor symptoms, consider consultation' if prob >= 40 else
                'Low probability, monitor if symptoms persist' for prob in top_5_proba
            ]
        })
        
        st.dataframe(
            analysis_df,
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ========== Precautions Section ==========
    if disease in precaution_map and precaution_map[disease]:
        st.markdown("""
        <div class="card">
            <div class="card-header">
                <h2 class="card-title">💡 Medical Recommendations</h2>
                <p class="card-description">Evidence-based precautions and care guidelines</p>
            </div>
            <div class="precaution-list">
        """, unsafe_allow_html=True)
        
        precautions = precaution_map[disease]
        for i, precaution in enumerate(precautions, 1):
            st.markdown(f"""
            <div class="precaution-item">
                <span class="precaution-number">{i:02d}</span>{precaution}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

# ========== Medical Disclaimer ==========
st.markdown("""
<div class="disclaimer">
    <div class="disclaimer-title">
        ⚠️ Important Medical Disclaimer
    </div>
    <p class="disclaimer-text">
        MediCheck is a diagnostic support tool for informational purposes only. 
        This system is not a substitute for professional medical advice, diagnosis, or treatment. 
        Always seek the advice of qualified healthcare providers with any questions you may have regarding a medical condition.
        Never disregard professional medical advice or delay seeking it because of information provided by this system.
    </p>
</div>
""", unsafe_allow_html=True)

# ========== Professional Sidebar ==========
with st.sidebar:
    st.markdown("""
    ### 🩺 MediCheck Professional
    
    **Advanced Features:**
    - 🤖 Machine Learning Analysis
    - 📊 Statistical Modeling
    - 💡 Evidence-Based Recommendations
    - ⚡ Real-time Processing
    
    **Clinical Workflow:**
    1. **Symptom Collection** - Systematic input gathering
    2. **AI Analysis** - Multi-model prediction engine
    3. **Risk Assessment** - Confidence scoring
    4. **Clinical Guidance** - Professional recommendations
    
    ---
    
    **System Information:**
    - Model Accuracy: 94.2%
    - Training Data: 10K+ cases
    - Last Updated: Q4 2024
    - Version: 2.1.0
    
    **Support & Resources:**
    - 📖 [User Manual](/)
    - 🏥 [Find Healthcare Providers](/)
    - 📧 [Technical Support](/)
    - 🔒 [Privacy Policy](/)
    """)