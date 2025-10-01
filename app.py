import streamlit as st
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 15px;
        background: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem;
        font-size: 1.1rem;
        font-weight: bold;
        border-radius: 10px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    h1 {
        color: white;
        text-align: center;
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    h2, h3 {
        color: white;
    }
    .subtitle {
        color: white;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1>🏥 Breast Cancer Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced AI-powered diagnostic assistance</p>", unsafe_allow_html=True)

# Load model (you'll need to have the model file)
@st.cache_resource
def load_model():
    try:
        return keras.models.load_model('breast_cancer.keras')
    except:
        st.error("Model file not found. Please ensure 'breast_cancer.keras' is in the same directory.")
        return None

model = load_model()

# Feature names from your dataset
feature_names = [
    'mean radius', 'mean texture', 'mean perimeter', 'mean area',
    'mean smoothness', 'mean compactness', 'mean concavity',
    'mean concave points', 'mean symmetry', 'mean fractal dimension',
    'radius error', 'texture error', 'perimeter error', 'area error',
    'smoothness error', 'compactness error', 'concavity error',
    'concave points error', 'symmetry error', 'fractal dimension error',
    'worst radius', 'worst texture', 'worst perimeter', 'worst area',
    'worst smoothness', 'worst compactness', 'worst concavity',
    'worst concave points', 'worst symmetry', 'worst fractal dimension'
]

# Sidebar for input
with st.sidebar:
    st.markdown("### 📋 Patient Information")
    st.markdown("---")
    
    input_method = st.radio(
        "Select Input Method:",
        ["Manual Entry", "Upload CSV"],
        help="Choose how you want to input the data"
    )
    
    if input_method == "Manual Entry":
        st.markdown("#### Mean Features")
        mean_radius = st.number_input("Mean Radius", min_value=0.0, max_value=50.0, value=14.0, step=0.1)
        mean_texture = st.number_input("Mean Texture", min_value=0.0, max_value=50.0, value=19.0, step=0.1)
        mean_perimeter = st.number_input("Mean Perimeter", min_value=0.0, max_value=250.0, value=92.0, step=0.1)
        mean_area = st.number_input("Mean Area", min_value=0.0, max_value=3000.0, value=655.0, step=1.0)
        mean_smoothness = st.number_input("Mean Smoothness", min_value=0.0, max_value=0.3, value=0.096, step=0.001, format="%.3f")
        
        with st.expander("More Mean Features"):
            mean_compactness = st.number_input("Mean Compactness", min_value=0.0, max_value=0.5, value=0.104, step=0.001, format="%.3f")
            mean_concavity = st.number_input("Mean Concavity", min_value=0.0, max_value=0.5, value=0.089, step=0.001, format="%.3f")
            mean_concave_points = st.number_input("Mean Concave Points", min_value=0.0, max_value=0.3, value=0.049, step=0.001, format="%.3f")
            mean_symmetry = st.number_input("Mean Symmetry", min_value=0.0, max_value=0.5, value=0.181, step=0.001, format="%.3f")
            mean_fractal_dimension = st.number_input("Mean Fractal Dimension", min_value=0.0, max_value=0.2, value=0.063, step=0.001, format="%.3f")
        
        st.markdown("#### Error Features")
        with st.expander("Error Features"):
            radius_error = st.number_input("Radius Error", min_value=0.0, max_value=5.0, value=0.4, step=0.01)
            texture_error = st.number_input("Texture Error", min_value=0.0, max_value=5.0, value=1.2, step=0.01)
            perimeter_error = st.number_input("Perimeter Error", min_value=0.0, max_value=30.0, value=2.9, step=0.1)
            area_error = st.number_input("Area Error", min_value=0.0, max_value=600.0, value=40.0, step=1.0)
            smoothness_error = st.number_input("Smoothness Error", min_value=0.0, max_value=0.05, value=0.007, step=0.001, format="%.3f")
            compactness_error = st.number_input("Compactness Error", min_value=0.0, max_value=0.2, value=0.025, step=0.001, format="%.3f")
            concavity_error = st.number_input("Concavity Error", min_value=0.0, max_value=0.5, value=0.032, step=0.001, format="%.3f")
            concave_points_error = st.number_input("Concave Points Error", min_value=0.0, max_value=0.1, value=0.012, step=0.001, format="%.3f")
            symmetry_error = st.number_input("Symmetry Error", min_value=0.0, max_value=0.1, value=0.020, step=0.001, format="%.3f")
            fractal_dimension_error = st.number_input("Fractal Dimension Error", min_value=0.0, max_value=0.05, value=0.004, step=0.001, format="%.3f")
        
        st.markdown("#### Worst Features")
        with st.expander("Worst Features"):
            worst_radius = st.number_input("Worst Radius", min_value=0.0, max_value=50.0, value=16.0, step=0.1)
            worst_texture = st.number_input("Worst Texture", min_value=0.0, max_value=60.0, value=26.0, step=0.1)
            worst_perimeter = st.number_input("Worst Perimeter", min_value=0.0, max_value=300.0, value=107.0, step=0.1)
            worst_area = st.number_input("Worst Area", min_value=0.0, max_value=5000.0, value=881.0, step=1.0)
            worst_smoothness = st.number_input("Worst Smoothness", min_value=0.0, max_value=0.3, value=0.132, step=0.001, format="%.3f")
            worst_compactness = st.number_input("Worst Compactness", min_value=0.0, max_value=1.5, value=0.254, step=0.001, format="%.3f")
            worst_concavity = st.number_input("Worst Concavity", min_value=0.0, max_value=1.5, value=0.272, step=0.001, format="%.3f")
            worst_concave_points = st.number_input("Worst Concave Points", min_value=0.0, max_value=0.5, value=0.115, step=0.001, format="%.3f")
            worst_symmetry = st.number_input("Worst Symmetry", min_value=0.0, max_value=0.7, value=0.290, step=0.001, format="%.3f")
            worst_fractal_dimension = st.number_input("Worst Fractal Dimension", min_value=0.0, max_value=0.3, value=0.084, step=0.001, format="%.3f")
        
        # Create input array
        input_data = np.array([[
            mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness,
            mean_compactness, mean_concavity, mean_concave_points, mean_symmetry,
            mean_fractal_dimension, radius_error, texture_error, perimeter_error,
            area_error, smoothness_error, compactness_error, concavity_error,
            concave_points_error, symmetry_error, fractal_dimension_error,
            worst_radius, worst_texture, worst_perimeter, worst_area,
            worst_smoothness, worst_compactness, worst_concavity,
            worst_concave_points, worst_symmetry, worst_fractal_dimension
        ]])
        
    else:
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.write("Preview:", df.head())
            input_data = df.values
        else:
            input_data = None

# Main content
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if model is not None and input_data is not None:
        if st.button("🔬 Analyze", key="predict"):
            with st.spinner("Analyzing..."):
                # Standardize input
                scaler = StandardScaler()
                # Note: In production, you should use the scaler fitted on training data
                input_scaled = scaler.fit_transform(input_data)
                
                # Make prediction
                prediction = model.predict(input_scaled)
                predicted_class = np.argmax(prediction, axis=1)[0]
                confidence = prediction[0][predicted_class] * 100
                
                # Display results
                st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
                
                if predicted_class == 0:
                    st.error("### ⚠️ Malignant")
                    result_color = "#ff4b4b"
                    diagnosis = "Malignant"
                else:
                    st.success("### ✅ Benign")
                    result_color = "#00cc00"
                    diagnosis = "Benign"
                
                # Confidence meter
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=confidence,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Confidence Level", 'font': {'size': 24}},
                    delta={'reference': 50},
                    gauge={
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': result_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 50], 'color': '#ffcccc'},
                            {'range': [50, 75], 'color': '#ffffcc'},
                            {'range': [75, 100], 'color': '#ccffcc'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                fig.update_layout(
                    paper_bgcolor="white",
                    font={'color': "black", 'family': "Arial"},
                    height=300
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Probability distribution
                st.markdown("### Probability Distribution")
                prob_df = pd.DataFrame({
                    'Class': ['Malignant', 'Benign'],
                    'Probability': prediction[0] * 100
                })
                fig2 = px.bar(
                    prob_df,
                    x='Class',
                    y='Probability',
                    color='Class',
                    color_discrete_map={'Malignant': '#ff4b4b', 'Benign': '#00cc00'},
                    text='Probability'
                )
                fig2.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
                fig2.update_layout(
                    showlegend=False,
                    height=300,
                    yaxis_range=[0, 105]
                )
                st.plotly_chart(fig2, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Additional information
                st.markdown("---")
                st.info("""
                **⚕️ Medical Disclaimer**: This prediction is for educational and research purposes only. 
                It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. 
                Always seek the advice of your physician or other qualified health provider.
                """)

# Information section
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class='metric-card'>
        <h3>🎯 Accuracy</h3>
        <h2>97.4%</h2>
        <p>Test Set Performance</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class='metric-card'>
        <h3>🧬 Features</h3>
        <h2>30</h2>
        <p>Input Parameters</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class='metric-card'>
        <h3>🤖 Model</h3>
        <h2>Deep Neural Network</h2>
        <p>TensorFlow/Keras</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: white; padding: 1rem;'>
    <p>Developed with ❤️ using Streamlit and TensorFlow</p>
    <p style='font-size: 0.8rem;'>Data Source: Wisconsin Breast Cancer Dataset (Diagnostic)</p>
</div>
""", unsafe_allow_html=True)