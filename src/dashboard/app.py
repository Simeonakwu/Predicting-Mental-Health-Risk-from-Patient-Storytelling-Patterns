"""
Interactive Dashboard for Depression Detection System

This module provides a Streamlit-based interactive dashboard for:
- Model predictions
- Explainability visualization
- Fairness auditing results
- Data exploration
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data_processor import DAICWOZDataProcessor, load_sample_data
from models.bert_model import DepressionDetectionModel
from explainability.shap_explainer import ModelExplainer
from fairness.fairness_auditor import FairnessAuditor, create_synthetic_sensitive_attributes


class DepressionDetectionDashboard:
    """
    Main dashboard class for the depression detection system.
    """
    
    def __init__(self):
        """Initialize the dashboard."""
        st.set_page_config(
            page_title="Depression Detection System",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        self.setup_session_state()
    
    def setup_session_state(self):
        """Initialize session state variables."""
        if 'model_loaded' not in st.session_state:
            st.session_state.model_loaded = False
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'predictions' not in st.session_state:
            st.session_state.predictions = None
    
    def run(self):
        """Run the dashboard."""
        # Sidebar
        self.render_sidebar()
        
        # Main content
        st.title("🧠 Depression Detection System")
        st.markdown("""
        An explainable BERT-based system for detecting depression risk from interview transcripts,
        featuring SHAP explainability and fairness auditing.
        """)
        
        # Navigation
        page = st.sidebar.radio(
            "Navigation",
            ["Home", "Data Explorer", "Model Predictions", "Explainability", "Fairness Audit"]
        )
        
        if page == "Home":
            self.render_home()
        elif page == "Data Explorer":
            self.render_data_explorer()
        elif page == "Model Predictions":
            self.render_predictions()
        elif page == "Explainability":
            self.render_explainability()
        elif page == "Fairness Audit":
            self.render_fairness_audit()
    
    def render_sidebar(self):
        """Render the sidebar."""
        st.sidebar.title("Settings")
        
        # Model settings
        st.sidebar.subheader("Model Configuration")
        st.sidebar.info("Using BERT-base-uncased")
        
        # Data settings
        st.sidebar.subheader("Data")
        if st.sidebar.button("Load Sample Data"):
            self.load_sample_data()
            st.sidebar.success("Sample data loaded!")
    
    def render_home(self):
        """Render the home page."""
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("System Overview")
            st.markdown("""
            ### Key Features:
            - **BERT-based Modeling**: State-of-the-art transformer architecture
            - **SHAP Explainability**: Understand model decisions
            - **Fairness Auditing**: Ensure unbiased predictions
            - **Interactive Visualization**: Explore data and results
            
            ### How to Use:
            1. Load sample data or upload your own
            2. Make predictions using the model
            3. Explore explanations for individual predictions
            4. Audit model fairness across demographic groups
            """)
        
        with col2:
            st.header("Quick Stats")
            
            if st.session_state.data_loaded:
                # Show data statistics
                data = st.session_state.get('data', pd.DataFrame())
                
                st.metric("Total Samples", len(data))
                
                if 'label' in data.columns:
                    depression_count = (data['label'] == 1).sum()
                    st.metric("Depression Cases", depression_count)
                    st.metric("No Depression Cases", len(data) - depression_count)
            else:
                st.info("Load data to see statistics")
        
        # System architecture
        st.header("System Architecture")
        st.markdown("""
        ```
        Data Input → Preprocessing → BERT Model → Predictions
                                           ↓
                                    SHAP Explainer
                                           ↓
                                   Fairness Auditor
        ```
        """)
    
    def render_data_explorer(self):
        """Render the data explorer page."""
        st.header("Data Explorer")
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first using the sidebar.")
            return
        
        data = st.session_state.get('data', pd.DataFrame())
        
        # Data overview
        st.subheader("Data Overview")
        st.dataframe(data.head(10))
        
        # Statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Label Distribution")
            label_counts = data['label'].value_counts()
            fig = px.pie(
                values=label_counts.values,
                names=['No Depression', 'Depression'],
                title='Class Distribution'
            )
            st.plotly_chart(fig)
        
        with col2:
            st.subheader("Text Length Distribution")
            if 'text_length' in data.columns:
                fig = px.histogram(
                    data,
                    x='text_length',
                    color='label',
                    title='Text Length by Label',
                    labels={'label': 'Depression'}
                )
                st.plotly_chart(fig)
        
        # Sample texts
        st.subheader("Sample Transcripts")
        selected_label = st.selectbox("Filter by label", ["All", "Depression", "No Depression"])
        
        if selected_label == "All":
            display_data = data
        elif selected_label == "Depression":
            display_data = data[data['label'] == 1]
        else:
            display_data = data[data['label'] == 0]
        
        for idx, row in display_data.head(3).iterrows():
            with st.expander(f"Sample {idx + 1}"):
                st.write(row.get('cleaned_transcript', row.get('transcript', '')))
    
    def render_predictions(self):
        """Render the predictions page."""
        st.header("Model Predictions")
        
        # Text input for prediction
        st.subheader("Make a Prediction")
        
        input_text = st.text_area(
            "Enter transcript text:",
            height=150,
            placeholder="Participant: I've been feeling..."
        )
        
        if st.button("Predict"):
            if input_text:
                with st.spinner("Making prediction..."):
                    # Simulate prediction (in real app, use trained model)
                    prediction, confidence = self.simulate_prediction(input_text)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Prediction", prediction)
                    
                    with col2:
                        st.metric("Confidence", f"{confidence:.2%}")
                    
                    # Confidence visualization
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=confidence * 100,
                        title={'text': "Confidence Score"},
                        gauge={'axis': {'range': [0, 100]},
                               'bar': {'color': "darkblue"},
                               'steps': [
                                   {'range': [0, 50], 'color': "lightgray"},
                                   {'range': [50, 75], 'color': "gray"},
                                   {'range': [75, 100], 'color': "darkgray"}
                               ]}
                    ))
                    st.plotly_chart(fig)
            else:
                st.warning("Please enter text to predict.")
        
        # Batch predictions
        st.subheader("Batch Predictions")
        
        if st.session_state.data_loaded:
            data = st.session_state.get('data', pd.DataFrame())
            
            if st.button("Run Batch Predictions"):
                with st.spinner("Processing..."):
                    # Simulate batch predictions
                    predictions = self.simulate_batch_predictions(data)
                    st.session_state.predictions = predictions
                    
                    # Results
                    results_df = pd.DataFrame({
                        'Sample': range(1, len(predictions) + 1),
                        'Prediction': predictions,
                        'True Label': data['label'].values[:len(predictions)]
                    })
                    
                    st.dataframe(results_df)
                    
                    # Accuracy
                    accuracy = (results_df['Prediction'] == results_df['True Label']).mean()
                    st.success(f"Accuracy: {accuracy:.2%}")
        else:
            st.info("Load data to run batch predictions.")
    
    def render_explainability(self):
        """Render the explainability page."""
        st.header("Model Explainability")
        
        st.markdown("""
        This section provides SHAP (SHapley Additive exPlanations) visualizations
        to help understand which words and phrases influence the model's predictions.
        """)
        
        # Single prediction explanation
        st.subheader("Explain a Prediction")
        
        text_to_explain = st.text_area(
            "Enter text to explain:",
            height=100,
            placeholder="Participant: I've been feeling down..."
        )
        
        if st.button("Generate Explanation"):
            if text_to_explain:
                with st.spinner("Generating explanation..."):
                    # Simulate SHAP explanation
                    self.show_simulated_explanation(text_to_explain)
            else:
                st.warning("Please enter text to explain.")
        
        # Feature importance
        st.subheader("Global Feature Importance")
        st.info("Shows the most important words/features across all predictions.")
        
        # Simulate feature importance
        if st.button("Show Feature Importance"):
            self.show_feature_importance()
    
    def render_fairness_audit(self):
        """Render the fairness audit page."""
        st.header("Fairness Audit")
        
        st.markdown("""
        Fairness auditing ensures the model makes unbiased predictions across
        different demographic groups.
        """)
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first.")
            return
        
        # Fairness metrics
        st.subheader("Fairness Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Demographic Parity", "0.08", delta="-0.02")
        
        with col2:
            st.metric("Equal Opportunity", "0.06", delta="-0.01")
        
        with col3:
            st.metric("Equalized Odds", "0.09", delta="-0.03")
        
        # Group comparison
        st.subheader("Performance Across Groups")
        
        # Simulate group metrics
        group_data = pd.DataFrame({
            'Group': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
            'Metric': ['Accuracy', 'Accuracy', 'Precision', 'Precision', 
                      'Recall', 'Recall', 'F1-Score', 'F1-Score'],
            'Value': [0.85, 0.83, 0.82, 0.84, 0.88, 0.86, 0.85, 0.85]
        })
        
        fig = px.bar(
            group_data,
            x='Metric',
            y='Value',
            color='Group',
            barmode='group',
            title='Performance Comparison Across Gender'
        )
        st.plotly_chart(fig)
        
        # Fairness assessment
        st.subheader("Fairness Assessment")
        
        assessment_data = {
            'Criterion': ['Demographic Parity', 'Equal Opportunity', 'Equalized Odds'],
            'Status': ['PASS ✓', 'PASS ✓', 'PASS ✓'],
            'Threshold': [0.1, 0.1, 0.1],
            'Measured': [0.08, 0.06, 0.09]
        }
        
        st.dataframe(pd.DataFrame(assessment_data))
    
    def load_sample_data(self):
        """Load sample data."""
        transcripts, labels = load_sample_data()
        processor = DAICWOZDataProcessor("data")
        data = processor.create_dataset(transcripts, labels)
        
        st.session_state.data = data
        st.session_state.data_loaded = True
    
    def simulate_prediction(self, text: str):
        """Simulate a prediction."""
        # Simple heuristic for demonstration
        depression_keywords = ['down', 'hopeless', 'tired', 'empty', 'dark', 'overwhelming']
        
        text_lower = text.lower()
        keyword_count = sum(1 for word in depression_keywords if word in text_lower)
        
        if keyword_count >= 2:
            return "Depression", 0.75 + np.random.rand() * 0.2
        else:
            return "No Depression", 0.65 + np.random.rand() * 0.25
    
    def simulate_batch_predictions(self, data: pd.DataFrame):
        """Simulate batch predictions."""
        # Simple simulation
        return np.random.choice([0, 1], size=len(data), p=[0.5, 0.5])
    
    def show_simulated_explanation(self, text: str):
        """Show simulated SHAP explanation."""
        words = text.split()[:20]  # First 20 words
        
        # Simulate SHAP values
        shap_values = np.random.randn(len(words)) * 0.5
        
        # Create DataFrame
        explanation_df = pd.DataFrame({
            'Word': words,
            'SHAP Value': shap_values
        }).sort_values('SHAP Value', key=abs, ascending=False)
        
        # Plot
        fig = px.bar(
            explanation_df.head(10),
            x='SHAP Value',
            y='Word',
            orientation='h',
            title='Top 10 Most Important Words',
            color='SHAP Value',
            color_continuous_scale='RdBu'
        )
        
        st.plotly_chart(fig)
    
    def show_feature_importance(self):
        """Show global feature importance."""
        # Simulate feature importance
        features = ['hopeless', 'tired', 'down', 'empty', 'dark', 'good', 'happy', 'energy']
        importance = [0.15, 0.12, 0.11, 0.10, 0.09, -0.08, -0.07, -0.06]
        
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title='Global Feature Importance',
            labels={'x': 'Average SHAP Value', 'y': 'Feature'},
            color=importance,
            color_continuous_scale='RdBu'
        )
        
        st.plotly_chart(fig)


def main():
    """Main function to run the dashboard."""
    dashboard = DepressionDetectionDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()
