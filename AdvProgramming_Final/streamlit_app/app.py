"""
NFL Game Prediction & Analysis Dashboard
Main landing page and navigation
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

# Page config
st.set_page_config(
    page_title="NFL Prediction Dashboard",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">🏈 NFL Game Prediction Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning Analysis of 2024 Season</p>', unsafe_allow_html=True)

# Introduction
st.markdown("""
Welcome to the **NFL Game Prediction Dashboard**! This interactive application showcases 
machine learning models trained on 2022-2024 NFL data to predict game outcomes, analyze 
team performance, and explore the relationship between team salary and success.
""")

st.markdown("---")

# Key Metrics Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="🎯 Best Model Accuracy",
        value="72.9%",
        delta="Logistic Regression",
        help="Accuracy on 2024 test games"
    )

with col2:
    st.metric(
        label="📊 Games Analyzed",
        value="2,176",
        delta="2024 Season",
        help="Test set games (one season)"
    )

with col3:
    st.metric(
        label="🏆 Best Ranking Model",
        value="ρ = 0.830",
        delta="Random Forest",
        help="Spearman correlation for team rankings"
    )

with col4:
    st.metric(
        label="💰 Salary Correlation",
        value="r = 0.475",
        delta="Moderate",
        help="Pearson correlation between salary and wins"
    )

st.markdown("---")

# Features Section
st.header("📱 Dashboard Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### 🏈 Game Predictor
    - Select any two teams for head-to-head prediction
    - See win probabilities from all 3 models
    - Adjust team statistics to see impact
    - Compare home vs away predictions
    """)
    
    st.markdown("""
    ### 📊 Model Comparison
    - View confusion matrices for all models
    - Compare accuracy, precision, and recall
    - Explore feature importance rankings
    - Understand why each model succeeded or failed
    """)

with col2:
    st.markdown("""
    ### 💰 Salary Analysis
    - Explore relationship between spending and wins
    - Interactive scatter plots with team labels
    - Compare predicted vs actual correlations
    - See which teams over/underperform their payroll
    """)
    
    st.markdown("""
    ### 🏆 Team Rankings
    - View predicted vs actual team rankings
    - Identify biggest surprises (over/underperformers)
    - Compare rankings across all models
    - Interactive bar charts for all 32 teams
    """)

st.markdown("---")

# Model Performance Summary
st.header("🎯 Model Performance Summary")

performance_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'KNN', 'Stacked Ensemble'],
    'Accuracy': [72.9, 71.5, 61.2, 62.5],
    'Precision': [0.73, 0.71, 0.61, 0.62],
    'Recall': [0.73, 0.71, 0.61, 0.62],
    'F1-Score': [0.73, 0.71, 0.61, 0.62],
    'Spearman ρ': [0.800, 0.830, 0.601, 0.620]
}

df_performance = pd.DataFrame(performance_data)

# Style the dataframe
st.dataframe(
    df_performance.style.background_gradient(cmap='RdYlGn', subset=['Accuracy']),
    use_container_width=True
)

st.markdown("---")

# Key Findings
st.header("🔍 Key Findings")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    #### Model Performance
    - **Logistic Regression** achieved highest accuracy (72.9%)
    - Matches professional oddsmakers (65-70%)
    - **Random Forest** best for team rankings (ρ=0.830)
    - **KNN** failed due to curse of dimensionality
    """)

with col2:
    st.markdown("""
    #### Feature Importance
    - **Win %** and **Opponent Win %** dominate (23%)
    - **Rushing EPA** is 3rd most important (4.6%)
    - **Home field advantage** adds 1.6%
    - EPA features collectively add ~2-3% accuracy
    """)

with col3:
    st.markdown("""
    #### Salary Insights
    - Moderate correlation: **r = 0.475**
    - Salary explains only ~23% of win variance
    - **NFL parity mechanisms are working**
    - HOW you spend > HOW MUCH you spend
    """)

st.markdown("---")

# Navigation Guide
st.header("🧭 How to Navigate")

st.markdown("""
Use the **sidebar** on the left to navigate between pages:

1. **🏈 Game Predictor** - Make predictions for specific matchups
2. **📊 Model Comparison** - Deep dive into model performance
3. **💰 Salary Analysis** - Explore spending vs winning
4. **🏆 Team Rankings** - See how teams stack up

Each page is interactive—adjust sliders, select teams, and explore the data!
""")

st.markdown("---")

# Footer
st.markdown("""
---
### 📚 About This Project

This dashboard was created as part of a **Machine Learning course project** analyzing NFL game predictions.

**Data Sources:**
- NFL game data (2022-2024) from NFLverse
- Team salary data from Spotrac
- EPA (Expected Points Added) metrics from nflfastR

**Models Used:**
- Logistic Regression (linear classifier)
- Random Forest (ensemble method)
- K-Nearest Neighbors (instance-based learning)
- Stacked Ensemble (meta-model)

**Methodology:**
- Time-based train/test split (2022-2023 → 2024)
- Lagged features with .shift(1) to prevent data leakage
- Cross-validation for hyperparameter tuning
- Permutation importance for unbiased feature ranking

---
*Built with Streamlit 🎈 | Created by [Your Name]*
""")

# Sidebar info
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/thumb/a/a2/National_Football_League_logo.svg/1200px-National_Football_League_logo.svg.png", 
             width=150)
    st.markdown("### Navigation")
    st.info("Use the pages above to explore different aspects of the NFL prediction analysis.")
    
    st.markdown("### Quick Stats")
    st.markdown("""
    - **Training Data**: 4,344 games (2022-2023)
    - **Test Data**: 2,176 games (2024)
    - **Features**: 20 engineered metrics
    - **Teams**: All 32 NFL teams
    """)
    
    st.markdown("### Model Status")
    st.success("✅ Logistic Regression: 72.9%")
    st.success("✅ Random Forest: 71.5%")
    st.warning("⚠️ KNN: 61.2%")
    st.warning("⚠️ Stacked: 62.5%")
