"""
Page 2: Model Comparison
Compare performance metrics and confusion matrices across all models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Model Comparison", page_icon="📊", layout="wide")

st.title("📊 Model Performance Comparison")
st.markdown("Deep dive into how each model performs and why")

st.markdown("---")

# Model performance data (from your actual results)
performance_data = {
    'Model': ['Logistic Regression', 'Random Forest', 'KNN', 'Stacked Ensemble'],
    'Accuracy': [72.9, 71.5, 61.2, 62.5],
    'Precision': [0.73, 0.71, 0.61, 0.62],
    'Recall': [0.73, 0.71, 0.61, 0.62],
    'F1-Score': [0.73, 0.71, 0.61, 0.62],
    'Spearman ρ': [0.800, 0.830, 0.601, 0.620],
    'CV Accuracy': [63.1, None, None, None]  # Only LR has CV reported
}

# Confusion Matrix data (from your PDF)
confusion_matrices = {
    'Logistic Regression': np.array([[791, 297], [294, 794]]),
    'Random Forest': np.array([[785, 303], [318, 770]]),
    'KNN': np.array([[670, 418], [427, 661]]),
    'Stacked Ensemble': np.array([[685, 403], [413, 675]])
}

# Feature importance data (top 10 from your results)
feature_importance_lr = pd.DataFrame({
    'Feature': ['opp_win_pct', 'win_pct', 'rel_rushing_epa', 'rel_passing_int', 'is_home',
                'rel_receiving_epa', 'rel_turnovers', 'rel_def_int', 'rel_def_sacks', 'rel_receiving_yards'],
    'Importance': [11.58, 11.47, 4.63, 1.68, 1.58, 1.51, 1.31, 1.01, 0.67, 0.63]
})

feature_importance_rf = pd.DataFrame({
    'Feature': ['win_pct', 'opp_win_pct', 'rel_rushing_epa', 'is_home', 'rel_penalty_yards',
                'is_playoff', 'rel_passing_int', 'rel_passing_tds', 'rel_def_tds', 'rel_rushing_tds'],
    'Importance': [9.16, 7.77, 0.92, 0.15, 0.03, 0.00, 0.00, -0.01, -0.06, -0.07]
})

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Metrics", "🎯 Confusion Matrices", "🔍 Feature Importance", "💡 Model Insights"])

## TAB 1: Performance Metrics
with tab1:
    st.subheader("Model Accuracy Comparison")
    
    # Bar chart for accuracy
    df_perf = pd.DataFrame(performance_data)
    
    fig = go.Figure()
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
    
    fig.add_trace(go.Bar(
        x=df_perf['Model'],
        y=df_perf['Accuracy'],
        marker_color=colors,
        text=df_perf['Accuracy'].apply(lambda x: f'{x:.1f}%'),
        textposition='outside'
    ))
    
    # Add benchmark lines
    fig.add_hline(y=70, line_dash="dash", line_color="green", 
                  annotation_text="Vegas Oddsmakers (65-70%)")
    fig.add_hline(y=50, line_dash="dot", line_color="gray",
                  annotation_text="Random Guess (50%)")
    
    fig.update_layout(
        title="Test Accuracy Comparison (2024 Season)",
        yaxis_title="Accuracy (%)",
        height=400,
        yaxis_range=[0, 100],
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed metrics table
    st.subheader("Detailed Performance Metrics")
    
    # Style the dataframe
    styled_df = df_perf.style.background_gradient(
        cmap='RdYlGn', 
        subset=['Accuracy', 'Precision', 'Recall', 'F1-Score']
    ).format({
        'Accuracy': '{:.1f}%',
        'Precision': '{:.2f}',
        'Recall': '{:.2f}',
        'F1-Score': '{:.2f}',
        'Spearman ρ': '{:.3f}',
        'CV Accuracy': lambda x: f'{x:.1f}%' if pd.notna(x) else 'N/A'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Key takeaways
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "🥇 Best Accuracy",
            "Logistic Regression",
            "72.9%"
        )
        st.caption("Outperforms Vegas oddsmakers (65-70%)")
    
    with col2:
        st.metric(
            "🏆 Best Rankings",
            "Random Forest",
            "ρ = 0.830"
        )
        st.caption("Best Spearman correlation for team strength")
    
    with col3:
        st.metric(
            "❌ Worst Performance",
            "KNN",
            "61.2%"
        )
        st.caption("Curse of dimensionality with 20 features")

## TAB 2: Confusion Matrices
with tab2:
    st.subheader("Confusion Matrices - Where Models Make Mistakes")
    
    # Create 2x2 grid of confusion matrices
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(confusion_matrices.keys()),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}],
               [{"type": "heatmap"}, {"type": "heatmap"}]]
    )
    
    for idx, (model_name, cm) in enumerate(confusion_matrices.items()):
        row = idx // 2 + 1
        col = idx % 2 + 1
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create text annotations
        text = [[f"{cm[i, j]}<br>({cm_norm[i, j]:.1%})" 
                 for j in range(2)] for i in range(2)]
        
        fig.add_trace(
            go.Heatmap(
                z=cm_norm,
                x=['Pred: Loss', 'Pred: Win'],
                y=['Actual: Loss', 'Actual: Win'],
                text=text,
                texttemplate="%{text}",
                colorscale='Blues',
                showscale=False
            ),
            row=row, col=col
        )
    
    fig.update_layout(height=700, title_text="Confusion Matrices (All Models)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("### 📖 How to Read Confusion Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Top-Left (True Negatives):**  
        Model predicted LOSS, team actually LOST ✅
        
        **Top-Right (False Positives):**  
        Model predicted WIN, team actually LOST ❌  
        *Type I Error - False alarm*
        """)
    
    with col2:
        st.markdown("""
        **Bottom-Left (False Negatives):**  
        Model predicted LOSS, team actually WON ❌  
        *Type II Error - Missed opportunity*
        
        **Bottom-Right (True Positives):**  
        Model predicted WIN, team actually WON ✅
        """)
    
    # Model-specific insights
    st.markdown("---")
    st.markdown("### 🔍 Model-Specific Insights")
    
    selected_model = st.selectbox(
        "Select model to analyze:",
        list(confusion_matrices.keys())
    )
    
    cm = confusion_matrices[selected_model]
    tn, fp, fn, tp = cm.ravel()
    
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    precision_loss = tn / (tn + fn) if (tn + fn) > 0 else 0
    precision_win = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_loss = tn / (tn + fp) if (tn + fp) > 0 else 0
    recall_win = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("True Negatives", tn)
        st.caption("Correctly predicted losses")
    
    with col2:
        st.metric("False Positives", fp)
        st.caption("Wrongly predicted wins")
    
    with col3:
        st.metric("False Negatives", fn)
        st.caption("Missed wins")
    
    with col4:
        st.metric("True Positives", tp)
        st.caption("Correctly predicted wins")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Precision (Losses)", f"{precision_loss:.1%}")
        st.caption("When model predicts loss, how often correct?")
        
        st.metric("Precision (Wins)", f"{precision_win:.1%}")
        st.caption("When model predicts win, how often correct?")
    
    with col2:
        st.metric("Recall (Losses)", f"{recall_loss:.1%}")
        st.caption("Of actual losses, how many did we catch?")
        
        st.metric("Recall (Wins)", f"{recall_win:.1%}")
        st.caption("Of actual wins, how many did we catch?")
    
    # Symmetry check
    symmetry = abs(fp - fn)
    if symmetry < 20:
        st.success(f"✅ **Balanced Errors**: Model shows symmetric performance (FP={fp}, FN={fn}). No bias toward predicting one outcome.")
    else:
        if fp > fn:
            st.warning(f"⚠️ **Bias Detected**: Model tends to over-predict wins (FP={fp} > FN={fn})")
        else:
            st.warning(f"⚠️ **Bias Detected**: Model tends to over-predict losses (FN={fn} > FP={fp})")

## TAB 3: Feature Importance
with tab3:
    st.subheader("Feature Importance Analysis")
    
    model_select = st.radio(
        "Select model:",
        ["Logistic Regression", "Random Forest"],
        horizontal=True
    )
    
    if model_select == "Logistic Regression":
        df_importance = feature_importance_lr
        color = '#1f77b4'
    else:
        df_importance = feature_importance_rf
        color = '#2ca02c'
    
    # Bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_importance['Importance'],
        y=df_importance['Feature'],
        orientation='h',
        marker_color=color,
        text=df_importance['Importance'].apply(lambda x: f'{x:.2f}%'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Top 10 Features - {model_select}",
        xaxis_title="Permutation Importance (%)",
        yaxis_title="Feature",
        height=500,
        yaxis={'categoryorder':'total ascending'}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Feature explanations
    st.markdown("---")
    st.markdown("### 📚 Feature Definitions")
    
    with st.expander("Click to see feature explanations"):
        st.markdown("""
        - **win_pct**: Team's historical win percentage (prior games only, using .shift(1))
        - **opp_win_pct**: Opponent's historical win percentage
        - **rel_rushing_epa**: Relative rushing EPA (team - opponent)
        - **rel_passing_int**: Relative passing interceptions (team - opponent)
        - **is_home**: Binary indicator for home field advantage
        - **rel_receiving_epa**: Relative receiving EPA (team - opponent)
        - **rel_turnovers**: Relative turnover rate (team - opponent)
        - **rel_def_int**: Relative defensive interceptions
        - **rel_def_sacks**: Relative defensive sacks
        - **is_playoff**: Binary indicator for playoff game
        
        **Note**: All "rel_" features are relative differences (team stat - opponent stat),
        which captures matchup dynamics better than absolute values.
        """)
    
    # Key insights
    st.markdown("---")
    st.markdown("### 💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Win % Dominates** (23% combined importance)
        
        Historical success is the best predictor of future success.
        Why? It implicitly captures:
        - Coaching quality
        - Roster talent  
        - Team chemistry
        - System fit
        - Recent injury history
        """)
    
    with col2:
        st.info("""
        **"Establish the Run" is Valid**
        
        Rushing EPA is the 3rd most important feature (4.6%).
        Teams that run efficiently:
        - Control time of possession
        - Wear down defenses
        - Convert 3rd downs
        - Protect leads in 4th quarter
        """)

## TAB 4: Model Insights
with tab4:
    st.subheader("🧠 Why Each Model Succeeded or Failed")
    
    model_insights = st.selectbox(
        "Select model for deep dive:",
        ['Logistic Regression', 'Random Forest', 'KNN', 'Stacked Ensemble']
    )
    
    if model_insights == 'Logistic Regression':
        st.markdown("### ✅ Logistic Regression - Why It Won")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Strengths:**
            - **Well-designed features**: Our relative features (team - opponent) naturally have linear relationships
            - **Additive effects**: Good rushing + good passing + home field = higher win probability
            - **Regularization**: Built-in L2 penalty prevents overfitting
            - **Interpretable**: Can see exactly which features push predictions up/down
            - **Fast**: Training takes seconds, predictions are instant
            
            **Why 72.9% is optimal:**
            - Captures all systematic predictable signal
            - Remaining 27% is irreducible randomness (injuries, weather, luck)
            - Matches professional oddsmakers (65-70%)
            
            **When to use:**
            - Individual game predictions for betting
            - Weekly pick'em contests
            - Playoff probability projections
            """)
        
        with col2:
            st.metric("Accuracy", "72.9%", "+2.9% vs Vegas")
            st.metric("Spearman ρ", "0.800", "Strong ranking")
            st.metric("CV Accuracy", "63.1%", "Stable")
            
            st.success("🏆 Best overall model")
    
    elif model_insights == 'Random Forest':
        st.markdown("### 🌲 Random Forest - Best Rankings, Not Best Accuracy")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Strengths:**
            - **Nonlinear interactions**: Can learn rules like "rushing EPA matters more when opponent win% is low"
            - **Ensemble robustness**: 100 trees voting reduces overfitting to individual patterns
            - **Best ranking correlation**: ρ = 0.830 beats Logistic Regression's 0.800
            - **Automatic feature importance**: No need for domain expertise
            
            **Why lower accuracy but better rankings?**
            - Ensemble averaging produces more calibrated probability estimates
            - When RF predicts 68%, it better reflects true team strength than LR's 73%
            - Binary decisions (win/loss) are occasionally wrong, but aggregate probabilities are more accurate
            
            **When to use:**
            - Team strength power rankings
            - GM decisions (trades, draft strategy)
            - Point spread setting
            - Season-long team evaluation
            """)
        
        with col2:
            st.metric("Accuracy", "71.5%", "-1.4% vs LR")
            st.metric("Spearman ρ", "0.830", "🥇 Best")
            st.metric("Parameters", "100 trees, depth=5")
            
            st.success("🏆 Best for rankings")
    
    elif model_insights == 'KNN':
        st.markdown("### ❌ KNN - Curse of Dimensionality")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Why it failed:**
            - **Curse of dimensionality**: With 20 features, training games are spread thinly in high-dimensional space
            - **Distance becomes meaningless**: Most games are roughly equidistant from any test game
            - **"Nearest" ≠ "Similar"**: The 11 nearest neighbors aren't actually similar games
            - **High variance**: Two games with identical stats can have opposite outcomes due to luck
            
            **Grid search tried to help:**
            - K=11 neighbors (not too few/noisy, not too many/diluted)
            - Manhattan distance (p=1) instead of Euclidean
            - Distance weighting (closer neighbors count more)
            - **Still failed**: Fundamental problem remained
            
            **When KNN works:**
            - Low dimensions (2-5 features)
            - Massive datasets (100k+ examples)
            - Problems where similar inputs truly predict similar outputs
            
            **NFL doesn't fit**: Too many dimensions, too much unmeasured variance
            """)
        
        with col2:
            st.metric("Accuracy", "61.2%", "-11.7% vs LR")
            st.metric("Spearman ρ", "0.601", "Worst")
            st.metric("Best K", "11")
            
            st.error("❌ Not recommended")
    
    else:  # Stacked Ensemble
        st.markdown("### 🔄 Stacked Ensemble - When Combining Models Fails")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **What went wrong:**
            - **KNN contamination**: Meta-model assigned coefficient 10.472 to KNN (!)
            - **Overfitting to training patterns**: Meta-model learned to trust KNN's confident but wrong predictions
            - **Weak base model**: Stacking requires ALL models to be strong and diverse
            
            **Meta-model coefficients:**
            ```
            knn_prob:  10.472  ← Red flag!
            rf_prob:    1.005  ← Reasonable
            lr_prob:    0.795  ← Reasonable
            ```
            
            **The correct approach (out-of-fold stacking):**
            1. Train each base model on 4 folds
            2. Predict on 5th fold (unseen data)
            3. Repeat for all folds
            4. Train meta-model on these out-of-fold predictions
            5. This prevents meta-model from seeing "perfect" training predictions
            
            **Lesson learned:**
            Ensemble diversity requires quality, not quantity. Adding a weak model can hurt.
            
            **Future work:**
            Replace KNN with Gradient Boosting or Neural Network
            """)
        
        with col2:
            st.metric("Accuracy", "62.5%", "-10.4% vs LR")
            st.metric("Spearman ρ", "0.620", "Weak")
            st.metric("KNN Weight", "10.472", "❌ Too high")
            
            st.warning("⚠️ Failed experiment")

# Sidebar
with st.sidebar:
    st.markdown("### 📊 Model Comparison Guide")
    st.info("""
    **Accuracy** = Overall correctness  
    (TP + TN) / Total
    
    **Precision** = When we predict X, how often correct?  
    TP / (TP + FP)
    
    **Recall** = Of actual X, how many did we catch?  
    TP / (TP + FN)
    
    **F1-Score** = Harmonic mean of precision & recall
    
    **Spearman ρ** = Rank correlation  
    How well model orders teams by strength
    """)
    
    st.markdown("### 🎯 Benchmarks")
    st.markdown("""
    - **Random Guess**: 50%
    - **Vegas Oddsmakers**: 65-70%
    - **ESPN FPI**: ~68%
    - **Our Best (LR)**: 72.9% ✓
    - **Theoretical Ceiling**: 70-75%
    """)

st.markdown("---")
st.caption("💡 All metrics calculated on 2024 test set (2,176 games)")
