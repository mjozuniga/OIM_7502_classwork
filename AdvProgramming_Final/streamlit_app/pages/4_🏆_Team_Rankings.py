"""
Page 4: Team Rankings
Compare predicted vs actual team rankings across all models
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(page_title="Team Rankings", page_icon="🏆", layout="wide")

st.title("🏆 NFL Team Rankings 2024")
st.markdown("How did each model rank teams? Who were the biggest surprises?")

st.markdown("---")

# Team rankings data (from your actual results - page 21-22 of PDF)
rankings_data = {
    'Team': ['PHI', 'KC', 'DET', 'MIN', 'BUF', 'WAS', 'BAL', 'LAC', 'GB', 'SEA', 'HOU', 'LA',
             'TB', 'PIT', 'DEN', 'CIN', 'IND', 'ATL', 'MIA', 'ARI', 'DAL', 'SF', 'CHI', 'NYJ',
             'NO', 'CAR', 'NE', 'LV', 'JAX', 'CLE', 'NYG', 'TEN'],
    'Actual': list(range(1, 33)),
    'Logistic': [3, 5, 4, 1, 2, 18, 12, 10, 16, 6, 12, 19, 11, 14, 17, 16, 20, 28, 7, 21, 9, 22, 24, 26, 8, 20, 23, 29, 25, 30, 31, 32],
    'RandomForest': [5, 3, 2, 4, 1, 13, 10, 12, 16, 6, 14, 20, 8, 14, 22, 7, 18, 26, 9, 21, 15, 23, 29, 27, 10, 24, 25, 28, 25, 31, 32, 30],
    'KNN': [10, 5, 1, 7, 2, 19, 13, 20, 21, 29, 11, 14, 12, 15, 27, 4, 17, 15, 3, 22, 18, 8, 24, 28, 8, 23, 24, 16, 26, 30, 29, 32]
}

df = pd.DataFrame(rankings_data)

# Calculate rank differences
df['LR_Diff'] = df['Logistic'] - df['Actual']
df['RF_Diff'] = df['RandomForest'] - df['Actual']
df['KNN_Diff'] = df['KNN'] - df['Actual']

# Spearman correlations (from your results)
spearman_correlations = {
    'Logistic': 0.800,
    'RandomForest': 0.830,
    'KNN': 0.601
}

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["📊 Rankings Comparison", "🎯 Biggest Surprises", "📈 Visual Analysis", "🏅 Model Performance"])

## TAB 1: Rankings Comparison
with tab1:
    st.subheader("Team Rankings: All Models vs Actual")
    
    # Model selector
    model_to_show = st.multiselect(
        "Select models to display:",
        ['Actual', 'Logistic', 'RandomForest', 'KNN'],
        default=['Actual', 'Logistic', 'RandomForest']
    )
    
    # Create display dataframe
    display_cols = ['Team'] + model_to_show
    df_display = df[display_cols].copy()
    
    # Sort by actual rank
    df_display = df_display.sort_values('Actual' if 'Actual' in model_to_show else model_to_show[0])
    
    # Style the dataframe
    def highlight_rank(val):
        if val <= 7:  # Playoff teams
            return 'background-color: #90EE90'
        elif val <= 14:
            return 'background-color: #FFE5B4'
        elif val >= 25:
            return 'background-color: #FFB6C6'
        return ''
    
    styled_df = df_display.style.applymap(highlight_rank, subset=[col for col in display_cols if col != 'Team'])
    
    st.dataframe(styled_df, use_container_width=True, height=600)
    
    st.caption("🟢 Green = Playoff teams (1-7) | 🟡 Orange = Middle (8-14) | 🔴 Pink = Bottom (25-32)")
    
    # Summary stats
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Logistic Regression",
            f"ρ = {spearman_correlations['Logistic']:.3f}",
            "Good correlation"
        )
    
    with col2:
        st.metric(
            "Random Forest",
            f"ρ = {spearman_correlations['RandomForest']:.3f}",
            "🥇 Best correlation"
        )
    
    with col3:
        st.metric(
            "KNN",
            f"ρ = {spearman_correlations['KNN']:.3f}",
            "Weak correlation"
        )

## TAB 2: Biggest Surprises
with tab2:
    st.subheader("🎯 Biggest Prediction Surprises")
    
    model_select = st.selectbox(
        "Select model to analyze:",
        ['Logistic', 'RandomForest', 'KNN']
    )
    
    diff_col = f'{model_select}_Diff'
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"### 📈 Biggest Overestimates ({model_select})")
        st.caption("Teams model ranked too high (predicted better than actual)")
        
        overestimates = df.nsmallest(5, diff_col)[['Team', 'Actual', model_select, diff_col]]
        
        for _, row in overestimates.iterrows():
            st.error(
                f"**{row['Team']}**: Predicted #{int(row[model_select])} but actually #{int(row['Actual'])} "
                f"(off by {-int(row[diff_col])} spots)"
            )
    
    with col2:
        st.markdown(f"### 📉 Biggest Underestimates ({model_select})")
        st.caption("Teams model ranked too low (predicted worse than actual)")
        
        underestimates = df.nlargest(5, diff_col)[['Team', 'Actual', model_select, diff_col]]
        
        for _, row in underestimates.iterrows():
            st.success(
                f"**{row['Team']}**: Predicted #{int(row[model_select])} but actually #{int(row['Actual'])} "
                f"(off by {int(row[diff_col])} spots)"
            )
    
    st.markdown("---")
    
    # Explanation of surprises
    st.markdown("### 🤔 Why Do Surprises Happen?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **Overestimates (Model too optimistic):**
        
        Teams the model thought would be good, but weren't:
        - Mid-season QB injury or decline
        - Roster turnover from previous year
        - Coaching changes didn't work out
        - Schedule strength miscalculation
        - Defensive regression
        
        Example: If NO was predicted #8 but finished #25, they likely had key injuries or unexpected decline.
        """)
    
    with col2:
        st.info("""
        **Underestimates (Model too pessimistic):**
        
        Teams that exceeded expectations:
        - Breakout young QB/players
        - New coaching success
        - Easier schedule than expected
        - Good health luck
        - Better team chemistry
        
        Example: If DET was predicted #4 but finished #1, they likely had breakout performances or exceeded prior trends.
        """)

## TAB 3: Visual Analysis
with tab3:
    st.subheader("📈 Ranking Comparison Visualizations")
    
    # Bar chart: Predicted vs Actual for all teams
    model_viz = st.radio(
        "Select model:",
        ['Logistic', 'RandomForest', 'KNN'],
        horizontal=True
    )
    
    # Sort by actual rank
    df_viz = df.sort_values('Actual')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Actual Rank',
        x=df_viz['Team'],
        y=df_viz['Actual'],
        marker_color='orange',
        opacity=0.7
    ))
    
    fig.add_trace(go.Bar(
        name=f'{model_viz} Predicted Rank',
        x=df_viz['Team'],
        y=df_viz[model_viz],
        marker_color='steelblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title=f"{model_viz}: Predicted vs Actual Rankings (Spearman ρ = {spearman_correlations[model_viz]:.3f})",
        yaxis_title="Rank Position",
        xaxis_title="Teams (ordered by actual standings)",
        barmode='group',
        height=500,
        yaxis=dict(autorange="reversed")  # Lower rank = better
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot: Predicted vs Actual
    st.markdown("---")
    st.subheader("Predicted Rank vs Actual Rank")
    
    fig2 = px.scatter(
        df,
        x=model_viz,
        y='Actual',
        text='Team',
        title=f"{model_viz} Predicted Rank vs Actual Rank",
        labels={
            model_viz: 'Predicted Rank',
            'Actual': 'Actual Rank'
        }
    )
    
    # Add diagonal line (perfect prediction)
    fig2.add_trace(go.Scatter(
        x=[1, 32],
        y=[1, 32],
        mode='lines',
        name='Perfect Prediction',
        line=dict(dash='dash', color='gray')
    ))
    
    fig2.update_traces(textposition='top center', marker=dict(size=12))
    fig2.update_layout(height=600)
    fig2.update_xaxes(range=[0, 33], dtick=5)
    fig2.update_yaxes(range=[0, 33], dtick=5, autorange="reversed")
    
    st.plotly_chart(fig2, use_container_width=True)
    
    st.caption("Points closer to the diagonal line = better predictions")

## TAB 4: Model Performance
with tab4:
    st.subheader("🏅 Ranking Performance by Model")
    
    # Comparison metrics
    st.markdown("### 📊 Spearman Rank Correlation")
    
    corr_df = pd.DataFrame({
        'Model': ['Random Forest', 'Logistic Regression', 'KNN'],
        'Spearman ρ': [0.830, 0.800, 0.601],
        'Interpretation': ['Excellent', 'Very Good', 'Moderate']
    })
    
    fig = go.Figure(go.Bar(
        x=corr_df['Model'],
        y=corr_df['Spearman ρ'],
        marker_color=['#2ca02c', '#1f77b4', '#ff7f0e'],
        text=corr_df['Spearman ρ'].apply(lambda x: f'{x:.3f}'),
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Spearman Rank Correlation (Higher = Better)",
        yaxis_title="Spearman ρ",
        height=400,
        yaxis_range=[0, 1]
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Interpretation
    st.markdown("---")
    st.markdown("### 📖 What Does Spearman ρ Mean?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **Random Forest (ρ = 0.830)**
        
        Best at ordering teams by strength.
        
        Correctly ranks 83% of team-pair comparisons.
        
        **Why it wins:**
        - Ensemble voting smooths predictions
        - Better probability calibration
        - Captures team strength nuances
        
        **Use for**: Power rankings, GM decisions
        """)
    
    with col2:
        st.info("""
        **Logistic Regression (ρ = 0.800)**
        
        Very good at rankings, best at game-level accuracy.
        
        Correctly ranks 80% of team-pair comparisons.
        
        **Why it's close:**
        - Linear model still captures most signal
        - Well-designed features
        - Simple and interpretable
        
        **Use for**: Game predictions, betting
        """)
    
    with col3:
        st.warning("""
        **KNN (ρ = 0.601)**
        
        Struggles to rank teams accurately.
        
        Only correctly ranks 60% of team-pair comparisons.
        
        **Why it fails:**
        - Curse of dimensionality
        - Distance metrics unreliable
        - High variance in predictions
        
        **Use for**: Nothing (failed experiment)
        """)
    
    # Mean Absolute Error in ranks
    st.markdown("---")
    st.markdown("### 📏 Mean Absolute Ranking Error")
    
    mae_lr = df['LR_Diff'].abs().mean()
    mae_rf = df['RF_Diff'].abs().mean()
    mae_knn = df['KNN_Diff'].abs().mean()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Logistic Regression",
            f"{mae_lr:.1f} positions",
            f"Average error"
        )
    
    with col2:
        st.metric(
            "Random Forest",
            f"{mae_rf:.1f} positions",
            f"Average error"
        )
    
    with col3:
        st.metric(
            "KNN",
            f"{mae_knn:.1f} positions",
            f"Average error"
        )
    
    st.caption("Lower is better. This measures average number of positions off in rankings.")

# Sidebar
with st.sidebar:
    st.markdown("### 🏆 Ranking Metrics Explained")
    st.info("""
    **Spearman's ρ (Rho):**
    Measures how well rankings correlate.
    
    - ρ = 1.0: Perfect ranking
    - ρ = 0.8-0.9: Excellent
    - ρ = 0.6-0.8: Good
    - ρ < 0.6: Weak
    
    **Rank Difference:**
    Predicted rank - Actual rank
    
    - Negative: Model overestimated
    - Positive: Model underestimated
    - Zero: Perfect prediction
    """)
    
    st.markdown("### 📊 Model Comparison")
    st.markdown("""
    **Random Forest wins** at rankings:
    - ρ = 0.830 (best)
    - Better probability calibration
    - Superior for power rankings
    
    **Logistic Regression** close second:
    - ρ = 0.800
    - Best game-level accuracy (72.9%)
    - Better for betting applications
    
    **KNN fails** at rankings:
    - ρ = 0.601 (worst)
    - High-dimensional curse
    - Not recommended
    """)

st.markdown("---")
st.caption("💡 **Key Insight**: Random Forest's ensemble approach produces better-calibrated team strength estimates, making it superior for rankings despite slightly lower game accuracy.")
