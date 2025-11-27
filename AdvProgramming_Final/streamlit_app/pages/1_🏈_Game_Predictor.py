"""
Page 1: Game Predictor
Interactive NFL game prediction interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import plotly.graph_objects as go

st.set_page_config(page_title="Game Predictor", page_icon="🏈", layout="wide")

st.title("🏈 NFL Game Predictor")
st.markdown("Select two teams and predict the outcome using our trained models!")

st.markdown("---")

# NFL teams (all 32)
NFL_TEAMS = [
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
    'LAC', 'LA', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
    'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB', 'TEN', 'WAS'
]

TEAM_NAMES = {
    'ARI': 'Arizona Cardinals', 'ATL': 'Atlanta Falcons', 'BAL': 'Baltimore Ravens',
    'BUF': 'Buffalo Bills', 'CAR': 'Carolina Panthers', 'CHI': 'Chicago Bears',
    'CIN': 'Cincinnati Bengals', 'CLE': 'Cleveland Browns', 'DAL': 'Dallas Cowboys',
    'DEN': 'Denver Broncos', 'DET': 'Detroit Lions', 'GB': 'Green Bay Packers',
    'HOU': 'Houston Texans', 'IND': 'Indianapolis Colts', 'JAX': 'Jacksonville Jaguars',
    'KC': 'Kansas City Chiefs', 'LAC': 'Los Angeles Chargers', 'LA': 'Los Angeles Rams',
    'LV': 'Las Vegas Raiders', 'MIA': 'Miami Dolphins', 'MIN': 'Minnesota Vikings',
    'NE': 'New England Patriots', 'NO': 'New Orleans Saints', 'NYG': 'New York Giants',
    'NYJ': 'New York Jets', 'PHI': 'Philadelphia Eagles', 'PIT': 'Pittsburgh Steelers',
    'SF': 'San Francisco 49ers', 'SEA': 'Seattle Seahawks', 'TB': 'Tampa Bay Buccaneers',
    'TEN': 'Tennessee Titans', 'WAS': 'Washington Commanders'
}

# Sample team statistics (you'll replace with actual data)
TEAM_STATS = {
    'BUF': {'win_pct': 0.882, 'rushing_epa': 0.12, 'passing_epa': 0.18, 'turnovers': 0.8},
    'KC': {'win_pct': 0.882, 'rushing_epa': 0.08, 'passing_epa': 0.15, 'turnovers': 0.9},
    'DET': {'win_pct': 0.882, 'rushing_epa': 0.15, 'passing_epa': 0.12, 'turnovers': 1.0},
    'PHI': {'win_pct': 0.824, 'rushing_epa': 0.10, 'passing_epa': 0.14, 'turnovers': 0.7},
    'MIN': {'win_pct': 0.765, 'rushing_epa': 0.05, 'passing_epa': 0.16, 'turnovers': 0.6},
    # Add more teams... (simplified for example)
}

# For teams not in sample data, use league average
LEAGUE_AVG = {'win_pct': 0.500, 'rushing_epa': 0.00, 'passing_epa': 0.00, 'turnovers': 1.0}

def get_team_stats(team):
    return TEAM_STATS.get(team, LEAGUE_AVG)

# Layout: Two columns for team selection
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏠 Home Team")
    home_team = st.selectbox(
        "Select Home Team",
        NFL_TEAMS,
        format_func=lambda x: f"{x} - {TEAM_NAMES[x]}",
        key='home'
    )
    home_stats = get_team_stats(home_team)
    
    st.markdown(f"**{TEAM_NAMES[home_team]}**")
    st.metric("Win %", f"{home_stats['win_pct']:.1%}")
    st.metric("Rushing EPA", f"{home_stats['rushing_epa']:.3f}")
    st.metric("Passing EPA", f"{home_stats['passing_epa']:.3f}")

with col2:
    st.subheader("✈️ Away Team")
    away_team = st.selectbox(
        "Select Away Team",
        [t for t in NFL_TEAMS if t != home_team],
        format_func=lambda x: f"{x} - {TEAM_NAMES[x]}",
        key='away'
    )
    away_stats = get_team_stats(away_team)
    
    st.markdown(f"**{TEAM_NAMES[away_team]}**")
    st.metric("Win %", f"{away_stats['win_pct']:.1%}")
    st.metric("Rushing EPA", f"{away_stats['rushing_epa']:.3f}")
    st.metric("Passing EPA", f"{away_stats['passing_epa']:.3f}")

st.markdown("---")

# Advanced Options (Expander)
with st.expander("⚙️ Advanced Options - Adjust Team Stats"):
    st.markdown("**Adjust Home Team Stats:**")
    col1, col2 = st.columns(2)
    
    with col1:
        home_win_adj = st.slider("Home Team Win %", 0.0, 1.0, home_stats['win_pct'], 0.05)
        home_rush_adj = st.slider("Home Rushing EPA", -0.3, 0.3, home_stats['rushing_epa'], 0.01)
    
    with col2:
        home_pass_adj = st.slider("Home Passing EPA", -0.3, 0.3, home_stats['passing_epa'], 0.01)
        home_to_adj = st.slider("Home Turnovers/Game", 0.0, 3.0, home_stats['turnovers'], 0.1)
    
    st.markdown("**Adjust Away Team Stats:**")
    col1, col2 = st.columns(2)
    
    with col1:
        away_win_adj = st.slider("Away Team Win %", 0.0, 1.0, away_stats['win_pct'], 0.05)
        away_rush_adj = st.slider("Away Rushing EPA", -0.3, 0.3, away_stats['rushing_epa'], 0.01)
    
    with col2:
        away_pass_adj = st.slider("Away Passing EPA", -0.3, 0.3, away_stats['passing_epa'], 0.01)
        away_to_adj = st.slider("Away Turnovers/Game", 0.0, 3.0, away_stats['turnovers'], 0.1)

if'home' in home_stats:
    home_team = home_stats['home']
    home_win_adj = home_stats['win_pct']
    home_rush_adj = home_stats['rushing_epa']
    home_pass_adj = home_stats['passing_epa']
    home_to_adj = home_stats['turnovers']
    
    away_win_adj = away_stats['win_pct']
    away_rush_adj = away_stats['rushing_epa']
    away_pass_adj = away_stats['passing_epa']
    away_to_adj = away_stats['turnovers']

# Prediction Button
if st.button("🎯 Predict Game Outcome", type="primary", use_container_width=True):
    st.markdown("---")
    st.subheader("📊 Prediction Results")
    
    # Simulate predictions (replace with actual model predictions)
    # In real implementation, you'd load your models and use actual features
    
    # Simple logistic-style calculation for demo
    home_advantage = 0.58  # ~58% home win rate historically
    
    # Calculate relative strength
    win_diff = home_win_adj - away_win_adj
    rush_diff = home_rush_adj - away_rush_adj
    pass_diff = home_pass_adj - away_pass_adj
    to_diff = away_to_adj - home_to_adj  # Higher opponent turnovers = good
    
    # Weighted combination (simplified)
    base_prob = home_advantage + (win_diff * 0.4) + (rush_diff * 0.15) + (pass_diff * 0.10) + (to_diff * 0.05)
    
    # Clip to [0.15, 0.85] to be realistic
    lr_prob = np.clip(base_prob, 0.15, 0.85)
    rf_prob = np.clip(base_prob * 0.95 + np.random.uniform(-0.03, 0.03), 0.15, 0.85)
    knn_prob = np.clip(base_prob * 0.85 + np.random.uniform(-0.08, 0.08), 0.15, 0.85)
    
    # Display predictions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Logistic Regression")
        st.metric(
            label=f"{home_team} Win Probability",
            value=f"{lr_prob:.1%}",
            delta=f"{lr_prob - 0.5:.1%} vs 50/50"
        )
        if lr_prob > 0.5:
            st.success(f"✅ Predicts {home_team} wins")
        else:
            st.error(f"❌ Predicts {away_team} wins")
    
    with col2:
        st.markdown("### Random Forest")
        st.metric(
            label=f"{home_team} Win Probability",
            value=f"{rf_prob:.1%}",
            delta=f"{rf_prob - 0.5:.1%} vs 50/50"
        )
        if rf_prob > 0.5:
            st.success(f"✅ Predicts {home_team} wins")
        else:
            st.error(f"❌ Predicts {away_team} wins")
    
    with col3:
        st.markdown("### KNN")
        st.metric(
            label=f"{home_team} Win Probability",
            value=f"{knn_prob:.1%}",
            delta=f"{knn_prob - 0.5:.1%} vs 50/50"
        )
        if knn_prob > 0.5:
            st.success(f"✅ Predicts {home_team} wins")
        else:
            st.error(f"❌ Predicts {away_team} wins")
    
    # Visualization
    st.markdown("---")
    st.subheader("📈 Visual Comparison")
    
    # Create probability gauge chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Logistic Regression',
        x=['Home Win', 'Away Win'],
        y=[lr_prob, 1 - lr_prob],
        marker_color=['#1f77b4', '#ff7f0e']
    ))
    
    fig.add_trace(go.Bar(
        name='Random Forest',
        x=['Home Win', 'Away Win'],
        y=[rf_prob, 1 - rf_prob],
        marker_color=['#2ca02c', '#d62728']
    ))
    
    fig.add_trace(go.Bar(
        name='KNN',
        x=['Home Win', 'Away Win'],
        y=[knn_prob, 1 - knn_prob],
        marker_color=['#9467bd', '#8c564b']
    ))
    
    fig.update_layout(
        title=f"Win Probability: {home_team} (Home) vs {away_team} (Away)",
        yaxis_title="Probability",
        barmode='group',
        height=400,
        yaxis=dict(tickformat='.0%')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Consensus
    st.markdown("---")
    st.subheader("🏆 Model Consensus")
    
    avg_prob = (lr_prob + rf_prob + knn_prob) / 3
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if avg_prob > 0.6:
            st.success(f"### Strong Consensus: {home_team} favored")
            st.markdown(f"All models agree {home_team} is likely to win (avg: {avg_prob:.1%})")
        elif avg_prob < 0.4:
            st.error(f"### Strong Consensus: {away_team} favored")
            st.markdown(f"All models agree {away_team} is likely to win (avg: {1-avg_prob:.1%})")
        else:
            st.warning("### Close Game - Too Close to Call")
            st.markdown(f"Models are divided. This is a toss-up game.")
    
    with col2:
        st.metric("Average Probability", f"{avg_prob:.1%}")
        st.metric("Confidence Spread", f"{max(lr_prob, rf_prob, knn_prob) - min(lr_prob, rf_prob, knn_prob):.1%}")

st.markdown("---")

# Sidebar
with st.sidebar:
    st.markdown("### 🎯 How Predictions Work")
    st.info("""
    The models use 20 features including:
    - Historical win percentage
    - Opponent win percentage  
    - Rushing EPA (efficiency)
    - Passing EPA
    - Turnover differential
    - Home field advantage
    
    Each model uses different algorithms:
    - **LR**: Linear combination of features
    - **RF**: 100 decision trees voting
    - **KNN**: 11 similar historical games
    """)
    
    st.markdown("### 📌 Tips")
    st.markdown("""
    - Use **Advanced Options** to simulate different scenarios
    - Adjust win% to see impact of recent form
    - Modify EPA to test offensive improvements
    - Higher turnover rate = worse performance
    """)

# Footer
st.markdown("---")
st.caption("💡 **Note**: Predictions are based on historical data and statistical models. Actual game outcomes depend on many unmeasured factors like injuries, weather, and coaching decisions.")
