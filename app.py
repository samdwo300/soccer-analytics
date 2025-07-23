import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Serie A Value Analysis", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("player_team_salary_stats_2024-2025_SerieA.csv")

df = load_data()

st.title("Serie A 2024-25: Player Analytics Dashboard")

# Sidebar filters (as before)
teams = df['Squad'].dropna().unique()
selected_teams = st.sidebar.multiselect('Filter by Team:', sorted(teams), default=sorted(teams))

filtered = df[df['Squad'].isin(selected_teams)].copy()
if 'Salary_M_USD' in filtered.columns:
    min_salary, max_salary = float(filtered['Salary_M_USD'].min()), float(filtered['Salary_M_USD'].max())
    salary_range = st.sidebar.slider('Filter by Salary (Million USD):', min_value=float(min_salary), max_value=float(max_salary), value=(min_salary, max_salary), step=0.1)
    filtered = filtered[(filtered['Salary_M_USD'] >= salary_range[0]) & (filtered['Salary_M_USD'] <= salary_range[1])]
min_shots = st.sidebar.slider("Minimum Shots", min_value=int(filtered['Shots'].min()), max_value=int(filtered['Shots'].max()), value=20)
filtered = filtered[filtered['Shots'] >= min_shots]
# Games played filter (new)
if 'Playing Time_MP' in filtered.columns:
    min_matches = int(filtered['Playing Time_MP'].min())
    max_matches = int(filtered['Playing Time_MP'].max())
    games_played = st.sidebar.slider(
        "Minimum Games Played",
        min_value=min_matches,
        max_value=max_matches,
        value=min_matches
    )
    filtered = filtered[filtered['Playing Time_MP'] >= games_played]
# Minimum total goals filter
if 'Goals' in filtered.columns:
    min_goals = int(filtered['Goals'].min())
    max_goals = int(filtered['Goals'].max())
    min_total_goals = st.sidebar.slider(
        "Minimum Total Goals",
        min_value=min_goals,
        max_value=max_goals,
        value=min_goals
    )
    filtered = filtered[filtered['Goals'] >= min_total_goals]


# --- TABS ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "💵 Value-for-Money", 
    "⚡ Performance Only", 
    "📊 Team Barplot",
    "🔥 Heatmaps",
    "📦 Boxplots",
    "🏆 Top Scorers & Out/Underperformers",
    "🏟️ Team Profile",
    "💰 Wage Bill vs. Goals",
    "🎂 Age Distributions"
])

with tab1:
    st.subheader("Interactive Value-for-Money Scatter (Hover for Details)")
    fig = px.scatter(
        filtered,
        x='xG_per_shot',
        y='Shot_Conversion',
        size='Salary_M_USD',
        color='Value_Score',
        color_continuous_scale='RdYlGn',
        size_max=50,
        hover_data={
            'Player': True,
            'Squad': True,
            'Salary_M_USD': ':.2f',
            'Goals': True,
            'xG': ':.2f',
            'G_minus_xG': ':.2f',
            'Value_Score': ':.2f'
        },
        labels={
            'xG_per_shot': 'xG per Shot (Chance Quality)',
            'Shot_Conversion': 'Shot Conversion Rate',
            'Salary_M_USD': 'Salary ($M)',
            'Value_Score': 'Value Score (G-xG / $M)',
            'G_minus_xG': 'G - xG'
        },
        title='Serie A Player Value-for-Money: Interactive Hover Plot'
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.add_vline(x=filtered['xG_per_shot'].mean(), line_dash="dash", line_color="red", annotation_text="Avg xG/Shot")
    fig.add_hline(y=filtered['Shot_Conversion'].mean(), line_dash="dash", line_color="blue", annotation_text="Avg Shot Conv.")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Performance: Conversion Rate vs. Chance Quality (No Salary)")
    perf_fig = px.scatter(
        filtered,
        x='xG_per_shot',
        y='Shot_Conversion',
        size='Shots',
        color='G_minus_xG',
        color_continuous_scale='RdYlGn',
        size_max=50,
        hover_data={
            'Player': True,
            'Squad': True,
            'Goals': True,
            'xG': ':.2f',
            'G_minus_xG': ':.2f',
            'Shots': True,
        },
        labels={
            'xG_per_shot': 'xG per Shot (Chance Quality)',
            'Shot_Conversion': 'Shot Conversion Rate',
            'Shots': 'Total Shots',
            'G_minus_xG': 'G - xG'
        },
        title='Serie A Players: Conversion vs. Chance Quality vs. Over/Underperformance'
    )
    perf_fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    perf_fig.add_vline(x=filtered['xG_per_shot'].mean(), line_dash="dash", line_color="red", annotation_text="Avg xG/Shot")
    perf_fig.add_hline(y=filtered['Shot_Conversion'].mean(), line_dash="dash", line_color="blue", annotation_text="Avg Shot Conv.")
    st.plotly_chart(perf_fig, use_container_width=True)

with tab3:
    st.subheader("Team Total Goals: Barplot")
    team_goals = filtered.groupby('Squad')['Goals'].sum().sort_values(ascending=False).head(15)
    team_fig = px.bar(
        team_goals,
        x=team_goals.index,
        y=team_goals.values,
        labels={'x': 'Team', 'y': 'Total Goals'},
        title='Top 15 Teams by Total Goals'
    )
    st.plotly_chart(team_fig, use_container_width=True)

with tab4:
    st.subheader("Dynamic Heatmap: Explore Any Metric Across Categories")

    # Let user select X, Y, and value columns
    categorical_cols = [col for col in filtered.columns if filtered[col].dtype == 'O' and filtered[col].nunique() < 25 and col not in ['Player']]
    numeric_cols = [col for col in filtered.columns if pd.api.types.is_numeric_dtype(filtered[col]) and filtered[col].notnull().sum() > 0]

    if not categorical_cols or not numeric_cols:
        st.warning("Not enough suitable columns to make a heatmap.")
    else:
        x_axis = st.selectbox("X-Axis (Category)", options=categorical_cols, index=0)
        y_axis = st.selectbox("Y-Axis (Category)", options=[col for col in categorical_cols if col != x_axis], index=0)
        value_col = st.selectbox("Value (Numeric/Metric)", options=numeric_cols, index=numeric_cols.index('Goals') if 'Goals' in numeric_cols else 0)
        agg_func = st.selectbox("Aggregation", options=['sum','mean','median','count'], index=0)

        pivot = pd.pivot_table(
            filtered,
            index=y_axis,
            columns=x_axis,
            values=value_col,
            aggfunc=agg_func,
            fill_value=0
        )

        fig, ax = plt.subplots(figsize=(min(16,2+1.1*len(pivot.columns)),min(10,2+0.6*len(pivot.index))))
        sns.heatmap(pivot, annot=True, fmt=".1f" if agg_func != 'count' else 'd', cmap="YlGnBu", ax=ax)
        ax.set_title(f"{agg_func.capitalize()} of {value_col} by {y_axis} and {x_axis}")
        st.pyplot(fig)
        st.write("Tip: Try Team (Squad) vs. Position, or Team vs. Age Group, etc.")


with tab5:
    st.subheader("Boxplots: Value Score & Finishing by Team")
    fig, ax = plt.subplots(figsize=(12,6))
    # Show top 8 teams with most players
    top_teams = filtered['Squad'].value_counts().head(8).index
    sns.boxplot(data=filtered[filtered['Squad'].isin(top_teams)], x='Squad', y='Value_Score', ax=ax, palette='RdYlGn')
    ax.set_title("Distribution of Value Score by Team (Top 8 by player count)")
    st.pyplot(fig)
    st.write("Boxplot: G-xG by Position")
    if 'Pos' in filtered.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.boxplot(data=filtered, x='Pos', y='G_minus_xG', ax=ax, palette='coolwarm')
        ax.set_title("Distribution of G-xG by Position")
        st.pyplot(fig)

with tab6:
    st.subheader("Top 10 Scorers")
    top_scorers = filtered.sort_values('Goals', ascending=False).head(10)
    fig = px.bar(
        top_scorers,
        x='Player',
        y='Goals',
        color='Squad',
        labels={'Goals':'Goals'},
        title='Top 10 Scorers'
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 10 Overperformers (G-xG)")
    top_over = filtered.sort_values('G_minus_xG', ascending=False).head(10)
    fig2 = px.bar(
        top_over,
        x='Player',
        y='G_minus_xG',
        color='Squad',
        labels={'G_minus_xG':'G - xG'},
        title='Top 10 Overperformers'
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Top 10 Underperformers (G-xG)")
    top_under = filtered.sort_values('G_minus_xG').head(10)
    fig3 = px.bar(
        top_under,
        x='Player',
        y='G_minus_xG',
        color='Squad',
        labels={'G_minus_xG':'G - xG'},
        title='Top 10 Underperformers'
    )
    st.plotly_chart(fig3, use_container_width=True)

with tab7:
    st.subheader("Team Profile: Detailed View")
    team_choice = st.selectbox("Choose a team:", sorted(df['Squad'].dropna().unique()))
    team_df = df[df['Squad'] == team_choice]

    if team_df.empty:
        st.warning("No data for this team.")
    else:
        st.markdown(f"### {team_choice} - Key Stats ({team_df.shape[0]} players)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Goals", int(team_df['Goals'].sum()))
            st.metric("Total Shots", int(team_df['Shots'].sum()))
            st.metric("Total Salary ($M)", f"{team_df['Salary_M_USD'].sum():.1f}")
        with col2:
            st.metric("Avg xG per Shot", f"{team_df['xG_per_shot'].mean():.3f}")
            st.metric("Avg Value Score", f"{team_df['Value_Score'].mean():.3f}")
            st.metric("Avg G-xG", f"{team_df['G_minus_xG'].mean():.2f}")
        with col3:
            st.metric("Top Scorer", team_df.sort_values('Goals', ascending=False)['Player'].iloc[0])
            st.metric("Top Overperformer (G-xG)", team_df.sort_values('G_minus_xG', ascending=False)['Player'].iloc[0])
            st.metric("Top Value", team_df.sort_values('Value_Score', ascending=False)['Player'].iloc[0])

        st.markdown("#### Player Table")
        st.dataframe(team_df[['Player','Pos','Goals','xG','Shots','G_minus_xG','xG_per_shot','Shot_Conversion','Salary_M_USD','Value_Score']].sort_values('Goals', ascending=False))

        st.markdown("#### Player Value-for-Money Plot")
        team_fig = px.scatter(
            team_df,
            x='xG_per_shot',
            y='Shot_Conversion',
            size='Salary_M_USD',
            color='Value_Score',
            color_continuous_scale='RdYlGn',
            size_max=50,
            hover_data=['Player', 'Goals', 'xG', 'Salary_M_USD', 'Value_Score'],
            labels={
                'xG_per_shot': 'xG per Shot',
                'Shot_Conversion': 'Conversion Rate',
                'Salary_M_USD': 'Salary ($M)',
                'Value_Score': 'Value Score'
            },
            title=f'{team_choice} Player Value-for-Money'
        )
        st.plotly_chart(team_fig, use_container_width=True)

        # --- Position breakdown ---
        if 'Pos' in team_df.columns:
            st.markdown("#### Position Breakdown (Team)")
            pos_summary = team_df.groupby('Pos').agg({
                'Goals': 'sum',
                'Salary_M_USD': 'sum',
                'Value_Score': 'mean'
            }).sort_values('Goals', ascending=False).reset_index()
            st.dataframe(pos_summary)
            pos_fig = px.bar(pos_summary, x='Pos', y='Goals', color='Value_Score',
                             color_continuous_scale='RdYlGn', title='Goals & Value Score by Position')
            st.plotly_chart(pos_fig, use_container_width=True)
        # --- Age distribution ---
        if 'Age' in team_df.columns:
            st.markdown("#### Age Distribution (Team)")
            fig_age = px.histogram(
                team_df,
                x='Age',
                nbins=10,
                title=f"{team_choice} Age Distribution",
                labels={'Age':'Player Age'}
            )
            st.plotly_chart(fig_age, use_container_width=True)
            st.write("Mean age:", round(team_df['Age'].mean(), 1))
            st.write("Median age:", round(team_df['Age'].median(), 1))

with tab8:
    st.subheader("Wage Bill vs. Goals Scored (League)")
    league_team = df.groupby('Squad').agg({
        'Salary_M_USD':'sum',
        'Goals':'sum'
    }).reset_index()
    fig = px.scatter(
        league_team,
        x='Salary_M_USD',
        y='Goals',
        text='Squad',
        color='Goals',
        size='Salary_M_USD',
        color_continuous_scale='RdYlGn',
        labels={'Salary_M_USD':'Team Wage Bill ($M)', 'Goals':'Goals'},
        title='Team Wage Bill vs. Goals Scored'
    )
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)
    st.write("**Who gets the most goals per $ spent?**")
    league_team['Goals_per_M'] = league_team['Goals'] / league_team['Salary_M_USD']
    st.dataframe(league_team.sort_values('Goals_per_M', ascending=False)[['Squad','Salary_M_USD','Goals','Goals_per_M']])

with tab9:
    st.subheader("Age Distribution (League)")
    if 'Age' in df.columns:
        fig = px.box(df, x='Squad', y='Age', points="all", title="Team Age Distributions")
        st.plotly_chart(fig, use_container_width=True)
