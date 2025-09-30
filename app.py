import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Bike Sharing Demand Analysis",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('datahour.csv')
    df['dteday'] = pd.to_datetime(df['dteday'])
    return df

@st.cache_resource
def load_models():
    models = {}
    out_dir = 'outputs'
    try:
        models['scaler'] = joblib.load(os.path.join(out_dir, 'scaler.joblib'))
        models['pca'] = joblib.load(os.path.join(out_dir, 'pca.joblib'))
        models['kmeans'] = joblib.load(os.path.join(out_dir, 'kmeans.joblib'))
        models['rf_final'] = joblib.load(os.path.join(out_dir, 'rf_final.joblib'))
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
    return models

# Load data
df = load_data()
models = load_models()

# Sidebar navigation
st.sidebar.title("🚲 Bike Sharing Analysis")
page = st.sidebar.selectbox(
    "Choose a section",
    ["Overview", "Exploratory Data Analysis", "Demand Prediction", "Clustering Analysis", "Marketing Insights"]
)

# Main content
st.title("🚲 Bike Sharing Demand Analysis Dashboard")
st.markdown("---")

if page == "Overview":
    st.header("📊 Data Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Date Range", f"{df['dteday'].min().date()} to {df['dteday'].max().date()}")
    with col3:
        st.metric("Average Daily Rentals", f"{df['cnt'].mean():.0f}")

    st.subheader("Dataset Information")
    st.dataframe(df.head(), use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)

    st.subheader("Data Types & Missing Values")
    info_df = pd.DataFrame({
        'Data Type': df.dtypes,
        'Missing Values': df.isnull().sum(),
        'Missing %': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(info_df, use_container_width=True)

elif page == "Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")

    # Interactive selection of plots to display
    plot_options = st.multiselect(
        "Select plots to display:",
        ["Distribution of Total Rentals", "Average Rentals by Hour and Weekday",
         "Rentals by Season", "Correlation Matrix", "Outliers Detection"],
        default=["Distribution of Total Rentals", "Average Rentals by Hour and Weekday"]
    )

    for plot in plot_options:
        if plot == "Distribution of Total Rentals":
            st.subheader("Distribution of Total Rentals (cnt)")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df['cnt'], bins=50, kde=True, ax=ax)
            ax.set_title('Distribution of Total Rentals')
            st.pyplot(fig)

        elif plot == "Average Rentals by Hour and Weekday":
            st.subheader("Average Rentals by Hour and Weekday")
            if 'hr' in df.columns and 'weekday' in df.columns:
                pivot = df.pivot_table(index='hr', columns='weekday', values='cnt', aggfunc='mean')
                fig, ax = plt.subplots(figsize=(12, 8))
                sns.heatmap(pivot, cmap='viridis', annot=True, fmt='.0f', ax=ax)
                ax.set_title('Average Rentals by Hour and Weekday')
                st.pyplot(fig)

        elif plot == "Rentals by Season":
            st.subheader("Rentals by Season")
            if 'season' in df.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x='season', y='cnt', data=df, ax=ax)
                ax.set_title('Rentals Distribution by Season')
                ax.set_xticklabels(['Spring', 'Summer', 'Fall', 'Winter'])
                st.pyplot(fig)

        elif plot == "Correlation Matrix":
            st.subheader("Correlation Matrix")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            corr_matrix = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Correlation Matrix')
            st.pyplot(fig)

        elif plot == "Outliers Detection":
            st.subheader("Outliers Detection (IQR Method)")
            Q1 = df['cnt'].quantile(0.25)
            Q3 = df['cnt'].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df['cnt'] < (Q1 - 1.5*IQR)) | (df['cnt'] > (Q3 + 1.5*IQR))]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(y=df['cnt'], ax=ax)
            ax.scatter([0] * len(outliers), outliers['cnt'], color='red', s=50, label='Outliers')
            ax.set_title(f'Outliers Detection (IQR): {len(outliers)} outliers')
            ax.legend()
            st.pyplot(fig)
            st.write(f"Number of outliers: {len(outliers)} ({len(outliers)/len(df):.2%})")

elif page == "Demand Prediction":
    st.header("🎯 Demand Prediction")

    st.markdown("Use the Random Forest model to predict hourly bike rentals based on environmental factors.")

    # Input features
    col1, col2, col3 = st.columns(3)

    with col1:
        season = st.selectbox("Season", [1, 2, 3, 4], format_func=lambda x: ['Spring', 'Summer', 'Fall', 'Winter'][x-1])
        yr = st.selectbox("Year", [0, 1], format_func=lambda x: ['2011', '2012'][x])
        mnth = st.slider("Month", 1, 12, 6)
        hr = st.slider("Hour", 0, 23, 12)

    with col2:
        holiday = st.selectbox("Holiday", [0, 1], format_func=lambda x: ['No', 'Yes'][x])
        weekday = st.slider("Weekday", 0, 6, 3)
        workingday = st.selectbox("Working Day", [0, 1], format_func=lambda x: ['No', 'Yes'][x])
        weathersit = st.selectbox("Weather Situation", [1, 2, 3, 4],
                                format_func=lambda x: ['Clear', 'Mist', 'Light Rain/Snow', 'Heavy Rain/Snow'][x-1])

    with col3:
        temp = st.slider("Temperature (normalized)", 0.0, 1.0, 0.5)
        atemp = st.slider("Feeling Temperature (normalized)", 0.0, 1.0, 0.5)
        hum = st.slider("Humidity (normalized)", 0.0, 1.0, 0.5)
        windspeed = st.slider("Windspeed (normalized)", 0.0, 1.0, 0.2)

    # Create feature array
    features = [season, yr, mnth, hr, holiday, weekday, workingday, weathersit, temp, atemp, hum, windspeed]

    # Add engineered features
    hr_sin = np.sin(2*np.pi*hr/24)
    hr_cos = np.cos(2*np.pi*hr/24)
    month_sin = np.sin(2*np.pi*mnth/12)
    month_cos = np.cos(2*np.pi*mnth/12)
    weekday_sin = np.sin(2*np.pi*weekday/7)
    weekday_cos = np.cos(2*np.pi*weekday/7)
    is_weekend = 1 if weekday in [0, 6] else 0

    # Add lag features (using averages as placeholders)
    cnt_t_1 = df['cnt'].mean()
    cnt_t_24 = df['cnt'].mean()

    full_features = features + [hr_sin, hr_cos, month_sin, month_cos, weekday_sin, weekday_cos, is_weekend, cnt_t_1, cnt_t_24]

    if st.button("Predict Demand", type="primary"):
        if 'rf_final' in models:
            # Scale and predict
            features_scaled = models['scaler'].transform([full_features])
            prediction = models['rf_final'].predict(features_scaled)[0]

            st.success(f"Predicted Hourly Bike Rentals: **{prediction:.0f}**")

            # Show feature importance
            st.subheader("Feature Importance")
            feat_importances = pd.Series(models['rf_final'].feature_importances_,
                                       index=['season', 'yr', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                                              'weathersit', 'temp', 'atemp', 'hum', 'windspeed',
                                              'hr_sin', 'hr_cos', 'month_sin', 'month_cos', 'weekday_sin', 'weekday_cos',
                                              'is_weekend', 'cnt_t_1', 'cnt_t_24'])
            fig, ax = plt.subplots(figsize=(10, 6))
            feat_importances.nlargest(10).plot(kind='barh', ax=ax)
            ax.set_title('Top 10 Feature Importances')
            st.pyplot(fig)
        else:
            st.error("Random Forest model not found. Please ensure models are trained and saved.")

elif page == "Clustering Analysis":
    st.header("📊 Clustering Analysis")

    st.markdown("KMeans clustering analysis for segmenting rental patterns.")

    # Interactive selection of plots to display
    cluster_plot_options = st.multiselect(
        "Select clustering plots to display:",
        ["Cluster Summary", "Average Demand by Cluster", "PCA 2D Visualization", "Cluster Distribution"],
        default=["Cluster Summary", "Average Demand by Cluster"]
    )

    # Load cluster summary if available
    cluster_file = os.path.join('outputs', 'cluster_summary.csv')
    if os.path.exists(cluster_file):
        cluster_summary = pd.read_csv(cluster_file)

        for plot in cluster_plot_options:
            if plot == "Cluster Summary":
                st.subheader("Cluster Summary")
                st.dataframe(cluster_summary, use_container_width=True)

            elif plot == "Average Demand by Cluster":
                st.subheader("Average Demand by Cluster")
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(cluster_summary['cluster'], cluster_summary['mean_cnt'])
                ax.set_xlabel('Cluster')
                ax.set_ylabel('Mean Demand')
                ax.set_title('Average Demand by Cluster')
                st.pyplot(fig)

            elif plot == "PCA 2D Visualization":
                st.subheader("PCA 2D Visualization of Clusters")
                if 'pca' in models and 'kmeans' in models:
                    # Recreate PCA 2D data
                    features = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit',
                               'temp','atemp','hum','windspeed','hr_sin','hr_cos','month_sin','month_cos',
                               'weekday_sin','weekday_cos','is_weekend','cnt_t_1','cnt_t_24']
                    features = [f for f in features if f in df.columns]
                    X = df[features].copy()
                    X_scaled = models['scaler'].transform(X)
                    X_pca = models['pca'].transform(X_scaled)
                    pca_2d = models['pca'].transform(X_scaled)[:, :2]  # First 2 components
                    clusters = models['kmeans'].predict(X_pca)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(pca_2d[:, 0], pca_2d[:, 1], c=clusters, cmap='viridis', s=6, alpha=0.6)
                    ax.set_xlabel('PCA Component 1')
                    ax.set_ylabel('PCA Component 2')
                    ax.set_title('PCA 2D Visualization of Clusters')
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    st.pyplot(fig)
                else:
                    st.warning("PCA or KMeans models not found.")

            elif plot == "Cluster Distribution":
                st.subheader("Cluster Distribution")
                if 'pca' in models and 'kmeans' in models:
                    features = ['season','yr','mnth','hr','holiday','weekday','workingday','weathersit',
                               'temp','atemp','hum','windspeed','hr_sin','hr_cos','month_sin','month_cos',
                               'weekday_sin','weekday_cos','is_weekend','cnt_t_1','cnt_t_24']
                    features = [f for f in features if f in df.columns]
                    X = df[features].copy()
                    X_scaled = models['scaler'].transform(X)
                    X_pca = models['pca'].transform(X_scaled)
                    clusters = models['kmeans'].predict(X_pca)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    cluster_counts = pd.Series(clusters).value_counts().sort_index()
                    ax.bar(cluster_counts.index, cluster_counts.values)
                    ax.set_xlabel('Cluster')
                    ax.set_ylabel('Number of Observations')
                    ax.set_title('Cluster Distribution')
                    st.pyplot(fig)
                else:
                    st.warning("PCA or KMeans models not found.")
    else:
        st.warning("Cluster summary not found. Please run the clustering analysis first.")

    st.subheader("Clustering Insights")
    st.markdown("""
    - **High Demand Clusters**: Prioritize maintenance and bike rotations
    - **Low Demand Clusters**: Schedule maintenance during off-peak times
    - **Peak Hour Clusters**: Focus on operational efficiency during busy periods
    """)

elif page == "Marketing Insights":
    st.header("📈 Marketing Insights")

    st.markdown("Data-driven marketing strategies for user acquisition and conversion.")

    # Load marketing strategy if available
    strategy_file = os.path.join('outputs', 'marketing_strategy.json')
    if os.path.exists(strategy_file):
        with open(strategy_file, 'r') as f:
            strategy = json.load(f)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Optimal Timing")
            st.markdown(f"**Conversion Windows**: {strategy['conversion_windows']}")
            st.markdown(f"**Acquisition Windows**: {strategy['acquisition_windows']}")
            st.markdown(f"**Retention Windows**: {strategy['retention_windows']}")

        with col2:
            st.subheader("Performance Metrics")
            st.metric("Current Conversion Rate", f"{strategy['current_conversion_rate']:.1f}%")
            st.metric("Max Conversion Rate", f"{strategy['max_conversion_rate']:.1f}%")
            st.metric("Improvement Potential", f"+{strategy['improvement_potential']:.1f} pts")

        st.subheader("Key Recommendations")
        st.markdown(f"""
        - **Casual Users**: Focus on {strategy['casual_top_feature']}
        - **Registered Users**: Focus on {strategy['registered_top_feature']}
        - **Best Weather**: Condition {strategy['best_weather_conversion']}
        """)

        st.subheader("Campaign Strategies")
        st.markdown("""
        - **Conversion**: Target casual users during optimal hours
        - **Acquisition**: Focus on peak casual activity times
        - **Retention**: Engage registered users during their peak usage
        """)
    else:
        st.warning("Marketing strategy not found. Please run the marketing analysis first.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Data Science for Bike Sharing Demand Analysis")
