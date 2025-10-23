import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍️ Mall Customer Segmentation Dashboard")
st.markdown("Cluster customers based on demographics and spending behavior using KMeans.")

# Upload file
uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type="csv")

if uploaded_file is not None:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    # Drop CustomerID
    df_clean = df.drop("CustomerID", axis=1)

    # Encode Genre
    le = LabelEncoder()
    df_clean['Genre'] = le.fit_transform(df_clean['Genre'])

    # Scale
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    # Sidebar: number of clusters
    st.sidebar.subheader("Clustering Parameters")
    k = st.sidebar.slider("Number of clusters (k)", 2, 10, 5)

    # KMeans
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    st.subheader("Clustered Data")
    st.dataframe(df.head())

    # Visualization: Income vs Spending Score
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(x="Annual Income (k$)", y="Spending Score (1-100)", hue="Cluster", data=df, palette="Set1", ax=ax)
    plt.title("Customer Segments")
    st.pyplot(fig)

    # Cluster profile
    st.subheader("Cluster Profiles")
    profile = df.groupby("Cluster").mean()
    profile['Count'] = df['Cluster'].value_counts()
    st.dataframe(profile)

    # Download clustered dataset
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Clustered Data", csv, "clustered_customers.csv", "text/csv")
else:
    st.info("Please upload the Mall_Customers.csv file to begin.")
