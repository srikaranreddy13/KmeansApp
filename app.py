import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# 1️⃣ APP TITLE & DESCRIPTION
# --------------------------------------------------
st.set_page_config(page_title="Customer Segmentation Dashboard", layout="wide")

st.title("🟢 Customer Segmentation Dashboard")
st.markdown(
    "This system uses **K-Means Clustering** to group customers based on their purchasing behavior and similarities."
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
df = pd.read_csv("Wholesale customers data.csv")

# Remove non-spending columns
df_spending = df.drop(columns=["Channel", "Region"])

numerical_features = df_spending.columns.tolist()

# --------------------------------------------------
# 2️⃣ INPUT SECTION (SIDEBAR)
# --------------------------------------------------
st.sidebar.header("🔧 Clustering Controls")

feature_1 = st.sidebar.selectbox(
    "Select Feature 1",
    numerical_features,
    index=0
)

feature_2 = st.sidebar.selectbox(
    "Select Feature 2",
    numerical_features,
    index=1
)

k = st.sidebar.slider(
    "Number of Clusters (K)",
    min_value=2,
    max_value=10,
    value=3
)

random_state = st.sidebar.number_input(
    "Random State (Optional)",
    value=42,
    step=1
)

run_button = st.sidebar.button("🟦 Run Clustering")

# --------------------------------------------------
# VALIDATION
# --------------------------------------------------
if feature_1 == feature_2:
    st.warning("⚠ Please select **two different features**.")
    st.stop()

# --------------------------------------------------
# 3️⃣ CLUSTERING ACTION
# --------------------------------------------------
if run_button:

    selected_data = df_spending[[feature_1, feature_2]]

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(selected_data)

    kmeans = KMeans(n_clusters=k, random_state=random_state)
    clusters = kmeans.fit_predict(scaled_data)

    df_result = selected_data.copy()
    df_result["Cluster"] = clusters

    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # --------------------------------------------------
    # 4️⃣ VISUALIZATION SECTION
    # --------------------------------------------------
    st.subheader("📊 Customer Cluster Visualization")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=df_result[feature_1],
        y=df_result[feature_2],
        hue=df_result["Cluster"],
        palette="Set2",
        ax=ax
    )

    ax.scatter(
        centers[:, 0],
        centers[:, 1],
        c="black",
        s=250,
        marker="X",
        label="Cluster Centers"
    )

    ax.set_xlabel(feature_1)
    ax.set_ylabel(feature_2)
    ax.set_title("Customer Segments Based on Selected Features")
    ax.legend()

    st.pyplot(fig)

    # --------------------------------------------------
    # 5️⃣ CLUSTER SUMMARY SECTION
    # --------------------------------------------------
    st.subheader("📋 Cluster Summary")

    summary = (
        df_result
        .groupby("Cluster")
        .agg(
            Count=("Cluster", "count"),
            Avg_Feature_1=(feature_1, "mean"),
            Avg_Feature_2=(feature_2, "mean")
        )
        .reset_index()
    )

    st.dataframe(summary)

    # --------------------------------------------------
    # 6️⃣ BUSINESS INTERPRETATION SECTION
    # --------------------------------------------------
    st.subheader("💡 Business Interpretation")

    for _, row in summary.iterrows():
        cluster_id = int(row["Cluster"])
        count = int(row["Count"])

        st.markdown(
            f"""
            🟢 **Cluster {cluster_id}**  
            Customers in this group show **similar spending behavior** in *{feature_1}* and *{feature_2}*.  
            This segment contains **{count} customers** and represents a distinct purchasing pattern
            that can be targeted with customized business strategies.
            """
        )

    # --------------------------------------------------
    # 7️⃣ USER GUIDANCE / INSIGHT BOX
    # --------------------------------------------------
    st.info(
        "📌 Customers within the same cluster exhibit similar purchasing behaviour "
        "and can be targeted with similar promotions, pricing strategies, and inventory planning."
    )

else:
    st.info("👈 Select features and click **Run Clustering** to view results.")
