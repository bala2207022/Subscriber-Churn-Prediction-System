import os
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(
    page_title="Netflix Subscriber Churn Dashboard",
    layout="wide"
)

st.title("🎬 Netflix Subscriber Churn Prediction Dashboard")

# ----------------- LOAD DATA -----------------
# Try data/final_user_risk.csv first, fallback to local
if os.path.exists("data/final_user_risk.csv"):
    CSV_PATH = "data/final_user_risk.csv"
else:
    CSV_PATH = "final_user_risk.csv"

df = pd.read_csv(CSV_PATH)

# Safety: make sure types are correct
if "predicted_churn" in df.columns:
    df["predicted_churn"] = df["predicted_churn"].astype(int)
if "churn_probability" in df.columns:
    df["churn_probability"] = df["churn_probability"].astype(float)

df["churn_label"] = df["predicted_churn"].map({0: "No Churn", 1: "Churn"})

# ----------------- TOP METRICS -----------------
total_users = len(df)
total_churners = int(df["predicted_churn"].sum())
churn_rate = (total_churners / total_users) * 100 if total_users > 0 else 0
avg_risk = df["churn_probability"].mean() if "churn_probability" in df.columns else 0

st.markdown("### 📊 Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Users", f"{total_users:,}")
col2.metric("Predicted Churners", f"{total_churners:,}")
col3.metric("Churn Rate", f"{churn_rate:.1f}%")
col4.metric("Average Churn Probability", f"{avg_risk:.3f}")

st.markdown("---")

# ----------------- HIGH RISK USERS TABLE -----------------
st.markdown("### 🔝 Top High-Risk Users")

# Let user adjust what "high risk" means
threshold = st.slider(
    "High-Risk Threshold (churn probability ≥)",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.01,
)

high_risk = (
    df[df["churn_probability"] >= threshold]
    .sort_values(by="churn_probability", ascending=False)
    .head(20)
)

st.caption(f"Showing users with churn_probability ≥ {threshold:.2f}")
st.dataframe(
    high_risk[
        [
            "user_id",
            "plan_type",
            "churn_probability",
            "predicted_churn",
            "account_age_days",
            "days_since_last_login",
            "total_watch_time_30d",
            "avg_rating",
            "num_ratings",
        ]
    ],
    use_container_width=True
)

st.markdown("---")

# ----------------- LAYOUT FOR GRAPHS -----------------
left_col, right_col = st.columns(2)

# ----- LEFT: Churn probability distribution -----
with left_col:
    st.markdown("### Churn Probability Distribution")

    prob_chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            alt.X("churn_probability:Q", bin=alt.Bin(maxbins=30), title="Churn Probability"),
            alt.Y("count():Q", title="Number of Users"),
            color=alt.Color("churn_label:N", title="Predicted Class"),
            tooltip=["count()", "churn_label"]
        )
        .properties(height=350)
        .interactive()
    )

    st.altair_chart(prob_chart, use_container_width=True)

# ----- RIGHT: Churn rate by plan type -----
with right_col:
    st.markdown("###  Churn Rate by Plan Type")

    if "plan_type" in df.columns:
        plan_group = (
            df.groupby("plan_type", as_index=False)
            .agg(
                total_users=("user_id", "count"),
                churners=("predicted_churn", "sum")
            )
        )
        plan_group["churn_rate"] = (
            plan_group["churners"] / plan_group["total_users"] * 100
        )

        plan_chart = (
            alt.Chart(plan_group)
            .mark_bar()
            .encode(
                x=alt.X("plan_type:N", title="Plan Type"),
                y=alt.Y("churn_rate:Q", title="Churn Rate (%)"),
                color=alt.Color("plan_type:N", legend=None),
                tooltip=["plan_type", "total_users", "churners", alt.Tooltip("churn_rate:Q", format=".1f")]
            )
            .properties(height=350)
        )

        st.altair_chart(plan_chart, use_container_width=True)
    else:
        st.info("No 'plan_type' column found in data.")

st.markdown("---")

# ----------------- SCATTER: ACCOUNT AGE vs CHURN PROBABILITY -----------------
st.markdown("###  Account Age vs Churn Probability")

if "account_age_days" in df.columns:
    scatter_chart = (
        alt.Chart(df)
        .mark_circle(size=60, opacity=0.7)
        .encode(
            x=alt.X("account_age_days:Q", title="Account Age (days)"),
            y=alt.Y("churn_probability:Q", title="Churn Probability"),
            color=alt.Color("churn_label:N", title="Predicted Class"),
            tooltip=[
                "user_id",
                "plan_type",
                "account_age_days",
                "churn_probability",
                "predicted_churn",
            ],
        )
        .properties(height=350)
        .interactive()
    )
    st.altair_chart(scatter_chart, use_container_width=True)
else:
    st.info("No 'account_age_days' column found in data.")

st.markdown("---")

# ----------------- USER SEARCH SECTION -----------------
st.markdown("### 🔎 Search Subscriber Details")

user_id_list = df["user_id"].unique().tolist()
selected_user = st.selectbox("Select User ID", user_id_list)

user_row = df[df["user_id"] == selected_user].copy()

if not user_row.empty:
    st.write("#### Subscriber Profile")
    show_cols = [
        "user_id",
        "plan_type",
        "age",
        "account_age_days",
        "days_since_last_login",
        "total_watch_time_30d",
        "avg_rating",
        "num_ratings",
        "churn_probability",
        "predicted_churn",
    ]
    existing_cols = [c for c in show_cols if c in user_row.columns]
    st.dataframe(user_row[existing_cols], use_container_width=True)

    st.write("#### Risk Interpretation")
    prob = float(user_row["churn_probability"].iloc[0])
    pred = int(user_row["predicted_churn"].iloc[0])

    if pred == 1:
        st.error(
            f"This user is predicted to **CHURN** "
            f"(probability ≈ {prob:.3f}). Consider targeted retention actions."
        )
    else:
        st.success(
            f"This user is predicted to **STAY** "
            f"(probability ≈ {prob:.3f})."
        )
