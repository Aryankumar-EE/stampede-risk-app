import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Crowd Risk Predictor", layout="wide")

st.title("🚨 Crowd Stampede Risk Prediction System")
st.markdown("### AI-based Pre-Event Crowd Safety Analysis")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("hybrid_crowd_dataset.csv")

# ---------------- TRAIN MODEL ----------------
X = df[['Density', 'Exits', 'EventType']]
y = df['Risk']

model = DecisionTreeClassifier(max_depth=4)
model.fit(X, y)

# ---------------- DATA VISUALIZATION ----------------
st.subheader("📊 Dataset Insights")

fig = px.scatter(df, x="Density", y="Risk", title="Density vs Risk")
st.plotly_chart(fig)

fig2 = px.histogram(df, x="Density", title="Density Distribution")
st.plotly_chart(fig2)

# ---------------- INPUT SECTION ----------------
st.subheader("🧾 Enter Event Details")

col1, col2 = st.columns(2)

with col1:
    crowd_size = st.number_input("Expected Crowd Size", 100, 100000, 5000)
    exits = st.slider("Number of Exits", 1, 10, 3)

with col2:
    area = st.number_input("Area Size (m²)", 100, 10000, 1000)
    event_type = st.selectbox("Event Type", ["Religious", "Concert", "Sports"])

event_map = {"Religious": 0, "Concert": 1, "Sports": 2}
event_val = event_map[event_type]

# ---------------- PREDICTION ----------------
if st.button("Check Risk"):

    density = crowd_size / area
    prediction = model.predict([[density, exits, event_val]])[0]

    st.subheader("📊 Analysis Result")

    col3, col4 = st.columns(2)

    with col3:
        st.metric("Crowd Density", f"{round(density,2)} people/m²")

    with col4:
        st.metric("Number of Exits", exits)

    # Density progress bar
    st.progress(min(int(density * 10), 100))

    # Density level indicator
    if density < 2:
        st.success("🟢 Low Density (Safe)")
    elif density < 4:
        st.warning("🟡 Moderate Density")
    else:
        st.error("🔴 High Density (Danger Zone)")

    # Final prediction
    if prediction == 1:
        st.error("🚨 HIGH RISK OF STAMPEDE")

        st.subheader("⚠️ Recommendations")

        if density > 4:
            st.write("👉 Reduce crowd size or increase area")

        if exits <= 3:
            st.write("👉 Increase number of exits")

        if event_val == 0:
            st.write("👉 Extra crowd control needed for religious events")

        st.write("👉 Deploy more security personnel")
        st.write("👉 Use barricades for crowd control")

    else:
        st.success("✅ SAFE CONDITIONS")
        st.info("Maintain current safety arrangements.")

# ---------------- STYLING ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0e1117;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)
