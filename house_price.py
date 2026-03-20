
import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="wide"
)

# -------------------- PREMIUM CSS --------------------
page_bg = """
<style>

/* Background Image with 50% Overlay */
[data-testid="stAppViewContainer"] {
    background-image: 
        linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
        url("https://images.unsplash.com/photo-1600607687939-ce8a6c25118c");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* Transparent Header */
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Glass Container */
[data-testid="stAppViewContainer"] > .main {
    background: rgba(0, 0, 0, 0.4);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(10px);
}

/* Title */
h1 {
    color: #FFD700 !important;
    text-align: center;
    font-size: 55px !important;
}

/* Subheadings */
h2, h3 {
    color: #00FFFF !important;
    font-size: 30px !important;
}

/* Text */
p, label {
    color: #E6E6FA !important;
    font-size: 20px !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: rgba(0, 0, 0, 0.95);
}
[data-testid="stSidebar"] * {
    color: #90EE90 !important;
}

/* Button */
.stButton>button {
    background-color: #FF4500;
    color: white;
    font-size: 20px;
    border-radius: 12px;
    height: 55px;
    width: 220px;
    transition: 0.3s;
}
.stButton>button:hover {
    background-color: #ff2200;
    transform: scale(1.05);
}

/* Feature Importance Table Styling */
.feature-table th {
    background-color: #FFD700 !important;  /* Gold headings */
    color: black !important;
    font-size: 18px !important;
}
.feature-table td {
    color: #00FF7F !important;  /* Spring Green values */
    font-size: 16px !important;
    font-weight: bold;
}

</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.title("🏡 House Price Prediction System")
st.markdown("### 💰 Predict House Prices using Machine Learning (Linear Regression)")

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    return pd.read_csv("house_price.csv")

try:
    df = load_data()
except:
    st.error("❌ Dataset file 'house_price.csv' not found!")
    st.stop()

# -------------------- TRAIN MODEL --------------------
X = df[['Size_sqft', 'Bedrooms', 'Age_years']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.header("🏠 Enter House Details")

size = st.sidebar.number_input("📏 Size (sqft)", 500, 10000, 2000)
bedrooms = st.sidebar.slider("🛏 Bedrooms", 1, 10, 3)
age = st.sidebar.number_input("🏗 Age (years)", 0, 100, 5)

# -------------------- PREDICTION --------------------
st.subheader("🔮 Prediction Result")

if st.button("🚀 Predict Price"):
    with st.spinner("🔄 Predicting... Please wait"):
        time.sleep(2)
        new_data = np.array([[size, bedrooms, age]])
        prediction = model.predict(new_data)

    st.success(f"🏷 Estimated House Price: ₹ {prediction[0]:,.2f}")
    st.balloons()

    # Download option
    result_df = pd.DataFrame({
        "Size_sqft": [size],
        "Bedrooms": [bedrooms],
        "Age_years": [age],
        "Predicted Price": [prediction[0]]
    })

    st.download_button(
        label="📥 Download Prediction",
        data=result_df.to_csv(index=False),
        file_name="prediction.csv",
        mime="text/csv"
    )

# -------------------- MODEL PERFORMANCE --------------------
st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)

with col1:
    r2 = r2_score(y_test, y_pred)
    st.metric("📈 R² Score", f"{r2:.2f}")

with col2:
    mae = mean_absolute_error(y_test, y_pred)
    st.metric("📉 Mean Absolute Error", f"₹ {mae:,.2f}")

# Smart Feedback
if r2 > 0.8:
    st.success("🔥 Excellent Model Performance!")
elif r2 > 0.6:
    st.info("👍 Good Model Performance")
else:
    st.warning("⚠ Model Needs Improvement")

# -------------------- FEATURE IMPORTANCE --------------------
st.subheader("📌 Feature Importance")

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_
})

# Apply custom CSS class to table
st.markdown(coefficients.style.set_table_attributes('class="feature-table"').to_html(), unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("👩‍💻 Developed by Shivali Singh 🇮🇳 using Streamlit & Scikit-Learn")
