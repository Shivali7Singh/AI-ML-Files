

# -------------------- IMPORT LIBRARIES --------------------
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="SOTY Prediction System", page_icon="🏆", layout="wide")

# -------------------- BACKGROUND IMAGE CSS --------------------
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1503676260728-1c00da094a0b");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

/* Titles */
h1 {
    color: #ffcc00;
    text-align: center;
    font-size: 60px !important;
}
h2, h3 {
    color: #00ffff;
    font-size: 36px !important;
}

/* Text */
p, label, .stMarkdown {
    color: #ffffff !important;
    font-size: 22px !important;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #ff512f, #dd2476);
    color: white;
    font-size: 20px;
    border-radius: 12px;
    height: 3em;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #141e30, #243b55);
    color: #ffffff;
}

/* Table Styling */
thead tr th {
    background-color: #00ffff !important;
    color: #000000 !important;
    font-size: 20px !important;
    text-align: center !important;
}
tbody tr td {
    font-size: 18px !important;
    color: #ffcc00 !important;
    text-align: center !important;
    background-color: rgba(0,0,0,0.6) !important;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
df = pd.read_csv("NSTI Student_data.csv")

X = df[["Study Hours", "Attendance", "Marks", "Sports"]]
y = df[["SOTY"]]

le = LabelEncoder()
y_enc = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# -------------------- SIDEBAR --------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Select Page", ["🏠 Home", "📊 Analytics", "🔍 Prediction"])

st.sidebar.markdown("### 🎯 Model Accuracy")
st.sidebar.success(f"{accuracy*100:.2f}%")

# ================= HOME =================
if page == "🏠 Home":
    st.title("🏆 Student Of The Year Prediction System")

    st.write("""
    This Machine Learning project predicts whether a student 
    can become **Student Of The Year (SOTY)** 
    based on performance indicators.
    """)

    st.subheader("📂 Dataset Preview")
    st.dataframe(df.style.set_properties(**{
        'background-color': '#1c1c1c',
        'color': '#ffcc00',
        'border-color': '#00ffff'
    }))

    st.subheader("📊 Quick Insights")
    col1, col2, col3 = st.columns(3)
    col1.metric("Average Study Hours", f"{df['Study Hours'].mean():.2f}")
    col2.metric("Average Attendance (%)", f"{df['Attendance'].mean():.2f}")
    col3.metric("Average Marks (%)", f"{df['Marks'].mean():.2f}")

# ================= ANALYTICS =================
elif page == "📊 Analytics":
    st.title("📊 Data Analytics Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Study Hours vs Marks")
        fig1 = plt.figure()
        plt.scatter(df["Study Hours"], df["Marks"], color="#00ffff")
        plt.xlabel("Study Hours")
        plt.ylabel("Marks")
        plt.title("Study vs Marks")
        st.pyplot(fig1)

    with col2:
        st.subheader("SOTY Distribution")
        fig2 = plt.figure()
        df["SOTY"].value_counts().plot(kind="bar", color="#ffcc00")
        plt.xlabel("SOTY")
        plt.ylabel("Count")
        plt.title("SOTY Count")
        st.pyplot(fig2)

    st.subheader("🌟 Feature Importance")
    importance = model.feature_importances_
    fig3 = plt.figure()
    plt.bar(X.columns, importance, color="#dd2476")
    plt.xticks(rotation=30)
    plt.title("Feature Importance")
    st.pyplot(fig3)

    st.subheader("📌 Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig4 = plt.figure()
    plt.imshow(cm, cmap="coolwarm")
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha="center", va="center", color="white")
    st.pyplot(fig4)

# ================= PREDICTION =================
elif page == "🔍 Prediction":
    st.title("🔍 Predict SOTY")

    study_hours = st.slider("Study Hours", 0, 12, 2)
    attendance = st.slider("Attendance (%)", 0, 100, 90)
    marks = st.slider("Marks (%)", 0, 100, 88)
    sports = st.slider("Sports Participation (0-5)", 0, 5, 2)

    if st.button("Predict SOTY"):
        new_data = np.array([[study_hours, attendance, marks, sports]])
        prediction = model.predict(new_data)
        result = "Yes" if prediction[0] == 1 else "No"

        if result == "Yes":
            st.balloons()
            st.success("🏆 Congratulations! Student can become SOTY!")
        else:
            st.error("❌ Student may not become SOTY.")

        report = f"""
        -------- Student Report --------
        Study Hours: {study_hours}
        Attendance: {attendance}
        Marks: {marks}
        Sports: {sports}

        Predicted SOTY: {result}
        """
        st.download_button("📥 Download Report", report, file_name="SOTY_Report.txt")

