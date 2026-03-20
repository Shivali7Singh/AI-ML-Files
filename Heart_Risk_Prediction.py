
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Medical Heart Risk Dashboard", layout="wide")

# ---------------- MEDICAL DASHBOARD CSS ----------------
st.markdown("""
<style>

/* HEART BACKGROUND IMAGE */

[data-testid="stAppViewContainer"]{
background-image: url("https://images.unsplash.com/photo-1505751172876-fa1923c5c528");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}

/* 50% TRANSPARENCY CONTAINER */

.block-container{
background-color: rgba(255,255,255,0.50);
padding:2rem;
border-radius:12px;
}

h1{
color:#b30000;
text-align:center;
font-weight:900;
}

/* Dashboard Cards */

.card{
background:white;
padding:20px;
border-radius:12px;
box-shadow:0px 4px 10px rgba(0,0,0,0.15);
text-align:center;
}

.card h3{
color:#333;
}

.metric{
font-size:28px;
font-weight:bold;
color:#b30000;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background-color:#f7fbff;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("❤️ Heart Disease Risk Medical Dashboard")
st.write("AI system predicts **Heart Disease Risk** using Logistic Regression.")

# ---------------- LOAD DATA ----------------
df = pd.read_csv("heart_risk.csv")

# ---------------- FEATURES ----------------
X = df[['Age','Cholesterol','BloodPressure','Sugar','MaxHeartRate','Smoking']]
y = df['Risk']

# ---------------- TRAIN TEST ----------------
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42)

# ---------------- MODEL ----------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# ---------------- DASHBOARD METRICS ----------------
col1,col2,col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="card">
    <h3>Total Patients</h3>
    <div class="metric">{len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="card">
    <h3>Model Accuracy</h3>
    <div class="metric">{accuracy*100:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="card">
    <h3>High Risk Cases</h3>
    <div class="metric">{df['Risk'].sum()}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- DATASET PREVIEW ----------------
st.subheader("📊 Dataset Preview")

styled_df = df.head().style \
.set_table_styles([
{'selector': 'th',
 'props': [('color', 'white'),
           ('background-color', '#b30000'),
           ('font-size', '16px'),
           ('text-align','center')]},

{'selector': 'td',
 'props': [('color', '#003366'),
           ('font-weight','bold'),
           ('text-align','center')]}
])

st.table(styled_df)

# ---------------- DATASET STATISTICS ----------------
st.subheader("📊 Dataset Statistics")

styled_stats = df.describe().style \
.set_table_styles([
{'selector': 'th',
 'props': [('color', 'white'),
           ('background-color', '#004d66'),
           ('font-size', '16px'),
           ('text-align','center')]},

{'selector': 'td',
 'props': [('color', '#660066'),
           ('font-weight','bold'),
           ('text-align','center')]}
])

st.table(styled_stats)

# ---------------- CONFUSION MATRIX ----------------
st.subheader("📈 Confusion Matrix")

cm = confusion_matrix(y_test,y_pred)

fig,ax = plt.subplots(figsize=(4,4))
ax.imshow(cm)

for i in range(len(cm)):
    for j in range(len(cm)):
        ax.text(j,i,cm[i,j],ha="center",va="center",fontsize=12)

plt.xlabel("Predicted")
plt.ylabel("Actual")

st.pyplot(fig)

# ---------------- GRAPH ----------------
st.subheader("📉 Actual vs Predicted Risk")

fig2,ax2 = plt.subplots(figsize=(6,3))

ax2.bar(range(len(y_test[:10])),y_test[:10],label="Actual")
ax2.bar(range(len(y_pred[:10])),y_pred[:10],alpha=0.7,label="Predicted")

plt.legend()
plt.xlabel("Samples")
plt.ylabel("Risk")

st.pyplot(fig2)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🫀 Patient Details")

age = st.sidebar.slider("Age",20,80,40)
chol = st.sidebar.slider("Cholesterol",100,300,180)
bp = st.sidebar.slider("Blood Pressure",80,200,120)
sugar = st.sidebar.slider("Sugar Level",70,200,100)
heart = st.sidebar.slider("Max Heart Rate",60,220,150)
smoke = st.sidebar.selectbox("Smoking",[0,1])

# ---------------- PREDICTION ----------------
if st.sidebar.button("Predict Heart Risk"):

    new_patient = [[age,chol,bp,sugar,heart,smoke]]

    prediction = model.predict(new_patient)

    st.subheader("🩺 Prediction Result")

    if prediction[0]==1:
        st.error("⚠ High Heart Disease Risk")
    else:
        st.success("✅ Low Heart Disease Risk")