
# =========================================
# 1. Import Libraries
# =========================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# =========================================
# 2. Page Config
# =========================================
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")

# =========================================
# 3. Background CSS
# =========================================
page_bg_img = """
<style>
[data-testid="stAppViewContainer"]{
background-image: linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
url("https://photos.peopleimages.com/picture/202403/3034503-abstract-stock-market-and-finance-background-design-for-business-economy-and-global-inflation.-graphic-index-or-marketing-strategy-graphic-wallpaper-for-banking-investment-growth-and-forex-trading-fit_400_400.jpg");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}

html, body, [class*="css"]{
color:#e0e0e0 !important;
font-size:18px;
}

h1{color:#FFD700 !important;}
h2,h3,h4{color:#FFA500 !important;}

/* Table */
thead tr th{
background-color:#b30000 !important;
color:orange !important;
text-align:center !important;
}
tbody tr td{
background-color:#5c0000 !important;
color:#f5f5f5 !important;
text-align:center !important;
}

/* KPI Card */
div[data-testid="metric-container"]{
background: linear-gradient(135deg,#ffffff,#ffe9b3);
border:3px solid orange;
padding:18px;
border-radius:15px;
box-shadow:0px 0px 18px rgba(255,165,0,0.8);
text-align:center;
}

/* 🔥 FIX: Dark White Text for Metrics */
[data-testid="stMetricLabel"]{
color:#e0e0e0 !important;
font-weight:bold;
}

[data-testid="stMetricValue"]{
color:#e0e0e0 !important;
font-size:30px !important;
font-weight:bold !important;
}

[data-testid="stMetricDelta"]{
color:#e0e0e0 !important;
}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# =========================================
# 4. Title
# =========================================
st.markdown("<h1 style='text-align:center;'>📈 Stock Market Prediction Dashboard</h1>", unsafe_allow_html=True)

# =========================================
# 5. Load Dataset
# =========================================
df = pd.read_csv("stock_price_prediction.csv", encoding="latin1")

# =========================================
# 6. Data Cleaning
# =========================================
df.replace("#VALUE!", np.nan, inplace=True)
df = df.drop(["Stock","Date","Month / Y"], axis=1, errors="ignore")

cols = ["Open","High","Low","Price"]
for col in cols:
    df[col] = df[col].astype(str).str.replace(",", "")
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Vol."] = df["Vol."].astype(str).str.replace(",", "")
df["Vol."] = df["Vol."].str.replace("K","000")
df["Vol."] = df["Vol."].str.replace("M","000000")
df["Vol."] = pd.to_numeric(df["Vol."], errors="coerce")

df = df.fillna(method="ffill")

# =========================================
# 7. Dataset Preview
# =========================================
st.subheader("📊 Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Dataset", csv, "cleaned_stock.csv")

# =========================================
# 8. Key Metrics
# =========================================
st.subheader("📊 Key Metrics")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Average Price", f"{df['Price'].mean():.2f}")
c2.metric("Max Price", f"{df['Price'].max():.2f}")
c3.metric("Min Price", f"{df['Price'].min():.2f}")
c4.metric("Avg Volume", f"{df['Vol.'].mean():.0f}")

# =========================================
# 9. Feature Engineering
# =========================================
df["Price_Range"] = df["High"] - df["Low"]
df["Target"] = (df["Price"].shift(-1) > df["Price"]).astype(int)
df = df.dropna()

# =========================================
# 10. Train Test Split
# =========================================
X = df[["Open","High","Low","Vol.","Price_Range"]]
y = df["Target"]

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# =========================================
# 11. Train Models
# =========================================
log_model = LogisticRegression(max_iter=300)
rf_model = RandomForestClassifier(n_estimators=300,random_state=42)

log_model.fit(X_train,y_train)
rf_model.fit(X_train,y_train)

log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

log_acc = accuracy_score(y_test,log_pred)
rf_acc = accuracy_score(y_test,rf_pred)

# =========================================
# 12. Model Accuracy
# =========================================
st.subheader("🧠 Model Accuracy")
c1,c2 = st.columns(2)
c1.metric("Random Forest", f"{rf_acc*100:.2f}%")
c2.metric("Logistic Regression", f"{log_acc*100:.2f}%")

# =========================================
# Sidebar Prediction
# =========================================
st.sidebar.header("📊 Prediction")

open_price = st.sidebar.number_input("Open Price",0.0)
high_price = st.sidebar.number_input("High Price",0.0)
low_price = st.sidebar.number_input("Low Price",0.0)
volume = st.sidebar.number_input("Volume",0)

price_range = high_price - low_price

model_choice = st.sidebar.selectbox("Model",["Random Forest","Logistic Regression"])

prediction_result = None

if st.sidebar.button("Predict"):
    data = np.array([[open_price,high_price,low_price,volume,price_range]])
    
    if model_choice == "Random Forest":
        prediction_result = rf_model.predict(data)[0]
    else:
        prediction_result = log_model.predict(data)[0]

# =========================================
# Prediction Result BELOW Accuracy
# =========================================
if prediction_result is not None:
    st.subheader("🔮 Prediction Result")
    if prediction_result == 1:
        st.balloons()
        st.success("📈 Stock Price Likely to Increase")
    else:
        st.error("📉 Stock Price Likely to Decrease")

# =========================================
# 13. Model Comparison
# =========================================
model_df = pd.DataFrame({
    "Model":["Random Forest","Logistic Regression"],
    "Accuracy":[rf_acc*100,log_acc*100]
})

fig_model = px.bar(model_df,x="Model",y="Accuracy",color="Model")
st.plotly_chart(fig_model,use_container_width=True)

# =========================================
# 14. Stock Analysis
# =========================================
st.subheader("📈 Stock Analysis")

df["MA20"] = df["Price"].rolling(20).mean()

col1,col2 = st.columns(2)

with col1:
    fig1 = px.line(df,y="Price")
    st.plotly_chart(fig1,use_container_width=True)

with col2:
    fig2 = px.line(df,y=["Price","MA20"])
    st.plotly_chart(fig2,use_container_width=True)

# =========================================
# 15. Candlestick Chart
# =========================================
st.subheader("🕯️ Candlestick Chart")

fig3 = go.Figure(data=[go.Candlestick(
    x=df.index,
    open=df["Open"],
    high=df["High"],
    low=df["Low"],
    close=df["Price"]
)])
st.plotly_chart(fig3,use_container_width=True)

# =========================================
# 16. Feature Importance
# =========================================
st.subheader("🔥 Feature Importance")

importance = pd.DataFrame({
    "Feature":X.columns,
    "Importance":rf_model.feature_importances_
}).sort_values(by="Importance",ascending=False)

fig4 = px.bar(importance,x="Feature",y="Importance",color="Importance")
st.plotly_chart(fig4,use_container_width=True)

# =========================================
# 17. Correlation Heatmap
# =========================================
st.subheader("🔥 Correlation")

corr = df.corr(numeric_only=True)

fig_corr = px.imshow(corr,text_auto=True,color_continuous_scale="RdBu")
st.plotly_chart(fig_corr,use_container_width=True)

# =========================================
# 18. Confusion Matrix
# =========================================
st.subheader("📊 Confusion Matrix")

cm = confusion_matrix(y_test,rf_pred)

fig_cm = px.imshow(cm,text_auto=True,color_continuous_scale="Oranges")
st.plotly_chart(fig_cm,use_container_width=True)

# =========================================
# 19. Footer
# =========================================
st.markdown("---")
st.markdown("<center style='color:white'>📊 Stock Market Prediction Dashboard | Built with Streamlit & Machine Learning</center>",unsafe_allow_html=True)
