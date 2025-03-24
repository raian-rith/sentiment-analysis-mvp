import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# 📌 Step 1: Dummy Sender Names
lead_senders = ["John Smith", "Emily Davis", "Michael Johnson", "Sophia Martinez"] * 6 + ["David Brown"]
client_senders = ["Mark Thompson", "Hannah Perez", "Chris Evans", "Laura Rodriguez"] * 6 + ["Jacob Parker"]

# Unique Emails for Leads and Clients
lead_emails = ["I love your marketing!", "Interested in SEO services.", "Frustrated with no response."] * 8 + ["Need flexible contract."]
client_emails = ["SEO is amazing!", "Ads not converting well.", "Performance tracking is off."] * 8 + ["Retention is declining."]

# 📌 Step 2: Adjust Email Timestamp Distribution
base_year = 2024
high_volume_dates = [
    datetime.datetime(base_year, 2, 10),
    datetime.datetime(base_year, 5, 15),
    datetime.datetime(base_year, 8, 20),
    datetime.datetime(base_year, 11, 5),
    datetime.datetime(base_year, 12, 25)
]

timestamps = []
for _ in range(50):
    if random.random() < 0.5:  # 50% chance to assign to a high-volume date
        timestamps.append(random.choice(high_volume_dates))
    else:  # Otherwise, distribute across random dates
        timestamps.append(datetime.datetime(base_year, random.randint(1, 12), random.randint(1, 28)))

# Create DataFrame
data = {
    "Sender": lead_senders[:25] + client_senders[:25],
    "Email_Text": lead_emails[:25] + client_emails[:25],
    "Sender_Type": ["Lead"] * 25 + ["Current Client"] * 25,
    "Timestamp": timestamps
}

df = pd.DataFrame(data)

# 📌 Step 3: AI-Powered Sentiment Analysis
df["Sentiment"] = df["Email_Text"].apply(lambda x: "Positive" if sia.polarity_scores(x)["compound"] > 0.2 else 
                                         "Negative" if sia.polarity_scores(x)["compound"] < -0.2 else "Neutral")

# 📌 Step 4: AI-Driven Urgency Assignment
def determine_urgency(text, sentiment):
    urgent_keywords = ["urgent", "asap", "immediate", "not working", "fix this", "need help", "still waiting", "unacceptable"]
    text_lower = text.lower()
    
    if any(word in text_lower for word in urgent_keywords):
        return "Urgent"
    
    if sentiment == "Negative":
        return "Urgent"
    
    if sentiment == "Neutral":
        return "Normal"
    
    return "Low Priority"

df["Urgency"] = df.apply(lambda row: determine_urgency(row["Email_Text"], row["Sentiment"]), axis=1)

# 📌 Step 5: Streamlit Dashboard
st.set_page_config(page_title="AI-Powered Customer Insights", layout="wide")
st.title("📊 AI-Powered Customer Insights Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filters")
sender_filter = st.sidebar.selectbox("Filter by Sender Type", ["All", "Lead", "Current Client"])
sentiment_filter = st.sidebar.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])
urgency_filter = st.sidebar.selectbox("Filter by Urgency", ["All", "Urgent", "Normal", "Low Priority"])

# Apply Filters
filtered_df = df.copy()
if sender_filter != "All":
    filtered_df = filtered_df[filtered_df["Sender_Type"] == sender_filter]
if sentiment_filter != "All":
    filtered_df = filtered_df[filtered_df["Sentiment"] == sentiment_filter]
if urgency_filter != "All":
    filtered_df = filtered_df[filtered_df["Urgency"] == urgency_filter]

st.subheader("📩 Filtered Email Dataset")
st.dataframe(filtered_df)

# 📌 Sentiment Distribution
st.subheader("📊 Sentiment Distribution")
st.write("This bar chart shows how many emails are positive, neutral, or negative.")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Sentiment", palette="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 Urgency Breakdown
st.subheader("⏳ Urgency Levels by Sender Type")
st.write("This chart helps understand how urgency is distributed among leads vs. current clients.")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Urgency", hue="Sender_Type", palette="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 Sentiment Over Time
st.subheader("📈 Sentiment Trend Over Time")
st.write("Tracking how customer sentiment changes over time.")

sentiment_over_time = df.groupby(["Timestamp", "Sentiment"]).size().unstack().fillna(0)
st.line_chart(sentiment_over_time)

# 📌 Urgency vs. Sentiment Correlation
st.subheader("🔗 Urgency vs. Sentiment Correlation")
st.write("This heatmap helps identify whether urgent emails are mostly positive or negative.")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pd.crosstab(df["Urgency"], df["Sentiment"]), annot=True, fmt="d", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 AI-Detected High-Risk Clients
st.subheader("⚠️ High-Risk Clients")
st.write("Clients with negative sentiment & urgent requests.")
st.dataframe(df[(df["Sender_Type"] == "Current Client") & (df["Sentiment"] == "Negative")])

st.subheader("🔍 Key Insights")
st.write("- **Most urgent emails come from customers with negative sentiment.**")
st.write("- **Tracking sentiment trends can help in proactive engagement.**")
st.write("- **Understanding frequent complaint words helps in improving service.**")
