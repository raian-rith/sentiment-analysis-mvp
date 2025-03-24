import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
from textblob import TextBlob
import nltk

nltk.download('stopwords')

# Unique, business-related email dataset
email_data = [
    {"text": "Could you please provide details on your enterprise pricing plans?", "sender": "Lead", "theme": "Pricing Inquiry"},
    {"text": "We love the results we've seen with your SEO service! Keep it up!", "sender": "Current Client", "theme": "Positive Feedback"},
    {"text": "I'm unhappy with the delay in resolving my account issue.", "sender": "Current Client", "theme": "Support Issue"},
    {"text": "We'd like to understand more about your lead generation services.", "sender": "Lead", "theme": "General Inquiry"},
    {"text": "Interested in upgrading our existing content marketing package.", "sender": "Current Client", "theme": "Upgrade Request"},
    {"text": "There's a discrepancy in the latest invoice we received.", "sender": "Current Client", "theme": "Billing Issue"},
    {"text": "Can you send us case studies for your PPC campaigns?", "sender": "Lead", "theme": "Case Study Request"},
    {"text": "Awaiting a response from our previous call regarding social media strategies.", "sender": "Lead", "theme": "Follow-up Needed"},
    {"text": "Our recent email marketing campaign didn't achieve desired results.", "sender": "Current Client", "theme": "Campaign Performance"},
    {"text": "Thanks to your marketing efforts, our quarterly sales increased significantly!", "sender": "Current Client", "theme": "Success Story"}
]

# Generate 100 distinct emails from the dataset
data = {
    "Email_Text": [item["text"] for item in random.choices(email_data, k=100)],
    "Sender_Type": [item["sender"] for item in random.choices(email_data, k=100)],
    "Urgency": [random.choice(["Urgent", "Normal", "Low Priority"]) for _ in range(100)],
    "Theme": [item["theme"] for item in random.choices(email_data, k=100)],
    "Timestamp": [datetime.datetime(2024, random.randint(1, 3), random.randint(1, 28)) for _ in range(100)]
}

df = pd.DataFrame(data)

# Sentiment Analysis
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"

df["Sentiment"] = df["Email_Text"].apply(get_sentiment)

# Lead Scoring Function
def calculate_lead_score(row):
    score = 0
    if row["Sender_Type"] == "Lead":
        score += 20
    if row["Sentiment"] == "Positive":
        score += 30
    elif row["Sentiment"] == "Negative":
        score -= 20
    if row["Urgency"] == "Urgent":
        score += 40
    elif row["Urgency"] == "Low Priority":
        score -= 10
    return score

df["Lead_Score"] = df.apply(calculate_lead_score, axis=1)

# Streamlit App
st.title("📊 Sentiment Analysis Dashboard")
st.sidebar.header("Filters")

# Filters
sender_type_filter = st.sidebar.selectbox("Filter by Sender Type", ["All", "Lead", "Current Client"])
if sender_type_filter != "All":
    df = df[df["Sender_Type"] == sender_type_filter]

urgency_filter = st.sidebar.selectbox("Filter by Urgency Level", ["All", "Urgent", "Normal", "Low Priority"])
if urgency_filter != "All":
    df = df[df["Urgency"] == urgency_filter]

# Display Data
st.subheader("📩 Filtered Email Dataset")
st.dataframe(df)

# Sentiment Distribution Chart
st.subheader("📊 Sentiment Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["Sentiment"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# Sentiment Breakdown by Urgency
st.subheader("⏳ Sentiment by Urgency Level")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df["Urgency"], hue=df["Sentiment"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# Sentiment Over Time
st.subheader("📈 Sentiment Trend Over Time")
sentiment_time = df.groupby(["Timestamp", "Sentiment"]).size().unstack().fillna(0)
st.line_chart(sentiment_time)

# Common Themes
st.subheader("📌 Most Common Email Themes")
theme_counts = df["Theme"].value_counts()
st.bar_chart(theme_counts)

# Top Leads
st.subheader("🎯 Top Priority Leads")
st.dataframe(df[df["Sender_Type"] == "Lead"].sort_values(by="Lead_Score", ascending=False).head(10))

# Churn Risk
st.subheader("⚠️ Clients at Risk of Churn")
df["Churn_Risk"] = df.apply(lambda x: "High" if (x["Sender_Type"] == "Current Client" and x["Sentiment"] == "Negative") else "Low", axis=1)
st.dataframe(df[df["Churn_Risk"] == "High"])

st.write("🔍 **Insights Summary:**")
st.write("- Track sentiment trends to monitor customer satisfaction.")
st.write("- Identify high-priority leads quickly.")
st.write("- Understand prevalent client concerns.")
st.write("- Highlight clients who may churn.")
