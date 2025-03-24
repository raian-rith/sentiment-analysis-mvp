import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from collections import Counter

nltk.download('stopwords')

# Generate Dummy Data
emails = [
    "I need help understanding the pricing for your services.",
    "Your team has been amazing! Our engagement rates have skyrocketed.",
    "I am very disappointed with the slow response to my support ticket. Please fix this ASAP.",
    "We are interested in learning more about your content marketing solutions.",
    "My company has been using your services, and we want to explore upgrading our plan.",
    "I have an issue with my invoice. It seems incorrect, and I need clarification.",
    "We are considering your agency for our marketing needs. Can you share case studies?",
    "I'm still waiting for a follow-up from your team about our last discussion.",
    "Our campaign isn't performing as expected. Can we schedule a call to discuss improvements?",
    "Thank you for the amazing support! Our sales numbers have improved since working with you."
]

themes = ["Pricing Inquiry", "Positive Feedback", "Support Issue", "General Inquiry", "Upgrade Request",
          "Billing Issue", "Case Study Request", "Follow-up Needed", "Campaign Performance", "Success Story"]

# Create dataset with 100 random emails
data = {
    "Email_Text": random.choices(emails, k=100),
    "Sender_Type": [random.choice(["Lead", "Current Client"]) for _ in range(100)],
    "Urgency": [random.choice(["Urgent", "Normal", "Low Priority"]) for _ in range(100)],
    "Theme": [random.choice(themes) for _ in range(100)],
    "Timestamp": [datetime.datetime(2024, random.randint(1, 3), random.randint(1, 28)) for _ in range(100)]
}

df = pd.DataFrame(data)

# Sentiment Analysis Function
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
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

# AI-Suggested Responses
def generate_response(row):
    if row["Sentiment"] == "Negative":
        return "We're sorry for the inconvenience. Our team is looking into this and will resolve it ASAP."
    elif row["Theme"] == "Pricing Inquiry":
        return "Thank you for reaching out! Here’s a link to our pricing details: [Insert Link]"
    elif row["Theme"] == "Upgrade Request":
        return "Great to hear you're interested in upgrading! Let's schedule a quick call to discuss your options."
    elif row["Theme"] == "Support Issue":
        return "We appreciate your patience. Our support team will follow up with you shortly."
    else:
        return "Thank you for your message! Our team will review and get back to you soon."

df["Suggested_Response"] = df.apply(generate_response, axis=1)

# Identify Clients at Risk of Churning
df["Churn_Risk"] = df.apply(lambda row: "High" if (row["Sender_Type"] == "Current Client" and row["Sentiment"] == "Negative") else "Low", axis=1)

# Keyword Extraction Function
def extract_keywords(text_list):
    stop_words = set(stopwords.words('english'))
    words = " ".join(text_list).lower().split()
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    return Counter(filtered_words).most_common(10)

positive_words = extract_keywords(df[df["Sentiment"] == "Positive"]["Email_Text"])
negative_words = extract_keywords(df[df["Sentiment"] == "Negative"]["Email_Text"])

# Streamlit App
st.title("📊 Sentiment Analysis Dashboard")
st.sidebar.header("Filters")

# Filter Options
sender_type_filter = st.sidebar.selectbox("Filter by Sender Type", ["All", "Lead", "Current Client"])
if sender_type_filter != "All":
    df = df[df["Sender_Type"] == sender_type_filter]

urgency_filter = st.sidebar.selectbox("Filter by Urgency Level", ["All", "Urgent", "Normal", "Low Priority"])
if urgency_filter != "All":
    df = df[df["Urgency"] == urgency_filter]

# Display Filtered Data
st.subheader("📩 Email Dataset (Filtered)")
st.dataframe(df)

# Sentiment Distribution Chart
st.subheader("📊 Sentiment Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["Sentiment"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# Sentiment Breakdown by Urgency
st.subheader("⏳ Sentiment Breakdown by Urgency Level")
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(x=df["Urgency"], hue=df["Sentiment"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# Sentiment Over Time
st.subheader("📈 Sentiment Trend Over Time")
sentiment_over_time = df.groupby(["Timestamp", "Sentiment"]).size().unstack().fillna(0)
st.line_chart(sentiment_over_time)

# Most Common Themes
st.subheader("📌 Most Common Email Themes")
theme_counts = df["Theme"].value_counts()
st.bar_chart(theme_counts)

# Display Top Leads
st.subheader("🎯 Top High-Priority Leads")
st.dataframe(df[df["Sender_Type"] == "Lead"].sort_values(by="Lead_Score", ascending=False).head(10))

# AI-Suggested Responses
st.subheader("✉️ AI-Suggested Responses")
st.dataframe(df[["Email_Text", "Theme", "Sentiment", "Suggested_Response"]])

# Churn Risk Analysis
st.subheader("⚠️ High-Risk Clients (May Churn)")
st.dataframe(df[df["Churn_Risk"] == "High"])

# Keyword Analysis
st.subheader("🔍 Common Words in Positive Emails")
st.write(positive_words)

st.subheader("⚠️ Common Words in Negative Emails")
st.write(negative_words)

st.write("🔍 **Insights:**")
st.write("- Track sentiment trends over time.")
st.write("- Identify high-priority leads for sales.")
st.write("- Discover common customer concerns.")
st.write("- Detect clients at risk of leaving.")
