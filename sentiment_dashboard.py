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
from wordcloud import WordCloud

# Download stopwords for NLP processing
nltk.download('stopwords')

# 📌 STEP 1: Improved Dummy Emails (Balanced Sentiments)
lead_emails = [
    "I'm interested in learning how your agency can help generate more leads for my business.",
    "Can you provide more details on your SEO services? I'm evaluating different agencies.",
    "Do you have case studies of companies in my industry that you have worked with?",
    "Our marketing budget is limited, and I want to ensure we get the best return on investment.",
    "I'm not convinced your services are different from competitors. What makes you stand out?",
    "I've tried other agencies, but I haven’t seen results. Can you prove your strategy works?",
    "Your case studies seem outdated. Do you have any recent success stories?",
    "I reached out last week, but I haven’t heard back. Is this the level of support I can expect?"
]

client_emails = [
    "We need a more effective paid ad strategy. Can we optimize our current campaigns?",
    "Our website traffic has dropped in the last month. Can you analyze and suggest improvements?",
    "Can we schedule a meeting to review last quarter’s performance and plan for the next?",
    "We have a new product launch coming up. Can you help us with a targeted campaign?",
    "The latest reports are missing some key metrics. Can you update them with more insights?",
    "Your team has been fantastic! We've seen a 30% increase in conversions since working with you.",
    "We're frustrated with the response time on support requests. This needs to improve.",
    "The campaign is underperforming, and we need immediate action. What are our options?",
    "Your team promised a content plan two weeks ago, and we're still waiting. This is unacceptable.",
    "I feel like our concerns aren’t being prioritized. Should we consider other agencies?"
]

# Generate Dataset
data = {
    "Email_Text": [random.choice(lead_emails) if random.random() > 0.5 else random.choice(client_emails) for _ in range(100)],
    "Sender_Type": [random.choice(["Lead", "Current Client"]) for _ in range(100)],
    "Timestamp": [datetime.datetime(2024, random.randint(1, 3), random.randint(1, 28)) for _ in range(100)]
}

df = pd.DataFrame(data)

# 📌 STEP 2: Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"

df["Sentiment"] = df["Email_Text"].apply(get_sentiment)

# 📌 STEP 3: AI-Driven Urgency Assignment
def determine_urgency(text, sentiment):
    urgent_keywords = ["urgent", "asap", "immediate", "not working", "fix this", "need help", "still waiting", "unacceptable"]
    text_lower = text.lower()
    
    # If critical words appear, mark as Urgent
    if any(word in text_lower for word in urgent_keywords):
        return "Urgent"

    # If sentiment is negative, increase likelihood of urgency
    if sentiment == "Negative":
        return "Urgent"
    
    # If sentiment is neutral, assume normal priority
    if sentiment == "Neutral":
        return "Normal"
    
    # If sentiment is positive, assume low priority
    return "Low Priority"

df["Urgency"] = df.apply(lambda row: determine_urgency(row["Email_Text"], row["Sentiment"]), axis=1)

# 📌 STEP 4: Streamlit Dashboard Layout
st.set_page_config(page_title="Customer Insights Dashboard", layout="wide")

st.title("📊 Customer Sentiment & Lead Prioritization Dashboard")

# 📌 Filters Section
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
st.subheader("📊 Sentiment Distribution Across Emails")
st.write("This chart provides an overview of customer sentiment. A high number of negative messages may indicate customer dissatisfaction, while positive messages suggest strong engagement.")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=filtered_df["Sentiment"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 Urgency Breakdown
st.subheader("⏳ Urgency Levels in Emails")
st.write("Analyzing urgency levels helps prioritize customer interactions.")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=filtered_df["Urgency"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 Sentiment Trend Over Time
st.subheader("📈 Sentiment Trends Over Time")
sentiment_over_time = filtered_df.groupby(["Timestamp", "Sentiment"]).size().unstack().fillna(0)
st.line_chart(sentiment_over_time)

# 📌 Word Cloud Section
st.subheader("🌟 Word Cloud - Most Common Words")
text_combined = " ".join(filtered_df["Email_Text"]).lower()
wordcloud = WordCloud(stopwords=set(stopwords.words('english')), background_color="white", width=800, height=400).generate(text_combined)
st.image(wordcloud.to_array())

# 📌 High-Risk Clients
st.subheader("⚠️ High-Risk Clients (Churn Warning)")
st.write("These clients have expressed negative sentiment and may require retention efforts.")
st.dataframe(filtered_df[(filtered_df["Sender_Type"] == "Current Client") & (filtered_df["Sentiment"] == "Negative")])

st.subheader("🔍 Key Insights")
st.write("- **Increase Engagement** with high-priority leads.")
st.write("- **Improve Response Time** for urgent client requests.")
st.write("- **Address Negative Feedback** to reduce churn risk.")
st.write("- **Optimize Messaging** based on frequently used words.")

