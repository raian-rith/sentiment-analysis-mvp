import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download("vader_lexicon")
nltk.download("stopwords")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# 📌 STEP 1: Unique Dummy Emails for Leads and Clients (No Reuse, No Random Selection)
lead_emails = [
    "I am considering digital marketing services. How do you help startups?",
    "What industries do you specialize in for lead generation?",
    "How do you measure success in a marketing campaign?",
    "Can I see some real-world case studies before making a decision?",
    "Do you offer a free consultation before signing up?",
    "I am unsure if inbound marketing is the right fit for my business. Can you advise?",
    "How does your pricing compare to competitors?",
    "How long does it take to see results from an SEO campaign?",
    "Can I get a breakdown of your PPC advertising approach?",
    "What type of reporting do you provide to track performance?",
    "Do you have experience working with B2B SaaS companies?",
    "How flexible are your contract terms? Is there a trial period?",
    "I need help understanding Google Ads better. Do you offer training?",
    "How do you ensure that leads generated are high quality?",
    "I am evaluating multiple agencies. What sets you apart?",
    "Are there any hidden costs I should be aware of?",
    "What platforms do you use for social media advertising?",
    "How do you handle content strategy for niche industries?",
    "What ROI should I expect in the first three months?",
    "I’ve tried marketing before and it didn’t work. Why should I trust this?",
    "Can I start small and scale up later?",
    "Do you handle email marketing automation?",
    "What kind of creative assets do you provide?",
    "How does your team stay updated on industry trends?",
    "Do you offer ongoing strategy adjustments based on campaign performance?"
]

client_emails = [
    "Our Google Ads campaigns aren't converting as expected. Can you review them?",
    "We need to optimize our website for better organic search rankings.",
    "Can you help us improve email engagement rates?",
    "We launched a new product, but our campaign isn't driving sales.",
    "Our team is struggling to create engaging content. Can you assist?",
    "We need to adjust our marketing approach for a new audience segment.",
    "Can we schedule a strategy session to realign our goals?",
    "We need better lead nurturing workflows in our CRM.",
    "Social media engagement is declining. Any recommendations?",
    "Our website bounce rate is too high. What can we do?",
    "We need a full performance report for the last six months.",
    "Our competitors seem to be ranking higher on Google. What are they doing differently?",
    "Can you help us identify why our cost per acquisition is increasing?",
    "We need new ad creatives that better resonate with our audience.",
    "Our YouTube ads aren't performing well. Can we adjust targeting?",
    "We want to run A/B tests on different ad variations. How do we do that?",
    "Our analytics tracking seems inaccurate. Can you verify our data?",
    "We need a fresh approach for holiday season promotions.",
    "Our previous content calendar didn’t perform well. What should change?",
    "We have budget constraints. How do we maximize ad spend efficiency?",
    "Our webinar campaign didn't attract enough registrations. What went wrong?",
    "We need an updated competitor analysis with fresh insights.",
    "Our landing pages aren’t converting well. Can you optimize them?",
    "Our brand messaging seems inconsistent across channels. Can you unify it?",
    "What’s the best approach for remarketing to previous site visitors?"
]

# 📌 STEP 2: Generate Data Without Random Selection
data = {
    "Email_Text": lead_emails + client_emails,  # 25 leads, 25 clients
    "Sender_Type": ["Lead"] * 25 + ["Current Client"] * 25,  # Assign correct sender type
    "Timestamp": [datetime.datetime(2024, random.randint(1, 3), random.randint(1, 28)) for _ in range(50)]
}

df = pd.DataFrame(data)

# 📌 STEP 3: Improved Sentiment Analysis (VADER + Rule-Based NLP)
def get_sentiment(text):
    sentiment_score = sia.polarity_scores(text)["compound"]
    
    negative_phrases = [
        "not happy", "not satisfied", "not working", "not impressed", "should we consider", 
        "concerns aren’t being prioritized", "waiting too long", "frustrated", "very disappointed"
    ]
    
    if any(phrase in text.lower() for phrase in negative_phrases):
        return "Negative"

    if sentiment_score > 0.2:
        return "Positive"
    elif sentiment_score < -0.2:
        return "Negative"
    else:
        return "Neutral"

df["Sentiment"] = df["Email_Text"].apply(get_sentiment)

# 📌 STEP 4: AI-Driven Urgency Assignment
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

# 📌 STEP 5: Streamlit Dashboard
st.set_page_config(page_title="Customer Insights Dashboard", layout="wide")
st.title("📊 Customer Sentiment & Lead Prioritization Dashboard")

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

# Sentiment Distribution
st.subheader("📊 Sentiment Distribution Across Emails")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=filtered_df["Sentiment"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# Urgency Breakdown
st.subheader("⏳ Urgency Levels in Emails")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=filtered_df["Urgency"], palette="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("🔍 Key Insights")
st.write("- **Increase Engagement** with high-priority leads.")
st.write("- **Improve Response Time** for urgent client requests.")
st.write("- **Address Negative Feedback** to reduce churn risk.")
st.write("- **Optimize Messaging** based on frequently used words.")
