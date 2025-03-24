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

### 📌 STEP 1: Improved Dummy Emails (More Realistic, Includes Negative Sentiments)
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

themes = [
    "Lead Inquiry", "SEO Interest", "Case Study Request", "Marketing Strategy", 
    "Client Performance Review", "Ad Optimization", "Website Traffic Issue", 
    "Campaign Planning", "Product Launch", "Reporting Issues"
]

# Generate Dataset with Balanced Sentiments
data = {
    "Email_Text": [random.choice(lead_emails) if random.random() > 0.5 else random.choice(client_emails) for _ in range(100)],
    "Sender_Type": [random.choice(["Lead", "Current Client"]) for _ in range(100)],
    "Urgency": [random.choice(["Urgent", "Normal", "Low Priority"]) for _ in range(100)],
    "Theme": [random.choice(themes) for _ in range(100)],
    "Timestamp": [datetime.datetime(2024, random.randint(1, 3), random.randint(1, 28)) for _ in range(100)]
}

df = pd.DataFrame(data)

### 📌 STEP 2: Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"

df["Sentiment"] = df["Email_Text"].apply(get_sentiment)

### 📌 STEP 3: Lead Scoring
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

### 📌 STEP 4: Streamlit Dashboard
st.set_page_config(page_title="Customer Insights Dashboard", layout="wide")

st.title("📊 Customer Sentiment & Lead Prioritization Dashboard")

# Layout: Create Columns for Better Structure
col1, col2 = st.columns([1, 2])

# Top Leads Section
with col1:
    st.subheader("🎯 Top High-Priority Leads")
    st.write("These are the most promising leads based on sentiment, urgency, and engagement level.")
    st.dataframe(df[df["Sender_Type"] == "Lead"].sort_values(by="Lead_Score", ascending=False).head(10))

# Sentiment Overview Section
with col2:
    st.subheader("📊 Overall Sentiment Distribution")
    st.write("A snapshot of customer sentiment across all messages.")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=df["Sentiment"], palette="coolwarm", ax=ax)
    st.pyplot(fig)

# Word Cloud Section
st.subheader("🌟 Word Cloud - Most Common Topics (Leads vs. Clients)")
col3, col4 = st.columns(2)

def generate_wordcloud(text_list):
    words = " ".join(text_list).lower()
    wordcloud = WordCloud(stopwords=set(stopwords.words('english')), background_color="white", width=800, height=400).generate(words)
    return wordcloud

with col3:
    st.write("🔍 Most Used Words by **Leads**")
    lead_texts = df[df["Sender_Type"] == "Lead"]["Email_Text"]
    if not lead_texts.empty:
        wordcloud_lead = generate_wordcloud(lead_texts)
        st.image(wordcloud_lead.to_array())
    else:
        st.write("No lead messages available.")

with col4:
    st.write("💼 Most Used Words by **Clients**")
    client_texts = df[df["Sender_Type"] == "Current Client"]["Email_Text"]
    if not client_texts.empty:
        wordcloud_client = generate_wordcloud(client_texts)
        st.image(wordcloud_client.to_array())
    else:
        st.write("No client messages available.")

# AI-Suggested Responses Section
st.subheader("✉️ AI-Suggested Responses for Emails")
st.dataframe(df[["Email_Text", "Theme", "Sentiment", "Suggested_Response"]])

# Churn Risk Analysis Section
st.subheader("⚠️ High-Risk Clients (May Churn)")
st.write("These clients have expressed negative sentiment and may require retention efforts.")
st.dataframe(df[df["Sender_Type"] == "Current Client"][(df["Sentiment"] == "Negative")])

st.write("🔍 **Key Takeaways:**")
st.write("- **Increase Engagement** with high-priority leads to improve conversion rates.")
st.write("- **Improve Response Time** for urgent client requests.")
st.write("- **Address Negative Feedback** to prevent churn.")
st.write("- **Enhance Marketing Messaging** based on frequently used keywords.")

