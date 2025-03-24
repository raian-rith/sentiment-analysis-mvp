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

# Download stopwords for word processing
nltk.download('stopwords')

# Realistic Dummy Emails for Leads & Clients
lead_emails = [
    "I'm interested in learning how your agency can help generate more leads for my business.",
    "Can you provide more details on your SEO services? I'm evaluating different agencies.",
    "Do you have case studies of companies in my industry that you have worked with?",
    "What kind of content marketing strategies do you specialize in?",
    "I'm looking for a long-term marketing partner. What sets you apart from other agencies?"
]

client_emails = [
    "We need a more effective paid ad strategy. Can we optimize our current campaigns?",
    "Our website traffic has dropped in the last month. Can you analyze and suggest improvements?",
    "Can we schedule a meeting to review last quarter’s performance and plan for the next?",
    "We have a new product launch coming up. Can you help us with a targeted campaign?",
    "The latest reports are missing some key metrics. Can you update them with more insights?"
]

themes = [
    "Lead Inquiry", "SEO Interest", "Case Study Request", "Marketing Strategy", "Client Performance Review",
    "Ad Optimization", "Website Traffic Issue", "Campaign Planning", "Product Launch", "Reporting Issues"
]

# Create dataset with 100 random emails
data = {
    "Email_Text": [random.choice(lead_emails) if random.random() > 0.5 else random.choice(client_emails) for _ in range(100)],
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
    elif row["Theme"] == "Lead Inquiry":
        return "Thanks for reaching out! We’d love to discuss how we can help. Let's schedule a call!"
    elif row["Theme"] == "SEO Interest":
        return "Our SEO services focus on boosting organic traffic. We’ll send over some details!"
    elif row["Theme"] == "Case Study Request":
        return "Sure! We have case studies relevant to your industry. We'll send them over."
    elif row["Theme"] == "Client Performance Review":
        return "Let's schedule a call to review last quarter’s performance and set new goals."
    else:
        return "Thank you for your message! Our team will review and get back to you soon."

df["Suggested_Response"] = df.apply(generate_response, axis=1)

# Identify Clients at Risk of Churning
df["Churn_Risk"] = df.apply(lambda row: "High" if (row["Sender_Type"] == "Current Client" and row["Sentiment"] == "Negative") else "Low", axis=1)

# Function to Generate Word Cloud
def generate_wordcloud(text_list):
    words = " ".join(text_list).lower()
    wordcloud = WordCloud(stopwords=set(stopwords.words('english')), background_color="white", width=800, height=400).generate(words)
    return wordcloud

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

# Word Cloud for Leads
st.subheader("🌟 Word Cloud - Most Used Words by Leads")
lead_texts = df[df["Sender_Type"] == "Lead"]["Email_Text"]
if not lead_texts.empty:
    wordcloud_lead = generate_wordcloud(lead_texts)
    st.image(wordcloud_lead.to_array())
else:
    st.write("No lead messages available.")

# Word Cloud for Clients
st.subheader("💼 Word Cloud - Most Used Words by Clients")
client_texts = df[df["Sender_Type"] == "Current Client"]["Email_Text"]
if not client_texts.empty:
    wordcloud_client = generate_wordcloud(client_texts)
    st.image(wordcloud_client.to_array())
else:
    st.write("No client messages available.")

# Display Top Leads
st.subheader("🎯 Top High-Priority Leads")
st.dataframe(df[df["Sender_Type"] == "Lead"].sort_values(by="Lead_Score", ascending=False).head(10))

# AI-Suggested Responses
st.subheader("✉️ AI-Suggested Responses")
st.dataframe(df[["Email_Text", "Theme", "Sentiment", "Suggested_Response"]])

# Churn Risk Analysis
st.subheader("⚠️ High-Risk Clients (May Churn)")
st.dataframe(df[df["Churn_Risk"] == "High"])

st.write("🔍 **Insights:**")
st.write("- Track sentiment trends over time.")
st.write("- Identify high-priority leads for sales.")
st.write("- Discover common customer concerns.")
st.write("- Detect clients at risk of leaving.")
