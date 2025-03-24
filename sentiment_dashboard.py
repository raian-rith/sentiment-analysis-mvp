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
from openai import OpenAI

# Download necessary NLTK resources
nltk.download("vader_lexicon")
nltk.download("stopwords")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# OpenAI API (For "Talk to Your Data" Feature)
OPENAI_API_KEY = "your-api-key-here"  # Replace with your OpenAI key
client = OpenAI(api_key=OPENAI_API_KEY)

# 📌 STEP 1: Dummy Sender Names
lead_senders = [
    "John Smith", "Emily Davis", "Michael Johnson", "Sophia Martinez", "David Brown",
    "Olivia Taylor", "James Wilson", "Isabella Moore", "Benjamin Anderson", "Charlotte White",
    "Daniel Harris", "Ava Thomas", "Matthew Lewis", "Sophia Robinson", "Lucas Scott",
    "Liam King", "Emma Young", "Henry Walker", "Ethan Allen", "Mia Hall",
    "Noah Adams", "Grace Wright", "William Clark", "Amelia Mitchell", "Elijah Carter"
]

client_senders = [
    "Mark Thompson", "Hannah Perez", "Chris Evans", "Laura Rodriguez", "Jacob Parker",
    "Natalie Stewart", "Ryan Brooks", "Chloe Foster", "Nathaniel Reed", "Samantha Ross",
    "Tyler Jenkins", "Alyssa Barnes", "Evan Ward", "Jessica Bailey", "Brandon Cooper",
    "Sophia Rivera", "Aaron Green", "Madison Simmons", "Zachary Murphy", "Katherine Hughes",
    "Jason Ramirez", "Victoria Cox", "Andrew Butler", "Isabelle Torres", "Caleb Patterson"
]

# 📌 STEP 2: Unique Emails with Correct Senders
lead_emails = [
    "I am considering digital marketing services. How do you help startups?",
    "What industries do you specialize in for lead generation?",
    "How do you measure success in a marketing campaign?",
    "Can I see some real-world case studies before making a decision?",
    "Do you offer a free consultation before signing up?"
]

client_emails = [
    "Our Google Ads campaigns aren't converting as expected. Can you review them?",
    "We need to optimize our website for better organic search rankings.",
    "Can you help us improve email engagement rates?",
    "We launched a new product, but our campaign isn't driving sales.",
    "Our team is struggling to create engaging content. Can you assist?"
]

# 📌 STEP 3: Generate Data with AI Features
data = {
    "Sender": lead_senders + client_senders,
    "Email_Text": lead_emails + client_emails,  
    "Sender_Type": ["Lead"] * len(lead_senders) + ["Current Client"] * len(client_senders),
    "Timestamp": [datetime.datetime(2024, random.randint(1, 3), random.randint(1, 28)) for _ in range(50)]
}

df = pd.DataFrame(data)

# 📌 STEP 4: AI-Powered Sentiment Analysis (VADER + Custom NLP)
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

# 📌 STEP 5: AI-Driven Urgency Assignment
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

# 📌 STEP 6: Streamlit Dashboard with AI Query Feature
st.set_page_config(page_title="AI-Powered Customer Insights", layout="wide")
st.title("🤖 AI-Powered Customer Insights & Query System")

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

# 📌 AI Query Section: "Talk to Your Data"
st.subheader("💬 Talk to Your Data")
query = st.text_input("Ask a question (e.g., 'Show me urgent leads')")

if query:
    response = client.Completions.create(
        model="gpt-4",
        prompt=f"You are a data analyst. Given this dataset: {df.to_dict()}, filter the data based on this user request: {query}",
        max_tokens=200
    )
    result = response["choices"][0]["text"]
    
    st.write("### AI Response:")
    st.write(result)

# 📌 Urgency Breakdown
st.subheader("⏳ Urgency Levels")
st.write("This shows how urgency varies across leads and clients.")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(x=df["Urgency"], hue=df["Sender_Type"], palette="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 Sentiment Trends
st.subheader("📈 Sentiment Over Time")
st.write("Understand how customer sentiment has changed over time.")
sentiment_over_time = df.groupby(["Timestamp", "Sentiment"]).size().unstack().fillna(0)
st.line_chart(sentiment_over_time)

# 📌 AI-Detected High-Risk Clients
st.subheader("⚠️ High-Risk Clients")
st.write("Clients with negative sentiment & urgent requests.")
st.dataframe(df[(df["Sender_Type"] == "Current Client") & (df["Sentiment"] == "Negative")])

st.subheader("🔍 Key Insights")
st.write("- **AI can filter emails based on natural language queries.**")
st.write("- **Clients with urgent and negative sentiment should be prioritized.**")
st.write("- **Understanding sentiment trends helps with proactive customer engagement.**")
