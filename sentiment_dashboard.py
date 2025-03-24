import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import nltk
import openai  # Correct OpenAI import
import os
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download("vader_lexicon")

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# OpenAI API Key (Use Environment Variable)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Store key securely
openai.api_key = OPENAI_API_KEY  # Set API Key for OpenAI

# 📌 Step 4: Sentiment Analysis Function
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

# 📌 Step 5: AI-Driven Urgency Assignment
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

# 📌 Step 6: Streamlit Dashboard
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

if query and OPENAI_API_KEY:
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst."},
                {"role": "user", "content": f"Given this dataset: {df.to_dict()}, filter the data based on this request: {query}"}
            ],
            max_tokens=200
        )
        result = response["choices"][0]["message"]["content"]
        
        st.write("### AI Response:")
        st.write(result)
    
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")

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
