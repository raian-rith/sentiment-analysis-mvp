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

# 📌 Step 2: Unique Emails with Balanced Sentiments
lead_emails = [
    "I love your marketing approach and I think it would be a great fit for my company!",
    "I’m interested in your SEO services, but I need more details before deciding.",
    "I'm frustrated that I haven’t received any response after my initial inquiry!",
    "Can you explain your pricing structure? I'm evaluating multiple agencies.",
    "What industries do you specialize in for lead generation?",
    "I've read great reviews about your agency. Excited to get started!",
    "I was recommended to your agency. How do you measure campaign success?",
    "How long does it typically take to see results from an SEO campaign?",
    "I need a flexible contract. Are your services month-to-month?",
    "I had a bad experience with another agency. What makes you different?"
]

client_emails = [
    "Your team is amazing! We've seen a huge boost in traffic thanks to your SEO strategy.",
    "Our Google Ads aren’t converting well. Can you review them?",
    "We need better performance tracking. Some metrics seem off.",
    "Our engagement on social media has dropped. What can we do?",
    "We need a complete brand refresh. Can you help with that?",
    "We’re seeing higher bounce rates on our site. Can you investigate?",
    "I need a more aggressive email marketing strategy. Can you implement that?",
    "Our cost-per-click is too high. How do we optimize spending?",
    "The last campaign didn’t perform well. We need urgent improvements!",
    "Our competitors seem to be outbidding us in ads. Can we adjust targeting?"
]

# 📌 Step 3: Adjust Email Timestamp Distribution
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
    "Sender": lead_senders[:10] + client_senders[:10],
    "Email_Text": lead_emails[:10] + client_emails[:10],
    "Sender_Type": ["Lead"] * 10 + ["Current Client"] * 10,
    "Timestamp": timestamps[:20]
}

df = pd.DataFrame(data)

# 📌 AI-Powered Sentiment Analysis
df["Sentiment"] = df["Email_Text"].apply(lambda x: "Positive" if sia.polarity_scores(x)["compound"] > 0.2 else 
                                         "Negative" if sia.polarity_scores(x)["compound"] < -0.2 else "Neutral")

# 📌 AI-Driven Urgency Assignment
def determine_urgency(text, sentiment):
    urgent_keywords = ["urgent", "asap", "immediate", "not working", "fix this", "need help", "still waiting", "unacceptable"]
    if any(word in text.lower() for word in urgent_keywords):
        return "Urgent"
    if sentiment == "Negative":
        return "Urgent"
    return "Normal"

df["Urgency"] = df.apply(lambda row: determine_urgency(row["Email_Text"], row["Sentiment"]), axis=1)

# 📌 Streamlit Dashboard
st.set_page_config(page_title="AI-Powered Customer Insights", layout="wide")
st.title("📊 AI-Powered Customer Insights Dashboard")

# Sidebar Filters
st.sidebar.header("🔍 Filters")
sender_filter = st.sidebar.selectbox("Filter by Sender Type", ["All", "Lead", "Current Client"])
sentiment_filter = st.sidebar.selectbox("Filter by Sentiment", ["All", "Positive", "Neutral", "Negative"])
urgency_filter = st.sidebar.selectbox("Filter by Urgency", ["All", "Urgent", "Normal"])

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

# 📌 Sentiment Distribution (Pie Chart)
st.subheader("📊 Sentiment Breakdown")
sentiment_counts = df["Sentiment"].value_counts()
fig, ax = plt.subplots(figsize=(5, 5))
ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=["green", "gray", "red"])
st.pyplot(fig)

# 📌 Urgency Levels Over Time (Line Chart)
st.subheader("📈 Urgency Trends Over Time")
urgency_trends = df.groupby(["Timestamp", "Urgency"]).size().unstack().fillna(0)
st.line_chart(urgency_trends)

# 📌 Email Volume by Date (Bar Chart)
st.subheader("📅 Email Volume Over Time")
email_counts = df["Timestamp"].value_counts().sort_index()
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x=email_counts.index, y=email_counts.values, ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# 📌 Top Senders by Email Volume (Bar Chart)
st.subheader("📊 Top Senders by Email Count")
top_senders = df["Sender"].value_counts().head(10)
fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(y=top_senders.index, x=top_senders.values, ax=ax)
st.pyplot(fig)

st.subheader("🔍 Key Insights")
st.write("- **Urgency trends help understand peak escalation periods.**")
st.write("- **Sentiment analysis provides insights into customer satisfaction.**")
st.write("- **Top senders highlight key customers engaging frequently.**")
