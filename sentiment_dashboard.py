import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob

# Generate dummy data (same as before)
import random

# Predefined sample texts for each theme
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

# Create dataset
data = {
    "Email_Text": random.choices(emails, k=100),
    "Sender_Type": [random.choice(["Lead", "Current Client"]) for _ in range(100)],
    "Urgency": [random.choice(["Urgent", "Normal", "Low Priority"]) for _ in range(100)],
    "Theme": [random.choice(themes) for _ in range(100)]
}

df = pd.DataFrame(data)

# Sentiment Analysis Function
def get_sentiment(text):
    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    return "Positive" if polarity > 0.2 else "Negative" if polarity < -0.2 else "Neutral"

# Apply sentiment analysis
df["Sentiment"] = df["Email_Text"].apply(get_sentiment)

# Streamlit App
st.title("📊 Sentiment Analysis Dashboard")
st.sidebar.header("Filters")

# Filter by Lead/Client
sender_type_filter = st.sidebar.selectbox("Filter by Sender Type", ["All", "Lead", "Current Client"])
if sender_type_filter != "All":
    df = df[df["Sender_Type"] == sender_type_filter]

# Filter by Urgency
urgency_filter = st.sidebar.selectbox("Filter by Urgency Level", ["All", "Urgent", "Normal", "Low Priority"])
if urgency_filter != "All":
    df = df[df["Urgency"] == urgency_filter]

# Display filtered data
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

# Most Common Themes
st.subheader("📌 Most Common Email Themes")
theme_counts = df["Theme"].value_counts()
st.bar_chart(theme_counts)

st.write("🔍 **Insights:**")
st.write("- See how sentiment varies between leads & current clients.")
st.write("- Identify urgency trends (Are urgent emails mostly negative?).")
st.write("- Find out which themes dominate customer inquiries.")

