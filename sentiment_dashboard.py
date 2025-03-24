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

# 📌 STEP 2: Unique Emails for Leads and Clients with Balanced Sentiments
lead_emails = [
    "I love your marketing approach and I think it would be a great fit for my company!",  # Positive
    "I’m interested in your SEO services, but I need more details before deciding.",  # Neutral
    "I'm frustrated that I haven’t received any response after my initial inquiry!",  # Negative
    "Can you explain your pricing structure? I'm evaluating multiple agencies.",  # Neutral
    "What industries do you specialize in for lead generation?",  # Neutral
    "I've read great reviews about your agency. Excited to get started!",  # Positive
    "I was recommended to your agency. How do you measure campaign success?",  # Neutral
    "How long does it typically take to see results from an SEO campaign?",  # Neutral
    "I need a flexible contract. Are your services month-to-month?",  # Neutral
    "I had a bad experience with another agency. What makes you different?",  # Negative
    "I’m looking for an aggressive lead generation strategy. Can you help?",  # Neutral
    "Your client testimonials look great. How do you ensure similar results?",  # Positive
    "I tried running PPC ads before, but it was a disaster. What do you suggest?",  # Negative
    "What type of reports do you provide? I want full transparency.",  # Neutral
    "I need to start a campaign ASAP. How quickly can you onboard new clients?",  # Urgent
    "I heard your content marketing is effective. Can I see some samples?",  # Neutral
    "Your agency was recommended by a colleague. Let’s schedule a consultation.",  # Positive
    "I am unsure if I need paid ads or organic SEO. Can you guide me?",  # Neutral
    "I want to focus on video marketing. Do you offer services for YouTube?",  # Neutral
    "I saw your ad but couldn't find case studies on your website. Can you share?",  # Neutral
    "What ROI should I expect in the first six months?",  # Neutral
    "I need help building my brand from scratch. Do you work with startups?",  # Neutral
    "I tried LinkedIn Ads, but got no results. Can you optimize them?",  # Negative
    "Do you offer social media marketing for e-commerce brands?",  # Neutral
    "What’s the best platform for generating high-quality B2B leads?",  # Neutral
]

client_emails = [
    "Your team is amazing! We've seen a huge boost in traffic thanks to your SEO strategy.",  # Positive
    "Our Google Ads aren’t converting well. Can you review them?",  # Neutral
    "We need better performance tracking. Some metrics seem off.",  # Neutral
    "Our engagement on social media has dropped. What can we do?",  # Neutral
    "We need a complete brand refresh. Can you help with that?",  # Positive
    "We’re seeing higher bounce rates on our site. Can you investigate?",  # Negative
    "I need a more aggressive email marketing strategy. Can you implement that?",  # Neutral
    "Our cost-per-click is too high. How do we optimize spending?",  # Neutral
    "The last campaign didn’t perform well. We need urgent improvements!",  # Urgent & Negative
    "Our competitors seem to be outbidding us in ads. Can we adjust targeting?",  # Neutral
    "What do you recommend for our Black Friday promotion?",  # Neutral
    "We need a new strategy for better lead nurturing in our CRM.",  # Neutral
    "Our YouTube ads aren’t converting well. Can we refine our approach?",  # Neutral
    "Our LinkedIn outreach isn’t generating meetings. Any suggestions?",  # Neutral
    "We want to expand into a new market. Can you guide our marketing strategy?",  # Positive
    "Can we schedule a quarterly performance review?",  # Neutral
    "We need a competitor analysis to benchmark our performance.",  # Neutral
    "Our video marketing results are mixed. Can we analyze what’s working?",  # Neutral
    "Our retargeting ads aren’t performing as expected. What adjustments are needed?",  # Negative
    "We need new creatives for our display ads. Can your team handle that?",  # Neutral
    "What’s the best budget allocation for next quarter?",  # Neutral
    "Our customer retention is declining. Can we adjust messaging?",  # Negative
    "We’re launching a new product. Can you create a multi-channel campaign?",  # Positive
    "Our blog content isn’t getting traction. How do we improve reach?",  # Neutral
    "We need more automation in our email workflows. Can you implement that?",  # Neutral
]

# 📌 Step 3: Create Dataframe
data = {
    "Sender": lead_senders[:25] + client_senders[:25],
    "Email_Text": lead_emails + client_emails,
    "Sender_Type": ["Lead"] * 25 + ["Current Client"] * 25,
    "Timestamp": [datetime.datetime(2024, random.randint(1, 3), random.randint(1, 28)) for _ in range(50)],
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
