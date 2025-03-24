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
    "Email_Text": lead_emails[:25] + client_emails[:25],
    "Sender_Type": ["Lead"] * 25 + ["Current Client"] * 25,
    "Timestamp": [datetime.datetime(year=2024, month=random.randint(1, 12), day=random.randint(1, 28)) for _ in range(50)]  # ✅ FIXED
}

df = pd.DataFrame(data)

# 📌 Step 2: AI-Powered Sentiment Analysis (VADER)
df["Sentiment"] = df["Email_Text"].apply(lambda x: "Positive" if sia.polarity_scores(x)["compound"] > 0.2 else 
                                         "Negative" if sia.polarity_scores(x)["compound"] < -0.2 else "Neutral")

# 📌 Step 3: AI-Driven Urgency Assignment
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

# 📌 Step 4: Streamlit Dashboard
st.set_page_config(page_title="AI-Powered Customer Insights", layout="wide")
st.title("📊 AI-Powered Customer Insights Dashboard")

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

st.subheader("📩 Filtered Emails")
st.dataframe(filtered_df)

# 📌 Sentiment Distribution
st.subheader("📊 Sentiment Distribution")
st.write("This bar chart shows how many emails are positive, neutral, or negative.")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Sentiment", palette="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 Urgency Breakdown
st.subheader("⏳ Urgency Levels by Sender Type")
st.write("This chart helps understand how urgency is distributed among leads vs. current clients.")

fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Urgency", hue="Sender_Type", palette="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 Top Words in Negative Emails
st.subheader("🚨 Common Words in Negative Emails")
st.write("Identifying common complaints based on the most frequently used words in negative emails.")

negative_emails = " ".join(df[df["Sentiment"] == "Negative"]["Email_Text"].values)
negative_words = [word for word in negative_emails.split() if word.lower() not in nltk.corpus.stopwords.words("english")]

word_freq = Counter(negative_words)
word_freq_df = pd.DataFrame(word_freq.items(), columns=["Word", "Frequency"]).sort_values(by="Frequency", ascending=False)

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(data=word_freq_df.head(10), x="Frequency", y="Word", palette="Reds_r", ax=ax)
st.pyplot(fig)

# 📌 Sentiment Over Time
st.subheader("📈 Sentiment Trend Over Time")
st.write("Tracking how customer sentiment changes over time.")

sentiment_over_time = df.groupby(["Timestamp", "Sentiment"]).size().unstack().fillna(0)
st.line_chart(sentiment_over_time)

# 📌 Urgency vs. Sentiment Correlation
st.subheader("🔗 Urgency vs. Sentiment Correlation")
st.write("This helps identify whether urgent emails are mostly positive or negative.")

fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pd.crosstab(df["Urgency"], df["Sentiment"]), annot=True, fmt="d", cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 AI-Detected High-Risk Clients
st.subheader("⚠️ High-Risk Clients")
st.write("Clients with negative sentiment & urgent requests.")
st.dataframe(df[(df["Sender_Type"] == "Current Client") & (df["Sentiment"] == "Negative")])

st.subheader("🔍 Key Insights")
st.write("- **Most urgent emails come from customers with negative sentiment.**")
st.write("- **Understanding frequent complaint words helps in improving service.**")
st.write("- **Tracking sentiment trends can help in proactive engagement.**")
