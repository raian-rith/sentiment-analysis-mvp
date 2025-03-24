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
    "I had a bad experience with another agency. What makes you different?",
    "I’m looking for an aggressive lead generation strategy. Can you help?",
    "Your client testimonials look great. How do you ensure similar results?",
    "I tried running PPC ads before, but it was a disaster. What do you suggest?",
    "What type of reports do you provide? I want full transparency.",
    "I need to start a campaign ASAP. How quickly can you onboard new clients?",
    "I heard your content marketing is effective. Can I see some samples?",
    "Your agency was recommended by a colleague. Let’s schedule a consultation.",
    "I am unsure if I need paid ads or organic SEO. Can you guide me?",
    "I want to focus on video marketing. Do you offer services for YouTube?",
    "I saw your ad but couldn't find case studies on your website. Can you share?",
    "What ROI should I expect in the first six months?",
    "I need help building my brand from scratch. Do you work with startups?",
    "I tried LinkedIn Ads, but got no results. Can you optimize them?",
    "Do you offer social media marketing for e-commerce brands?",
    "What’s the best platform for generating high-quality B2B leads?"
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
    "Our competitors seem to be outbidding us in ads. Can we adjust targeting?",
    "What do you recommend for our Black Friday promotion?",
    "We need a new strategy for better lead nurturing in our CRM.",
    "Our YouTube ads aren’t converting well. Can we refine our approach?",
    "Our LinkedIn outreach isn’t generating meetings. Any suggestions?",
    "We want to expand into a new market. Can you guide our marketing strategy?",
    "Can we schedule a quarterly performance review?",
    "We need a competitor analysis to benchmark our performance.",
    "Our video marketing results are mixed. Can we analyze what’s working?",
    "Our retargeting ads aren’t performing as expected. What adjustments are needed?",
    "We need new creatives for our display ads. Can your team handle that?",
    "What’s the best budget allocation for next quarter?",
    "Our customer retention is declining. Can we adjust messaging?",
    "We’re launching a new product. Can you create a multi-channel campaign?",
    "Our blog content isn’t getting traction. How do we improve reach?",
    "We need more automation in our email workflows. Can you implement that?"
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
    "Sender": lead_senders[:25] + client_senders[:25],
    "Email_Text": lead_emails[:25] + client_emails[:25],
    "Sender_Type": ["Lead"] * 25 + ["Current Client"] * 25,
    "Timestamp": timestamps
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
st.subheader("📩 Email Dataset")
st.dataframe(df)

# 📌 Sentiment Distribution
st.subheader("📊 Sentiment Distribution")
fig, ax = plt.subplots(figsize=(6, 4))
sns.countplot(data=df, x="Sentiment", palette="coolwarm", ax=ax)
st.pyplot(fig)

# 📌 Urgency vs. Sentiment Correlation
st.subheader("🔗 Urgency vs. Sentiment Correlation")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(pd.crosstab(df["Urgency"], df["Sentiment"]), annot=True, fmt="d", cmap="coolwarm", ax=ax)
st.pyplot(fig)

st.subheader("🔍 Key Insights")
st.write("- **Urgent emails often have negative sentiment.**")
st.write("- **Tracking sentiment trends helps improve response strategies.**")
