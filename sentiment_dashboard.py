import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import datetime
from textblob import TextBlob
import nltk

nltk.download('stopwords')

# Unique, business-related email dataset
email_data = [
    {"text": "Could you please provide details on your enterprise pricing plans?", "sender": "Lead", "theme": "Pricing Inquiry", "urgency": "Normal"},
    {"text": "We love the results we've seen with your SEO service! Keep it up!", "sender": "Current Client", "theme": "Positive Feedback", "urgency": "Low Priority"},
    {"text": "I'm unhappy with the delay in resolving my account issue.", "sender": "Current Client", "theme": "Support Issue", "urgency": "Urgent"},
    {"text": "We'd like to understand more about your lead generation services.", "sender": "Lead", "theme": "General Inquiry", "urgency": "Normal"},
    {"text": "Interested in upgrading our existing content marketing package.", "sender": "Current Client", "theme": "Upgrade Request", "urgency": "Normal"},
    {"text": "There's a discrepancy in the latest invoice we received.", "sender": "Current Client", "theme": "Billing Issue", "urgency": "Urgent"},
    {"text": "Can you send us case studies for your PPC campaigns?", "sender": "Lead", "theme": "Case Study Request", "urgency": "Normal"},
    {"text": "Awaiting a response from our previous call regarding social media strategies.", "sender": "Lead", "theme": "Follow-up Needed", "urgency": "Urgent"},
    {"text": "Our recent email marketing campaign didn't achieve desired results.", "sender": "Current Client", "theme": "Campaign Performance", "urgency": "Normal"},
    {"text": "Thanks to your marketing efforts, our quarterly sales increased significantly!", "sender": "Current Client", "theme": "Success Story", "urgency": "Low Priority"},
    {"text": "Could we schedule a meeting to discuss our new product launch campaign?", "sender": "Lead", "theme": "General Inquiry", "urgency": "Normal"},
    {"text": "Our website traffic has significantly increased since using your SEO services.", "sender": "Current Client", "theme": "Success Story", "urgency": "Low Priority"},
    {"text": "I'm experiencing issues logging into my account; please resolve urgently.", "sender": "Current Client", "theme": "Support Issue", "urgency": "Urgent"},
    {"text": "Can you provide detailed pricing information for your premium support plans?", "sender": "Lead", "theme": "Pricing Inquiry", "urgency": "Normal"},
    {"text": "We'd like to upgrade our social media management package.", "sender": "Current Client", "theme": "Upgrade Request", "urgency": "Normal"},
    {"text": "Our recent billing statement has some unexpected charges.", "sender": "Current Client", "theme": "Billing Issue", "urgency": "Urgent"},
    {"text": "Please send over some success stories regarding your video marketing efforts.", "sender": "Lead", "theme": "Case Study Request", "urgency": "Normal"},
    {"text": "Still awaiting confirmation on our latest campaign proposal.", "sender": "Lead", "theme": "Follow-up Needed", "urgency": "Urgent"},
    {"text": "We didn't see the expected ROI from our recent ad campaign.", "sender": "Current Client", "theme": "Campaign Performance", "urgency": "Normal"},
    {"text": "Appreciate the proactive support your team provided during our event!", "sender": "Current Client", "theme": "Positive Feedback", "urgency": "Low Priority"},
    {"text": "Do you have resources outlining your content strategy process?", "sender": "Lead", "theme": "General Inquiry", "urgency": "Normal"},
    {"text": "We've decided to expand our service package based on your recommendations.", "sender": "Current Client", "theme": "Upgrade Request", "urgency": "Normal"},
    {"text": "The support ticket I opened hasn't been answered yet, please expedite.", "sender": "Current Client", "theme": "Support Issue", "urgency": "Urgent"},
    {"text": "Can your team handle a large-scale marketing campaign?", "sender": "Lead", "theme": "General Inquiry", "urgency": "Normal"},
    {"text": "Thank you for quickly resolving the billing error last month.", "sender": "Current Client", "theme": "Positive Feedback", "urgency": "Low Priority"},
    {"text": "We'd like a follow-up on our discussion about influencer marketing.", "sender": "Lead", "theme": "Follow-up Needed", "urgency": "Urgent"},
    {"text": "Your team significantly improved our online engagement rates.", "sender": "Current Client", "theme": "Success Story", "urgency": "Low Priority"},
    {"text": "I noticed inconsistencies in our latest service invoice.", "sender": "Current Client", "theme": "Billing Issue", "urgency": "Urgent"},
    {"text": "Could you provide examples of successful email marketing campaigns?", "sender": "Lead", "theme": "Case Study Request", "urgency": "Normal"},
    {"text": "Our Facebook ads aren't performing as expected.", "sender": "Current Client", "theme": "Campaign Performance", "urgency": "Normal"},
    {"text": "Requesting details on volume discounts for large-scale projects.", "sender": "Lead", "theme": "Pricing Inquiry", "urgency": "Normal"},
    {"text": "Thanks for your guidance on our social media strategy; results are impressive.", "sender": "Current Client", "theme": "Positive Feedback", "urgency": "Low Priority"},
    {"text": "Our issue regarding account access remains unresolved.", "sender": "Current Client", "theme": "Support Issue", "urgency": "Urgent"},
    {"text": "Looking to learn more about your analytics and reporting capabilities.", "sender": "Lead", "theme": "General Inquiry", "urgency": "Normal"},
    {"text": "We want to upgrade to a more comprehensive analytics package.", "sender": "Current Client", "theme": "Upgrade Request", "urgency": "Normal"},
    {"text": "There's a delay in receiving our monthly performance reports.", "sender": "Current Client", "theme": "Support Issue", "urgency": "Urgent"},
    {"text": "Could you share successful case studies from the healthcare sector?", "sender": "Lead", "theme": "Case Study Request", "urgency": "Normal"},
    {"text": "Still awaiting your response regarding our campaign improvement suggestions.", "sender": "Lead", "theme": "Follow-up Needed", "urgency": "Urgent"},
    {"text": "Our latest LinkedIn campaign didn't achieve the targeted engagement.", "sender": "Current Client", "theme": "Campaign Performance", "urgency": "Normal"},
    {"text": "Your team’s innovative approach greatly enhanced our brand visibility.", "sender": "Current Client", "theme": "Success Story", "urgency": "Low Priority"},
    {"text": "Please clarify the details of our most recent invoice.", "sender": "Current Client", "theme": "Billing Issue", "urgency": "Urgent"},
    {"text": "What’s your typical turnaround time for marketing proposals?", "sender": "Lead", "theme": "General Inquiry", "urgency": "Normal"},
    {"text": "The custom package your team designed has exceeded our expectations.", "sender": "Current Client", "theme": "Positive Feedback", "urgency": "Low Priority"},
    {"text": "Our Instagram ads seem to be underperforming compared to previous months.", "sender": "Current Client", "theme": "Campaign Performance", "urgency": "Normal"},
    {"text": "Could you provide details about your premium client support services?", "sender": "Lead", "theme": "Pricing Inquiry", "urgency": "Normal"},
    {"text": "We're impressed by your team's timely and effective issue resolution.", "sender": "Current Client", "theme": "Positive Feedback", "urgency": "Low Priority"},
    {"text": "We urgently need assistance with a recurring technical problem.", "sender": "Current Client", "theme": "Support Issue", "urgency": "Urgent"},
    {"text": "Please follow up regarding our previous inquiry about your CRM integration.", "sender": "Lead", "theme": "Follow-up Needed", "urgency": "Urgent"},
    {"text": "We want to explore upgrading our subscription based on recent growth.", "sender": "Current Client", "theme": "Upgrade Request", "urgency": "Normal"},
    {"text": "Can you send detailed success stories related to your event marketing?", "sender": "Lead", "theme": "Case Study Request", "urgency": "Normal"}

]

sender_names = ["Alice Johnson", "Bob Smith", "Charlie Lee", "Dana White", "Evelyn Green", "Frank Brown", "Grace Hall", "Henry Adams", "Isabella Turner", "Jack Scott"]

# Create 50 distinct emails
data = {
    "Sender_Name": [random.choice(sender_names) for _ in email_data],
    "Email_Text": [item["text"] for item in email_data],
    "Sender_Type": [item["sender"] for item in email_data],
    "Urgency": [item["urgency"] for item in email_data],
    "Theme": [item["theme"] for item in email_data],
    "Timestamp": [datetime.datetime(2024, random.randint(1, 3), random.randint(1, 28)) for _ in range(len(email_data))]
}

df = pd.DataFrame(data)

# Sentiment Analysis
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity > 0.2 else "Negative" if polarity < -0.01 else "Neutral"

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

# Streamlit App
st.title("📊 Sentiment Analysis Dashboard")
st.sidebar.header("Filters")

# Filters
sender_type_filter = st.sidebar.selectbox("Filter by Sender Type", ["All", "Lead", "Current Client"])
if sender_type_filter != "All":
    df = df[df["Sender_Type"] == sender_type_filter]

urgency_filter = st.sidebar.selectbox("Filter by Urgency Level", ["All", "Urgent", "Normal", "Low Priority"])
if urgency_filter != "All":
    df = df[df["Urgency"] == urgency_filter]

# Display Data
st.subheader("📩 Filtered Email Dataset")
st.dataframe(df)

# Sentiment Distribution Chart
st.subheader("📊 Sentiment Distribution")

# Create figure and axis
fig, ax = plt.subplots(figsize=(8, 5))

# Create a countplot with enhanced styling
sns.countplot(
    data=df,
    x="Sentiment",
    palette="viridis",
    edgecolor="black",
    linewidth=1.5,
    ax=ax
)

# Title and labels with improved styling
ax.set_xlabel("Sentiment", fontsize=13, labelpad=10)
ax.set_ylabel("Count", fontsize=13, labelpad=10)

# Adding value labels on bars
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f'{height}', 
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=12, color='black', fontweight='bold')

# Removing spines for a cleaner look
sns.despine(left=True, bottom=True)

# Adding gridlines
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Display plot in Streamlit
st.pyplot(fig)


# Sentiment Breakdown by Urgency
# Enhanced Sentiment Breakdown by Urgency
st.subheader("⏳ Sentiment by Urgency Level")

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Create countplot with enhanced styling
sns.countplot(
    data=df,
    x="Urgency",
    hue="Sentiment",
    palette="mako",
    edgecolor="black",
    linewidth=1.2,
    ax=ax
)

# Enhanced title and labels
ax.set_xlabel("Urgency Level", fontsize=13, labelpad=10)
ax.set_ylabel("Count", fontsize=13, labelpad=10)

# Add annotations to bars
for p in ax.patches:
    height = p.get_height()
    if height > 0:
        ax.annotate(f'{height}', 
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    fontsize=11, color='black', fontweight='bold')

# Remove unnecessary spines and add gridlines
sns.despine(left=True, bottom=True)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Legend placement and styling
ax.legend(title="Sentiment", title_fontsize='12', fontsize='11', frameon=False, bbox_to_anchor=(1.05, 1), loc='upper left')

# Display plot in Streamlit
st.pyplot(fig)

# Sentiment Over Time
st.subheader("📈 Sentiment Trend Over Time")
sentiment_time = df.groupby(["Timestamp", "Sentiment"]).size().unstack().fillna(0)
st.line_chart(sentiment_time)

# Common Themes
st.subheader("📌 Most Common Email Themes")
theme_counts = df["Theme"].value_counts()
st.bar_chart(theme_counts)

# Top Leads
st.subheader("🎯 Top Priority Leads")
st.dataframe(df[df["Sender_Type"] == "Lead"].sort_values(by="Lead_Score", ascending=False).head(10))

# Churn Risk
st.subheader("⚠️ Clients at Risk of Churn")
df["Churn_Risk"] = df.apply(lambda x: "High" if (x["Sender_Type"] == "Current Client" and x["Sentiment"] == "Negative") else "Low", axis=1)
st.dataframe(df[df["Churn_Risk"] == "High"])

st.write("🔍 **Insights Summary:**")
st.write("- Track sentiment trends to monitor customer satisfaction.")
st.write("- Identify high-priority leads quickly.")
st.write("- Understand prevalent client concerns.")
st.write("- Highlight clients who may churn.")
