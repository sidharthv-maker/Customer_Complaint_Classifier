import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
def train_model_categ():
    data = pd.read_csv("data/dataset.csv")
    data["issue_description"] = data["issue_description"].fillna("")

    y = data.category
    X = data[["product","issue_description","priority","channel","region","customer_age","customer_gender","subscription_type","customer_tenure_months","previous_tickets","operating_system","browser","payment_method","language","preferred_contact_time","issue_complexity_score","customer_segment"]]

    X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=1)

    num_cols = ["customer_age","customer_tenure_months","previous_tickets","issue_complexity_score"]

    cat_cols = ["product","priority","channel","region","customer_gender","subscription_type","operating_system","browser","payment_method","language","preferred_contact_time","customer_segment"]

    text_col = "issue_description"

    numpre = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scalar", MinMaxScaler())
    ])

    catpre = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    textpre = CountVectorizer()

    preprocess = ColumnTransformer(transformers=[
        ('num', numpre, num_cols),
        ('cat', catpre, cat_cols),
        ('text', textpre, text_col)
    ])

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_val)

    # print(classification_report(y_val, pred))
    return pipeline

def train_model_prio():
    data = pd.read_csv("data/dataset.csv")
    data["issue_description"] = data["issue_description"].fillna("")

    y = data.priority
    X = data[["product","issue_description","channel","region","customer_age","customer_gender","subscription_type","customer_tenure_months","previous_tickets","operating_system","browser","payment_method","language","preferred_contact_time","issue_complexity_score","customer_segment"]]

    X_train, X_val, y_train, y_val = train_test_split(X,y,random_state=1)

    num_cols = ["customer_age","customer_tenure_months","previous_tickets","issue_complexity_score"]

    cat_cols = ["product","channel", "region","customer_gender","subscription_type","operating_system","browser","payment_method","language","preferred_contact_time","customer_segment"]

    text_col = "issue_description"

    numpre = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="median")),
        ("scalar", MinMaxScaler())
    ])

    catpre = Pipeline(steps=[
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore"))
    ])

    textpre = CountVectorizer()

    preprocess = ColumnTransformer(transformers=[
        ('num', numpre, num_cols),
        ('cat', catpre, cat_cols),
        ('text', textpre, text_col)
    ])

    model = LogisticRegression(max_iter=1000)

    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", model)
    ])

    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_val)

    # print(classification_report(y_val, pred))
    return pipeline

def get_team(category):
    team_map = {
        "Payment Problem": "Billing Support",
        "Refund Request": "Billing Support",
        "Subscription Cancellation": "Retention Team",
        "Login Issue": "Account Access Team",
        "Security Concern": "Security Team",
        "Bug Report": "Engineering Support",
        "Performance Issue": "Technical Support",
        "Data Sync Issue": "Integration Support",
        "Feature Request": "Product Team",
        "Account Suspension": "Trust & Safety Team"
    }
    return team_map.get(category, "General Support")


def get_esc(priority, category, confidence):
    high_risk_categories = [
        "Security Concern",
        "Account Suspension",
        "Payment Problem",
        "Data Sync Issue"
    ]
    if priority in ["Urgent", "High"]:
        return "Yes"
    if category in high_risk_categories and confidence < 0.70:
        return "Yes"
    return "No"


def resp(category):
    response_templates = {
        "Payment Problem": "We've received your payment concern. Our billing team will verify and update your plan shortly.",
        "Refund Request": "We've received your refund request. Our billing team will confirm eligibility shortly.",
        "Subscription Cancellation": "We've received your cancellation request. Our team will assist you with the next steps.",
        "Login Issue": "We're aware of your login trouble. Our team will verify and restore your access shortly.",
        "Security Concern": "We take this seriously. Our security team will review and guide you on next steps.",
        "Bug Report": "Thanks for reporting this. Our engineering team will investigate and update you shortly.",
        "Performance Issue": "We're aware of the performance issue. Our technical team will look into it shortly.",
        "Data Sync Issue": "We're aware of the sync issue. Our integration team will restore it shortly.",
        "Feature Request": "Thanks for the suggestion. Our product team will review it for future updates.",
        "Account Suspension": "We've received your concern. Our trust and safety team will review and share next steps."
    }
    return response_templates.get(category)