# Smart Customer Support Ticket Assistant

A machine learning-based support ticket assistant that classifies customer issues, predicts priority, assigns the right support team, and generates a suggested first response.

## Project Overview

This project uses Natural Language Processing and Machine Learning to analyze customer support tickets and automate the first level of ticket triage.

Instead of manually reading every ticket, the system takes a customer issue description and predicts:

- Ticket category
- Suggested priority
- Assigned support team
- Escalation requirement
- Suggested customer response

## Key Features

- Classifies customer support tickets into issue categories
- Predicts ticket priority
- Assigns tickets to the appropriate support team
- Flags tickets that may need escalation
- Generates a professional response template
- Uses both text and structured ticket information
- Built using a Scikit-learn machine learning pipeline

## Dataset

The dataset contains customer support ticket details such as:

- `product`
- `issue_description`
- `priority`
- `channel`
- `region`
- `customer_age`
- `customer_gender`
- `subscription_type`
- `customer_tenure_months`
- `previous_tickets`
- `operating_system`
- `browser`
- `payment_method`
- `language`
- `preferred_contact_time`
- `issue_complexity_score`
- `customer_segment`
- `category`

The main prediction targets are:

```
category
priority
```

## Machine Learning Approach

The project uses two separate ML pipelines:

# Category Prediction Model
Predicts the type of customer issue.
Example: Payment Problem, Bug Report, Security Concern, Data Sync Issue.
# Priority Prediction Model
Predicts ticket urgency.
Example: Low, Medium, High, Urgent.

Text data is processed using CountVectorizer, Structured categorical data is processed using OneHotEncoder and Numerical data is processed using SimpleImputer and MinMaxScaler.

## Future Improvements
- Replace CountVectorizer with TF-IDF
- Compare Logistic Regression with Naive Bayes and Linear SVM
- Add model confidence scores
- Save trained models using Joblib
- Add a Streamlit web interface
- Add ticket summary generation
- Add manual review flag for low-confidence predictions and filter through spam tickets