import pandas as pd
from model import train_model_categ, train_model_prio, resp, get_esc, get_team

cat_model = train_model_categ()
pri_model = train_model_prio()

issue_description = input("Issue description: ")

user_input = pd.DataFrame([{"product": "Web Portal","issue_description": issue_description,"priority": "Medium","channel": "Email","region": "South Asia","customer_age": 28,"customer_gender": "Male","subscription_type": "Pro","customer_tenure_months": 14,"previous_tickets": 2,"operating_system": "Windows","browser": "Chrome","payment_method": "Credit Card","language": "English","preferred_contact_time": "Evening","issue_complexity_score": 3,"customer_segment": "Individual",}])

pred_cat = cat_model.predict(user_input)[0]
pred_pri = pri_model.predict(user_input.drop("priority", axis=1))[0]

category_probs = cat_model.predict_proba(user_input)
confidence = category_probs.max()
assigned_team = get_team(pred_cat)
esc_needed = get_esc(pred_pri,pred_cat,confidence)
response = resp(pred_cat)

print(f"Category: {pred_cat}")
print(f"Priority: {pred_pri}")
print(f"Assigned Team: {assigned_team}")
print(f"Escalation Needed: {esc_needed}")
print(f"Response: {response}")