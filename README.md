# Credit_Card_Offer_Prediction
---

## Project Overview
This project aims to predict which customers are most likely to respond positively to a credit card offer based on historical customer data. By analyzing features such as demographic details, financial status, existing banking relationships, and previous interactions with offers, we develop a machine learning model that helps the bank target potential customers efficiently. This project helps optimize marketing efforts, improve conversion rates, and reduce unnecessary costs.

## Business Problem
Banks need to enhance their marketing strategies to identify customers who are likely to show interest in credit card offers. Instead of reaching out to all customers, which is time-consuming and costly, the goal is to build a predictive model that can accurately identify customers with a high likelihood of responding positively. This approach will help:
- **Improve Targeting Efficiency**: Focus marketing efforts on customers who are more likely to respond positively.
- **Increase Conversion Rates**: Reach the right prospects to enhance the success rate of credit card offers.
- **Optimize Marketing Costs**: Minimize marketing expenses by avoiding less promising customers.

## Dataset
The dataset used for this project includes customer information such as:
- **Cust_ID**: Unique identifier for each customer.
- **Gender**: Customer's gender (Male/Female).
- **Month_Income**: Customer's monthly income.
- **Age**: Customer's age.
- **Region_Code**: Geographic region code.
- **Occupation**: Type of occupation (Salaried, Self-Employed, Business, Student).
- **Credit_Score**: Customer's credit score.
- **Loan_Status**: Whether the customer has an active loan (Yes/No).
- **Existing_Credit_Cards**: Number of credit cards the customer already has.
- **Avg_Account_Balance**: Customer's average account balance.
- **Account_Category**: Account balance categorization (Savings Account, Current Account, Senior Citizens Account, Investment Account ).
- **Tenure_with_Bank**: Number of years the customer has been with the bank.
- **Lead_Outcome**: Target variable indicating if the customer showed interest in the credit card offer (Yes/No).

## Key Features
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and scaling features.
- **Exploratory Data Analysis (EDA)**: Understanding the key patterns in the data and the relationship between features and the `Lead_Outcome`.
- **Modeling**: Building and comparing machine learning classification models to predict whether a customer will respond positively. Models used include:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - XGBoost Classifier
  - Support Vector Classifier
- **Model Evaluation**: Performance metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1 Score

## Installation and Usage
To use this project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/MaleeshaAluwihare/Credit_Card_Offer_Prediction.git
    cd Credit_Card_Offer_Prediction
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the Streamlit app:
    ```bash
    streamlit run App.py
    ```

4. In the Streamlit app, provide customer input data in the form to predict whether the customer will respond positively to the credit card offer.

## Results
The best-performing model was **[model_name]**, which achieved an accuracy of **[accuracy_score]**, showing that the model can effectively identify customers who are likely to show interest in credit card offers.

## Conclusion
By predicting customer interest in credit card offers, the bank can make data-driven decisions to optimize marketing strategies, improve conversion rates, and minimize costs. This project demonstrates the potential of machine learning in solving real-world business challenges in the financial industry.

## Future Improvements
- Experiment with more advanced feature engineering techniques.
- Tune model hyperparameters for better performance.
- Deploy the model as an API for real-time predictions.

---
