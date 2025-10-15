# 🏅 Olympics Event Classification using Multinomial Naive Bayes

This project applies **Multinomial Naive Bayes**, a classic machine learning algorithm for text classification, to classify Olympic events based on their names into their corresponding sports. The model is trained using textual event data extracted from the **Olympic athlete events dataset**.

---

## 📘 Project Overview

The goal of this project is to build a text classification model that predicts the **sport category** given the **event name** from the Olympic Games dataset.  
By leveraging the `CountVectorizer` and `MultinomialNB` from Scikit-learn, the project demonstrates a practical NLP (Natural Language Processing) application in sports analytics.

---

## 📂 Dataset

- **Source:** `athlete_events.csv.zip`
- The dataset contains historical records of Olympic athletes and their respective events and sports.

**Important columns used:**
- `Event`: The name of the specific Olympic event (e.g., *100m Freestyle Men*).
- `Sport`: The general category of the event (e.g., *Swimming*).

---

## ⚙️ Project Workflow

1. **Data Loading & Preprocessing**
   - Load the dataset and display its shape.
   - Remove rows with missing `Event` or `Sport` values.
   - Filter out rare sports (appearing less than twice).

2. **Feature Extraction**
   - Convert event names into numerical vectors using **CountVectorizer**.

3. **Model Training**
   - Split the data into training and testing sets using `train_test_split`.
   - Train a **Multinomial Naive Bayes** classifier on the training data.

4. **Model Evaluation**
   - Predict sports on the test set.
   - Compute and display:
     - Accuracy score
     - Classification report (precision, recall, f1-score)
   - Visualize the distribution of sports in the dataset.

---

## 🧠 Machine Learning Model

- **Algorithm:** Multinomial Naive Bayes  
- **Feature Representation:** Bag of Words (Count Vectorization)
- **Libraries Used:**
  - `pandas`
  - `scikit-learn`
  - `matplotlib`

---

## 📊 Visualization

A bar chart is generated to visualize the **distribution of sports** in the dataset.  
The final accuracy score is also displayed on the chart.

---

