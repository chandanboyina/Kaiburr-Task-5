# Kaiburr Task 5: Consumer Complaint Text Classification

This repository contains a complete data science pipeline for multi-class text classification. The goal is to automatically categorize consumer complaints into four distinct financial product categories.

---

## Project Goal

To classify complaint narratives from the Consumer Financial Protection Bureau (CFPB) into one of the following four classes, based on the requirements:

| Numerical Label | Category Name |
| :---: | :--- |
| **0** | Credit reporting, repair, or other |
| **1** | Debt collection |
| **2** | Consumer Loan |
| **3** | Mortgage |

## Methodology
The pipeline uses Python, Pandas, scikit-learn, and NLTK to implement a standard machine learning workflow, which is best viewed in the accompanying Jupyter Notebook (Consumer_Complaint_Classifier.ipynb).

## Pipeline Steps

### 1. Explanatory Data Analysis (EDA) and Feature Engineering

* Action: Data is downloaded, filtered to the four target categories, and missing values are removed.
* Feature Engineering: The text categories (Product column) are mapped to numerical labels (0 to 3) to serve as the target variable (Target).
* Visualization: A bar chart is generated to show the distribution of the four target classes (class balance) .

  ![EDA](https://github.com/chandanboyina/Kaiburr-Task-5/blob/main/1.EDA.jpeg)

### Text Pre-Processing

* Action: Complaint narratives are cleaned by: lowercasing, removing punctuation and special characters, tokenizing, removing English stop words (e.g., 'the', 'a'), and lemmatizing words.

* Vectorization: The cleaned text is converted into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency), which weights words based on their importance. The data is then split into training and testing sets.

![](https://github.com/chandanboyina/Kaiburr-Task-5/blob/main/2.%20Text%20pre%20processing.jpeg)

### Selection of Multi Classification Model

* Action: Three standard multi-class classifiers suitable for sparse text data are initialized and trained on the TF-IDF feature vectors:
  * Logistic Regression

  * Linear Support Vector Classification (LinearSVC)

  * Multinomial Naive Bayes (MultinomialNB)
 
  ![](https://github.com/chandanboyina/Kaiburr-Task-5/blob/main/3.%20model%20trainig.jpeg)

### Comparison of Model Performance

* Action: The performance of all three trained models is compared based on key metrics (Accuracy, Precision, Recall, F1-Score).
* Visualization: A bar chart is generated to visually compare the Weighted F1-Scores of the models, clearly identifying the best performer.

  ![](https://github.com/chandanboyina/Kaiburr-Task-5/blob/main/4.comparision.jpeg)

### 5. Model Evaluation

* Action: For each model, a detailed Classification Report is generated on the test set.
* Output: The report provides metric scores (Precision, Recall, F1-Score) for each individual category (0 through 3), allowing for an in-depth analysis of where the model performs well and where it struggles. The model with the highest overall F1-Score is selected as the final model.

![](https://github.com/chandanboyina/Kaiburr-Task-5/blob/main/5.%20model%20evaluation.jpeg)

### 6. Prediction

* Action: The best-performing model from the evaluation is used to classify a new, unseen complaint text.
* Output: The final result is the predicted category name (e.g., 'Mortgage') for the new input text.

![](https://github.com/chandanboyina/Kaiburr-Task-5/blob/main/6.%20Prediction.jpeg)

## How to Run the Project

This section provides instructions for setting up your environment and executing the text classification pipeline.

### 1. Prerequisites

Dataset URL: https://files.consumerfinance.gov/ccdb/complaints.csv.zip 

You need **Python 3.x** and the following data science libraries installed:
* `pandas`
* `scikit-learn`
* `nltk`
* `requests`
* `seaborn`
* `matplotlib`

### 2. Environment Setup

You can install all necessary dependencies using pip:
```bash
pip install -r requirements.txt 
# OR manually:
pip install pandas scikit-learn nltk requests seaborn matplotlib
```
### Execution 

 If using a standard Python script
 
 ```bash
python Consumer_Complaint_Classifier.py
```

If using a Jupyter Notebook (Recommended for visualization)

```bash
jupyter notebook Consumer_Complaint_Classifier.ipynb
```
