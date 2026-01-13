# Titanic-Surivival-Prediction

Absolutely! Here's a professional and structured README tailored for your Titanic project based on everything you’ve done. You can copy this into a `README.md` file:

---

# Titanic Survival Prediction Project

**Author:** Joshua Jacob
**Project:** Who Will Survive? – Titanic Disaster Analysis

## Overview

The Titanic disaster of April 15, 1912, claimed over 1,500 lives out of 2,224 passengers onboard. This project explores the factors that influenced survival using passenger data. We aim to identify which features—such as gender, socio-economic status, family size, or luck—played the biggest role in survival, and build predictive models to estimate the likelihood of survival for passengers.

---

## Dataset

* **Training set:** `train.csv` (891 entries)
* **Test set:** `test.csv` (418 entries)

**Key Columns:**

| Column      | Description                                                          |
| ----------- | -------------------------------------------------------------------- |
| PassengerId | Unique ID for each passenger                                         |
| Survived    | Survival status (0 = No, 1 = Yes)                                    |
| Pclass      | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)                             |
| Name        | Passenger name                                                       |
| Sex         | Gender                                                               |
| Age         | Age in years                                                         |
| SibSp       | # of siblings/spouses aboard                                         |
| Parch       | # of parents/children aboard                                         |
| Ticket      | Ticket number                                                        |
| Fare        | Passenger fare                                                       |
| Cabin       | Cabin number (many missing values)                                   |
| Embarked    | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

---

## Data Cleaning & Preprocessing

1. **Handling Missing Values**

   * `Cabin` dropped due to excessive missingness (train: 204/891, test: 91/418).
   * `Age` missing values filled with mean age.
   * `Embarked` missing values filled with mode (most common port).
   * `Fare` missing value in the test set filled using the mean fare grouped by `Pclass` and `Embarked`.

2. **Outlier Detection**

   * Checked top 10 and bottom 10 values for `Age`, `Fare`, `SibSp`, and `Parch`.
   * No abnormal outliers were removed as extreme values were valid (e.g., high fares for first-class passengers, large family sizes).

3. **Feature Engineering**

   * `FamilySize` = `SibSp` + `Parch` + 1
   * `IsAlone` = 1 if `FamilySize` == 1, else 0
   * Extracted passenger `Title` from `Name` and grouped rare titles under `'Rare'`.
   * Converted categorical variables (`Sex`, `Embarked`, `Title`) to dummy variables for modeling.

---

## Exploratory Data Analysis (EDA)

Key insights:

* **Gender:** Female passengers had a significantly higher survival rate than males (~50% higher).
* **Class:** 1st class passengers had the highest survival (~62%), followed by 2nd class (~48%), then 3rd class (~24%).
* **Embarkation Port:** Cherbourg passengers had the highest survival (~55%), followed by Queenstown (~39%) and Southampton (~34%).
* **Family Size:** Families of size 4 had the highest survival (~73%), while solo passengers had lower survival.
* **Age & Fare:** Children (<10) and elderly passengers had higher survival rates. Higher fares correlated with higher survival.
* **Correlation Analysis:**

  * Positive correlation with survival: `Fare` (+0.26)
  * Negative correlation: `Pclass` (-0.31)
  * Family size variables (`SibSp`, `Parch`, `FamilySize`) are strongly inter-correlated.

Visualizations included:

* Survival rate by `Sex`, `Pclass`, `Embarked`, and `FamilySize`.
* Stacked histograms for `Age` and `Fare` distributions by survival.
* Correlation heatmap.
* Survival rate by passenger `Title`.

---

## Modeling

Three models were built to predict survival:

1. **Logistic Regression**

   * Binary classifier predicting survival probability based on features.

2. **Linear Regression**

   * Regression model predicting continuous survival scores, converted to binary outcomes with a 0.5 threshold.

3. **K-Nearest Neighbors (KNN)**

   * Predicts survival by majority vote of the 6 nearest passengers in feature space.

**Validation (80/20 split)**

| Model               | Accuracy | Precision | Recall | F1-score |
| ------------------- | -------- | --------- | ------ | -------- |
| Logistic Regression | 84.9%    | 77.8%     | 83.6%  | 0.81     |
| Linear Regression   | 85.5%    | 79.7%     | 82.1%  | 0.81     |
| KNN (k=6)           | 74.9%    | 69.0%     | 59.7%  | 0.64     |

**5-Fold Cross-Validation**

| Model               | Mean Accuracy ± Std |
| ------------------- | ------------------- |
| Logistic Regression | 82.5% ± 2.4%        |
| Linear Regression   | 83.1% ± 2.1%        |
| KNN                 | 71.2% ± 0.9%        |

**Observations:**

* Logistic and Linear Regression were the most consistent and accurate.
* KNN underperformed due to sensitivity to distance metrics and feature scaling.
* Slight drop in accuracy under CV suggests some overfitting in initial train/validation split.

---

## Conclusion

* **Most Important Predictors:** `Sex`, `Pclass`, `Fare`, `Title`, and `Embarked`.
* Logistic and Linear Regression are reliable models for this dataset.
* Family size, alone status, and socio-economic status also impact survival.
* Future improvements could include:

  * Hyperparameter tuning for KNN
  * Feature scaling for distance-based methods
  * Using ensemble models (Random Forest, Gradient Boosting) for potentially higher accuracy

---

## How to Run

1. Clone the repository.
2. Ensure `train.csv` and `test.csv` are in the project directory.
3. Install dependencies:

   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
   ```
4. Run the Jupyter Notebook `Titanic_Survival_Prediction.ipynb` to see the full workflow, from cleaning to modeling and evaluation.

---

## References

* Kaggle Titanic Dataset: [https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)
* Scikit-learn documentation: [https://scikit-learn.org/stable/documentation.html](https://scikit-learn.org/stable/documentation.html)

---
