# Credit-Card-Fraud-Detection
 
Detect fraudulent credit card transactions using classical machine learning models trained on a highly imbalanced dataset. The project compares a Decision Tree, Random Forest, and XGBoost with imbalance-aware settings and evaluates performance using precision, recall, F1-score, and confusion matrices.

---

## Dataset

- Source: Kaggle CreditCard Dataset
- Rows: **284,807**
- Fraudulent (Class=1): **492**
- Genuine (Class=0): **284,315**
- Fraud rate: **0.1727%**

---

### 3) Modeling (Imbalance-Aware)
Models used in `DS650CT3.py`:
- **Decision Tree** with `class_weight="balanced"`
- **Random Forest (100 trees)** with `class_weight="balanced"`
- **XGBoost** with `scale_pos_weight = (#genuine / #fraud)`

---

## Results

> Because fraud is extremely underrepresented, **accuracy is not enough**. The evaluation focuses on precision, recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1 |
|------|----------|-----------|--------|----|
| Decision Tree | 0.9990 | 0.7619 | 0.6486 | 0.7007 |
| Random Forest | 0.9994 | 0.9720 | 0.7027 | 0.8157 |
| XGBoost | 0.9995 | 0.8939 | 0.7973 | 0.8429 |

 
- **Decision Tree** is simplest but misses more fraud cases (lower recall).
- **Random Forest** achieves very high precision (few false alarms) but recall is lower than XGBoost.
- **XGBoost** provides the **best balance** and detects the most fraud cases (highest recall) while keeping precision strong.


### Best Model Confusion Matrix (XGBoost)
![Confusion Matrix - XGBoost](images/confusion_matrix_xgboost.png)
---

## Key Takeaways

- **Extreme class imbalance:** 284,807 transactions with only **492 fraud cases (~0.1727%)**. This makes accuracy alone misleading.  
- **Best overall model in this project:** **XGBoost** achieved the best balance between precision and recall (F1 = **0.8429**).  
- **Why this matters:** In fraud detection, **recall** is critical to catch fraud (reduce false negatives), while **precision** helps avoid flagging too many legitimate transactions (reduce false positives).

## Note

This project was developed as part of my Master of Data Science program, within the
Predictive Analytics for Business course.

## Contact
For any questions, please contact me:

- [LinkedIn](https://www.linkedin.com/in/mashael-alsogair-97b754230/)

Thank you!

