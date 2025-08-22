# Website Fingerprinting Attack Framework

This repository implements a machine learning pipeline for **website fingerprinting (WF) attacks** using packet capture (PCAP) data. It supports a variety of classical ML models as well as a deep learning model (TikTok).

---

## 📁 Project Structure

```
data/
├── models/                 # Saves the models (.pkl) 
├── pcaps/                  # !!!PUT HERE!!! the .pcap files
├── results/                # Stores the metrics from the wf attacks
src/
├── 1_validate_pcaps.py     # Validates and preprocesses raw PCAP files
├── 2_extract_features.py   # Extracts features from validated PCAPs
├── 3_wf_attack.py          # Trains and evaluates ML models on extracted features
````

---

## 📝 Requirements

- Python 3.10+
- [dpkt](https://pypi.org/project/dpkt/)
- [numpy](https://numpy.org/)
- [pandas](https://pandas.pydata.org/)
- [scikit-learn](https://scikit-learn.org/)
- [xgboost](https://xgboost.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [joblib](https://joblib.readthedocs.io/)

Install dependencies:

```bash
pip install dpkt numpy pandas scikit-learn xgboost matplotlib seaborn joblib
````

---

## 🧩 Workflow

### 1. Prepare PCAP Files

Place your `.pcap` files in the `data/pcaps/` directory.
Each file must follow the naming convention:

```
x_y.pcap
```

Where:  
- **x** – the index of the website (e.g., its rank in the Tranco Top 1000 list)  
- **y** – the sample number for that website  

Example:  
- `5_1.pcap` → first sample of the 5th-ranked website  
- `250_3.pcap` → third sample of the 250th-ranked website  

--- 

### 2. Validate PCAPs

```bash
cd src
python 1_validate_pcaps.py
```

* Checks the integrity of PCAP files.
* Filters out corrupted or incomplete captures.

---

### 3. Extract Features

```bash
python 2_extract_features.py
```

* Converts validated PCAPs into numerical feature vectors suitable for ML.
* Output is a CSV containing features and labels (website/component).

---

### 4. Train and Evaluate Models

```bash
python 3_wf_attack.py
```

Supported models:

| Model              | Notes                                         |
| ------------------ | --------------------------------------------- |
| GradientBoosting   | `sklearn.ensemble.GradientBoostingClassifier` |
| DecisionTree       | `sklearn.tree.DecisionTreeClassifier`         |
| RandomForest       | `sklearn.ensemble.RandomForestClassifier`     |
| XGBoost            | `xgboost.XGBClassifier`                       |
| ExtraTrees         | `sklearn.ensemble.ExtraTreesClassifier`       |
| LogisticRegression | `sklearn.linear_model.LogisticRegression`     |
| NaiveBayes         | `sklearn.naive_bayes.GaussianNB`              |
| KNN                | `sklearn.neighbors.KNeighborsClassifier`      |
| SVM                | `sklearn.svm.SVC`                             |
---

## 📊 Evaluation

The framework provides thorough evaluation for classical ML models:

- **Data Splitting:**  
  Train/test split is performed **per website**, ensuring that samples from the same website do not appear in both train and test sets. This prevents over-optimistic metrics due to data leakage.

- **Metrics:**  
  - **Accuracy** – overall correctness of predictions  
  - **Precision, Recall, F1-score** – detailed per-class performance  
  - **Confusion Matrix** – visual overview of misclassifications

- **Cross-Validation:**  
  - Supported for ML models to assess model stability and robustness.

---

## 📚 References

* Feature extraction inspired by and adapted from: [**"Effective Detection of Multimedia Protocol Tunneling using Machine Learning"**](https://github.com/dmbb/MPTAnalysis/blob/master/CovertCastAnalysis/extractFeatures.py), USENIX Security Symposium, 2018.
