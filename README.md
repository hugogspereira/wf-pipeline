# Website Fingerprinting Framework

This repository implements a machine learning and deep learning pipeline (:warning: WIP :warning:) for **website fingerprinting (WF) attacks** using packet capture (PCAP) data.

---

## 📁 Project Structure

```
data/
├── models/                 # Saves the models (.pkl) 
├── pcaps/                  # Stores the .pcap files
├── results/                # Stores the metrics from the wf attacks
| 
src-ml/
├── 1_validate_pcaps.py     # Validates and preprocesses raw PCAP files
├── 2_extract_features.py   # Extracts features from validated PCAPs
├── 3_wf_attack.py          # Trains and evaluates ML models on extracted features
| 
src-dl/
├── 1_validate_pcaps.py     # Validates and preprocesses raw PCAP files
├── 2_extract_features.py   # Extracts features (based on Wang14-style) from validated PCAPs
├── RF/                     # Trains and evaluates based on Robust Fingerprinting model (RF) on extracted features
    ├── img/
    ├── RF/                 # For info, see the README.md there
        ├── ...
        ├── extract-list.py
        ├── extract-all.py
        ├── train.py
        ├── train_10fold.py
        ├── test.py
````

:warning: The code is intended for research purposes ONLY! :warning:

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

#### Move to the correct directory:

For ML:
```bash
cd src-ml
```
For DL:
```bash
cd src-dl
```

Then, validate the pcap files:
```bash
python 1_validate_pcaps.py
```

* Checks the integrity of PCAP files.
* Filters out corrupted or incomplete captures.

---

### 3. Extract Features

```bash
python 2_extract_features.py
```

* Converts validated PCAPs into numerical feature vectors.
* Output is a CSV (for ML) or a WANG14 format files (for DL) containing features and labels (website/component).

---

### 4. Train and Evaluate Models

#### In case of ML:
```bash
python 3_wf_attack.py
```

#### In case of DL:
:warning: Follow the instructions presented in the README.md inside the respective model to be used.

<br>

Supported models:

| Model              | From                                          |
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
|--------------------| --------------------------------------------- |
| RF                 | `https://github.com/robust-fingerprinting/RF` |
---

## 📚 References

* Referenced for inspiring our feature extraction which was adapted from it: [**"Effective Detection of Multimedia Protocol Tunneling using Machine Learning"**](https://github.com/dmbb/MPTAnalysis/blob/master/CovertCastAnalysis/extractFeatures.py), USENIX Security Symposium, 2018.
* Referenced for implementing the **Robust Fingerprinting (RF) model**, which inspired the deep learning architecture and training/evaluation pipeline for website fingerprinting in this repository: [**"Subverting Website Fingerprinting Defenses with Robust Traffic Representation"**](https://github.com/robust-fingerprinting/RF), USENIX Security Symposium, 2023.
