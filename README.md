# 🏍️ Biker_Comment_Analyzer

## 🎯 Project Overview

**Biker_Comment_Analyzer** is an NLP-based project designed to extract and classify the main topics from feedback provided by delivery bikers. The goal is to identify recurring pain points (e.g., pricing, insurance, commission) and help business/product teams make informed decisions based on real biker comments.

---

## 🧩 Dataset

- **Format**: CSV (`.csv`)
- **Key column**: `comment`
- **Data size**: ~XXXX comments (replace with actual count)
- **Example rows**:

''' csv
comment
"قیمت خیلی کمه، اصلاً نمی‌صرفه"
"چرا بیمه نداریم؟"
"کمیسیون شما خیلی بالاست"
'''

## 🧠 Methodologies

- ** 🔹 Unsupervised Topic Modeling (BERTopic) 
 
  - Model: BERTopic with ParsBERT embeddings

  - Goal: Cluster similar comments into coherent topics

  - Outcome: Each comment gets assigned a topic label (e.g., "pricing", "insurance")

- ** 🔹 Supervised Text Classification

  - Model: ParsBERT + Classifier (e.g., Logistic Regression, SVM, or MLP)
    
  - Training data: Manually labeled sample (e.g., 200 comments)

  - Goal: Automatically tag new biker comments with predefined topics

## 🛠 Tools & Libraries
- Python 3.10+

- pandas

- scikit-learn

- transformers (HooshvareLab/bert-fa-base-uncased)

- BERTopic

- umap-learn, hdbscan

- tqdm, matplotlib, seaborn

## 🔄 Workflow
- ✅ Step 1: Load and Clean Data 
  - Remove null/empty comments
  - Normalize Persian text (ی ← ي, ک ← ك, ... if needed)
  - Optional: remove emojis, URLs, and unwanted punctuation


- ✅ Step 2: Unsupervised Topic Modeling 
  - Generate sentence embeddings using ParsBERT
  - Cluster embeddings using HDBSCAN
  - Use BERTopic to extract keywords per topic
  - Assign each comment a topic label


- ✅ Step 3: Manual Labeling (for supervised approach)
  - Sample ~200 comments
  - Define labels like: pricing, insurance, commission, customer_behavior
  - Save labeled data in labeled_comments.csv


- ✅ Step 4: Train Classifier (Supervised)
  - Convert comments to embeddings
  - Train a classifier on labeled data
  - Evaluate performance with accuracy, F1-score, and confusion matrix


- ✅ Step 5: Predict on New Data 
  - Use trained classifier to assign topic labels to unlabeled comments
  - Save results as predicted_comments.csv

## 📊 Outputs
- CSV file with each comment and its detected topic
- Topic distribution visualization
- Sample comments for each topic


## 📌 Future Improvements
- Improve topic accuracy with more labeled data
- Deploy as an API for live classification 
- Integrate with business dashboards 
- Add sentiment analysis layer


## 👤 Project Info
Project: Biker_Comment_Analyzer

Author: [Mehdi Habibian]

Start Date: July 2025

Status: In Progress