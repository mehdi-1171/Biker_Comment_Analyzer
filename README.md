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

## Implementation

### 🔍 3.1 Label Extraction via TF-IDF

To support meaningful **manual labeling**, we first extracted unsupervised keywords using the TF-IDF algorithm. This provided an overview of recurring terms and themes in user comments.

#### 🧪 Process Overview

- ✅ **Text preprocessing** was done using the [Hazm](https://github.com/sobhe/hazm) library:
  - Normalized Persian text
  - Tokenized into words
  - Removed stopwords and custom non-informative words (e.g., `سلام`, `ممنون`)
  - 🧠 POS tagging was applied to extract only nouns from each comment.
  - 🧹 Lemmatized the nouns to their base forms (e.g., `رفت` → `رفتن`).
  - 🧩 Created **2- to 5-word n-grams** from lemmatized nouns.
  - 📊 Calculated **TF-IDF scores** to identify high-value phrases across the dataset.
  - 🏷️ Extracted top-ranked n-grams as candidate labels, such as:
    > `قیمت بالا`, `زمان تحویل`, `پشتیبانی ضعیف`

#### 📝 Outcome

- Used the extracted phrases to define a **multi-label schema** for annotation.
- Manually labeled ~200 sample comments using the suggested categories.
> Final Result:
> 
> مبلغ کرایه: 2379.2240  
> بیمه تامین: 199.5220  
> اسنپ فود: 905.3307  
> بیمه تامین: 618.8005  
> تسویه لحظه: 582.0610  
> مبدا مقصد: 579.3380  
> کرایه کرایه: 544.9816  
> مقصد منتخب: 532.3066  
> هزینه سفر: 509.2082  
> فاصله مبدا: 505.1568  
> لغو سفر: 482.2130  
> سفر طرح: 474.7676  
> موتور سوار: 455.6272  
> اسنپ شاپ: 446.6013  
> سهمیه بنزین: 366.8005  
> سفر اسنپ: 350.4944  
> نرم افزار: 345.8108  
> اسنپ کار: 312.7900  
> انتخاب مقصد: 312.0974  
> طرح سفر: 301.8259  
> درخواست اسنپ: 281.9594  
> طرح تشویق: 272.7084

#### 📝 Labels:
We use the TF-IDF algorithm to extract candidate labels from user comments. The resulting labels are manually labeled by a team of experts. The final list of labels includes:

> - Pricing  
> - Fuel  
> - Cancelation  
> - Incentive  
> - Commission  
> - App  
> - Origin Distance  
> - Desired Destination  
> - Insurance  
> - Instant Cashout  
> - Other


---

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