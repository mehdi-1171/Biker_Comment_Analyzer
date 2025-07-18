"""
    A class for extracting keywords from Persian comments using NLP techniques.

"""
from pandas import read_excel, DataFrame
from numpy import array, asarray
from sklearn.feature_extraction.text import TfidfVectorizer
from hazm import Normalizer, word_tokenize, stopwords_list, POSTagger, Lemmatizer
import os

from utility import DATA_PATH


class KeyWordExtraction:
    FILE_NAME = "comments.xlsx"

    def __init__(self):
        """
               Initializes required components for preprocessing and keyword extraction.
        """
        self.df = DataFrame()
        self.normalizer = Normalizer()
        self.stopwords = set(stopwords_list())
        self.tagger = POSTagger(model=os.path.join(DATA_PATH, "resources/pos_tagger.model"))
        self.lemmatizer = Lemmatizer()
        self.tokenized = []
        self.feature_names = array([])
        self.tfidf_matrix = None

    def read_data(self):
        """
               Reads the Excel file and keeps only the "comment" column.
               Drops rows with missing values.
        """
        self.df = read_excel(os.path.join(DATA_PATH, self.FILE_NAME))
        self.df = self.df[["comment"]].dropna().reset_index(drop=True)

    def normalize_text(self):
        """
             Normalizes text (e.g., fixes spacing, punctuation, common replacements).
        """
        self.df["normalized"] = self.df["comment"].apply(self.normalizer.normalize)

    def tokenize_text(self):
        """
        Tokenizes normalized text into words.
        """
        self.df["tokens"] = self.df["normalized"].apply(word_tokenize)

    def eliminate_stopwords(self):
        """
        Removes stopwords and manually defined frequent words from the tokens.
        Filters out tokens with length <= 1.
        """
        custom_stopwords = self.stopwords.union({'سلام', 'ممنون', 'خواهش', 'افزایش', 'کاهش'})
        self.df["filtered"] = self.df["tokens"].apply(
            lambda tokens: [w for w in tokens if w not in custom_stopwords and len(w) > 1]
        )

    def extract_nouns(self):
        """
        Extracts only noun tokens using POS tagging, then lemmatizes them.
        """
        def keep_nouns(tokens):
            tagged = self.tagger.tag(tokens)
            nouns = [w for w, tag in tagged if tag.startswith("N") or tag == "Ne"]
            lemmatized = [self.lemmatizer.lemmatize(w).split("#")[0] for w in nouns]
            return lemmatized

        self.df["nouns"] = self.df["filtered"].apply(keep_nouns)

    def tf_idf(self):
        """
        Calculates TF-IDF matrix from the noun tokens using 2- to 5-gram phrases.
        Limits to the top 1000 features.
        """
        corpus = self.df["nouns"].apply(lambda x: ' '.join(x)).tolist()
        vectorizer = TfidfVectorizer(ngram_range=(2, 5), max_features=1000)
        self.tfidf_matrix = vectorizer.fit_transform(corpus)
        self.feature_names = vectorizer.get_feature_names_out()

    def top_words(self, top_n=40):
        """
        Extracts and prints the top N keyword phrases based on TF-IDF scores.

        :param top_n: Number of top-ranked phrases to return
        """
        summed = asarray(self.tfidf_matrix.sum(axis=0)).ravel()
        scores = list(zip(self.feature_names, summed))
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        for word, score in sorted_scores[:top_n]:
            print(f"{word}: {score:.4f}")

    def run(self):
        """
        Runs the full keyword extraction pipeline:
        loading data, preprocessing, POS tagging, TF-IDF, and output.
        """
        self.read_data()
        self.normalize_text()
        self.tokenize_text()
        self.eliminate_stopwords()
        self.extract_nouns()
        self.tf_idf()
        return self.top_words()


""" TEST """
k = KeyWordExtraction()
k.run()


""" Result simple tf-idf: """
# 🔑 Top 20 Keywords:
# کرایه: 8774.2351
# اسنپ: 3542.9436
# قیمت: 3007.5199
# سفر: 2870.3018
# طرح: 2525.0969
# مبلغ: 2441.0173
# باکس: 2010.4909
# کمیسیون: 1897.2216
# سرویس: 1773.9573
# مقصد: 1695.1729
# مسیر: 1457.3849
# هزینه: 1417.4845
# راننده: 1394.6229
# درخواست: 1367.6115
# موتور: 1354.2148
# کار: 1307.9143
# پاداش: 1170.195
# مبدا: 1081.0158
# کمه: 1064.2538
# بیمه: 1032.0796

""" N -gram"""
# مبلغ کرایه: 2379.2240
# اسنپ باکس: 2291.6665
# اسنپ فود: 905.3307
# کرایه هارو: 867.3568
# کرایه اسنپ: 802.6109
# کرایه کمه: 798.3876
# کرایه پایینه: 706.8056
# قیمت کرایه: 688.4342
# بیمه تامین: 618.8005
# کرایه کمیسیون: 597.5375
# اسنپ مارک: 593.4369
# تسویه لحظه: 582.0610
# مبدا مقصد: 579.3380
# کرایه کرایه: 544.9816
# مقصد منتخب: 532.3066
# نرخ کرایه: 519.0596
# هزینه سفر: 509.2082
# فاصله مبدا: 505.1568
# قیمت سفر: 497.2048
# لغو سفر: 482.2130
# سفر طرح: 474.7676
# موتور سوار: 455.6272
# اسنپ شاپ: 446.6013
# کرایه طرح: 445.3776
# کرایه مسیر: 442.6498
# موتور سیکل: 442.1992
# اسنپ دکتر: 410.7954
# سرویس اسنپ: 385.1991
# سهمیه بنزین: 366.8005
# سفر اسنپ: 350.4944
# نرم افزار: 345.8108
# هزار تومان: 339.3669
# اسنپ کار: 312.7900
# انتخاب مقصد: 312.0974
# کرایه اسنپ باکس: 303.0572
# طرح سفر: 301.8259
# کرایه سفر: 296.3706
# درخواست اسنپ: 281.9594
# ۲۰ درصد: 273.5326
# طرح تشویق: 272.7084

""" Extract label from keywords """
# Pricing
# Fuel
# Cancelation
# Incentive
# Commission
# App
# Origen Distance
# Desired Destination
# Insurance
# Instant Cashout
# Other
