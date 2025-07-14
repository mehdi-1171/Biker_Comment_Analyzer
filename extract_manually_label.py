from pandas import read_excel

from utility import DATA_PATH

# Load CSV
df = read_excel(DATA_PATH + "raw_comments.xlsx")

# Drop nulls or empty comments
df = df.dropna(subset=["comment"])
df = df[df["comment"].str.strip() != ""]

# Sample N rows for manual labeling (مثلاً 200 تا)
sampled_df = df.sample(n=200, random_state=42).reset_index(drop=True)
df = df[~df['comment'].isin(sampled_df['comment'])]
df.to_excel(DATA_PATH + "without_sampled.xlsx", index=False)
# Export to CSV for manual labeling
sampled_df.to_csv(DATA_PATH + "sampled_for_labeling.csv", index=False, encoding="utf-8-sig")


""" After Labeling: """

df = read_excel(DATA_PATH + "raw_labeled_comments.xlsx")
label_cols = [
    "Pricing", "Fuel", "Cancelation", "Incentive", "Commission",
    "Desired Destination", "App", "Insurance", "Instant Cashout",
    "Equipment", "call center", "Other"
]
""" Fill nan column to 0 and convert to int """
df[label_cols] = df[label_cols].fillna(0).astype(int)
df = df[["comment"] + label_cols]
df.to_excel(DATA_PATH + "labeled_comments.xlsx", index=False)
