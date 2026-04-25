import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("water_dataX.csv", encoding="latin1")

print("Initial shape:", df.shape)

# Standardize columns
df.columns = df.columns.str.strip().str.lower()

# Rename columns
rename_dict = {
    'd.o. (mg/l)': 'do',
    'b.o.d. (mg/l)': 'bod',
    'nitratenan n+ nitritenann (mg/l)': 'nitrate'
}
df = df.rename(columns=rename_dict)

# Required columns
required_cols = ['station code','locations','year','ph','do','bod','nitrate']
df = df[required_cols]

# -------------------------
# LOCATION HANDLING (FIXED)
# -------------------------

df['locations_full'] = df['locations']   # keep original

df['locations'] = df['locations'].str.upper().str.strip()
df['locations'] = df['locations'].str.replace(r',.*', '', regex=True)

# Remove duplicates (important)
df = df.drop_duplicates(subset=['station code','locations','year'])

# -------------------------
# CLEAN NUMERIC
# -------------------------

for col in ['ph','do','bod','nitrate']:
    df[col] = df[col].astype(str)
    df[col] = df[col].str.replace(r'[^0-9.]','',regex=True)
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.fillna(df.median(numeric_only=True), inplace=True)

# -------------------------
# WQI CALCULATION
# -------------------------

standards = {'ph':8.5,'do':5,'bod':5,'nitrate':50}
ideal = {'ph':7,'do':14.6,'bod':0,'nitrate':0}

k = 1 / sum([1/v for v in standards.values()])
weights = {p:k/standards[p] for p in standards}

for p in standards:
    qi = ((df[p] - ideal[p]) / (standards[p] - ideal[p])) * 100
    df[p+'_Qi'] = qi.clip(0,300)

df['wqi'] = sum(df[p+'_Qi'] * weights[p] for p in standards)
df['wqi'] = df['wqi'].abs().clip(0,300).round(0)

# -------------------------
# CLASSIFICATION
# -------------------------

def classify(wqi):
    if wqi <= 55: return "Good"
    elif wqi <= 105: return "Average"
    else: return "Bad"

df['water_quality'] = df['wqi'].apply(classify)

# -------------------------
# FINAL AGGREGATION
# -------------------------

df = df.sort_values(['station code','year'])

final_df = df.groupby(['station code','year']).agg({
    'wqi':'mean',
    'ph':'mean',
    'do':'mean',
    'bod':'mean',
    'nitrate':'mean',
    'locations':'first',
    'locations_full':'first'
}).reset_index()

final_df['water_quality'] = final_df['wqi'].apply(classify)

print("Processed shape:", final_df.shape)

final_df.to_csv("processed_data.csv", index=False)

print("✅ processed_data.csv created successfully!")
