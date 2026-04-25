import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================
# SETUP
# ==============================

os.makedirs("results", exist_ok=True)

df = pd.read_csv("processed_data.csv")
print(df.head())

# ==============================
# 1. WQI DISTRIBUTION
# ==============================

fig, ax = plt.subplots(figsize=(6,5))
sns.histplot(df['wqi'], bins=30, kde=True, ax=ax)

ax.set_title("WQI Distribution")
ax.set_xlabel("WQI")
ax.set_ylabel("Frequency")

fig.savefig("results/wqi_distribution.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ==============================
# 2. SINGLE STATION TREND
# ==============================

station = df['station code'].iloc[0]
station_data = df[df['station code'] == station]

fig, ax = plt.subplots(figsize=(6,5))
ax.plot(station_data['year'], station_data['wqi'], marker='o')

ax.set_title(f"WQI Trend for Station {station}")
ax.set_xlabel("Year")
ax.set_ylabel("WQI")
ax.grid()

fig.savefig("results/single_station_trend.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ==============================
# 3. MULTIPLE STATION TRENDS
# ==============================

fig, ax = plt.subplots(figsize=(6,5))

for station in df['station code'].unique()[:5]:
    temp = df[df['station code'] == station]
    ax.plot(temp['year'], temp['wqi'], label=station)

ax.set_title("WQI Trends (Multiple Stations)")
ax.set_xlabel("Year")
ax.set_ylabel("WQI")
ax.legend()

fig.savefig("results/multi_station_trends.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ==============================
# 4. CORRELATION HEATMAP (FIXED)
# ==============================

original_df = pd.read_csv("water_dataX.csv", encoding="latin1")

# 🔥 STANDARDIZE COLUMN NAMES
original_df.columns = original_df.columns.str.strip().str.lower()

# 🔥 RENAME SAME AS TRAINING FILE
rename_dict = {
    'd.o. (mg/l)': 'do',
    'b.o.d. (mg/l)': 'bod',
    'nitratenan n+ nitritenann (mg/l)': 'nitrate'
}

original_df = original_df.rename(columns=rename_dict)

# 🔥 SAFE COLUMN SELECTION
numeric_cols = [col for col in ['ph', 'do', 'bod', 'nitrate'] if col in original_df.columns]

# 🔥 CLEAN NUMERIC VALUES
for col in numeric_cols:
    original_df[col] = original_df[col].astype(str)
    original_df[col] = original_df[col].str.replace(r'[^0-9.]', '', regex=True)
    original_df[col] = pd.to_numeric(original_df[col], errors='coerce')

original_df.fillna(original_df.median(numeric_only=True), inplace=True)

# 🔥 COMPUTE CORRELATION
corr = original_df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)

ax.set_title("Correlation Heatmap")

fig.savefig("results/correlation_heatmap.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ==============================
# 5. WATER QUALITY COUNT
# ==============================

fig, ax = plt.subplots(figsize=(6,5))
sns.countplot(x='water_quality', data=df, ax=ax)

ax.set_title("Water Quality Categories")

fig.savefig("results/water_quality_count.png", dpi=300, bbox_inches='tight')
plt.close(fig)

# ==============================
# 6. ACTUAL VS PREDICTED (FROM SAVED FILE)
# ==============================

import os

pred_path = "results/predictions.csv"

if os.path.exists(pred_path):

    pred_df = pd.read_csv(pred_path)

    fig, ax = plt.subplots(figsize=(6,5))

    ax.plot(pred_df['actual'][:100], label="Actual")
    ax.plot(pred_df['predicted'][:100], label="Predicted")

    ax.set_title("Actual vs Predicted WQI")
    ax.legend()

    fig.savefig("results/actual_vs_predicted.png", dpi=300, bbox_inches='tight')
    plt.close(fig)

    print("✅ Actual vs Predicted plot saved")

else:
    print("⚠️ predictions.csv not found. Run train_model.py first.")

# ==============================
# DONE
# ==============================

print("\n🎉 All plots saved successfully in 'results/' folder!")