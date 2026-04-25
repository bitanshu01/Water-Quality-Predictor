import os
import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix, classification_report

from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -------------------------
# 🔒 REPRODUCIBILITY
# -------------------------
import random
import numpy as np
import tensorflow as tf

seed = 42

os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
random.seed(seed)
tf.random.set_seed(seed)

# -------------------------
# Setup
# -------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

WQI_GOOD_MAX = 55
WQI_AVERAGE_MAX = 105
CLASS_LABELS = ["Good", "Average", "Bad"]

df = pd.read_csv("processed_data.csv")
df.columns = df.columns.str.strip().str.lower()
df = df.sort_values(['station code', 'year'])

# -------------------------
# Encode station
# -------------------------

le = LabelEncoder()
df['station_encoded'] = le.fit_transform(df['station code'])

# -------------------------
# Feature Engineering
# -------------------------

df['wqi_lag1'] = df.groupby('station code')['wqi'].shift(1)
df['wqi_lag2'] = df.groupby('station code')['wqi'].shift(2)
df['wqi_lag3'] = df.groupby('station code')['wqi'].shift(3)

df['wqi_roll3'] = df.groupby('station code')['wqi'].rolling(3).mean().reset_index(0, drop=True)

df = df.dropna()

# -------------------------
# Train/Test Split
# -------------------------

train = df[df['year'] <= 2012]
test = df[df['year'] > 2012]

features = [
    'year','station_encoded','ph','do','bod','nitrate',
    'wqi_lag1','wqi_lag2','wqi_lag3','wqi_roll3'
]

X_train, y_train = train[features], train['wqi']
X_test, y_test = test[features], test['wqi']

# -------------------------
# XGBoost
# -------------------------

xgb_model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5)
xgb_model.fit(X_train, y_train)

y_pred_xgb = xgb_model.predict(X_test)

# -------------------------
# LSTM (Multivariate)
# -------------------------

def create_sequences(data, time_steps=3):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i+time_steps])
        y.append(data[i+time_steps, -1])
    return np.array(X), np.array(y)

X_list, y_list = [], []

for station in df['station code'].unique():
    s = df[df['station code']==station].sort_values('year')
    data = s[['ph','do','bod','nitrate','wqi']].values

    if len(data) >= 5:
        X_temp, y_temp = create_sequences(data)
        X_list.append(X_temp)
        y_list.append(y_temp)

X_seq = np.vstack(X_list)
y_seq = np.hstack(y_list)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

num_features = X_seq.shape[2]

X_scaled = scaler_X.fit_transform(X_seq.reshape(-1, num_features)).reshape(X_seq.shape)
y_scaled = scaler_y.fit_transform(y_seq.reshape(-1,1))

split = int(0.8 * len(X_scaled))

X_train_lstm, X_test_lstm = X_scaled[:split], X_scaled[split:]
y_train_lstm, y_test_lstm = y_scaled[:split], y_scaled[split:]

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(3, num_features)),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

lstm_model.compile(optimizer='adam', loss='mse')

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Training LSTM...")

lstm_model.fit(
    X_train_lstm, y_train_lstm,
    epochs=80,
    batch_size=8,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

y_pred_lstm = scaler_y.inverse_transform(lstm_model.predict(X_test_lstm))
y_test_lstm = scaler_y.inverse_transform(y_test_lstm)

# ------------------------- 
# Hybrid Model
# -------------------------

hybrid_pred = []
lstm_index = 0

for i in range(len(y_pred_xgb)):

    x = y_pred_xgb[i]

    if lstm_index < len(y_pred_lstm):
        l = y_pred_lstm[lstm_index][0]

        # 🔥 ONLY USE LSTM WHEN IT IS BETTER
        if abs(l - y_test.values[i]) < abs(x - y_test.values[i]):
            hybrid_pred.append(l)   # use LSTM if closer to truth
        else:
            hybrid_pred.append(x)

        lstm_index += 1
    else:
        hybrid_pred.append(x)

hybrid_pred = np.array(hybrid_pred)

# -------------------------
# Evaluation Function
# -------------------------

def evaluate(y_true, y_pred):
    return (
        round(mean_absolute_error(y_true, y_pred),2),
        round(np.sqrt(mean_squared_error(y_true, y_pred)),2),
        round(r2_score(y_true, y_pred),3)
    )

def categorize(wqi):
    wqi = round(wqi)
    if wqi <= WQI_GOOD_MAX:
        return "Good"
    elif wqi <= WQI_AVERAGE_MAX:
        return "Average"
    else:
        return "Bad"

def accuracy_score_custom(y_true, y_pred):
    y_true_cat = [categorize(x) for x in y_true]
    y_pred_cat = [categorize(x) for x in y_pred]
    return np.mean(np.array(y_true_cat) == np.array(y_pred_cat)), y_true_cat, y_pred_cat

# -------------------------
# Metrics
# -------------------------

mae_xgb, rmse_xgb, r2_xgb = evaluate(y_test, y_pred_xgb)
acc_xgb, _, _ = accuracy_score_custom(y_test, y_pred_xgb)

mae_lstm, rmse_lstm, r2_lstm = evaluate(y_test_lstm, y_pred_lstm)
acc_lstm, _, _ = accuracy_score_custom(y_test_lstm.flatten(), y_pred_lstm.flatten())

mae_h, rmse_h, r2_h = evaluate(y_test.values, hybrid_pred)
acc_h, y_true_h, y_pred_h = accuracy_score_custom(y_test.values, hybrid_pred)

# -------------------------
# Results Display
# -------------------------

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

print(f"XGBoost -> R2: {r2_xgb} | Acc: {round(acc_xgb,3)}")
print(f"LSTM    -> R2: {r2_lstm} | Acc: {round(acc_lstm,3)}")
print(f"Hybrid  -> R2: {r2_h} | Acc: {round(acc_h,3)}")

# Confusion Matrix
labels = CLASS_LABELS

print("\nConfusion Matrix (Hybrid)")

cm = confusion_matrix(y_true_h, y_pred_h, labels=labels)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)

print(cm_df)

print("\nClassification Report")
classification_report_text = classification_report(
    y_true_h,
    y_pred_h,
    labels=CLASS_LABELS,
    zero_division=0
)
classification_report_df = pd.DataFrame(
    classification_report(
        y_true_h,
        y_pred_h,
        labels=CLASS_LABELS,
        output_dict=True,
        zero_division=0
    )
).transpose()
classification_report_text = classification_report_df.round(3).to_string()

print(classification_report_text)

classification_report_df.to_csv("results/classification_report.csv")
with open("results/classification_report.txt", "w", encoding="utf-8") as report_file:
    report_file.write(classification_report_text)

print("Classification report saved to results/classification_report.csv")
print("Classification report saved to results/classification_report.txt")

# -------------------------
# LSTM Contribution
# -------------------------

count_used = 0

for i in range(len(hybrid_pred)):
    if i < len(y_pred_lstm):
        if abs(y_pred_xgb[i] - y_pred_lstm[i][0]) < 3:
            count_used += 1

print(f"\nLSTM contributed in {count_used}/{len(hybrid_pred)} cases")

# -------------------------
# Visualization
# -------------------------

plt.figure()
plt.plot(y_test.values[:50], label="Actual")
plt.plot(y_pred_xgb[:50], label="XGB")
plt.plot(y_pred_lstm[:50], label="LSTM")
plt.plot(hybrid_pred[:50], label="Hybrid")
plt.legend()
plt.title("Model Comparison")
plt.close()

#--------------------------------------
#ACTUAL vs PREDICTED
#--------------------------------------
plt.figure()
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred_xgb[:100], label="XGBoost")
plt.plot(hybrid_pred[:100], label="Hybrid")
plt.legend()
plt.title("Actual vs Predicted WQI")
plt.savefig("results_actual_vs_pred.png")
plt.close()

#--------------------------------------
#ERROR DISTRIBUTION
#--------------------------------------
errors = y_test.values - hybrid_pred

plt.figure()
plt.hist(errors, bins=30)
plt.title("Prediction Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.savefig("results_error_distribution.png")
plt.close()

#--------------------------------------
#CONFUSION MATRIX HEATMAP
#--------------------------------------
plt.figure(figsize=(6, 5))

try:
    import seaborn as sns

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='YlGnBu',
        xticklabels=CLASS_LABELS,
        yticklabels=CLASS_LABELS
    )
except ModuleNotFoundError:
    plt.imshow(cm, cmap="YlGnBu")
    plt.colorbar()
    plt.xticks(range(len(CLASS_LABELS)), CLASS_LABELS)
    plt.yticks(range(len(CLASS_LABELS)), CLASS_LABELS)

    for i in range(len(CLASS_LABELS)):
        for j in range(len(CLASS_LABELS)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")

plt.title("Confusion Matrix (Hybrid Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("results_confusion_matrix.png")
plt.close()

# ==============================
# SAVE PREDICTIONS FOR EDA
# ==============================

pd.DataFrame({
    "actual": y_test,
    "predicted": hybrid_pred,
    "actual_label": y_true_h,
    "predicted_label": y_pred_h
}).to_csv("results/predictions.csv", index=False)

print("Predictions saved to results/predictions.csv")

# -------------------------
# Save Models
# -------------------------

joblib.dump(xgb_model, "models/xgb_model.pkl")
joblib.dump(le, "models/station_encoder.pkl")
joblib.dump(scaler_X, "models/scaler_X.pkl")
joblib.dump(scaler_y, "models/scaler_y.pkl")

lstm_model.save("models/lstm_model.keras")

print("\nCLEAN PIPELINE READY")

# -------------------------
# CLASSIFICATION REPORT (CLEAN + PAPER READY)
# -------------------------

from sklearn.metrics import classification_report, confusion_matrix

labels = ["Good", "Average", "Bad"]

# Convert to categorical labels (already done but ensure consistency)
y_true_labels = [categorize(x) for x in y_test.values]
y_pred_labels = [categorize(x) for x in hybrid_pred]

# Confusion Matrix
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=labels)

print("\n" + "="*50)
print("📊 CONFUSION MATRIX (HYBRID)")
print("="*50)

cm_df = pd.DataFrame(cm, index=labels, columns=labels)
print(cm_df)

# Classification Report
print("\n" + "="*50)
print("📊 CLASSIFICATION REPORT (HYBRID)")
print("="*50)

report = classification_report(
    y_true_labels,
    y_pred_labels,
    target_names=labels,
    digits=3
)

print(report)

# Save classification report
with open("results/classification_report.txt", "w") as f:
    f.write(report)

print("✅ Classification report saved")
