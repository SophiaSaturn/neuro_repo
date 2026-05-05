import pandas as pd
import numpy as np

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

#Load files
df14 = pd.read_csv("data_014.csv")
df15 = pd.read_csv("data_015.csv")

df14["participant"] = "P014"
df15["participant"] = "P015"

df = pd.concat([df14, df15], ignore_index=True)

#leave out practice blocks
df = df[df["block"] != "practice"].copy()
#leave out trials without responses (too slow)
df = df[df["response"].notna()].copy()

#Drop rows with missing values in critical columns
df = df.dropna(subset=["RT", "IT", "MT", "correct", "pressure", "block"])

#labels: normal = 0, pressure = 1
df["condition"] = df["pressure"].map({
    "normal": 0,
    "pressure": 1
})

#create group labels for LOGO-CV (participant + block)
df["group"] = df["participant"] + "_block_" + df["block"].astype(str)

#Features
features = ["IT", "MT", "correct"]

X = df[features].values
y = df["condition"].values
groups = df["group"].values

#LOGO-CV model
logo = LeaveOneGroupOut()

#Define a pipeline with scaling and logistic regression
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000)
)

scores = []
#Perform LOGO-CV
for train_idx, test_idx in logo.split(X, y, groups):
    model.fit(X[train_idx], y[train_idx]) #this line: trains LR on all blocks except one
    y_pred = model.predict(X[test_idx]) #this line: predicts on the left-out block
    score = balanced_accuracy_score(y[test_idx], y_pred)
    scores.append(score)


#calculating the mean, std and SE
mean_score = np.mean(scores)
std_score = np.std(scores)
se_score = np.std(scores) / np.sqrt(len(scores))


print(f"Number of folds: {len(scores)}")
print(f"Mean balanced accuracy: {mean_score}")
print(f"Std: {std_score}")
print(f"SE: {se_score}")


#please extract the coefficients from the last fitted model (the one trained on the last fold) and print them out
# Get the coefficients from the last fitted model
coefficients = model.named_steps['logisticregression'].coef_[0]
# Print the coefficients with their corresponding feature names
for feature, coef in zip(features, coefficients):
    print(f"{feature}: {coef}")


    