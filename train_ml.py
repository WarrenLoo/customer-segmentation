# %% imports
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from module import cramers_corrected_stat
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
import os

CSV_PATH = os.path.join(os.getcwd(), "dataset", "train.csv")

# %% Step 1) Data loading
df = pd.read_csv(CSV_PATH)

# %% Step 2) EDA
df.info()
df.describe().T

# categorical data
cat_col = list(df.columns[df.dtypes == "object"])
cat_col.append("Family_Size")

# continous data
con_col = list(df.columns[(df.dtypes == "int64") | (df.dtypes == "float64")])
con_col.remove("Family_Size")
con_col.remove("ID")

# target segmentation
df.groupby(["Segmentation", "Profession"]).agg({"Segmentation": "count"}).plot(
    kind="bar"
)
df.groupby(["Segmentation", "Gender"]).agg({"Segmentation": "count"}).plot(kind="bar")
df.groupby(["Segmentation", "Ever_Married"]).agg({"Segmentation": "count"}).plot(
    kind="bar"
)
df.groupby(["Segmentation", "Spending_Score"]).agg({"Segmentation": "count"}).plot(
    kind="bar"
)
df.groupby(["Segmentation", "Graduated"]).agg({"Segmentation": "count"}).plot(
    kind="bar"
)

# %%

# %% Step 3) Data cleaning
df.info()
df.isna().sum()

df = df.drop(labels=["ID"], axis=1)
# %%
# label encoding is for features & Target (ML)

le = LabelEncoder()

for cat in cat_col:
    if cat == "Family_Size":
        continue
    else:
        temp = df[cat]
        temp[df[cat].notnull()] = le.fit_transform(temp[df[cat].notnull()])
        df[cat] = pd.to_numeric(df[cat], errors="coerce")
        save_path = os.path.join(os.getcwd(), "model", cat + "_encoder.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(le, f)

# %% KNN imputation
# 0 NaN from this point onwards

column_names = df.columns

ki = KNNImputer(n_neighbors=5)
df = ki.fit_transform(df)  # this will convert dataframe into numpy array

# to convert back into dataframe format
df = pd.DataFrame(df, columns=column_names)
df.isna().sum()

# %% Step 4) Features Selection
# continous versus categorical [target]

for con in con_col:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con], axis=-1), df["Segmentation"])
    print(con)
    print(lr.score(np.expand_dims(df[con], axis=-1), df["Segmentation"]))

# categorical versus categorical [target]
for cat in cat_col:
    cm = pd.crosstab(df[cat], df["Segmentation"]).to_numpy()
    print(cat)
    print(cramers_corrected_stat(cm))

# %% PCA
pca = PCA(n_components=2)
pca_X = pca.fit_transform(df.drop(labels=["Segmentation"], axis=1))
print(pca.explained_variance_ratio_)

plt.figure()
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_)
plt.show()

plt.figure()
plt.scatter(pca_X[:, 0], pca_X[:, 1])
plt.xlabel("PCA Axis 1")
plt.ylabel("PCA Axis 2")
plt.show()

# %% to visualize only

kmeans = KMeans(n_clusters=4)  # 4 categories A,B,C,D
kmeans.fit(pca_X)
y_pred_km = kmeans.predict(pca_X)

for i in np.unique(y_pred_km):
    filtered_y_pred = pca_X[y_pred_km == i]
    plt.scatter(filtered_y_pred[:, 0], filtered_y_pred[:, 1], label=i)
    plt.legend()

# %% Model development

rf = RandomForestClassifier()
rf.fit(pca_X, df["Segmentation"])
y_pred_rf = rf.predict(pca_X)

print(classification_report(df["Segmentation"], y_pred_rf))


# %% Step 5) Data preprocessing
X = df.drop(labels=["Segmentation"], axis=1)
y = df["Segmentation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.7, shuffle=True, random_state=123
)
