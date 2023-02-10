# -*- coding: utf-8 -*-d
"""
Created on Sat Jun 11 14:04:33 2022

@author: warren
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as ss
import numpy as np

from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.impute import KNNImputer

#%% functions
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))



#%% Static
CSV_PATH = os.path.join(os.getcwd(),'train.csv')
GENDER_ENCODER_PATH = os.path.join(os.getcwd(),'gender_encoder.pkl')
MARRIED_ENCODER_PATH = os.path.join(os.getcwd(),'married_encoder.pkl')
PROFESSION_ENCODER_PATH = os.path.join(os.getcwd(),'profession_encoder.pkl')
GRADUATE_ENCODER_PATH = os.path.join(os.getcwd(),'graduate_encoder.pkl')




# %% EDA
# Step 1) Data Loading
df = pd.read_csv(CSV_PATH)

# Step 2) Data inspection
df.info()
df.describe()

# cat_column = ['Gender','Ever_Married','Graduate','Profession','Spending_Score'
#               ,'Var_1','Segmentation'] # object --> categorical

# cont_column = ['Age','Work_Experience','Family_Size'] #int/float

# EXTRA
cat_column = df.columns[df.dtypes=='object']
cont_column = df.columns[(df.dtypes=='int64') | (df.dtypes=='float64')]

# continous data
for i in cont_column:
    plt.figure()
    sns.distplot(df[i])
    plt.show()

# categorical data
for i in cat_column:
    plt.figure(figsize=(10,12))
    sns.countplot(df[i])
    plt.show()

df.groupby(['Segmentation','Profession']).agg({'Segmentation':
                                               'count'}).plot(kind='bar')

df.groupby(['Segmentation','Graduated','Gender',
            'Spending_Score']).agg({'Segmentation':'count'})

for i in cat_column:
    plt.figure()
    sns.countplot(df[i],hue=df['Segmentation'])
    plt.show()

# Regression analysis
# Cramer's V Categorical vs categorical
for i in cat_column:
    print(i)
    confussion_mat = pd.crosstab(df[i],df['Segmentation']).to_numpy()
    print(cramers_corrected_stat(confussion_mat)) 

# logistic regression categorical vs continous 

#%%
# Step 3) Data cleaning
# convert categorical columns --> integers 
# Convert Target column --> OHE

df_dummy = df.copy() # to duplicate the data

df_dummy = df_dummy.drop(labels=['ID'],axis=1)

df.median()

le = LabelEncoder()
df_dummy['Gender'] = le.fit_transform(df_dummy['Gender'])
with open(GENDER_ENCODER_PATH,'wb') as file :
    pickle.dump(le,file)

#%% easy way to convert categorical into integers
# print(df_dummy['Ever_Married'].unique()) # to get unique elements
# df_dummy['Ever_Married'] = df_dummy['Ever_Married'].map({'No':0,'Yes':1})


# print(df_dummy['Profession'].unique())
# df_dummy['Profession'] = df_dummy['Profession'].map({'Healthcare':0,
#                                                        'Engineer':1,
#                                                        'Entertainment':2})

# df_dummy.isna().sum()
#%% Elegant way to convert categorical data into integers
paths = [GENDER_ENCODER_PATH,MARRIED_ENCODER_PATH]

for index,i in enumerate(cat_column):
    temp = df_dummy[i]
    temp[temp.notnull()] = le.fit_transform(temp[temp.notnull()])
    df_dummy[i] = pd.to_numeric(temp,errors='coerce')
    # with open(paths[index],'wb') as  file:
    #     pickle.dump(le,file)
    
# Drop unneccessary

#%%
df_dummy.isna().sum()

# mode
df_dummy['Ever_Married'] = df_dummy['Ever_Married'].\
                    fillna(df_dummy['Ever_Married'].mode()[0]) # add [0]
                    
                    
df_dummy['Graduated'] = df_dummy['Graduated'].\
                    fillna(df_dummy['Graduated'].mode()[0])
                    
df_dummy['Profession'] = df_dummy['Profession'].\
                    fillna(df_dummy['Profession'].mode()[0])

df_dummy['Work_Experience'] = df_dummy['Work_Experience'].\
                        fillna(df_dummy['Work_Experience'].median())
                        
df_dummy['Family_Size'] = df_dummy['Family_Size'].\
                        fillna(df_dummy['Family_Size'].median())
                        
df_dummy['Var_1'] = df_dummy['Var_1'].fillna(df_dummy['Var_1'].mode()[0])

df_dummy.isna().sum()

#%%

knn_imp = KNNImputer()
df_dummy = knn_imp.fit_transform(df_dummy)
df_dummy = pd.DataFrame(df_dummy)
df_dummy.columns = df.drop(labels='ID',axis=1).columns

# to make sure there is no Decimal places in categorical data
df_dummy['Var_1'] = np.floor(df_dummy['Var_1'])
df_dummy['Graduated'] = np.floor(df_dummy['Graduated'])
df_dummy.isna().sum()

# Step 4) Features Selection
# Will not select Gender due low correlation 

for i in cat_column:
    print(i)
    confussion_mat = pd.crosstab(df_dummy[i],df_dummy['Segmentation']).to_numpy()
    print(cramers_corrected_stat(confussion_mat)) 

# Continous Categorical
from sklearn.linear_model import LogisticRegression
for con in cont_column: 
    if con == 'ID':
        pass
    else:
        print(con)
        lr = LogisticRegression()
        lr.fit(np.expand_dims(df_dummy[con],axis=-1),df_dummy['Segmentation'])
        print(lr.score(np.expand_dims(df_dummy[con],axis=-1),df_dummy['Segmentation']))



X = df_dummy.loc[:,['Ever_Married','Graduated','Profession','Spending_Score']]
y = df['Segmentation']



# Step 5) Data pre-processing
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))
# save as pickle

#%%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from tensorflow.keras.layers import Dense,Dropout,BatchNormalization
from tensorflow.keras import Sequential,Input

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=123)

# Model development

nb_features = np.shape(X)[1:]
nb_classes = len(np.unique(y_train,axis=0))
#%%
model = Sequential()
model.add(Input(shape=(nb_features)))
model.add(Dense(32,activation='relu',name='Hidden_Layer1'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu',name='Hidden_Layer2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(nb_classes,activation='softmax',name='Output_layer'))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='acc')

# callbacks

hit = model.fit(X_train,y_train,
                batch_size=64,
                validation_data=(X_test,y_test),
                epochs=10)


# Model evaluation 


#%% Bonus

# Features selection 
# PCA
X = df_dummy.drop(labels='Segmentation',axis=1)
column_names_X = X.columns

y = df_dummy['Segmentation']

from sklearn.decomposition import PCA

pca = PCA(2)
X = pca.fit_transform(X)

#%%
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4) # A,B,C,D
kmeans.fit(X)

label = kmeans.predict(X)

for i in np.unique(label):
    filtered_label = X[label==i]
    
    plt.scatter(filtered_label[:,0],filtered_label[:,1],label=i)
    plt.legend()

#%%
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X,y)

from sklearn.metrics import classification_report
y_true = y
y_pred = rfc.predict(X)

print(classification_report(y_true,y_pred))