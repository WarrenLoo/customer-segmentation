# %% imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import scipy.stats as ss
import seaborn as sns
import pandas as pd
import numpy as np 
import pickle 
import os 

CSV_PATH = os.path.join(os.getcwd(),'dataset','train.csv')

#%% function 


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

# %% Step 1) Data loading
df = pd.read_csv(CSV_PATH)

# %% Step 2) EDA
df.info()
df.describe().T

# categorical data
cat_col = list(df.columns[df.dtypes=='object'])
cat_col.append('Family_Size')

# continous data
con_col = list(df.columns[(df.dtypes=='int64')|(df.dtypes=='float64')])
con_col.remove('Family_Size')
con_col.remove('ID')

# target segmentation
df.groupby(['Segmentation','Profession']).agg({'Segmentation':'count'}).plot(kind='bar')
df.groupby(['Segmentation','Gender']).agg({'Segmentation':'count'}).plot(kind='bar')
df.groupby(['Segmentation','Ever_Married']).agg({'Segmentation':'count'}).plot(kind='bar')
df.groupby(['Segmentation','Spending_Score']).agg({'Segmentation':'count'}).plot(kind='bar')
df.groupby(['Segmentation','Graduated']).agg({'Segmentation':'count'}).plot(kind='bar')

#%% 

# for continous
for con in con_col:
    plt.figure()
    sns.distplot(df[con])
    plt.show()


# for categorical 
for cat in cat_col:
    plt.figure()
    sns.countplot(x=df[cat])
    plt.show()

# %% Step 3) Data cleaning
df.info()
df.isna().sum()

df = df.drop(labels=['ID'],axis=1)
#%%
# label encoding is for features & Target (ML)

le = LabelEncoder()

for cat in cat_col:
    if cat == 'Family_Size':
        continue
    else:
        temp = df[cat]
        temp[df[cat].notnull()] = le.fit_transform(temp[df[cat].notnull()])
        df[cat] = pd.to_numeric(df[cat],errors='coerce')
        save_path = os.path.join(os.getcwd(),'model',cat+'_encoder.pkl')
        with open(save_path,'wb') as f:
            pickle.dump(le,f)

#%% KNN imputation
# 0 NaN from this point onwards

column_names = df.columns

ki = KNNImputer(n_neighbors=5)
df = ki.fit_transform(df) # this will convert dataframe into numpy array

# to convert back into dataframe format
df = pd.DataFrame(df,columns=column_names)
df.isna().sum()

# %% Step 4) Features Selection
# continous versus categorical [target]

for con in con_col:
    lr = LogisticRegression()
    lr.fit(np.expand_dims(df[con],axis=-1),df['Segmentation'])
    print(con)
    print(lr.score(np.expand_dims(df[con],axis=-1),df['Segmentation']))

# categorical versus categorical [target]
for cat in cat_col:
    cm = pd.crosstab(df[cat],df['Segmentation']).to_numpy()
    print(cat)
    print(cramers_corrected_stat(cm))

# PCA to extract essential information 

#%% PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca_X = pca.fit_transform(df.drop(labels=['Segmentation'],axis=1))

plt.figure()
plt.scatter(pca_X[:,0],pca_X[:,1])
plt.xlabel('PCA Axis 1')
plt.ylabel('PCA Axis 2')
plt.show()

X,y = pca_X,df['Segmentation']


# %% Step 5) Data preprocessing
# X = df.drop(labels=['Segmentation'],axis=1)
# y = df['Segmentation']

from sklearn.preprocessing import OneHotEncoder # DL
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y,axis=-1))

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.7,shuffle=True,random_state=123)

#%% model development
from tensorflow.keras.layers import Dense,BatchNormalization,Dropout
from tensorflow.keras import Sequential, Input

input_shape = np.shape(X_train)[1:]
nb_class = len(np.unique(y_train,axis=0))

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Dense(64,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(nb_class,activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='acc')

# callbacks

# callbacks 
# tensorboard callback
import datetime
from tensorflow.keras.callbacks import TensorBoard,EarlyStopping

log_dir = os.path.join(os.getcwd(),'logs',datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = TensorBoard(log_dir=log_dir)

# early stopping callback 
es_callback = EarlyStopping(monitor='loss',patience=5)

hist = model.fit(X_train,y_train,batch_size=128,epochs=100,validation_data=(X_test,y_test),callbacks=[tb_callback,es_callback])

# %% Model Analysis

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.legend(['training loss','validation loss'])
plt.show()

plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.legend(['training loss','validation loss'])
plt.show()

#%%
# evauate model using confusion matrix
X_pred = model.predict(X_test) # to perform prediction 
X_pred = np.argmax(X_pred,axis=1) # returns the position with the highest value
y_test = np.argmax(y_test,axis=1)
y_train = np.argmax(y_train,axis=1)

#%%
# to display confusion matrix
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.metrics import classification_report


cm = confusion_matrix(y_test,X_pred)
cr = classification_report(y_test,X_pred)
print(cr)

labels = ohe.inverse_transform(np.unique(y,axis=0)) #['0','1','2','3','4','5','6']
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
disp.plot(cmap=plt.cm.Blues)
plt.show()

plt.figure()
sns.countplot(x=y_train)
plt.show()



