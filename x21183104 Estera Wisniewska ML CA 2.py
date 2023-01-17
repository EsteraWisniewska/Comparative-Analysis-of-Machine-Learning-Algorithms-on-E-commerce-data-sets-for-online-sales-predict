######## ESTERA WISNIEWSKA ########
### Comperative analysis of ML Algorithms for online sale prediction

# 1. Importing libraries
# 2. Loading Data set I - SMALL
#   2.1 Loading data set 1 - 'online shoppers intention'
#   2.2 EDA
#   2.3 Data preparation
#   2.4 Checking distribution
#   2.5 Correlation
#   2.6 ML - importing libraries
#   2.6.1 ML - full data set - raw
#   2.6.2 ML - full data set - normalized
#   2.6.3 ML - full data set - balanced
#   2.6.4 ML - full data set - Comparing the results
#   2.7.1 ML - selected features - raw
#   2.7.2 ML - selected features - normalized
#   2.7.3 ML - selected features - balnced
#   2.7.4 ML - slected features - Comparing the results

# 3. Loading Data set II - MEDIUM
#   3.1 Loading data set 2 - 'online shopping'
#   3.2 EDA
#   3.3 Data preparation
#   3.4 Checking distribution
#   3.5 Correlation
#   3.6.1 ML - full data set - raw
#   3.6.2 ML - full data set - balanced
#   3.6.3 ML - full data set - Coparing the results - balanced vs unbalanced
#   3.7.1 ML - selected features - raw, unbalanced
#   3.7.2 ML - selected features - balanced
#   3.7.3 ML - selected features - comparing the results of balanced and unbalanced
#   3.7.4 ML - comparing the results - full data vs slected features only (both balanced)

# 4. Loading Data set III - LARGE
#   4.1 Loading data set 2 - 'Online Shopping USA'
#   4.2 EDA
#   4.3 Data preparation
#   4.4 Checking distribution
#   4.5 Correlation
#   4.6.1 ML - full data set - unbalanced
#   4.6.2 ML - full data set - balanced
#   4.6.3 ML - full data set - Coparing the results
#   4.7.1 ML - full data set - balanced
#   4.7.2 ML - full data set - Coparing the results - balanced vs unbalanced
#   4.7.3 ML - selected features - comparing the results of balanced and unbalanced
#   4.8 ML - Comparing ML models execution time


##################################################
# 1. Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time


# 2. Loading Data set I - SMALL set (12000 x 18), binary class
df = pd.read_csv("C:\\ESTERA\\STUDIES\\Machine Learning\\CA 2\\online_shoppers_intention.csv")
# checking how data look like
df.info()

##################################################
# 2.2 EDA
# dropping missing values
df=df.dropna(axis=0)
df.isna().sum()

##################################################
# 2.3  DATA PREPARATION 
# Transforming bool columns to integers

# Mapping attributes - transforming a group of values into another group of values
# Mapping the nominal features into integer
VisitorType={'Returning_Visitor':3, 'New_Visitor':2, 'Other':1}
df['VisitorType']=df['VisitorType'].map(VisitorType)

# Mapping the ordinal feature into integer
Month={'Feb':2, 'Mar':3, 'May':5, 'Oct':10, 'June':6, 'Jul':7, 'Aug':8, 'Nov':11, 'Sep':9,'Dec':12}
df['Month']=df['Month'].map(Month)

# 'Revenue' & 'Weekend' column
df.loc[ df['Revenue'] == 'True', 'Revenue'] == 1
df.loc[ df['Revenue'] == 'False', 'Revenue'] == 0

df.loc[ df['Weekend'] == 'True', 'Weekend'] == 1
df.loc[ df['Weekend'] == 'False', 'Weekend'] == 0

# convert 'Revenue' column to integer
df['Revenue'] = df['Revenue'].astype(int)
df['Weekend'] = df['Weekend'].astype(int)

df.info()



#################################################
# 2.4 Checking distributions of attributes
for column in df:
        plt.figure(figsize=(7,1))
        sns.boxplot(data=df, x=column)



df.hist(bins=50, figsize=(20,15))
plt.show()


#################################################
#   2.5 Correlation
# checking correlations
# creating a heatmap
plt.figure(figsize=(15,10))
plot = sns.heatmap(df.corr().round(2), annot= True, cmap='BrBG')
plot.set_title('Heatmap')




#################################################
#   2.6 ML - importing libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve
#from sklearn import metrics
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.metrics import classification_report


##############################################
#   2.6.1 ML - full data set - raw
# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Split train-test data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


ML1 = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML1_columns = []
ML1_compare = pd.DataFrame(columns = ML1_columns)
row_index = 0

for alg in ML1:  
    
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML1_name = alg.__class__.__name__
    ML1_compare.loc[row_index,'MLA used'] = ML1_name
    ML1_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 4)
    ML1_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML1_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML1_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML1_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML1_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)

    row_index+=1

ML1_compare


#############################################
#   2.6.2 ML - full data set - normalized
### Data Normalization

#normalize values in every column
df_norm = (df-df.min())/ (df.max() - df.min())

#view normalized DataFrame
df_norm.head()

# Separating the independent variables from dependent variables
X = df_norm.iloc[:,:-1]
y = df_norm.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


ML2 = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML2_columns = []
ML2_compare = pd.DataFrame(columns = ML2_columns)
row_index = 0

for alg in ML2:  
    
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML2_name = alg.__class__.__name__
    ML2_compare.loc[row_index,'MLA used'] = ML2_name
    ML2_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 4)
    ML2_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML2_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML2_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML2_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML2_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)

    row_index+=1

ML2_compare



#########################################
#   2.6.3 ML - full data set - balanced

### Blanacing clases
df_SMOTE = df_norm

X = df_SMOTE.iloc[:,:-1]
y = df_SMOTE.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


# checking size of each class
# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# data is unbalanced, oversampling for training data will be done

SMOTE = SMOTE()

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))


ML3 = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML3_columns = []
ML3_compare = pd.DataFrame(columns = ML3_columns)
row_index = 0

for alg in ML3:  
    start = time.time()
    predicted = alg.fit(X_train_SMOTE, y_train_SMOTE).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML3_name = alg.__class__.__name__
    ML3_compare.loc[row_index,'MLA used'] = ML3_name
    ML3_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train_SMOTE, y_train_SMOTE), 4)
    ML3_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML3_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML3_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML3_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML3_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)
    end = time.time()
    print("The time of execution of ", alg, ' is: ',
      (end-start) * 10**3, "ms")
    row_index+=1

ML3_compare



#########################
#   2.6.4 ML - full data set - Comparint the results 

print(ML1_compare)
print(ML2_compare)
print(ML3_compare)

d = [['Raw', 0.512,0.773,0.841,0.884,0.919],
    ['Normalized',0.829,0.75,0.896,0.911,0.939],
    ['SMOTE',0.743,0.623,0.839,0.877,0.927]]
df_s = pd.DataFrame(d, columns =['Type','KNN','NB','Log','DT','RF'])
df_s

# visualising F-1 scores for each model and each data transformation
# displaying the results
palette = sns.color_palette("mako_r", 5)
g = sns.lineplot(data=df_s, markers=True, dashes=False, palette=palette)

g.set_xticks(range(len(df_s)))
g.set_xticklabels(['Raw','Normalized','SMOTE'])

plt.xlabel('Type')
plt.ylabel('F-1 Score')
plt.title('F-1 Scores - full dataset')



############ QUESTION 1
### Does the purchase of the product can be predicted ONLY by the time spent on the shop website?

#   2.7.1 ML - selected features - raw

# Raw data set - selecting only features with suration
df_1 = df[['Administrative_Duration','Informational_Duration','ProductRelated_Duration','Revenue']]


# Separating the independent variables from dependent variables
X = df_1.iloc[:,:-1]
y = df_1.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

ML1_2 = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML1_2_columns = []
ML1_2_compare = pd.DataFrame(columns = ML1_2_columns)
row_index = 0

for alg in ML1_2:  
    
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML1_2_name = alg.__class__.__name__
    ML1_2_compare.loc[row_index,'MLA used'] = ML1_2_name
    ML1_2_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 4)
    ML1_2_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML1_2_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML1_2_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML1_2_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML1_2_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)

    row_index+=1

ML1_2_compare

###################################################
#   2.7.2 ML - selected features - normalized

df_norm2 = df_norm[['Administrative_Duration','Informational_Duration','ProductRelated_Duration','Revenue']]
df_norm2

# Separating the independent variables from dependent variables
X = df_norm2.iloc[:,:-1]
y = df_norm2.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

ML2_2 = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML2_2_columns = []
ML2_2_compare = pd.DataFrame(columns = ML2_2_columns)
row_index = 0

for alg in ML2_2:  
    
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML2_2_name = alg.__class__.__name__
    ML2_2_compare.loc[row_index,'MLA used'] = ML2_2_name
    ML2_2_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 4)
    ML2_2_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML2_2_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML2_2_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML2_2_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML2_2_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)

    row_index+=1

ML2_2_compare


####################################################
#   2.7.3 ML - selected features - balnced

df_SMOTE2 = df_SMOTE[['Administrative_Duration','Informational_Duration','ProductRelated_Duration','Revenue']]
df_SMOTE2

X = df_SMOTE2.iloc[:,:-1]
y = df_SMOTE2.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

ML3_2 = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML3_2_columns = []
ML3_2_compare = pd.DataFrame(columns = ML3_2_columns)
row_index = 0

for alg in ML3_2:  
    
    predicted = alg.fit(X_train_SMOTE, y_train_SMOTE).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML3_2_name = alg.__class__.__name__
    ML3_2_compare.loc[row_index,'MLA used'] = ML3_2_name
    ML3_2_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train_SMOTE, y_train_SMOTE), 4)
    ML3_2_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML3_2_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML3_2_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML3_2_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML3_2_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)

    row_index+=1

ML3_2_compare



#######################################################
# # # Comparing f-1 results
#   2.7.4 ML - slected features - Comparing the results

print(ML1_2_compare)
print(ML2_2_compare)
print(ML3_2_compare)

du = [['Raw',0.151,0.173,0.038,0.236,0.152],
    ['Normalized',0.085,0.214,0.016,0.205,0.119],
    ['SMOTE',0.288,0.42,0.314,0.265,0.271]]
df_duration = pd.DataFrame(du, columns =['Type','KNN','NB','Log','DT','RF'])
df_duration

# visualising the results
palette = sns.color_palette("mako_r", 5)
g = sns.lineplot(data=df_duration, markers=True, dashes=False, palette=palette)

g.set_xticks(range(len(df_s)))
g.set_xticklabels(['Raw','Normalized','SMOTE'])

plt.xlabel('Type')
plt.ylabel('F-1 Score')
plt.title('F-1 Scores - duration features')





#################  DATA SET 2  ######################
## medium data set (42000 x 26), binary class, unbalanced 


#   3.1 Loading data set 2 - 'online shopping'
df = pd.read_csv("C:\\ESTERA\\STUDIES\\Machine Learning\\CA 2\\Online Shop.csv")
df.info()

###################################################
#   3.2 Data preparation
# dropping missing values
df=df.dropna(axis=0)
df.isna().sum()

# dropping unnecessary column
del df["UserID"]
del df["date"]
del df["time(s)"]
df.shape

# Appling one-hot encoding to the categorical features
categorical_features = ['device']
df = pd.get_dummies(df, columns = categorical_features)
df.info()

# mapping features to Integers 
df['location'] = df['location'].map({'Non EU': 0, 'EU': 1})
df['user'] = df['user'].map({'Returning': 0, 'New': 1})
df['Item'] = df['Item'].map({'Not ordered': 0, 'Ordered': 1,'Cancelled' : 2, 'Wishlist':3, 'Returned':4})
df['gender'] = df['gender'].map({'Female':2, 'Male' :1, 'Unknown':0})


def converter(column):
    if column == 'Yes':
        return 1
    else:
        return 0

# converting all 'Yes' and 'No' values to 1 and 0
df['basket_icon_click'] = df['basket_icon_click'].apply(converter)
df['basket_add_list'] = df['basket_add_list'].apply(converter)
df['basket_add_detail'] = df['basket_add_detail'].apply(converter)
df['sort_by'] = df['sort_by'].apply(converter)
df['image_picker'] = df['image_picker'].apply(converter)
df['account_page_click'] = df['account_page_click'].apply(converter)
df['promo_banner_click'] = df['promo_banner_click'].apply(converter)
df['detail_wishlist_add'] = df['detail_wishlist_add'].apply(converter)
df['list_size_dropdown'] = df['list_size_dropdown'].apply(converter)
df['closed_minibasket_click'] = df['closed_minibasket_click'].apply(converter)
df['checked_delivery_detail'] = df['checked_delivery_detail'].apply(converter)
df['checked_returns_detail'] = df['checked_returns_detail'].apply(converter)
df['sign_in'] = df['sign_in'].apply(converter)
df['saw_checkout'] = df['saw_checkout'].apply(converter)
df['saw_sizecharts'] = df['saw_sizecharts'].apply(converter)
df['saw_delivery'] = df['saw_delivery'].apply(converter)
df['saw_account_upgrade'] = df['saw_account_upgrade'].apply(converter)
df['saw_homepage'] = df['saw_homepage'].apply(converter)

df.info()


##########################################################
#   3.4 Checking distribution
# ### Whole data has a numeric value
df.describe()

#### Dataset do not have outliers, all values are between 0 to 1 <br> Two columns are from 0 to 2

# moving 'Item' column to last place
# 'Item' is our predicted column, does item was sold or not
last_column = df.pop('Item')
df.insert(23, 'Item', last_column)


##########################################################
#   3.5 Correlation

# checking correlations
# creating a heatmap
plt.figure(figsize=(15,10))
plot = sns.heatmap(df.corr().round(2), annot= True, cmap='BrBG')
plot.set_title('Heatmap')



###### ML 
#   3.6.1 ML - full data set - raw

# Separating the independent variables from dependent variables
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Split train-test data to 80:20 due to ‘Zero frequency’ problem
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


ML4 = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML4_columns = []
ML4_compare = pd.DataFrame(columns = ML4_columns)
row_index = 0

for alg in ML4:  
    
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML4_name = alg.__class__.__name__
    ML4_compare.loc[row_index,'MLA used'] = ML4_name
    ML4_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 4)
    ML4_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML4_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML4_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML4_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML4_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)

    row_index+=1

ML4_compare


##################################################################
#   3.6.2 ML - full data set - balanced
# BALANCING Labels classes

# summarize class distribution
print("Before oversampling: ",Counter(y_train))

# fit and apply the transform
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))

ML4_2 = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML4_2_columns = []
ML4_2_compare = pd.DataFrame(columns = ML4_2_columns)
row_index = 0

for alg in ML4:  
    start = time.time()
    predicted = alg.fit(X_train_SMOTE, y_train_SMOTE).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML4_2_name = alg.__class__.__name__
    ML4_2_compare.loc[row_index,'MLA used'] = ML4_2_name
    ML4_2_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train_SMOTE, y_train_SMOTE), 4)
    ML4_2_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML4_2_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML4_2_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML4_2_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML4_2_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)
    end = time.time()
    print("The time of execution of ", alg, ' is: ',
      (end-start) * 10**3, "ms")
    row_index+=1

ML4_2_compare



#################################################
# 3.6.3 ML - full data set - Coparing the results - balanced vs unbalanced

X = ['KNN','NB','LOG','DT','RF']
Unbalanced = ML4_compare['F1']
Balanced = ML4_2_compare['F1']

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.1, Unbalanced, 0.2, color='lightblue', label = 'Unbalanced')
plt.bar(X_axis + 0.1, Balanced, 0.2, color='steelblue', label = 'Balanced')

plt.xticks(X_axis, X)

plt.xlabel("Model")
plt.ylabel("F-1 Score")
plt.title("Models scores - full data set")
plt.legend(loc='lower right')
plt.show()

######################################################################
##### Question 2
##### Does purchase of the product can be predicted ONLY from customer browsing/clicking
##### on different areas in the shop webpage?


# 3.7.1 ML - selected features - unbalanced
df_pages = df[['basket_icon_click','basket_add_list','basket_add_detail','image_picker','account_page_click',
              'promo_banner_click','detail_wishlist_add','list_size_dropdown','list_size_dropdown',
              'checked_delivery_detail','checked_returns_detail','saw_checkout','saw_sizecharts','saw_delivery',
              'saw_homepage','Item']]

df_pages
# Separating the independent variables from dependent variables
X = df_pages.iloc[:,:-1]
y = df_pages.iloc[:,-1]

print("Before oversampling: ",Counter(y_train))


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


ML4_3 = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML4_3_columns = []
ML4_3_compare = pd.DataFrame(columns = ML4_3_columns)
row_index = 0

for alg in ML4_3:  
    
    predicted = alg.fit(X_train, y_train).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML4_3_name = alg.__class__.__name__
    ML4_3_compare.loc[row_index,'MLA used'] = ML4_3_name
    ML4_3_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train, y_train), 4)
    ML4_3_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML4_3_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML4_3_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML4_3_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML4_3_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)

    row_index+=1

ML4_3_compare



#########################################################################
# 3.7.2 ML - selected features - balanced

X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))

ML4_3_SMOTE = [KNeighborsClassifier(),GaussianNB(),LogisticRegression(),DecisionTreeClassifier(),RandomForestClassifier()]
ML4_3_SMOTE_columns = []
ML4_3_SMOTE_compare = pd.DataFrame(columns = ML4_2_columns)
row_index = 0

for alg in ML4_3_SMOTE:  
    
    predicted = alg.fit(X_train_SMOTE, y_train_SMOTE).predict(X_test)
    fp, tp, th = roc_curve(y_test, predicted)
    ML4_3_SMOTE_name = alg.__class__.__name__
    ML4_3_SMOTE_compare.loc[row_index,'MLA used'] = ML4_3_SMOTE_name
    ML4_3_SMOTE_compare.loc[row_index, 'Train Accuracy'] = round(alg.score(X_train_SMOTE, y_train_SMOTE), 4)
    ML4_3_SMOTE_compare.loc[row_index, 'Test Accuracy'] = round(alg.score(X_test, y_test), 4)
    ML4_3_SMOTE_compare.loc[row_index, 'Precission'] = round(precision_score(y_test, predicted), 4)
    ML4_3_SMOTE_compare.loc[row_index, 'Recall'] = round(recall_score(y_test, predicted), 4)
    ML4_3_SMOTE_compare.loc[row_index, 'F1'] = round(f1_score(y_test, predicted), 4)
    ML4_3_SMOTE_compare.loc[row_index, 'AUC'] = round(auc(fp, tp), 4)

    row_index+=1

ML4_3_SMOTE_compare


#############################################################
# 3.7.3 ML - selected features - comparing the results of balanced and unbalanced

X = ['KNN','NB','LOG','DT','RF']
Unbalanced = ML4_3_compare['F1']
Balanced = ML4_3_SMOTE_compare['F1']

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.1, Unbalanced, 0.2, color='lightblue', label = 'Unbalanced')
plt.bar(X_axis + 0.1, Balanced, 0.2, color='steelblue', label = 'Balanced')

plt.xticks(X_axis, X)
plt.xlabel("Model")
plt.ylabel("F-1 Score")
plt.title("Models scores - selected features ")
plt.legend(loc='lower right')
plt.show()


################################################
# 3.7.4 ML - comparing the results - full data vs slected features only (both balanced)

du = [['KNN',0.830,0.918],
    ['NB',0.804,0.860],
    ['LOG',0.877,0.887],
    ['DT',0.860,0.880],
    ['RF',0.878,0.888]]
df_duration = pd.DataFrame(du, columns =['Classyfier','Full data','Selected Features'])
df_duration

# visualising the results
palette = sns.color_palette("mako_r", 2)
g = sns.lineplot(data=df_duration, markers=True, dashes=False, palette=palette)

g.set_xticks(range(len(df_duration)))
g.set_xticklabels(['KNN','NB','LOG','DT','RF'])

plt.xlabel('Type')
plt.ylabel('F-1 Score')
plt.title('Models scores - full data set vs selected features only')




### Visualising the improvement for each classyfiers

Classifier = ['KNN','NB','LOG','DT','RF']
Difference = [0.88, 0.056, 0.01, 0.02, 0.01]

plt.bar(Classifier,Difference)

plt.xticks(X_axis, X)

plt.xlabel("Model")
plt.ylabel("F-1 Score")
plt.title("Models scores improvement")
plt.legend(loc='lower right')
plt.show()







####################################################
# 4.1 Loading Data set II - LARGE (286 000x10), multiclass, balanced

df = pd.read_csv("C:\\ESTERA\\STUDIES\\Machine Learning\\CA 2\\Online shop USA.csv")



######################################################
#   4.2 Data preparation
# dropping missing values
df=df.dropna(axis=0)
df.isna().sum()

df.info()

# mapping features to Integers
df['category'] = df['category'].map({"Mobiles & Tablets": 8, 'Books': 1, 'Beauty & Grooming':2,'Kids & Baby' :3, 
                                 "Women's Fashion":4, "Health & Sports":5, 'Others':6,'School & Education':7})
df['payment_method'] = df['payment_method'].map({'MasterCard': 1, 'PayPal': 2, 'Visa':3})
df['Name Prefix'] = df['Name Prefix'].map({'Prof.': 1, 'Hon.': 2, 'Doc.':3})
df['Gender'] = df['Gender'].map({'F':1, 'M':2})
df['Region'] = df['Region'].map({'South':1, 'Midwest':2, 'West':3, 'Northeast':4})
df['status'] = df['status'].map({'complete':1, 'refund':2, 'cancelled':3})

del df['State']


######################################################
#   4.3 EDA
df.info()

####################################################
#   4.4 Checking distribution
for column in df:
        plt.figure(figsize=(7,1))
        sns.boxplot(data=df, x=column)


df.hist(bins=50, figsize=(20,15))
plt.show()

###################################################
#   4.5 Correlation
# checking correlations
# creating a heatmap
plt.figure(figsize=(15,10))
plot = sns.heatmap(df.corr().round(2), annot= True, cmap='BrBG')
plot.set_title('Heatmap')


##################################################
# 4.6 ML

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# checking size of each class
#from collections import Counter

# summarize class distribution
print("Before oversampling: ",Counter(y_train))




##################################################
# 4.6.1 ML - full data sest - unbalanced

### KNN
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
cr_KNN = classification_report(y_test, y_pred)
print(cr_KNN)


### Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)
y_predNB = model.predict(X_test)
cr_NB = classification_report(y_test, y_predNB)
print(cr_NB)


##### Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_predLR = logreg.predict(X_test)
cr_LOG = classification_report(y_test, y_predLR)
print(cr_LOG)


##### DECISSION TREE
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)
y_predDT = decision_tree.predict(X_test)
cr_DT = classification_report(y_test, y_predDT)
print(cr_DT)


##### RANDOM FOREST
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)
y_predRF = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
cr_RF = classification_report(y_test, y_predRF)
print(cr_RF)


############################################
# 4.6.2 ML - full data set - comparing the results
# comparing models f1 score for class prediction - unbalanced
X = ['KNN','NB','LOG','DT','RF']
Completed = [0.32,0.26,0.14,0.6,0.59]
Returned = [0.35,0.5,0.54,0.56,0.57]
Cancelled = [0.33,0.48,0.64,0.77,0.78]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, Completed, 0.2, color='lightblue', label = 'Completed')
plt.bar(X_axis - 0.0, Returned, 0.2, color='steelblue', label = 'Returned')
plt.bar(X_axis + 0.2, Cancelled, 0.2, color='blue', label = 'Cancelled')

plt.xticks(X_axis, X)

plt.xlabel('Transaction')
plt.ylabel("F-1 Score")
plt.title("Models scores")
plt.legend(bbox_to_anchor =(1, .55), ncol = 1)
plt.show()






#############################################
# 4.7.1 ML - full data set - balanced

# Balancing classes with SMOTE
X_train_SMOTE, y_train_SMOTE = SMOTE.fit_resample(X_train, y_train)

# summarize class distribution
print("After oversampling: ",Counter(y_train_SMOTE))


### KNN
start = time.time()
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_SMOTE, y_train_SMOTE)
y_pred = knn.predict(X_test)
cr_KNN = classification_report(y_test, y_pred)
print(cr_KNN)
end = time.time()
print("The time of execution of KNN is :", (end-start) * 10**3, "ms")


### Naive Bayes
start = time.time()
model = GaussianNB()
model.fit(X_train_SMOTE, y_train_SMOTE)
y_predNB = model.predict(X_test)
cr_NB = classification_report(y_test, y_predNB)
print(cr_NB)
end = time.time()
print("The time of execution of NB is :", (end-start) * 10**3, "ms")


##### Logistic Regression
start = time.time()
logreg = LogisticRegression()
logreg.fit(X_train_SMOTE, y_train_SMOTE)
y_predLR = logreg.predict(X_test)
cr_LOG = classification_report(y_test, y_predLR)
print(cr_LOG)
end = time.time()
print("The time of execution of LOG is :", (end-start) * 10**3, "ms")


##### DECISSION TREE
start = time.time()
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train_SMOTE, y_train_SMOTE)
y_predDT = decision_tree.predict(X_test)
cr_DT = classification_report(y_test, y_predDT)
print(cr_DT)
end = time.time()
print("The time of execution of DT is :", (end-start) * 10**3, "ms")


##### RANDOM FOREST
start = time.time()
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train_SMOTE, y_train_SMOTE)
y_predRF = random_forest.predict(X_test)
random_forest.score(X_train, y_train)
cr_RF = classification_report(y_test, y_predRF)
print(cr_RF)
end = time.time()
print("The time of execution of RF is :", (end-start) * 10**3, "ms")



############################################################
#   4.7.2 ML - full data set - Coparing the results - balanced 
# comparing models f1 score for class prediction - balanced
X = ['KNN','NB','LOG','DT','RF']
Completed = [0.59,0.20,0.19,0.60,0.59]
Returned = [0.52,0.71,0.51,0.57,0.58]
Cancelled = [0.70,0.43,0.64,0.76,0.78]

X_axis = np.arange(len(X))

plt.bar(X_axis - 0.2, Completed, 0.2, color='lightblue', label = 'Completed')
plt.bar(X_axis - 0.0, Returned, 0.2, color='steelblue', label = 'Returned')
plt.bar(X_axis + 0.2, Cancelled, 0.2, color='blue', label = 'Cancelled')

plt.xticks(X_axis, X)

plt.xlabel('Classifier')
plt.ylabel("F-1 Score")
plt.title("Models scores - balanced")
plt.legend(bbox_to_anchor =(1, .55), ncol = 1)
plt.show()






############################################################
# 4.8 ML - Comparing ML models execution time
d = [['Small', 667.69,24.2,55.28,55.31,1297.49],
    ['Medium',67910.69,70.89,279.57,129.37,2270.64],
    ['Large',5554.50,135.38,2981.37,655.26,21203.82]]
df_s = pd.DataFrame(d, columns =['Type','KNN','NB','Log','DT','RF'])
df_s

# displaying the results
palette = sns.color_palette("mako_r", 5)
g = sns.lineplot(data=df_s, markers=True, dashes=False, palette=palette)

g.set_xticks(range(len(df_s)))
g.set_xticklabels(['Raw','Normalized','SMOTE'])

plt.xlabel('Type')
plt.ylabel('F-1 Score')
plt.title('F-1 Scores - full dataset')





### Comparing ML models F-1 score vs execution time
# data had assigned new values from 1 to 5
# 5 is the highest score, 1 the lowest
# 5 is the longest time, 1 the shortes

X = ['KNN','NB','LOG','DT','RF']
score = [2,1,4,3,5]
time = [4,1,2,3,5]
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.1, score, 0.2, color='lightblue', label = 'F-1 Score')
plt.bar(X_axis + 0.1, time, 0.2, color='steelblue', label = 'Execution Time')
plt.xticks(X_axis, X)
plt.xlabel("Classifier")
plt.title("Small dataset")
plt.legend()
plt.show()

X = ['KNN','NB','LOG','DT','RF']
score = [2,1,4,3,5] 
time = [5,1,3,2,4]
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.1, score, 0.2, color='lightblue', label = 'F-1 Score')
plt.bar(X_axis + 0.1, time, 0.2, color='steelblue', label = 'Execution Time')
plt.xticks(X_axis, X)
plt.xlabel("Classifier")
plt.title("Medium dataset")
plt.legend()
plt.show()

X = ['KNN','NB','LOG','DT','RF']
score = [3,1,2,4,5]
time = [4,1,3,2,5]
X_axis = np.arange(len(X))
plt.bar(X_axis - 0.1, score, 0.2, color='lightblue', label = 'F-1 Score')
plt.bar(X_axis + 0.1, time, 0.2, color='steelblue', label = 'Execution Time')
plt.xticks(X_axis, X)
plt.xlabel("Classifier")
plt.title("Large dataset")
plt.legend()
plt.show()

