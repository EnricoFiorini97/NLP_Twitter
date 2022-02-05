
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import re
import nltk
nltk.download('omw-1.4')


from asyncio import base_events
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV

accsTab = []
suggTab = [["numTot","numGood/numTot","avgSuggFollowed"]]
xygroups = []

groups = ['g1-new','g2-new','g3-new','g4-new','g5-new']
#groups = ['g1-new','g2-new','g3-new','g4-new','g5-new','g6-new']
#groups = ['g1','g2','g3','g4','g5','g6']
#groups = ['g1-new']

SEED = 1234
NRUNS = 10
TESTSIZE = 0.1

np.random.seed(SEED)

shuffles = model_selection.ShuffleSplit(n_splits=NRUNS, test_size=TESTSIZE, random_state=SEED)
base_path = "/home/enrico/Documents/repos/NLP_Twitter/raw_dataset/"

df_target_nlp = pd.DataFrame(columns=["FULL_TEXT"])
for group in ['g1-new','g2-new']:
    df1 = pd.read_csv(base_path+'threads-'+group+'-ng.txt',sep='\t')
    df2 = pd.read_csv(base_path+'threads-'+group+'-2020-ng.txt',sep='\t')
    df = pd.concat([df1,df2]) 
    for tweet in df["FULL_TEXT"]:
        df_target_nlp.loc[len(df_target_nlp)] = [tweet]

del df1
del df2
del df
#NLP model training

df_nlp = pd.read_csv("/home/enrico/Documents/repos/NLP_Twitter/POEC/dataset_2.csv", delimiter=',')

#Building up data for plotting
categories = list(df_nlp["CATEGORY"].unique())
try:
    categories.remove(np.nan)
except: pass

data = {}

for category in categories:
    data[category] = 0

for cat in df_nlp["CATEGORY"]:
    try:
        data[cat] += 1
    except Exception:
        pass
    
#Getting dataset composition
names = list(data.keys())
values = list(data.values())

#Plotting dataset composition
plt.bar(names[0], values[0], color="blue", label=names[0])
plt.bar(names[1], values[1], color="red", label=names[1])
plt.bar(names[2], values[2], color="black", label=names[2])
plt.bar(names[3], values[3], color="green", label=names[3])
plt.bar(names[4], values[4], color="yellow", label=names[4])
plt.bar(names[5], values[5], color="brown", label=names[5])
plt.bar(names[6], values[6], color="orange", label=names[6])
plt.xlabel("Categories")
plt.ylabel("Size")
plt.title("Dataset composition")
plt.legend()
plt.grid()
plt.show()

models = {}

#Special characted cleaning
df_nlp["F_CONTENT"] = df_nlp["F_CONTENT"].str.replace("'s", "")
df_nlp["F_CONTENT"] = df_nlp["F_CONTENT"].str.replace("’s", "")
df_nlp["F_CONTENT"] = df_nlp['F_CONTENT'].apply(lambda x: re.sub(r'http\S+',"", str(x)))
df_nlp["F_CONTENT"] = df_nlp["F_CONTENT"].str.strip().str.lower().str.replace('"','')

#TODO: modify punct sign list(?)

try:
    remove_punct = ""
    remove_punct = input("Do you want to remove punctuation? [Y/n] ")
    if remove_punct != "n":
        for punct_sign in list("?:!.,;'’"):
            df_nlp["F_CONTENT"] = df_nlp["F_CONTENT"].str.replace(punct_sign, ' ')
except Exception:
    pass

nltk.download('punkt')
nltk.download('wordnet')

#Lemmatization
wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_text_list = []
for row in range(len(df_nlp)):
    lemmatized_list = []
    text = df_nlp.loc[row]["F_CONTENT"]
    text_words = text.split(" ")

    for word in text_words:
        lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
    
    lemmatized_text_list.append(" ".join(lemmatized_list))

df_nlp["F_CONTENT"] = lemmatized_text_list

#Stopwords
nltk.download('stopwords')
stop_words = list(stopwords.words('english'))
for stop_word in stop_words:
    regex_stopword = r"\b" + stop_word + r"\b"
    df_nlp["F_CONTENT"] = df_nlp["F_CONTENT"].str.replace(regex_stopword, '')

category_codes = {categories[i] : i for i in range(len(categories))}
df_nlp["F_CAT_CODE"] = [cat.split("-")[0] for cat in df_nlp["F_NAME"]]
df_nlp = df_nlp.replace({"F_CAT_CODE" : category_codes})

X_train, X_test, y_train, y_test = train_test_split(df_nlp["F_CONTENT"], 
                                                        df_nlp["F_CAT_CODE"], 
                                                        test_size=0.15, 
                                                        random_state=8
                                                    )
#Check with TF-Idf_nlp, setting parameters
ngram_range = (1,2)
min_df_nlp = 1
max_df_nlp = 0.45
max_features = 1200

tfidf_nlp = TfidfVectorizer(encoding='utf-8',
                    ngram_range=ngram_range,
                    stop_words=None,
                    lowercase=False,
                    max_df=max_df_nlp,
                    min_df=min_df_nlp,
                    max_features=max_features,
                    norm='l2',
                    sublinear_tf=True)
                        
features_train = tfidf_nlp.fit_transform(X_train).toarray()
labels_train = y_train
print(features_train.shape)

features_test = tfidf_nlp.transform(X_test).toarray()
labels_test = y_test
print(features_test.shape)

#LOGISTIC REGRESSION
# Create the parameter grid based on the results of random search 
C = [float(x) for x in np.linspace(start = 0.6, stop = 1, num = 10)]
multi_class = ['multinomial']
solver = ['sag']
class_weight = ['balanced']
penalty = ['l2']

param_grid = {'C': C,
            'multi_class': multi_class,
            'solver': solver,
            'class_weight': class_weight,
            'penalty': penalty}

# Create a base model
lrc = LogisticRegression(random_state=8)

# Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

# Instantiate the grid search model
grid_search = GridSearchCV(estimator=lrc, 
                        param_grid=param_grid,
                        scoring='accuracy',
                        cv=cv_sets,
                        verbose=1)

# Fit the grid search to the data
grid_search.fit(features_train, labels_train)

print("The mean accuracy of a LR model is:")
print(grid_search.best_score_)

best_lrc = grid_search.best_estimator_
best_lrc.fit(features_train, labels_train)

# Training accuracy
print("The training accuracy of a LR model is: ")
print(accuracy_score(labels_train, best_lrc.predict(features_train)))

for group in groups:
    # Read dataset
    # df1 = pd.read_csv('threads-'+group+'.txt',sep='\t')     # dati 2018
    df1 = pd.read_csv(base_path+'threads-'+group+'-ng.txt',sep='\t')     # dati 2018 ng
    #print (len(df1.index))
    # df2 = pd.read_csv('threads-'+group+'-2020.txt',sep='\t')   # dati 2020
    df2 = pd.read_csv(base_path+'threads-'+group+'-2020-ng.txt',sep='\t')   # dati 2020 ng
    #print (len(df2.index))
    df = pd.concat([df1,df2]) 
    #print (len(df.index))
    #df = pd.read_csv('threads-all-new.txt',sep='\t')   # tutti i gruppi (diretti)
    #df = df.drop(['FOLLOWERS'], axis=1)
    #df = df.drop(['MUSEUM'], axis=1)
    #df = df.drop(['PARTOFDAY'], axis=1)
    #df = df.drop(['ISRETWEET'], axis=1)
    df = df.drop(['ENG','LONG'], axis=1)
    df = df.drop(['FULL_TEXT'], axis=1)
    #df = df.drop(['HASEMARK','HASNERMISC','DENSE','HASNERORG','HASNERLOC'], axis=1) # top 12
    #df = df.drop(['FULL_TEXT','ENG','LONG','DENSE','HASEMARK','HASNERORG','HASNERLOC','HASNERMISC'], axis=1) # top 11 old
    df['FOLLOWERS']=df['FOLLOWERS']/1000
    df['MUSEUM']=df['MUSEUM'].astype('category').cat.codes  # label encoding
    df['PARTOFDAY']=df['PARTOFDAY'].astype('category').cat.codes  # label encoding
    #df = pd.get_dummies( df, columns = ['MUSEUM'] )  # one-hot-encoding
    #df = pd.get_dummies( df, columns = ['PARTOFDAY'] )  # one-hot-encoding
    # df description
    #display(df.loc[:, df.columns != 'FORMULA'].describe())
    # print GOOD/BAD thresholds
    # print (df.NRETWEET.quantile([0.2,0.8]))
    # exclude central tweets
    df = df[ (df['NRETWEET'].rank(pct=True) < 0.2) | (df['NRETWEET'].rank(pct=True) > 0.8)] # elimina valori intermedi
    #df = df[ (df['NRETWEET'].rank(pct=True) < 0.2) | ((df['NRETWEET'].rank(pct=True) > 0.4) & (df['NRETWEET'].rank(pct=True) < 0.6)) | (df['NRETWEET'].rank(pct=True) > 0.8)] # elimina due buchi intermedi
    #print (len(df.index))
    
    # correlation plot
    #display (df.corr(method='spearman')['NRETWEET'])
    corr = df.corr(method='spearman')
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})
    
    # pair plot
    #sns.set(style="ticks")
    #sns.pairplot(df)
    
    # display(df.describe())
    # display(df.dtypes)
    # display(df.head())
    
    # target (y): 1 NRETWEET 2 NLIKE 3 NANS
    X, y = df.iloc[:,4:],df.iloc[:,1]   # X: prendo da col 5 a penultima
    y = np.where(df['NRETWEET'].rank(pct=True)>0.5, 'GOOD', 'BAD')  # y: 2 classi
    #y = np.where(df['NRETWEET'].rank(pct=True)>0.66, 'GOOD', np.where(df['NRETWEET'].rank(pct=True)>0.33, 'NO', 'BAD'))  # y: 3 classi
    #y = np.where(df['NRETWEET'].rank(pct=True)>0.8, 'GOOD', np.where(df['NRETWEET'].rank(pct=True)>0.2, 'NO', 'BAD'))  # y: 3 classi (sbilanciate)
    y = LabelEncoder().fit(y).transform(y)    # label encoding per target, 'BAD'->0, 'GOOD'->1
    xygroups.append([X,y])
 
    ## regression - xgboost
    #xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1,
    #                max_depth = 5, alpha = 10, n_estimators = 10)
    #y = df['NRETWEET']
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TESTSIZE, random_state=SEED)
    #xg_reg.fit(X_train,y_train)
    #preds = xg_reg.predict(X_test)
    #rmse = np.sqrt(mean_squared_error(y_test, preds))
    ## mape = np.mean(np.abs((np.array(y_test) - np.array(preds)) / np.array(y_test))) * 100
    #print("RMSE: %f" % (rmse))
    ## print("MAPE: %f" % (mape))

    # classification (xgb + sk classifiers)
    #cls = [KNeighborsClassifier(3),
         ##SVC(kernel="linear", C=0.025),
         ##SVC(gamma=2, C=1),
         ##GaussianProcessClassifier(1.0 * RBF(1.0)),
         #DecisionTreeClassifier(max_depth=5, random_state = np.random.seed(SEED)),
         #RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1,random_state = np.random.seed(SEED)),
         ##MLPClassifier(alpha=1, max_iter=1000),
         #AdaBoostClassifier(),
         ##GaussianNB(),
         #xgb.XGBClassifier()]
    cls = [xgb.XGBClassifier()]
    if not len(accsTab):
        accsTab.append([cl.__class__.__name__ for cl in cls])
    accs = []
    for cl in cls:
        #print (cl.__class__.__name__)
        print (".", end = '')
        #cl.fit(X_train,y_train)
        #preds = cl.predict(X_test)
        #accuracy = accuracy_score(y_test, preds)
        results = model_selection.cross_val_score(cl, X, y, cv=shuffles, scoring='f1')
        #results = model_selection.cross_val_score(cl, X, y, cv=shuffles, scoring='accuracy') 
        accs.append("%.2f%%" % (results.mean()*100.0))
    accsTab.append(accs)
    
    # sugg accuracy test
    KNN = 5
    NUMSUGG = 3
    #print ("\nSugg test")

    X_clean = X.drop(['FOLLOWERS', 'MUSEUM', 'PARTOFDAY', 'ISRETWEET','LENGTH'], axis = 1) 
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=TESTSIZE, random_state=SEED)
    cl = xgb.XGBClassifier()
    cl.fit(X_train,y_train)
    explainer = shap.TreeExplainer(cl)
    X_test_predict = X_test.copy()
    X_test_predict["PREDICTION"] = cl.predict(X_test)
    X_test_predict_bad = X_test_predict[X_test_predict["PREDICTION"]==0].drop(['PREDICTION'], axis = 1)
    X_train_predict = X_train.copy()
    #X_train_predict["PREDICTION"] = y_train
    X_train_predict["PREDICTION"] = cl.predict(X_train)
    X_train_predict_good = X_train_predict[X_train_predict["PREDICTION"]==1].drop(['PREDICTION'], axis = 1) 
    ##print (X_test_predict_bad.shape[0])
    ##print (X_train_predict_good.shape[0])
    ##print (X_test_predict_bad)
    numGood = 0
    numTot = 0
    numSuggFollowed = []

    for index in range(X_test_predict_bad.shape[0]):     # for each bad tweet
        badTweet = X_test_predict_bad.iloc[[index],:].copy()
        #prediction = cl.predict(badTweet)
        ## print (prediction[0])  # predicted value
        ## print (y_test.flat[0]) # expected value
        shap_values = explainer.shap_values(badTweet)[0].tolist()
        #print ("Shap values: "+str(shap_values))
        minSuggIndexes = sorted(range(len(shap_values)), key=lambda k: shap_values[k])  # indexes of features in order of lowest shap value (highest negative impact)
        #print ("MinSuggIndexes: "+str(minSuggIndexes))
        # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])
        neigh = NearestNeighbors()
        neigh.fit(X_train_predict_good)
        knnIdxs = neigh.kneighbors(badTweet, KNN, return_distance=False).tolist()[0]
        ##print (knnIdxs)
        neighs = X_train_predict_good.iloc[ knnIdxs , : ]
        #print ("Bad tweet: \n"+str(badTweet))
        #print ("Nearest neighbors: \n"+str(neighs))
        suggFollowed = 0
        modifiedTweet = badTweet.copy()
        for suggIdx in minSuggIndexes:   # suggestion generation / tweet modification cycle
            startSugg = int(badTweet[badTweet.columns[suggIdx]])  # sugg start value
            targetSugg = int(round(neighs[neighs.columns[suggIdx]].mean()))   # sugg target value
            colNameSugg = badTweet.columns[suggIdx]  # sugg column name
            modSugg = targetSugg    # sugg modification to perform (same as target for boolean features)
            if startSugg == targetSugg:
                continue
            if targetSugg-startSugg>1:   # non-boolean and far from target feature
                modSugg = int(round((targetSugg+startSugg)/2))
            #print ("SUGG: "+colNameSugg+" from "+str(startSugg)+" to "+str(targetSugg)+" mod "+str(modSugg))
            modifiedTweet[modifiedTweet.columns[suggIdx]] = modSugg   # ok mod
            #rndSuggIdx = np.random.randint(0,len(badTweet.columns)-1)
            #rndModSugg = 1 if int(badTweet[badTweet.columns[rndSuggIdx]])==0 else 0
            #modifiedTweet[modifiedTweet.columns[rndSuggIdx]] = rndModSugg # random mod
            suggFollowed += 1
            if suggFollowed == NUMSUGG:
                break
        numSuggFollowed.append(suggFollowed)
        oldPrediction = cl.predict(badTweet)[0]
        newPrediction = cl.predict(modifiedTweet)[0]
        if newPrediction:
            numGood += 1
        numTot += 1
        #print ("Modified tweet: \n"+str(modifiedTweet))
        #print ("Old prediction: "+str(oldPrediction))
        #print ("New prediction: "+str(newPrediction))
        #break
    #print ("\nnumTot: "+str(numTot))
    #print ("numGood/numTot: "+str(numGood/numTot*100)+"%")
    #print ("avg numSuggFollowed: "+str(sum(numSuggFollowed)/len(numSuggFollowed)))
    suggTab.append([str(numTot), "%.2f"%(numGood/numTot*100), "%.2f"%(sum(numSuggFollowed)/len(numSuggFollowed))])

suggTab = list(map(list, zip(*suggTab)))
suggDF = pd.DataFrame(suggTab,columns=['TestSugg']+groups)
suggDF = suggDF.style.hide_index()
#display (suggDF)

accsTab = list(map(list, zip(*accsTab)))
accsDF = pd.DataFrame(accsTab,columns=['Method']+groups)
accsDF = accsDF.style.hide_index()
accsDF



# SHAP

# load JS visualization code to notebook
shap.initjs()

for xy in xygroups:
    X = xy[0]
    y = xy[1]
    model = xgb.train({"learning_rate": 0.01}, xgb.DMatrix(X, label=y), 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    # 1 - visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    #display(shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:]))
    fig = shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:],show=False,matplotlib=True,figsize=(10,3),text_rotation=30)
    #plt.savefig('prova.pdf')





# 2 - visualize the training set predictions
#display(shap.force_plot(explainer.expected_value, shap_values, X))





# 3 - create a dependence plot to show the effect of a single feature across the whole dataset
#shap.dependence_plot("NMENTION", shap_values, X)




# 4 - summarize the effects of all the features
for xy in xygroups:
    X = xy[0]
    y = xy[1]
    model = xgb.train({"learning_rate": 0.01}, xgb.DMatrix(X, label=y), 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X)




# 5 - summarize bar plot
for xy in xygroups:
    X = xy[0]
    y = xy[1]
    model = xgb.train({"learning_rate": 0.01}, xgb.DMatrix(X, label=y), 100)
    #print (model.get_score(importance_type='gain'))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    shap.summary_plot(shap_values, X, plot_type="bar")
    df_shap_values = pd.DataFrame(data=shap_values).apply(abs).mean(axis = 0) 
    print (df_shap_values)



#xgboost feature importance
imp=[]
i=1
for xy in xygroups:
    X = xy[0]
    y = xy[1]
    model = xgb.train({"learning_rate": 0.01}, xgb.DMatrix(X, label=y), 100)
    d = model.get_score(importance_type='gain')
    sv = pd.DataFrame([d], columns=d.keys(), index=['Group_'+str(i)])
    imp.append(sv)
    #sv = sv.sort_values(by=[0],axis=1,ascending=False)
    i=i+1
sv = pd.concat(imp,sort=False)
#display(sv)


