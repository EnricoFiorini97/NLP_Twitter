# Data manipulation
import nltk
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import re
import xgboost as xgb

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier


def main() -> None:
    #Loading dataset
    df = pd.read_csv("dataset.csv", delimiter=',')

    #Building up data for plotting
    categories = list(df["CATEGORY"].unique())
    categories.remove(np.nan)

    data = {}

    for category in categories:
        data[category] = 0

    for cat in df["CATEGORY"]:
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
    plt.xlabel("Categories")
    plt.ylabel("Size")
    plt.title("Dataset composition")
    plt.legend()
    plt.grid()
    plt.show()

    models = {}

    #Special characted cleaning
    df["F_CONTENT"] = df["F_CONTENT"].str.replace("'s", "")
    df["F_CONTENT"] = df["F_CONTENT"].str.replace("’s", "")
    df["F_CONTENT"] = df['F_CONTENT'].apply(lambda x: re.sub(r'http\S+',"", str(x)))
    df["F_CONTENT"] = df["F_CONTENT"].str.strip().str.lower().str.replace('"','')

    #TODO: modify punct sign list(?)

    try:
        remove_punct = ""
        remove_punct = input("Do you want to remove punctuation? [Y/n] ")
        if remove_punct != "n":
            for punct_sign in list("?:!.,;'’"):
                df["F_CONTENT"] = df["F_CONTENT"].str.replace(punct_sign, ' ')
    except Exception:
        return
    
    nltk.download('punkt')
    nltk.download('wordnet')

    #Lemmatization
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatized_text_list = []
    for row in range(len(df)):
        lemmatized_list = []
        text = df.loc[row]["F_CONTENT"]
        text_words = text.split(" ")

        for word in text_words:
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))
        
        lemmatized_text_list.append(" ".join(lemmatized_list))

    df["F_CONTENT"] = lemmatized_text_list
   
    #Stopwords
    nltk.download('stopwords')
    stop_words = list(stopwords.words('english'))
    for stop_word in stop_words:
        regex_stopword = r"\b" + stop_word + r"\b"
        df["F_CONTENT"] = df["F_CONTENT"].str.replace(regex_stopword, '')

    category_codes = {categories[i] : i for i in range(len(categories))}
    df["F_CAT_CODE"] = [cat.split("-")[0] for cat in df["F_NAME"]]
    df = df.replace({"F_CAT_CODE" : category_codes})

    print(categories)
    
    X_train, X_test, y_train, y_test = train_test_split(df["F_CONTENT"], 
                                                            df["F_CAT_CODE"], 
                                                            test_size=0.15, 
                                                            random_state=8
                                                        )
    #Check with TF-IDF, setting parameters
    ngram_range = (1,2)
    min_df = 1
    max_df = 0.45
    max_features = 1200

    tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features=max_features,
                        norm='l2',
                        sublinear_tf=True)
                            
    features_train = tfidf.fit_transform(X_train).toarray()
    labels_train = y_train
    print(features_train.shape)

    features_test = tfidf.transform(X_test).toarray()
    labels_test = y_test
    print(features_test.shape)

    for category, category_id in category_codes.items():
        features_chi2 = chi2(features_train, labels_train == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}' category:".format(category))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))

        '''with open(f"../Features/{category}.txt", 'w') as f:
            f.write("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
            f.write("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))

    #WARNING: these pickles are intended to be printed out only in tuning and test phase, they will be removed later.

    #X_train
    with open('Pickles/X_train.pickle', 'wb') as output:
        pickle.dump(X_train, output)

    # X_test    
    with open('Pickles/X_test.pickle', 'wb') as output:
        pickle.dump(X_test, output)

    # y_train
    with open('Pickles/y_train.pickle', 'wb') as output:
        pickle.dump(y_train, output)
        
    # y_test
    with open('Pickles/y_test.pickle', 'wb') as output:
        pickle.dump(y_test, output)
        
    # df
    with open('Pickles/df.pickle', 'wb') as output:
        pickle.dump(df, output)
        
    # features_train
    with open('Pickles/features_train.pickle', 'wb') as output:
        pickle.dump(features_train, output)

    # labels_train
    with open('Pickles/labels_train.pickle', 'wb') as output:
        pickle.dump(labels_train, output)

    # features_test
    with open('Pickles/features_test.pickle', 'wb') as output:
        pickle.dump(features_test, output)

    # labels_test
    with open('Pickles/labels_test.pickle', 'wb') as output:
        pickle.dump(labels_test, output)
        
    # TF-IDF object
    with open('Pickles/tfidf.pickle', 'wb') as output:
        pickle.dump(tfidf, output)'''

        
    # Dataframe
    path_df = "Pickles/df.pickle"
    with open(path_df, 'rb') as data:
        df = pickle.load(data)

    # features_train
    path_features_train = "Pickles/features_train.pickle"
    with open(path_features_train, 'rb') as data:
        features_train = pickle.load(data)

    # labels_train
    path_labels_train = "Pickles/labels_train.pickle"
    with open(path_labels_train, 'rb') as data:
        labels_train = pickle.load(data)

    # features_test
    path_features_test = "Pickles/features_test.pickle"
    with open(path_features_test, 'rb') as data:
        features_test = pickle.load(data)

    # labels_test
    path_labels_test = "Pickles/labels_test.pickle"
    with open(path_labels_test, 'rb') as data:
        labels_test = pickle.load(data)

    
    #GRID SEARCH CV
    # Create the parameter grid based on the results of random search 
    bootstrap = [False]
    max_depth = [30, 40, 50]
    max_features = ['sqrt']
    min_samples_leaf = [1, 2, 4]
    min_samples_split = [5, 10, 15]
    n_estimators = [800]

    param_grid = {
        'bootstrap': bootstrap,
        'max_depth': max_depth,
        'max_features': max_features,
        'min_samples_leaf': min_samples_leaf,
        'min_samples_split': min_samples_split,
        'n_estimators': n_estimators
    }

    # Create a base model
    rfc = RandomForestClassifier(random_state=8)

    # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
    cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=rfc, 
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=cv_sets,
                            verbose=1)

    # Fit the grid search to the data
    grid_search.fit(features_train, labels_train)
    print("The mean accuracy of a GridSearchCV model is:")
    print(grid_search.best_score_)
    models["GridSearchCV"] = grid_search.best_score_

    #SVC
    svc_0 =svm.SVC(random_state=8)
    # C
    C = [.0001, .001, .01]

    # gamma
    gamma = [.0001, .001, .01, .1, 1, 10, 100]

    # degree
    degree = [1, 2, 3, 4, 5]

    # kernel
    kernel = ['linear', 'rbf', 'poly']

    # probability
    probability = [True]

    # Create the random grid
    random_grid = {'C': C,
                'kernel': kernel,
                'gamma': gamma,
                'degree': degree,
                'probability': probability
                }

    # First create the base model to tune
    svc = svm.SVC(random_state=8)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=svc,
                                    param_distributions=random_grid,
                                    n_iter=50,
                                    scoring='accuracy',
                                    cv=3, 
                                    verbose=1, 
                                    random_state=8)

    # Fit the random search model
    random_search.fit(features_train, labels_train)

    print("The mean accuracy of a SCV model  is:")
    print(random_search.best_score_)
    models["SupportVectorMachine"] = random_search.best_score_


    knnc_0 =KNeighborsClassifier()
    # Create the parameter grid 
    n_neighbors = [int(x) for x in np.linspace(start = 1, stop = 500, num = 100)]

    param_grid = {'n_neighbors': n_neighbors}

    # Create a base model
    knnc = KNeighborsClassifier()

    # Manually create the splits in CV in order to be able to fix a random_state (GridSearchCV doesn't have that argument)
    cv_sets = ShuffleSplit(n_splits = 3, test_size = .33, random_state = 8)

    # Instantiate the grid search model
    grid_search = GridSearchCV(estimator=knnc, 
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=cv_sets,
                            verbose=1)

    # Fit the grid search to the data
    grid_search.fit(features_train, labels_train)
    print("The mean accuracy of a KNN model is:")
    print(grid_search.best_score_)
    models["KNeighborsClassifier"] = grid_search.best_score_



    #MULTINOMIAL NAIVE BAYES
    mnbc = MultinomialNB()
    mnbc.fit(features_train, labels_train)
    mnbc_pred = mnbc.predict(features_test)

    print("The mean accuracy of MNB a model is: ")
    print(accuracy_score(labels_train, mnbc.predict(features_train)))

    # Test accuracy
    print("The test accuracy of MNB a model is: ")
    print(accuracy_score(labels_test, mnbc_pred))

    models["MultinomialNB"] = accuracy_score(labels_train, mnbc.predict(features_train))


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
    lrc_pred = best_lrc.predict(features_test)

    # Training accuracy
    print("The training accuracy of a LR model is: ")
    print(accuracy_score(labels_train, best_lrc.predict(features_train)))
    models["LogisticRegression"] = accuracy_score(labels_train, best_lrc.predict(features_train))

    model_names, model_scores  = list(models.keys()), list(models.values())
     
    plt.bar(model_names[0], model_scores[0], color="blue", label=model_names[0])
    plt.bar(model_names[1], model_scores[1], color="red", label=model_names[1])
    plt.bar(model_names[2], model_scores[2], color="black", label=model_names[2])
    plt.bar(model_names[3], model_scores[3], color="green", label=model_names[3])
    plt.bar(model_names[4], model_scores[4], color="yellow", label=model_names[4])
    plt.xlabel("Model name")
    plt.ylabel("Accuracy score")
    plt.title("Various adopted models scores")
    plt.legend()
    plt.grid()
    plt.show()

 
if __name__ == "__main__":
    main()