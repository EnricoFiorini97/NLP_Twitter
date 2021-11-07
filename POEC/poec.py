# Data manipulation
import nltk
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit


def main() -> None:
    #Loading dataset
    df = pd.read_csv("dataset.csv", delimiter=',')

    #Building up data for plotting
    categories = df["CATEGORY"].unique()
    data = {}
    for category in categories:
        data[category] = 0

    for cat in df["CATEGORY"]:
        data[cat] += 1

    #Getting dataset composition
    names = list(data.keys())
    values = list(data.values())
    
    #Plotting dataset composition

    print(names)
    
    plt.bar(names[0], values[0], color="blue", label=names[0])
    plt.bar(names[1], values[1], color="red", label=names[1])
    plt.bar(names[2], values[2], color="black", label=names[2])
    plt.bar(names[3], values[3], color="green", label=names[3])
    plt.bar(names[4], values[4], color="yellow", label=names[4])
    plt.bar(names[5], values[5], color="orange", label=names[5])
    plt.xlabel("Categories")
    plt.ylabel("Size")
    plt.title("Dataset composition")
    plt.legend()
    plt.grid()
    #plt.show()

    #Special characted cleaning
    df["F_CONTENT"] = df["F_CONTENT"].str.replace("'s", "")
    df["F_CONTENT"] = df["F_CONTENT"].str.replace("’s", "")
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

    category_codes = {categories[i]:i for i in range(len(categories))}
    df["F_CAT_CODE"] = [cat.split("-")[0] for cat in df["F_NAME"]]
    df = df.replace({"F_CAT_CODE":category_codes})
    
    X_train, X_test, y_train, y_test = train_test_split(df["F_CONTENT"], 
                                                    df["F_CAT_CODE"], 
                                                    test_size=0.15, 
                                                    random_state=8)

    #Check with TF-IDF, setting parameters
    ngram_range = (1,2)
    min_df = 0.05
    max_df = 0.90
    max_features = 300

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

    for Product, category_id in sorted(category_codes.items()):
        features_chi2 = chi2(features_train, labels_train == category_id)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print("# '{}' category:".format(Product))
        print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
        print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-2:])))

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
        pickle.dump(tfidf, output)

        
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

    rf_0 = RandomForestClassifier(random_state = 8)

    # n_estimators
    n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 5)]

    # max_features
    max_features = ['auto', 'sqrt']

    # max_depth
    max_depth = [int(x) for x in np.linspace(20, 100, num = 5)]
    max_depth.append(None)

    # min_samples_split
    min_samples_split = [2, 5, 10]

    # min_samples_leaf
    min_samples_leaf = [1, 2, 4]

    # bootstrap
    bootstrap = [True, False]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}

    pprint(random_grid)


    # First create the base model to tune
    rfc = RandomForestClassifier(random_state=8)

    # Definition of the random search
    random_search = RandomizedSearchCV(estimator=rfc,
                                    param_distributions=random_grid,
                                    n_iter=50,
                                    scoring='accuracy',
                                    cv=3, 
                                    verbose=1, 
                                    random_state=8)

    # Fit the random search model
    random_search.fit(features_train, labels_train)
    print("The best hyperparameters from Random Search are:")
    print(random_search.best_params_)
    print("")
    print("The mean accuracy of a model with these hyperparameters is:")
    print(random_search.best_score_)

if __name__ == "__main__":
    main()