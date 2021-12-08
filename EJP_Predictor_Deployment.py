import pandas as pd  # dataframe
import numpy as np  # algebre
import matplotlib.pyplot as plt  # visualisation
import seaborn as sns  # visualisation
import imblearn


function_to_apply = np.log

plt.rcParams['figure.figsize'] = (12, 10)


def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 1],  # consommation == 1
            X[y == l, 2],  # temperature == 2
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.xlabel('consommation')
    plt.ylabel('temperature')

    plt.legend(loc='upper right')
    plt.show()


def OverSample(sampler, label, X, y, ratio=None):
    os = sampler()
    X_sm, y_sm = os.fit_resample(X, y)

    # plot_2d_space(X_sm, y_sm, label)
    return X_sm, y_sm


def df_x_y(X_us, y_us, labels):
    X_us = pd.DataFrame(X_us, columns=labels)
    y_us = pd.DataFrame(y_us, columns=['JourPointe'])
    X_us['jour'] = X_us['jour'].astype('int32')
    X_us['StockRestant'] = X_us['StockRestant'].astype('int32')
    # y_us['JourPointe'] = y_us['JourPointe'].astype('int32')
    return X_us, y_us


from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from imblearn.combine import SMOTETomek

scaler = StandardScaler()
imputer = KNNImputer(n_neighbors=2, weights="uniform")


def preprocess(df, past_or_future, function_to_apply=None):
    if function_to_apply != None: df['consommation'] = function_to_apply(df.consommation)  # for lices
    if past_or_future == "passe":
        X = df[df.columns[:-1]]
        y = df.JourPointe
        # impute
        imputer.fit_transform(X, y)
        X = imputer.transform(X)

        # scale
        # scaler.fit_transform(X,y)
        # X = scaler.transform(X)

        print('np.all(np.isfinite(X)) :', np.all(np.isfinite(X)))
        # oversample then undersample(separe blue(0) from orange(1))
        X_os_us, y_os_us = OverSample(SMOTETomek, 'SMOTE + Tomek links', X, y, ratio='auto')

        # return to dataframe
        X_os_us, y_os_us = df_x_y(X_os_us, y_os_us, df.columns[:-1])

        # create balanced df
        df_balanced = pd.concat([X_os_us, y_os_us], axis=1, join="inner")
    else:
        df_balanced = df

    return df_balanced


from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=1000, learning_rate=0.05)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline


def create_pripeline(df_columns):
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('imputer', KNNImputer(n_neighbors=2, weights="uniform"))
    ])
    # Bundle preprocessing for numerical data
    numerical_cols = df_columns[:-1]  # All X columns are numerical

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
        ])
    ## Bundle preprocessing and modeling code in a pipeline
    my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('model', model)
                                  ])

    return my_pipeline


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc


def evaluate_model(my_pipeline, X_train, y_train, X_valid, y_valid):
    my_pipeline.fit(X_train, y_train)
    PredictedOutput = my_pipeline.predict(X_valid)
    test_score = my_pipeline.score(X_valid, y_valid)
    test_recall = recall_score(y_valid, PredictedOutput, pos_label=1)
    test_precesion = precision_score(y_valid, PredictedOutput, pos_label=1)
    test_f1_score = f1_score(y_valid, PredictedOutput, pos_label=1)
    fpr, tpr, thresholds = roc_curve(y_valid, PredictedOutput, pos_label=1)
    test_auc = auc(fpr, tpr)

    print("Test accuracy ", test_score)
    print("Test recall", test_recall)  # true positive/(true positive + false negative)
    print("Test precesion", test_precesion)  # true positive/(true positive + false positive)
    print("Test f1 score", test_f1_score)
    print("Test AUC", test_auc)
    print(confusion_matrix(y_true=y_valid, y_pred=PredictedOutput))


from sklearn.model_selection import train_test_split


def split_data(df):
    # Separate target (y) from predictors(X)
    X = df.drop(['JourPointe'], axis=1)
    y = df.JourPointe
    # Divide data into training and validation subsets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25)
    return X_train, X_valid, y_train, y_valid


def pipeline_trained(name_poste_source, function_to_apply):
    # import and balance data
    df = pd.read_excel(name_poste_source)
    df_balanced = preprocess(df, "passe", function_to_apply)

    # create pipeline
    my_pipeline = create_pripeline(df.columns)
    # evaluate model
    X_train_balanced, X_valid_balanced, y_train_balanced, y_valid_balanced = split_data(df_balanced)
    evaluate_model(my_pipeline, X_train_balanced, y_train_balanced, X_valid_balanced, y_valid_balanced)

    # train model on whole data
    X_balanced = df_balanced[df.columns[:-1]]
    y_balanced = df_balanced.JourPointe
    my_pipeline.fit(X_balanced, y_balanced)
    return my_pipeline


def predict_future(my_pipeline, name_poste_source, function_to_apply):
    df_future = pd.read_excel(url_future_complet + name_poste_source)
    print(df_future)
    df_future = preprocess(df_future, "future", function_to_apply)
    print(df_future)
    Predictions = my_pipeline.predict(df_future)
    print(Predictions)
    return Predictions

