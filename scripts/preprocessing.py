import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.impute import SimpleImputer

def preprocessing(file_path='/Users/woojinheo/Desktop/github/SHAP/data/train.csv', test_size=0.3):
    '''
    # train_data
    df_train, X_train, X_valid, y_train, y_valid = preprocessing(test_size=0.3, file_path='/Users/woojinheo/Desktop/github/SHAP/data/train.csv')

    # test_data
    df_test, X_test, _, y_test, _ = preprocessing(test_size=0, file_path='/Users/woojinheo/Desktop/github/SHAP/data/test.csv')
    '''
    df = pd.read_csv(file_path, index_col=0)

    # Target column transformation
    df['satisfaction'] = df['satisfaction'].map({"neutral or dissatisfied": 0, "satisfied": 1})


    # Prepare features and target
    X = df.drop(columns=["satisfaction", "Unnamed: 0", "id"], errors='ignore')
    y = df["satisfaction"]

    # Encode categorical features
    categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
    X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

    # Handle missing values using imputation
    imputer = SimpleImputer(strategy="median")
    X = imputer.fit_transform(X)

    # Split data into training and test sets
    if test_size==0:
        X_train = X
        y_train = y
        X_valid, y_valid = None, None
    else:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    return df, X_train, X_valid, y_train, y_valid
