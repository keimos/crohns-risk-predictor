import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def create_preprocessor():
    """Creates a scikit-learn preprocessor pipeline."""
    
    # Define features
    numeric_features = ['age', 'bmi', 'crp', 'fecal_calprotectin', 'wbc', 'diet_score', 'stress_level']
    categorical_features = ['sex', 'family_crohns', 'smoking_status', 'nod2_mutation', 'atg16l1_mutation']

    # Numeric pipeline: Impute missing (if any) -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Impute -> OneHot
    # Note: Many of these are binary 0/1, but treating as categorical is safer for future proofing 
    # or if we want to handle unknown categories gracefully.
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

def load_data(path="data/synthetic_data.csv"):
    """Loads data and separates features and target."""
    df = pd.read_csv(path)
    X = df.drop(columns=['risk_label'])
    y = df['risk_label']
    return X, y