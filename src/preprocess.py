import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(path="data/synthetic_data.csv"):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Feature & Target
    X = df.drop(columns=['risk_label'])
    y = df['risk_label']

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.select_dtypes(include=['float64', 'int64']))

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test