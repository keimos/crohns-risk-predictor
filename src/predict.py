import joblib
import pandas as pd
import os

class RiskPredictor:
    def __init__(self, model_path='models/crohns_model.joblib'):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please run train.py first.")
        self.model = joblib.load(model_path)

    def predict(self, patient_data):
        """
        Predicts the risk of Crohn's disease for a given patient.
        
        Args:
            patient_data (dict): Dictionary containing patient features.
            
        Returns:
            dict: Prediction result containing 'risk_label' (0 or 1) and 'probability'.
        """
        # Convert dict to DataFrame
        df = pd.DataFrame([patient_data])
        
        # Predict
        prediction = self.model.predict(df)[0]
        probability = self.model.predict_proba(df)[0][1]
        
        return {
            'risk_label': int(prediction),
            'probability': float(probability)
        }

if __name__ == "__main__":
    # Example usage
    predictor = RiskPredictor()
    sample_patient = {
        'age': 30,
        'sex': 1, # Male
        'bmi': 22.5,
        'family_crohns': 1,
        'crp': 5.0,
        'fecal_calprotectin': 150.0,
        'wbc': 7.0,
        'smoking_status': 0,
        'diet_score': 4,
        'stress_level': 6,
        'nod2_mutation': 0,
        'atg16l1_mutation': 0
    }
    
    result = predictor.predict(sample_patient)
    print("Prediction Result:", result)
