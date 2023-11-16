# Import the necessary libraries
import warnings
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
warnings.simplefilter("ignore")

# Step 1: Data Preparation
class DataPreparation:
    def __init__(self, csv_file):
        self.data, self.symptoms = self.load_data(csv_file)

    def load_data(self, csv_file):
        # Load the data from the CSV file
        df_comb = pd.read_csv(csv_file)

        # Split the data into features and labels
        X = df_comb.iloc[:, 1:]
        Y = df_comb.iloc[:, 0:1]

        # Convert the dataframes to lists for further processing
        features = X.values.tolist()
        labels = Y.values.tolist()

        # Get the list of all symptoms
        symptoms = df_comb.columns[1:].tolist()

        return list(zip(labels, features)), symptoms

    def split_data(self):
        # Split data into features (symptoms) and labels (conditions)
        labels, features = zip(*self.data)

        # Split data into training and testing sets
        features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2, random_state=42)

        return features_train, features_test, labels_train, labels_test

# Step 2: Feature Extraction
class FeatureExtraction:
    def __init__(self, symptoms):
        self.symptoms = symptoms

    def transform_features(self, user_input):
        # Check if all symptoms in user_input exist in self.symptoms
        for symptom in user_input:
            if symptom not in self.symptoms:
                raise ValueError(f"Invalid symptom: {symptom}. Please go back and retype your symptoms.")
        user_input_transformed = [1 if symptom in user_input else 0 for symptom in self.symptoms]
        return [user_input_transformed]

# Step 3: Model Training (K-Nearest Neighbors)
class KNNModel:
    def __init__(self):
        self.model = KNeighborsClassifier(n_jobs=4, n_neighbors=17, weights='uniform')

    def train_model(self, features_train, labels_train):
        self.model.fit(features_train, labels_train)

    def predict(self, features_test):
        return self.model.predict(features_test)

# Step 4: Main Program
def prediction (input_text):
    # Data Preparation
    data_preparation = DataPreparation('dis_sym_dataset_comb.csv')
    features_train, features_test, labels_train, labels_test = data_preparation.split_data()

    # Model Training
    knn_model = KNNModel()
    knn_model.train_model(features_train, labels_train)

    # Feature Extraction
    feature_extraction = FeatureExtraction(data_preparation.symptoms)

    # Take user input for symptoms
    user_input = input_text.split(", ")
    try:
        user_input_transformed = feature_extraction.transform_features(user_input)
    except ValueError as e:
        return str(e)

    # Predict the top 5 conditions for user input
    predicted_conditions = knn_model.model.classes_
    predicted_probabilities = knn_model.model.predict_proba(user_input_transformed)[0]
    sorted_predictions = [(condition, prob) for condition, prob in zip(predicted_conditions, predicted_probabilities)]
    sorted_predictions = sorted(sorted_predictions, key=lambda x: x[1], reverse=True)

    output = "Top 5 Predicted Conditions:\r\n"
    for i, (condition, prob) in enumerate(sorted_predictions[:5], 1):
        output += f"{i}. Name of illness: {condition}\n Prediction Accuracy: {prob*100}%\r\n\n"

    return(output)
