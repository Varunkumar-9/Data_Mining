  
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def process_dataset(file_path):
    # Load the dataset
    df = pd.read_csv(file_path, header=None)

    # Separating the target column and features
    target = df.iloc[:, 0].map({'p': 1, 'e': 0})
    features = df.iloc[:, 1:]

    # One-hot encode the features
    encoder = OneHotEncoder(sparse=False)
    features_encoded = encoder.fit_transform(features)

    # Convert features to binary by converting to int
    features_encoded = features_encoded.astype(int)

    # Combine the class label with the one-hot encoded features
    final_data = pd.concat([target, pd.DataFrame(features_encoded)], axis=1)

    # Split the dataset into training (70%), validation (15%), and testing (15%)
    train, temp = train_test_split(final_data, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Save the datasets
    train.to_csv('training.txt', index=False, header=False)
    val.to_csv('val.txt', index=False, header=False)
    test.to_csv('testing.txt', index=False, header=False)



# Replace 'path_to_file.csv' with the path to your dataset file
process_dataset('C:/Users/Desktop/VarunBejjenki/agaricus-lepiota.data')