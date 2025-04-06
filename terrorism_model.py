import pandas as pd
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score
import folium
import os
import joblib

class TerrorismModel:
    def __init__(self):
        self.african_countries = [
            "Algeria", "Angola", "Benin", "Botswana", "Burkina Faso", "Burundi", "Cabo Verde", 
            "Cameroon", "Central African Republic", "Chad", "Comoros", "Democratic Republic of the Congo", 
            "Djibouti", "Egypt", "Equatorial Guinea", "Eritrea", "Eswatini", "Ethiopia", "Gabon", 
            "Gambia", "Ghana", "Guinea", "Guinea-Bissau", "Ivory Coast", "Kenya", "Lesotho", "Liberia", 
            "Libya", "Madagascar", "Malawi", "Mali", "Mauritania", "Mauritius", "Morocco", "Mozambique", 
            "Namibia", "Niger", "Nigeria", "Republic of the Congo", "Rwanda", "São Tomé and Príncipe", 
            "Senegal", "Seychelles", "Sierra Leone", "Somalia", "South Africa", "South Sudan", "Sudan", 
            "Tanzania", "Togo", "Tunisia", "Uganda", "Zambia", "Zimbabwe"
        ]
        self.selected_columns = [
            'iyear', 'imonth', 'iday', 'latitude', 'longitude', 'country',
            'country_txt', 'region', 'region_txt', 'provstate', 'city', 'multiple',
            'success', 'suicide', 'targtype1','targtype1_txt', 'weaptype1',
            'weaptype1_txt', 'gname', 'attacktype1_txt', 'attacktype1', 'target1',
            'natlty1', 'natlty1_txt', 'nkill', 'property', 'dbsource'
        ]
        self.label_mapping = {
            1: 'Assassination',
            2: 'Armed Assault',
            3: 'Bombing/Explosion',
            6: 'Kidnapping',
            7: 'Facility Attack',
            9: 'Unknown'
        }

    def preprocess_data(self, df):
        # Filter data for African countries and select the required columns
        africa_df = df[df['country_txt'].isin(self.african_countries)]
        attack_type_df = africa_df[self.selected_columns]
        return attack_type_df

    def fill_null_values(self, df):
        # Fill null values separately for latitude, longitude, and number of kills
        avg_lat_long = df.groupby('country_txt')[['latitude', 'longitude']].mean().reset_index()
        df = df.merge(avg_lat_long, on='country_txt', suffixes=('', '_avg'))
        df['latitude'].fillna(df['latitude_avg'], inplace=True)
        df['longitude'].fillna(df['longitude_avg'], inplace=True)
        df.drop(columns=['latitude_avg', 'longitude_avg'], inplace=True)

        avg_nkill = df.groupby('country_txt')['nkill'].mean().reset_index()
        df = df.merge(avg_nkill, on='country_txt', suffixes=('', '_avg'))
        df['nkill'].fillna(df['nkill_avg'], inplace=True)
        df.drop(columns=['nkill_avg'], inplace=True)

        # Drop rows with missing values in important categorical columns
        df.dropna(subset=['provstate', 'city', 'target1'], inplace=True)
        df['natlty1'].fillna('Unknown', inplace=True)
        df['natlty1_txt'].fillna('Unknown', inplace=True)
        return df

    def encode_categorical_data(self, df):
        # Encode high-cardinality categorical columns using category codes
        categorical_cols_high_cardinality = ['provstate', 'city', 'gname', 'target1', 'natlty1', 'natlty1_txt']
        for col in categorical_cols_high_cardinality:
            df[col + '_encoded'] = df[col].astype('category').cat.codes
        df.drop(columns=categorical_cols_high_cardinality, inplace=True)

        # Drop other irrelevant object columns
        object_columns_to_drop = ['country_txt', 'region_txt', 'targtype1_txt', 'weaptype1_txt', 'attacktype1_txt']
        df.drop(columns=object_columns_to_drop, inplace=True)

        # One-hot encode the 'dbsource' column
        one_hot_encoded_dbsource = pd.get_dummies(df['dbsource'], prefix='dbsource')
        df = pd.concat([df, one_hot_encoded_dbsource], axis=1)
        df.drop(columns=['dbsource'], inplace=True)
        df.columns = df.columns.str.replace('dbsource_', '')
        return df

    def train_model(self, X_train, y_train):
        # Train a LightGBM model with specified hyperparameters
        lgbm = LGBMClassifier(learning_rate=0.1, max_depth=15, num_leaves=40, verbose=-1)
        lgbm.fit(X_train, y_train)
        return lgbm

    def deploy_model(self, df, training=True):
        # Preprocess the data
        df = self.preprocess_data(df)

        # Temporal split to avoid data leakage: Training data (1970-2015) and Test data (2016-2020)
        train_df = df[df['iyear'] <= 2015]
        test_df = df[df['iyear'] > 2015]

        # Fill missing values separately for training and test data
        train_df = self.fill_null_values(train_df)
        test_df = self.fill_null_values(test_df)

        # Encode categorical data separately for training and test data
        train_df = self.encode_categorical_data(train_df)
        test_df = self.encode_categorical_data(test_df)

        # Remove certain classes from the dataset
        classes_to_remove = [4, 5, 8]
        train_df = train_df[~train_df['attacktype1'].isin(classes_to_remove)]
        test_df = test_df[~test_df['attacktype1'].isin(classes_to_remove)]

        # Split data into features (X) and target (y)
        X_train = train_df.drop(columns=['attacktype1'])
        y_train = train_df['attacktype1']

        X_test = test_df.drop(columns=['attacktype1'])
        y_test = test_df['attacktype1']

        if training or not os.path.exists('terrorism_model.pkl'):
            # Train the model if training mode is enabled or model does not exist
            trained_model = self.train_model(X_train, y_train)
            joblib.dump(trained_model, 'terrorism_model.pkl')
        else:
            # Load the model if it's already trained
            trained_model = joblib.load('terrorism_model.pkl')

        # Predict on the test set
        y_pred = trained_model.predict(X_test)

        # Generate a classification report with text labels
        target_names = [self.label_mapping[i] for i in sorted(self.label_mapping.keys())]
        report = classification_report(y_test, y_pred, target_names=target_names)
        accuracy = accuracy_score(y_test, y_pred)

        # Visualize the predictions on the map
        X_test['predicted_attacktype1'] = y_pred
        map_html = self.visualize_predictions(X_test)

        return trained_model, report, accuracy, map_html

    def visualize_predictions(self, df):
        # Create a map centered on Africa
        africa_map = folium.Map(location=[1.650801, 10.267895], zoom_start=4)

        # Mapping of attack types to labels
        attack_type_labels = {
            1: 'Assassination',
            2: 'Armed Assault',
            3: 'Bombing/Explosion',
            6: 'Kidnapping',
            7: 'Facility Attack',
            9: 'Unknown'
        }
        
        # Add markers to the map for each attack
        for _, row in df.iterrows():
            folium.Marker(
                location=[row['latitude'], row['longitude']],
                popup=f"Attack Type: {attack_type_labels.get(row['predicted_attacktype1'], 'Other')}",
                icon=folium.Icon(color="red", icon="info-sign")
            ).add_to(africa_map)

        # Return the HTML representation of the map
        return africa_map._repr_html_()
