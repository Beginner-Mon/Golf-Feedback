import pandas as pd

class DataPreprocessor:
    def __init__(self, filepath):
        self.filepath = filepath
        self.data = None

    def load_data(self):
        self.data = pd.read_csv(self.filepath)

    def clean_data(self):
        # Example cleaning: drop rows with missing values
        self.data.dropna(inplace=True)

    def normalize_data(self):
        # Example normalization: min-max scaling
        numeric_cols = self.data.select_dtypes(include=['float64', 'int64']).columns
        self.data[numeric_cols] = (self.data[numeric_cols] - self.data[numeric_cols].min()) / (self.data[numeric_cols].max() - self.data[numeric_cols].min())

    def preprocess(self):
        self.load_data()
        self.clean_data()
        self.normalize_data()
        return self.data

if __name__ == "__main__":
    preprocessor = DataPreprocessor("../data/CaddieSet.csv")
    preprocessor.load_data()
    print(preprocessor.data.head())