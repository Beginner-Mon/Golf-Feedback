import pandas as pd
import numpy as np
from paths import DATA_DIR
class DataProcessor:
    def __init__(self, file_path):
        """
        Initialize the DataProcessor with the path to the CSV file.
        
        :param file_path: str, path to the CSV file (e.g., 'CaddieSet.csv')
        """
        self.df = pd.read_csv(file_path)
        self.faceon = None
        self.dtl = None

    def split_data(self):
        """Split the data into FACEON and DTL views."""
        self.faceon = self.df[self.df['View'].str.upper() == 'FACEON'].copy()
        self.dtl = self.df[self.df['View'].str.upper() == 'DTL'].copy()

    def remove_empty_columns(self):
        """Remove columns that are entirely empty."""
        if self.faceon is not None:
            self.faceon.dropna(axis=1, how='all', inplace=True)
        if self.dtl is not None:
            self.dtl.dropna(axis=1, how='all', inplace=True)

    def fill_missing_with_mode(self):
        """Fill missing values using the mode of each column."""
        def impute_group(group):
            for col in group.columns:
                if col == 'View':
                    continue
                
                # Try numeric first
                numeric_series = pd.to_numeric(group[col], errors='coerce')
                numeric_series.replace([pd.NA, pd.NaT, float('inf'), float('-inf')], pd.NA, inplace=True)
                
                if numeric_series.notnull().any():  # Numeric column
                    mode_val = numeric_series.mode()
                    fill_value = mode_val[0] if not mode_val.empty else 0
                    group[col] = numeric_series.fillna(fill_value)
                else:  # Non-numeric column
                    mode_val = group[col].mode()
                    fill_value = mode_val[0] if not mode_val.empty else ''
                    group[col] = group[col].fillna(fill_value)
            return group

        if self.faceon is not None:
            self.faceon = impute_group(self.faceon)
        if self.dtl is not None:
            self.dtl = impute_group(self.dtl)

    def create_binary_targets(self):
        """
        Create binary columns for DirectionAngle and SpinAxis:
        - DirectionAngle_binary: 1 if within ±6°, else 0
        - SpinAxis_binary: 1 if within ±10°, else 0
        """
        def add_binary_cols(df):
            if 'DirectionAngle' in df.columns:
                df['DirectionAngle_binary'] = np.where(df['DirectionAngle'].between(-6, 6), 1, 0)
            if 'SpinAxis' in df.columns:
                df['SpinAxis_binary'] = np.where(df['SpinAxis'].between(-10, 10), 1, 0)
            return df

        if self.faceon is not None:
            self.faceon = add_binary_cols(self.faceon)
        if self.dtl is not None:
            self.dtl = add_binary_cols(self.dtl)

    def process_all(self):
        """
        Run full pipeline:
        1. Split data
        2. Remove empty columns
        3. Fill missing values
        4. Create binary classification targets
        """
        self.split_data()
        self.remove_empty_columns()
        self.fill_missing_with_mode()
        self.create_binary_targets()


# Example usage
if __name__ == "__main__":
    processor = DataProcessor(DATA_DIR / "CaddieSet.csv")
    processor.process_all()

    # Save cleaned datasets
    processor.faceon.to_csv(DATA_DIR / "faceon_cleaned.csv", index=False)
    processor.dtl.to_csv(DATA_DIR / "dtl_cleaned.csv", index=False)

    print(f"FACEON data: {processor.faceon.shape[0]} rows, {processor.faceon.shape[1]} columns")
    print(f"DTL data: {processor.dtl.shape[0]} rows, {processor.dtl.shape[1]} columns")
    print("Missing values after processing:", processor.faceon.isnull().sum().sum() + processor.dtl.isnull().sum().sum())
