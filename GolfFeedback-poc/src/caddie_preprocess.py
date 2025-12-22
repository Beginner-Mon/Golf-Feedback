import pandas as pd

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
        """
        Split the data into two DataFrames based on the 'View' column:
        - self.faceon: Rows where View is 'FACEON'
        - self.dtl: Rows where View is 'DTL'
        """
        self.faceon = self.df[self.df['View'].str.upper() == 'FACEON'].copy()
        self.dtl = self.df[self.df['View'].str.upper() == 'DTL'].copy()

    def remove_empty_columns(self):
        """
        Remove columns that are entirely empty (all NaN) from each split DataFrame.
        """
        if self.faceon is not None:
            self.faceon.dropna(axis=1, how='all', inplace=True)
        if self.dtl is not None:
            self.dtl.dropna(axis=1, how='all', inplace=True)

    def fill_missing_with_mode(self):
        """
        Fill missing values (NaN) or invalid non-numeric entries in each column 
        using the most frequent valid value (mode) within each split group.
        
        - For numeric columns (even if containing strings like '#NAME?'), 
          coerce to numeric and impute with the mode of valid numbers.
        - For non-numeric columns, impute with the most frequent string value.
        - If no mode exists, falls back to a reasonable default (0 for numeric, '' for object).
        """
        def impute_group(group):
            for col in group.columns:
                if col == 'View':  # Skip View column
                    continue
                
                # Handle numeric columns with possible bad strings
                numeric_series = (
                    pd.to_numeric(group[col], errors='coerce')
                    .replace([pd.NA, pd.NaT, float('inf'), float('-inf')], pd.NA)
                )

                
                if numeric_series.notnull().any():  # It's a numeric column
                    mode_val = numeric_series.mode()
                    fill_value = mode_val[0] if not mode_val.empty else 0
                    group[col] = numeric_series.fillna(fill_value)
                else:
                    # Purely categorical/string column
                    mode_val = group[col].mode()
                    fill_value = mode_val[0] if not mode_val.empty else ''
                    group[col] = group[col].fillna(fill_value)
            return group

        if self.faceon is not None:
            self.faceon = impute_group(self.faceon)
        if self.dtl is not None:
            self.dtl = impute_group(self.dtl)

    def process_all(self):
        """
        Convenience method to run the full pipeline: split -> remove empty columns -> fill missing values.
        """
        self.split_data()
        self.remove_empty_columns()
        self.fill_missing_with_mode()

# Example usage
if __name__ == "__main__":
    processor = DataProcessor('../data/CaddieSet.csv')
    processor.process_all()
    
    # Save the cleaned datasets if desired
    processor.faceon.to_csv('faceon_cleaned.csv', index=False)
    processor.dtl.to_csv('dtl_cleaned.csv', index=False)
    
    print(f"FACEON data: {processor.faceon.shape[0]} rows, {processor.faceon.shape[1]} columns")
    print(f"DTL data: {processor.dtl.shape[0]} rows, {processor.dtl.shape[1]} columns")
    print("Missing values after processing:", processor.faceon.isnull().sum().sum() + processor.dtl.isnull().sum().sum())