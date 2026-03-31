from typing import List, Dict
import pandas as pd
import base64

class CyberQuizPreprocessor:
    def __init__(
        self, 
        file_path
    ): 
        self.file_path = file_path

    def load_file(self):
        try:
            df = pd.read_csv(self.file_path + '/data.csv', encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.file_path + '/data.csv', encoding='latin1')
        
        df['qid'] = df.index
        
        return df

    def preprocess_image_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the 'image' column in the dataframe:
        - Replace 'NA' with None (null).
        - Split comma-separated paths in the 'image' column.
        - Concatenate each path with self.file_path and store them as a list.

        Args:
            df (pd.DataFrame): The input dataframe with an 'image' column.

        Returns:
            pd.DataFrame: The dataframe with the processed 'image' column.
        """
        if 'image' in df.columns:
            def process_image_value(value):
                if pd.isna(value):  # Check for NaN values
                    return None
                paths = value.split(',')
                return [f"{self.file_path}/{path.strip()}" for path in paths]

            df['image'] = df['image'].apply(process_image_value)
        return df

    def preprocess_na_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess specific columns ('context', 'e', 'f') in the dataframe:
        - Replace 'NA' with None (null) for these columns.

        Args:
            df (pd.DataFrame): The input dataframe with the specified columns.

        Returns:
            pd.DataFrame: The dataframe with 'NA' values replaced by None in the specified columns.
        """
        columns_to_process = ['context', 'e', 'f']
        for column in columns_to_process:
            if column in df.columns:
                df[column] = df[column].replace('NA', None)
        return df

    def encode_image_paths(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert lists of image paths in the 'image' column to lists of base64 encoded strings.

        Args:
            df (pd.DataFrame): The input dataframe with an 'image' column containing lists of paths.

        Returns:
            pd.DataFrame: The dataframe with the 'image' column updated to contain base64 encoded strings.
        """
        if 'image' in df.columns:
            def encode_paths(paths):
                if paths is None:
                    return None
                return [base64.b64encode(path.encode('utf-8')).decode('utf-8') for path in paths]

            df['image'] = df['image'].apply(encode_paths)
        return df

    def run(self) -> pd.DataFrame:
        """
        Perform the full preprocessing pipeline:
        1. Load the data.
        2. Preprocess the 'image' column.
        3. Preprocess 'context', 'e', and 'f' columns.
        4. Encode image paths to base64 strings.

        Returns:
            pd.DataFrame: The fully preprocessed dataframe.
        """
        # Step 1: Load the data
        df = self.load_file()

        # Step 2: Preprocess the 'image' column
        df = self.preprocess_image_column(df)

        # Step 3: Preprocess 'context', 'e', and 'f' columns
        df = self.preprocess_na_columns(df)

        # Step 4: Encode image paths to base64 strings
        df = self.encode_image_paths(df)

        return df