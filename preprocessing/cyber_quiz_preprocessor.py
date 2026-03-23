from typing import List, Dict
import pandas as pd

class CyberQuizPreprocessor:
    def __init__(
        self, 
        file_path
    ): 
        self.file_path = file_path

    def process(self):
        try:
            df = pd.read_csv(self.file_path, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.file_path, encoding='latin1')
        # Keep all domains/question types for the cybersecurity quiz dataset.
        df = df.drop(['Title', 'Explanation', 'Status', 'Tags', 'System', 'CorrectLevel'], axis=1, errors='ignore')
        df = df.rename(columns={
            'QuestionType': 'question_type', 
            'Domain': 'domain', 
            'QuestionText': 'question', 
            'Difficulty': 'difficulty', 
            'Regulation': 'regulation',
            'OptionA': 'option_A', 
            'OptionB': 'option_B',
            'OptionC': 'option_C',  
            'OptionD': 'option_D', 
            'OptionE': 'option_E', 
            'OptionF': 'option_F', 
            'CorrectOption': 'ground_truth'
        })
        df["order"] = df.index
        return df