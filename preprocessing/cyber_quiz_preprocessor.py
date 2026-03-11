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
        df = df[df['Domain'].isin(['Handling', 'Privacy'])]
        df = df.drop(['QuestionType', 'Title', 'Explanation', 'Status', 'Tags', 'System', 'CorrectLevel'], axis=1)
        df = df.rename(columns={
            'Domain': 'domain', 
            'ContextText': 'question', 
            'Difficulty': 'difficulty', 
            'QuestionId': 'id', 
            'Regulation': 'regulation', 
            'OptionA': 'option_A', 
            'OptionB': 'option_B',
            'OptionC': 'option_C',  
            'OptionD': 'option_D', 
            'CorrectOption': 'ground_truth'
        })
        return df