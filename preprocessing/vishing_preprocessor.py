from typing import List, Dict
import pandas as pd
import os
from pathlib import Path

from core.azure_transcribe import AzureTranscriber
from core.config import Settings

settings = Settings()

class VishingPreprocessor:
    def __init__(
        self, 
        file_path
    ): 
        self.file_path = file_path
        self.transcriber = AzureTranscriber(settings.AZURE_AI_PROJECT_API_KEY, "westus", "en-US")

    def process(self):
        folder_path = Path(f"{self.file_path}")
        df = pd.DataFrame(columns=['id', 'file_name', 'transcript', 'ground_truth'])
        for file in list(folder_path.glob('*.mp3')):
            result = self.transcriber.transcribe(file)
            if "phish" in str(file):
                df.loc[len(df)] = [f'p{len(df) + 1}', str(file), result.text, 'phishing']
            else:
                df.loc[len(df)] = [f'r{len(df) + 1}', str(file), result.text, 'real']
        return df