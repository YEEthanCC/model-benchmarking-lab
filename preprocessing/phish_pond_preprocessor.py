from typing import List, Dict
import pandas as pd
import os
from pathlib import Path

from core.azure_transcribe import AzureTranscriber
from core.config import Settings

settings = Settings()

class PhishPondPreprocessor:
    def __init__(
        self, 
        file_path
    ): 
        self.file_path = file_path
        self.transcriber = AzureTranscriber(settings.AZURE_AI_PROJECT_API_KEY, "westus", "en-US")

    def process(self):
        phishing_folder_path = Path(f"{self.file_path}/fish")
        real_folder_path = Path(f"{self.file_path}/Real")
        df = pd.DataFrame(columns=['id', 'transcript', 'details', 'ground_truth'])
        for file in list(phishing_folder_path.glob('*.mp3')):
            result = self.transcriber.transcribe(file)
            df.loc[len(df)] = [f"p{len(df) + 1}", result.text, result.details, 'phishing']
        for file in list(real_folder_path.glob('*.mp3')):
            result = self.transcriber.transcribe(file)
            df.loc[len(df)] = [f"r{len(df) + 1 - len(list(phishing_folder_path.glob('*.mp3')))}", result.text, result.details, 'real']
        return df