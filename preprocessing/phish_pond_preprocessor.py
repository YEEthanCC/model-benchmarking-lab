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
        files = list(phishing_folder_path.glob('*.mp3')) + list(real_folder_path.glob('*.mp3'))
        df = pd.DataFrame(columns=['transcript', 'details'])
        for file in files:
            result =self.transcriber.transcribe(file)
            df.loc[len(df)] = [result.text, result.details]
        return df