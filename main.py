"""Run experiments comparing agents across models, tools, and datasets."""
import argparse
import itertools
import json
import os
import time
from pathlib import Path
import asyncio

from orchestration.data_lockdown_pipelines import DataLockdownPipeline
from orchestration.phish_pond_pipeline import PhishPondPipeline
from orchestration.cyber_quiz_pipelines import CyberQuizPipeline
from core.azure_client import  get_azure_client


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    file_path = args.file_path
    mode = args.mode
    models = args.models

    if not os.path.exists(file_path):
        raise Exception("Invalid file path")
    if not mode in ["data-lockdown", "phish-pond", "cyber-quiz"]:
        raise Exception("Unsupported mode")
    
    client = get_azure_client()
    
    if mode == "data-lockdown":
        pipeline = DataLockdownPipeline(client, models, file_path)
    elif mode == "phish-pond":
        pipeline = PhishPondPipeline(client, models, file_path)
    elif mode == "cyber-quiz":
        pipeline = CyberQuizPipeline(client, models, file_path)

    pipeline.execute()



if __name__ == "__main__":
    # asyncio.run(main())
    main()
