import sys
from pathlib import Path

# Ensure project root is on sys.path when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization.data_lockdown_visualizer import DataLockdownVisualizer
import pandas as pd

def main():
        df = pd.read_csv('results/data_lockdown_20260306_092121/detailed_results.csv')
        viz = DataLockdownVisualizer(df, Path('results/data_lockdown_20260306_092121'))
        # viz.plot_overall_accuracy()
        viz.generate_all()

if __name__ == "__main__":
    main()