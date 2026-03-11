import sys
from pathlib import Path

# Ensure project root is on sys.path when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization.phish_pond_visualizer import PhishPondVisualizer
import pandas as pd

def main():
        df = pd.read_csv('results/phish_pond_20260310_121717/detailed_results.csv')
        viz = PhishPondVisualizer(df, Path('results/phish_pond_20260310_121717'))
        # viz.plot_overall_accuracy()
        viz.generate_all()

if __name__ == "__main__":
    main()