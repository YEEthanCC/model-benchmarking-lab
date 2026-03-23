import sys
from pathlib import Path

# Ensure project root is on sys.path when running this script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualization.cyber_quiz_visualizer import CyberQuizVisualizer
import pandas as pd

def main():
        df = pd.read_csv('results/cyber_quiz_20260323_124825/detailed_results.csv')
        viz = CyberQuizVisualizer(df, Path('results/cyber_quiz_20260323_124825'))
        viz.generate_all()

if __name__ == "__main__":
    main()