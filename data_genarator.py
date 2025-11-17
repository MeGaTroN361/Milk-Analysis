# data_generator.py
"""
Generates a synthetic training dataset for the milk-analysis project.
Usage:
    python data_generator.py --rows 2000 --out training_dataset_2000.csv
"""
import argparse
import pandas as pd
import numpy as np

def generate_dataset(num_records=4000, dataset_csv="../training_dataset.csv", seed=42):
    health_states = {
        "healthy": {
            "temp": (37.5, 38.4),
            "ph": (6.5, 6.8),
            "turbidity": (0.1, 0.6),
            "conductivity": (4.0, 5.4),
            "heart_rate": (60, 80),
            "fat": (3.8, 4.8),
            "p": 0.60
        },
        "mastitis_risk": {
            "temp": (38.0, 39.2),
            "ph": (6.8, 7.2),
            "turbidity": (0.6, 1.6),
            "conductivity": (5.6, 7.0),
            "heart_rate": (70, 95),
            "fat": (2.4, 3.6),
            "p": 0.15
        },
        "ketosis_risk": {
            "temp": (36.8, 38.4),
            "ph": (6.3, 6.9),
            "turbidity": (0.1, 0.8),
            "conductivity": (4.8, 6.0),
            "heart_rate": (50, 80),
            "fat": (5.0, 7.0),
            "p": 0.10
        },
        "metritis_risk": {
            "temp": (38.2, 39.5),
            "ph": (6.6, 7.4),
            "turbidity": (0.6, 2.0),
            "conductivity": (5.5, 7.5),
            "heart_rate": (70, 110),
            "fat": (2.5, 4.0),
            "p": 0.075
        },
        "udder_infection": {
            "temp": (38.4, 39.8),
            "ph": (6.9, 7.6),
            "turbidity": (1.0, 3.0),
            "conductivity": (6.0, 9.0),
            "heart_rate": (80, 120),
            "fat": (2.0, 3.5),
            "p": 0.075
        }
    }

    states = list(health_states.keys())
    probabilities = np.array([health_states[s]['p'] for s in states])
    probabilities = probabilities / probabilities.sum()

    rng = np.random.default_rng(seed=seed)
    rows = []
    for _ in range(num_records):
        status = rng.choice(states, p=probabilities)
        params = health_states[status]
        row = {
            "temperature": round(rng.uniform(*params["temp"]), 2),
            "ph": round(rng.uniform(*params["ph"]), 2),
            "turbidity": round(rng.uniform(*params["turbidity"]), 3),
            "conductivity": round(rng.uniform(*params["conductivity"]), 2),
            "heart_rate": round(rng.uniform(*params["heart_rate"]), 1),
            "fat_content": round(rng.uniform(*params["fat"]), 2),
            "status": status
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(dataset_csv, index=False)
    print(f"Generated {len(df)} rows and saved to '{dataset_csv}'")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for milk analysis")
    parser.add_argument('--rows', type=int, default=4000, help='Number of rows to generate (default 4000)')
    parser.add_argument('--out', type=str, default='training_dataset.csv', help='Output CSV filename')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()

    if args.rows < 2000:
        print("Warning: recommended at least 2000 rows for stable training.")

    generate_dataset(num_records=args.rows, dataset_csv=args.out, seed=args.seed)
