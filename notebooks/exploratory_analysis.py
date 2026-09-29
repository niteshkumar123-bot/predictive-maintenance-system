import matplotlib.pyplot as plt
import pandas as pd

from src.utils import FIGURES_DIR, logger
from src.data_preprocessing import load_or_generate_data
from src.feature_engineering import engineer_features


def run_eda():
    """Generate lightweight EDA figures using matplotlib only."""
    logger.info("Running automated Exploratory Data Analysis (EDA)...")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df = engineer_features(load_or_generate_data())

    # 1. Failure class distribution
    counts = df["failure"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar([str(i) for i in counts.index], counts.values)
    plt.title("Failure Class Distribution")
    plt.xlabel("Failure")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_distribution.png", bbox_inches="tight")
    plt.close()

    # 2. Correlation heatmap
    numeric_df = df.select_dtypes(include=["number"])
    corr = numeric_df.corr()
    plt.figure(figsize=(12, 10))
    plt.imshow(corr, aspect="auto")
    plt.colorbar(label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=7)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # 3. Failure vs temperature
    plt.figure(figsize=(6, 4))
    for label in sorted(df["failure"].unique()):
        values = df.loc[df["failure"] == label, "temperature"]
        plt.boxplot(values, positions=[label], widths=0.5)
    plt.xticks(sorted(df["failure"].unique()), [str(x) for x in sorted(df["failure"].unique())])
    plt.xlabel("Failure")
    plt.ylabel("Temperature")
    plt.title("Failure vs Temperature")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_vs_temperature.png", bbox_inches="tight")
    plt.close()

    # 4. Failure vs vibration
    plt.figure(figsize=(6, 4))
    for label in sorted(df["failure"].unique()):
        values = df.loc[df["failure"] == label, "vibration"]
        plt.boxplot(values, positions=[label], widths=0.5)
    plt.xticks(sorted(df["failure"].unique()), [str(x) for x in sorted(df["failure"].unique())])
    plt.xlabel("Failure")
    plt.ylabel("Vibration")
    plt.title("Failure vs Vibration")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_vs_vibration.png", bbox_inches="tight")
    plt.close()

    # 5. Operating hours by failure class
    plt.figure(figsize=(8, 5))
    for label in sorted(df["failure"].unique()):
        values = df.loc[df["failure"] == label, "operating_hours"]
        plt.hist(values, bins=30, alpha=0.55, label=f"Failure={label}")
    plt.xlabel("Operating Hours")
    plt.ylabel("Count")
    plt.title("Operating Hours Distribution by Failure Status")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "failure_vs_operating_hours.png", bbox_inches="tight")
    plt.close()

    logger.info(f"EDA plots successfully saved to {FIGURES_DIR}")


if __name__ == "__main__":
    run_eda()
