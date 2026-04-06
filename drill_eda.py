"""Core Skills Drill — Descriptive Analytics

Compute summary statistics, plot distributions, and create a correlation
heatmap for the sample sales dataset.

Usage:
    python drill_eda.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_summary(df):
    """Compute summary statistics for all numeric columns.

    Args:
        df: pandas DataFrame with at least some numeric columns

    Returns:
        DataFrame containing count, mean, median, std, min, max
        for each numeric column. Save the result to output/summary.csv.
    """
    summary = df.select_dtypes(include="number").agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    summary.to_csv("output/summary.csv")
    return summary


def plot_distributions(df, columns, output_path):
    """Create a 2x2 subplot figure with histograms for the specified columns.

    Args:
        df: pandas DataFrame
        columns: list of 4 column names to plot (use numeric columns)
        output_path: file path to save the figure (e.g., 'output/distributions.png')

    Returns:
        None — saves the figure to output_path
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, col in zip(axes.flatten(), columns):
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_correlation(df, output_path):
    """Compute Pearson correlation matrix and visualize as a heatmap.

    Args:
        df: pandas DataFrame with numeric columns
        output_path: file path to save the figure (e.g., 'output/correlation.png')

    Returns:
        None — saves the figure to output_path
    """
    corr = df.select_dtypes(include="number").corr(method="pearson")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main():
    """Load data, compute summary, and generate all plots."""
    os.makedirs("output", exist_ok=True)

    df = pd.read_csv("data/sample_sales.csv")
    compute_summary(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()[:4]
    plot_distributions(df, numeric_cols, "output/distributions.png")
    plot_correlation(df, "output/correlation.png")


if __name__ == "__main__":
    main()
