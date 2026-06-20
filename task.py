"""
Housing Price Analysis — Internship Project Week 1
=====================================================
A single end-to-end script covering Tasks 1–5 from the original notebook:

  Task 1 — Load data, inspect shape/target/features, check missing values
  Task 2 — Clean data (drop NA/duplicates), encode categorical columns
  Task 3 — Train/test split
  Task 4 — Train & evaluate Linear Regression and Random Forest models
  Task 5 — Visualizations + written conclusions

Improvements over the original notebook version:
  * Runs top-to-bottom as a plain .py file (no notebook required)
  * Friendly error message if Housing.csv is missing
  * Metrics are saved to a CSV (model_metrics.csv) in addition to being printed
  * Adds a 4th chart: feature importance comparison (Linear coefficients vs
    Random Forest importances) — useful extra context not in the original
  * Adds a predicted-vs-actual scatter plot for both models (5th chart)
  * Cross-validation (5-fold) added for more robust performance estimates
  * Light docstrings/section banners for readability
  * Final written summary saved to summary.txt as well as printed to console

Usage:
    python housing_price_analysis.py
    (Place Housing.csv in the same folder, or pass a path as the first CLI arg)
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
pd.set_option("display.width", 120)


def banner(title: str) -> None:
    """Print a readable section header to the console."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Task 1 — Load & Inspect
# ---------------------------------------------------------------------------
def load_and_inspect(csv_path: str) -> pd.DataFrame:
    banner("TASK 1 — Load & Inspect Data")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Could not find '{csv_path}'.\n"
            f"Please place 'Housing.csv' in the same directory as this script, "
            f"or pass the correct path as a command-line argument, e.g.:\n"
            f"    python housing_price_analysis.py /path/to/Housing.csv"
        )

    df = pd.read_csv(csv_path)

    print("\nFirst 10 rows:")
    print(df.head(10))

    print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")

    print("\nTarget  : price")
    print("Features:", [c for c in df.columns if c != "price"])

    missing = df.isnull().sum().rename("Missing Values").to_frame()
    print("\nMissing values per column:")
    print(missing)

    return df


# ---------------------------------------------------------------------------
# Task 2 — Clean & Encode
# ---------------------------------------------------------------------------
def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    banner("TASK 2 — Clean & Encode Data")

    df = df.dropna().copy()

    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicate rows removed: {before - len(df)}")

    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    print(f"Categorical columns: {cat_cols}")

    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # Convert any boolean dummy columns to int (0/1) for cleanliness
    for c in df.columns:
        if df[c].dtype == bool:
            df[c] = df[c].astype(int)

    print(f"\nFinal shape after encoding: {df.shape}")
    print(df.head())

    return df


# ---------------------------------------------------------------------------
# Task 3 — Train/Test Split
# ---------------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    banner("TASK 3 — Train/Test Split")

    X = df.drop(columns=["price"]).astype(float)
    y = df["price"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train : {X_train.shape}  |  Test : {X_test.shape}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Task 4 — Train & Evaluate Models
# ---------------------------------------------------------------------------
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    """Fit a model, score it, and run 5-fold CV for a more robust R^2 estimate."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")

    print(f"\n{name}")
    print(f"  MAE        : {mae:,.0f}")
    print(f"  RMSE       : {rmse:,.0f}")
    print(f"  R^2 (test) : {r2:.4f}")
    print(f"  R^2 (5-fold CV on train): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return {
        "model": model,
        "y_pred": y_pred,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "cv_r2_mean": cv_scores.mean(),
        "cv_r2_std": cv_scores.std(),
    }


def train_and_evaluate(X_train, X_test, y_train, y_test):
    banner("TASK 4 — Train & Evaluate Models")

    lr = LinearRegression()
    lr_results = evaluate_model("Linear Regression", lr, X_train, X_test, y_train, y_test)

    rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
    rf_results = evaluate_model("Random Forest Regressor", rf, X_train, X_test, y_train, y_test)

    summary = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Random Forest"],
            "MAE": [f"{lr_results['mae']:,.0f}", f"{rf_results['mae']:,.0f}"],
            "RMSE": [f"{lr_results['rmse']:,.0f}", f"{rf_results['rmse']:,.0f}"],
            "R2 Score (test)": [round(lr_results["r2"], 4), round(rf_results["r2"], 4)],
            "R2 Score (5-fold CV)": [
                round(lr_results["cv_r2_mean"], 4),
                round(rf_results["cv_r2_mean"], 4),
            ],
        }
    ).set_index("Model")

    print("\nModel comparison summary:")
    print(summary)

    return lr_results, rf_results, summary


# ---------------------------------------------------------------------------
# Task 5 — Visualizations
# ---------------------------------------------------------------------------
def make_visualizations(df, y_test, lr_results, rf_results, X_train, output_dir="charts"):
    banner("TASK 5 — Visualizations")

    os.makedirs(output_dir, exist_ok=True)

    # Chart 1: Price Distribution
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df["price"], bins=40, color="steelblue", edgecolor="white")
    ax.set_title("Distribution of House Prices")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart1_price_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir}/chart1_price_distribution.png")

    # Chart 2: Correlation Heatmap
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=False, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart2_correlation_heatmap.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir}/chart2_correlation_heatmap.png")

    # Chart 3: Residual Distribution
    residuals_lr = y_test - lr_results["y_pred"]
    residuals_rf = y_test - rf_results["y_pred"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(residuals_lr, bins=40, color="steelblue", edgecolor="white", alpha=0.6, label="Linear Regression")
    ax.hist(residuals_rf, bins=40, color="darkorange", edgecolor="white", alpha=0.6, label="Random Forest")
    ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
    ax.set_title("Residual Distribution \u2014 Linear Regression vs Random Forest")
    ax.set_xlabel("Prediction Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart3_residual_distribution.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir}/chart3_residual_distribution.png")

    # Chart 4 (NEW): Predicted vs Actual scatter for both models
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, (name, results, color) in zip(
        axes,
        [
            ("Linear Regression", lr_results, "steelblue"),
            ("Random Forest", rf_results, "darkorange"),
        ],
    ):
        ax.scatter(y_test, results["y_pred"], alpha=0.5, color=color, edgecolor="white", s=40)
        lims = [min(y_test.min(), results["y_pred"].min()), max(y_test.max(), results["y_pred"].max())]
        ax.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_title(f"{name}: Predicted vs Actual")
        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart4_predicted_vs_actual.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir}/chart4_predicted_vs_actual.png")

    # Chart 5 (NEW): Feature importance comparison
    feature_names = X_train.columns
    lr_coefs = pd.Series(lr_results["model"].coef_, index=feature_names)
    # Normalize linear coefficients to comparable scale (abs value, then normalize to sum to 1)
    lr_importance = lr_coefs.abs() / lr_coefs.abs().sum()
    rf_importance = pd.Series(rf_results["model"].feature_importances_, index=feature_names)

    importance_df = pd.DataFrame(
        {"Linear Regression (|coef| normalized)": lr_importance, "Random Forest": rf_importance}
    ).sort_values("Random Forest", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 7))
    importance_df.plot(kind="barh", ax=ax, color=["steelblue", "darkorange"])
    ax.set_title("Feature Importance Comparison")
    ax.set_xlabel("Relative Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "chart5_feature_importance.png"), dpi=150)
    plt.close(fig)
    print(f"Saved {output_dir}/chart5_feature_importance.png")

    return importance_df


# ---------------------------------------------------------------------------
# Summary / Conclusions
# ---------------------------------------------------------------------------
def write_summary(summary_df, importance_df, output_path="summary.txt"):
    top_features = importance_df.sort_values("Random Forest", ascending=False).head(5)
    top_feature_names = ", ".join(top_features.index.tolist())

    text = f"""
HOUSING PRICE ANALYSIS — SUMMARY
=================================

Model Comparison
-----------------
{summary_df.to_string()}

Which features influence price most?
-------------------------------------
Based on Random Forest importances, the top features are: {top_feature_names}.
Both models broadly agree that area, bathrooms, air conditioning, and
furnishing status are the strongest drivers of price, with stories and
preferred area (location) also playing a meaningful role. Larger homes with
more bathrooms and AC command higher prices, while amenities like a
guestroom or hot water heating have a smaller effect.

How accurate was the model, in plain terms?
---------------------------------------------
Linear Regression and Random Forest were both evaluated using a held-out
test set and 5-fold cross-validation for a more robust estimate of
generalization performance. See the table above for exact MAE, RMSE, and R^2
values.

What surprised us?
-------------------
On this relatively small dataset (545 rows), the simpler Linear Regression
model can generalize comparably to or better than Random Forest, which
suggests price scales fairly linearly with the key features (especially
area). The added flexibility of Random Forest can introduce noise rather
than signal on smaller datasets like this one.

Recommendation for a real estate business
--------------------------------------------
Since area, bathroom count, and air conditioning are among the biggest price
levers, prioritize collecting and prominently displaying these details in
listings to capture buyer attention quickly and price homes more accurately.
"""

    with open(output_path, "w") as f:
        f.write(text)

    print(text)
    print(f"(Summary also saved to {output_path})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "Housing.csv"

    df_raw = load_and_inspect(csv_path)
    df_clean = clean_and_encode(df_raw)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    lr_results, rf_results, summary_df = train_and_evaluate(X_train, X_test, y_train, y_test)
    importance_df = make_visualizations(df_clean, y_test, lr_results, rf_results, X_train)

    # Save metrics to CSV
    summary_df.to_csv("model_metrics.csv")
    print("\nSaved model_metrics.csv")

    banner("FINAL SUMMARY")
    write_summary(summary_df, importance_df)


if __name__ == "__main__":
    main()
