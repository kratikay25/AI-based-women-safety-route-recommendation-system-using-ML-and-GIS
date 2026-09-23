import os
import pandas as pd


class DatasetEDA:

    def __init__(self, path):

        # Load dataset
        self.df = pd.read_csv(path)

        # Create output folders
        os.makedirs("outputs/reports", exist_ok=True)
        os.makedirs("outputs/plots", exist_ok=True)

    # --------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------
    def dataset_summary(self):

        print("\n" + "=" * 60)
        print("DATASET SHAPE")
        print(self.df.shape)

        print("\n" + "=" * 60)
        print("COLUMN NAMES")
        print(self.df.columns.tolist())

        print("\n" + "=" * 60)
        print("DATA TYPES")
        print(self.df.dtypes)

        print("\n" + "=" * 60)
        print("FIRST FIVE ROWS")
        print(self.df.head())

        print("\n" + "=" * 60)
        print("LAST FIVE ROWS")
        print(self.df.tail())

        print("\n" + "=" * 60)
        print("DESCRIPTIVE STATISTICS")

        stats = self.df.describe(include="all")

        print(stats)

        stats.to_csv(
            "outputs/reports/descriptive_statistics.csv",
            index=True
        )

    # --------------------------------------------------
    # MISSING VALUES
    # --------------------------------------------------
    def missing_values(self):

        print("\n" + "=" * 60)
        print("MISSING VALUES")

        missing = self.df.isnull().sum()

        print(missing)

        missing.to_csv(
            "outputs/reports/missing_values.csv",
            header=["Missing Values"]
        )

    # --------------------------------------------------
    # DUPLICATE ROWS
    # --------------------------------------------------
    def duplicate_rows(self):

        duplicates = self.df.duplicated().sum()

        print("\n" + "=" * 60)
        print("DUPLICATE ROWS")

        print(f"Total Duplicate Rows : {duplicates}")

    # --------------------------------------------------
    # UNIQUE VALUES
    # --------------------------------------------------
    def unique_values(self):

        print("\n" + "=" * 60)
        print("UNIQUE VALUES")

        for column in self.df.columns:

            print("\n" + "-" * 40)

            print(f"Column : {column}")

            unique_count = self.df[column].nunique()

            print(f"Unique Values : {unique_count}")

            # Print values only if there are few categories
            if unique_count <= 15:
                print(self.df[column].unique())

    # --------------------------------------------------
    # SAVE DATASET INFO
    # --------------------------------------------------
    def save_summary(self):

        summary = pd.DataFrame({

            "Column": self.df.columns,
            "Data Type": self.df.dtypes.astype(str),
            "Missing Values": self.df.isnull().sum().values,
            "Unique Values": self.df.nunique().values

        })

        summary.to_csv(
            "outputs/reports/dataset_summary.csv",
            index=False
        )

        print("\nDataset summary saved successfully.")


# ======================================================
# MAIN
# ======================================================

if __name__ == "__main__":

    eda = DatasetEDA("data/merged_dataset.csv")

    eda.dataset_summary()

    eda.missing_values()

    eda.duplicate_rows()

    eda.unique_values()

    eda.save_summary()