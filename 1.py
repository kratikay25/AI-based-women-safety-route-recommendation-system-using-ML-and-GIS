import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class EDA:

    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

        os.makedirs("outputs/plots", exist_ok=True)
        os.makedirs("outputs/reports", exist_ok=True)

    def basic_info(self):
        print("=" * 50)
        print(self.df.head())

        print("=" * 50)
        print(self.df.shape)

        print("=" * 50)
        print(self.df.info())

        print("=" * 50)
        print(self.df.describe())

    def missing_values(self):

        missing = self.df.isnull().sum()

        missing.to_csv("outputs/reports/missing_values.csv")

        plt.figure(figsize=(12,6))

        sns.heatmap(self.df.isnull(),
                    cbar=False)

        plt.title("Missing Values")

        plt.savefig("outputs/plots/missing_values.png")

        plt.close()

    def duplicate_values(self):

        duplicates = self.df.duplicated().sum()

        print(f"Duplicate Rows : {duplicates}")

    def numerical_distribution(self):

        numerical = self.df.select_dtypes(include=["int64","float64"])

        numerical.hist(figsize=(18,15))

        plt.tight_layout()

        plt.savefig("outputs/plots/numerical_distribution.png")

        plt.close()

    def categorical_distribution(self):

        categorical = self.df.select_dtypes(include="object")

        for col in categorical.columns:

            plt.figure(figsize=(10,6))

            sns.countplot(
                y=self.df[col],
                order=self.df[col].value_counts().index
            )

            plt.title(col)

            plt.tight_layout()

            plt.savefig(f"outputs/plots/{col}.png")

            plt.close()

    def correlation(self):

        corr = self.df.corr(numeric_only=True)

        plt.figure(figsize=(12,10))

        sns.heatmap(
            corr,
            annot=True,
            cmap="coolwarm"
        )

        plt.savefig("outputs/plots/correlation.png")

        plt.close()

    def outliers(self):

        numerical = self.df.select_dtypes(include=["int64","float64"])

        for col in numerical.columns:

            plt.figure(figsize=(8,4))

            sns.boxplot(x=self.df[col])

            plt.title(col)

            plt.savefig(f"outputs/plots/{col}_boxplot.png")

            plt.close()

    def location_plot(self):

        plt.figure(figsize=(10,10))

        plt.scatter(
            self.df["Longitude"],
            self.df["Latitude"],
            s=2
        )

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.savefig("outputs/plots/location.png")

        plt.close()

    def crime_density_map(self):

        plt.figure(figsize=(10,10))

        plt.scatter(
            self.df["Longitude"],
            self.df["Latitude"],
            c=self.df["Crime_Density"],
            cmap="Reds",
            s=4
        )

        plt.colorbar(label="Crime Density")

        plt.savefig("outputs/plots/crime_density.png")

        plt.close()

    def run(self):

        self.basic_info()

        self.missing_values()

        self.duplicate_values()

        self.numerical_distribution()

        self.categorical_distribution()

        self.correlation()

        self.outliers()

        self.location_plot()

        self.crime_density_map()


if __name__ == "__main__":

    eda = EDA("data/merged_dataset.csv")

    eda.run()