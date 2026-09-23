import os
import pandas as pd
import matplotlib.pyplot as plt

# Create output folder
os.makedirs("outputs/plots", exist_ok=True)


class DataVisualization:

    def __init__(self, path):
        self.df = pd.read_csv(path)

    # -------------------------------------------------
    # Correlation Heatmap
    # -------------------------------------------------
    def correlation_heatmap(self):

        corr = self.df.select_dtypes(include=["number"]).corr()

        plt.figure(figsize=(12, 10))

        plt.imshow(corr, cmap="coolwarm")

        plt.colorbar()

        plt.xticks(range(len(corr.columns)), corr.columns,
                   rotation=90)

        plt.yticks(range(len(corr.columns)), corr.columns)

        plt.title("Correlation Heatmap")

        plt.tight_layout()

        plt.savefig(
            "outputs/plots/correlation_heatmap.png",
            dpi=300
        )

        plt.close()

        print("✓ Correlation Heatmap Saved")

    # -------------------------------------------------
    # Boxplots
    # -------------------------------------------------
    def boxplots(self):

        numerical = self.df.select_dtypes(include=["number"])

        for column in numerical.columns:

            plt.figure(figsize=(8,4))

            plt.boxplot(
                numerical[column].dropna(),
                vert=False
            )

            plt.title(column)

            plt.tight_layout()

            plt.savefig(
                f"outputs/plots/{column}_boxplot.png",
                dpi=300
            )

            plt.close()

        print("✓ Boxplots Saved")

    # -------------------------------------------------
    # Police Distance Histogram
    # -------------------------------------------------
    def police_distance(self):

        plt.figure(figsize=(10,6))

        plt.hist(
            self.df["Police_Distance_m"],
            bins=40
        )

        plt.xlabel("Distance (m)")
        plt.ylabel("Frequency")
        plt.title("Police Distance Distribution")

        plt.tight_layout()

        plt.savefig(
            "outputs/plots/police_distance_distribution.png",
            dpi=300
        )

        plt.close()

        print("✓ Police Distance Distribution Saved")

    # -------------------------------------------------
    # Hospital Distance Histogram
    # -------------------------------------------------
    def hospital_distance(self):

        plt.figure(figsize=(10,6))

        plt.hist(
            self.df["Hospital_Distance_m"],
            bins=40
        )

        plt.xlabel("Distance (m)")
        plt.ylabel("Frequency")
        plt.title("Hospital Distance Distribution")

        plt.tight_layout()

        plt.savefig(
            "outputs/plots/hospital_distance_distribution.png",
            dpi=300
        )

        plt.close()

        print("✓ Hospital Distance Distribution Saved")

    # -------------------------------------------------
    # Crime Hotspots
    # -------------------------------------------------
    def crime_hotspots(self):

        plt.figure(figsize=(8,8))

        scatter = plt.scatter(
            self.df["Longitude"],
            self.df["Latitude"],
            c=self.df["Crime_Density"],
            s=4
        )

        plt.colorbar(scatter)

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.title("Crime Density Map")

        plt.tight_layout()

        plt.savefig(
            "outputs/plots/crime_hotspots.png",
            dpi=300
        )

        plt.close()

        print("✓ Crime Hotspots Saved")

    # -------------------------------------------------
    # Road Network
    # -------------------------------------------------
    def road_network(self):

        plt.figure(figsize=(8,8))

        plt.scatter(
            self.df["Longitude"],
            self.df["Latitude"],
            s=2
        )

        plt.xlabel("Longitude")
        plt.ylabel("Latitude")

        plt.title("Road Network")

        plt.tight_layout()

        plt.savefig(
            "outputs/plots/road_network.png",
            dpi=300
        )

        plt.close()

        print("✓ Road Network Saved")

    # -------------------------------------------------
    # Run All
    # -------------------------------------------------
    def run(self):

        self.correlation_heatmap()

        self.boxplots()

        self.police_distance()

        self.hospital_distance()

        self.crime_hotspots()

        self.road_network()


if __name__ == "__main__":

    viz = DataVisualization("data/merged_dataset.csv")

    viz.run()