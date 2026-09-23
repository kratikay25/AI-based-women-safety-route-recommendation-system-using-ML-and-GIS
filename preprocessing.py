import os
import re
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class DataPreprocessor:

    def __init__(self, path):

        self.df = pd.read_csv(path)

        os.makedirs("outputs/processed", exist_ok=True)

    # ----------------------------------------------------
    # Remove Unnecessary Columns
    # ----------------------------------------------------
    def remove_columns(self):

        columns_to_remove = [
            "Road_ID",
            "u",
            "v",
            "name",
            "maxspeed"
        ]

        self.df.drop(
            columns=columns_to_remove,
            inplace=True,
            errors="ignore"
        )

        print("✓ Removed unnecessary columns")

    # ----------------------------------------------------
    # Convert Boolean to Integer
    # ----------------------------------------------------
    def convert_boolean(self):

        self.df["oneway"] = self.df["oneway"].astype(int)

        print("✓ Converted oneway to 0 and 1")

    # ----------------------------------------------------
    # Clean Highway Column
    # ----------------------------------------------------
    def clean_highway(self):

        def extract_highway(value):

            if pd.isna(value):
                return "unknown"

            value = str(value)

            # Remove brackets and quotes
            value = re.sub(r"[\[\]']", "", value)

            value = value.strip()

            # Split using comma OR whitespace
            parts = re.split(r"[,\s]+", value)

            # Remove empty strings
            parts = [p for p in parts if p != ""]

            if len(parts) == 0:
                return "unknown"

            # Keep only first highway type
            return parts[0]

        self.df["highway"] = self.df["highway"].apply(extract_highway)

        print("✓ Highway column cleaned")

    # ----------------------------------------------------
    # One Hot Encoding
    # ----------------------------------------------------
    def encode_highway(self):

        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore"
        )

        encoded = encoder.fit_transform(self.df[["highway"]])

        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(["highway"])
        )

        self.df = pd.concat(
            [
                self.df.drop(columns=["highway"]),
                encoded_df
            ],
            axis=1
        )

        print("✓ Highway encoded")

    # ----------------------------------------------------
    # Remove Missing Values
    # ----------------------------------------------------
    def remove_missing(self):

        self.df.dropna(inplace=True)

        print("✓ Missing values removed")

    # ----------------------------------------------------
    # Save Dataset
    # ----------------------------------------------------
    def save_dataset(self):

        output_path = "outputs/processed/processed_dataset.csv"

        self.df.to_csv(
            output_path,
            index=False
        )

        print("\n✓ Processed dataset saved successfully!")

        print("\nFinal Shape :", self.df.shape)

        print("\nFirst Five Rows:\n")

        print(self.df.head())

    # ----------------------------------------------------
    # Complete Pipeline
    # ----------------------------------------------------
    def run(self):

        self.remove_columns()

        self.convert_boolean()

        self.clean_highway()

        self.encode_highway()

        self.remove_missing()

        self.save_dataset()


# --------------------------------------------------------
# MAIN
# --------------------------------------------------------

if __name__ == "__main__":

    processor = DataPreprocessor(
        "data/merged_dataset.csv"
    )

    processor.run()