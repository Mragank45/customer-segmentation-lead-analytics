"""Run the whole pipeline: generate sample data -> segment customers -> analyse leads."""
from src import generate_data, lead_analysis, segmentation


def main() -> None:
    generate_data.main()

    print("\n=== Customer segments (RFM + K-Means) ===")
    print(segmentation.run().to_string())

    print("\n=== Lead conversion by source ===")
    print(lead_analysis.run().to_string())

    print("\nCharts and CSVs saved in /outputs")


if __name__ == "__main__":
    main()
