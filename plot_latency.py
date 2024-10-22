# test_latency_and_plot.py
import requests
import time
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm  # Import tqdm for progress bar

BASE_URL = "http://ece-444-pra5-env-2.eba-ujsdbkn2.us-east-1.elasticbeanstalk.com/predict"

# Test data (two fake news, two real news examples)
test_data = [
    {"input": "This is fake news", "gt": "FAKE"},
    {"input": "This is a real data", "gt": "REAL"},
    {"input": "Politicians are lying to you", "gt": "FAKE"},
    {"input": "This is real data", "gt": "REAL"}
]

def record_performance(test_case, filename_index):
    csv_filename = f"performance_{filename_index}.csv"
    
    # Open a CSV file to record timestamps
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Call_Number", "Start_Time", "End_Time", "Elapsed_Time(ms)"])

        # Use tqdm to show progress for 100 API calls
        for i in tqdm(range(100), desc=f"Processing {test_case['input']}"):
            start_time = datetime.now()
            response = requests.post(
                BASE_URL,
                json={"article": test_case["input"]},
                headers={"Content-Type": "application/json"}
            )
            end_time = datetime.now()

            assert response.status_code == 200

            elapsed_time = (end_time - start_time).total_seconds() * 1000  # Convert to milliseconds
            writer.writerow([i + 1, start_time, end_time, elapsed_time])

            time.sleep(0.1)  # Slight delay between API calls to avoid overwhelming the server

    # Generate a boxplot for the elapsed time
    generate_boxplot(csv_filename)

def generate_boxplot(csv_file):
    # Read the performance CSV file
    df = pd.read_csv(csv_file)

    # Create a boxplot for the elapsed time
    plt.figure(figsize=(10, 6))
    plt.boxplot(df["Elapsed_Time(ms)"])
    plt.title(f"Latency Performance (ms) for: {csv_file}")
    plt.ylabel("Time (ms)")

    # Save the boxplot
    plt.savefig(csv_file.replace(".csv", "_boxplot.png"))
    # plt.show()

    # Calculate and print average performance
    avg_performance = df["Elapsed_Time(ms)"].mean()
    print(f"Average Performance for {csv_file}: {avg_performance:.2f} ms")

if __name__ == "__main__":
    for filename_index, test_case in enumerate(test_data, start=1):
        record_performance(test_case, filename_index)
