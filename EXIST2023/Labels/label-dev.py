import csv
import json

# Load the JSON object from the file
with open('EXIST2023_dev_task1_gold_hard.json') as f:
    data = json.load(f)

# Create a dictionary to store the label values as 1 or 0
label_dict = {}
for key in data:
    label_dict[key] = 1 if data[key]["hard_label"] == "YES" else 0

# Write the dictionary to a CSV file
with open('labels_dev.csv', 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["label"])
    writer.writeheader()
    for key in label_dict:
        writer.writerow({"label": label_dict[key]})
