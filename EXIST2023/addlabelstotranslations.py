import pandas as pd

#Step 1: Read the first CSV file with two columns (the one with text and translated text)
data = pd.read_csv("translated_dev_es.csv")

#Step 2: Read the second CSV file with labels
labels = pd.read_csv("labels_dev.csv", usecols=[0])

#Step 3: Merge the two dataframes on index or a common column
merged_data = pd.concat([data, labels], axis=1)

# write the merged data to a new CSV file
merged_data.to_csv("translated_dev_es_labeled.csv", index=False)