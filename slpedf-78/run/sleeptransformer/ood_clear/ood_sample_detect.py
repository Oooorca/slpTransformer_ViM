import numpy as np
import hdf5storage
import csv

# Load the ood_indices_list_temazepam.csv file
ood_indices_list = []
with open('f50_ood_indices_list_sc.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)
    next(csv_reader)  # Skip the header
    for row in csv_reader:
        ood_indices_list.append(int(row[0]))


# Process the ood_indices_list and create a set to store unique values
processed_values = set()
# Process the indices and add the processed values to the set
for index in ood_indices_list:
    processed_value = (index // 21) if (index % 21 == 0) else (index // 21) + 1
    processed_values.add(processed_value)

# Convert the set to a sorted list
processed_values_list = sorted(list(processed_values))
print(processed_values_list)

# Save the processed values to ood_row.csv
with open('f50_ood_row.csv', 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Processed Values'])
    for value in processed_values_list:
        csv_writer.writerow([value])

