import numpy as np
import hdf5storage
import csv

input_file_path = "test_list.txt"
output_file_path = "ori_row.csv"

# Read data from the input file
with open(input_file_path, 'r') as input_file:
    lines = input_file.readlines()

# Process the data and calculate cumulative sums
cumulative_sums = []
current_sum = 0
for line in lines:
    value = int(line.strip())
    current_sum += value
    print(current_sum)
    cumulative_sums.append(current_sum)


# Save the cumulative sums to the output CSV file
with open(output_file_path, 'w', newline='') as output_file:
    for value in cumulative_sums:
        output_file.write(f"{value}\n")

print(f"Cumulative sums saved to '{output_file_path}'")
