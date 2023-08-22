# Read and process ori_row.csv
ori_file_path = "ori_row.csv"
ood_file_path = "f50_ood_row.csv"

# Read and process ori_row.csv
def read_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        next(file)  # Skip the first line
        for line in file:
            value = int(line.strip())
            data.append(value)
    return data

def read_ori_csv(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            value = int(line.strip())
            data.append(value)
    return data

ori_data = read_ori_csv(ori_file_path)
increment = 20  # The increment value

A = [value - (i + 1) * increment for i, value in enumerate(ori_data)] #12940->12640
print(A)
A = [value - i * 20 for i, value in enumerate(A)] #模拟场景通过真值表找到区间
print(A)

# Calculate intervals in A
A_intervals = [(0, A[0])]
for i in range(len(A) - 1):
    interval = (A[i], A[i + 1])
    A_intervals.append(interval)

print(A_intervals)
# Read and process ood_row.csv
ood_data = read_csv(ood_file_path)
# Adjust ood_data based on intervals in A
adjusted_data = []
j = 0  # Counter for interval index

print(ood_data)

for value in ood_data:
    while j < len(A_intervals) and value > A_intervals[j][1]:
        j += 1
    adjusted_value = value + j * 20
    adjusted_data.append(adjusted_value)

print(adjusted_data)

# Save adjusted data to a new CSV file
adjusted_file_path = "f50_ood_cleared_row.csv"
with open(adjusted_file_path, 'w', newline='') as file:
    for value in adjusted_data:
        file.write(f"{value}\n")

print(f"Adjusted data saved to '{adjusted_file_path}'")
