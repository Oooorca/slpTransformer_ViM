clc;
clear all;

% Load yt and yh
load('labels_yt.mat');
load('labels_yh.mat');

% Specify the file containing excluded rows
excluded_row_file = 'f50_ood_cleared_row.csv';

% Load excluded rows from the file
excluded_rows = csvread(excluded_row_file);

% Compare yh and yt
mismatched_indices = find(yh ~= yt);
matched_indices = find(yh == yt);

% Find values in excluded_rows that are in matched_indices
values_in_matched = excluded_rows(ismember(excluded_rows, matched_indices));

% Find values in excluded_rows that are not in matched_indices
values_in_mismatched = setdiff(excluded_rows, values_in_matched)

% Calculate the count of excluded_rows and mismatched_indices
num_excluded_rows = numel(excluded_rows);
num_mismatched_indices = numel(mismatched_indices);
num_matched_indices = numel(matched_indices);

% Calculate the count of such values
num_values_in_matched = numel(values_in_matched);

% Display the results
fprintf('Number of excluded rows: %d\n', num_excluded_rows);
fprintf('Number of mismatched indices: %d\n', num_mismatched_indices);
fprintf('Number of matched indices: %d\n', num_matched_indices);
fprintf('Number of values in "%s" that are included in matched indices: %d\n', excluded_row_file, num_values_in_matched);

% Extract yt and yh values corresponding to excluded_rows
yt_excluded = yt(excluded_rows);
yh_excluded = yh(excluded_rows);

% Create a matrix with excluded_rows, yt_excluded, and yh_excluded
combined_data = [excluded_rows, yt_excluded, yh_excluded];

% Save combined_data to a new variable
save('combined_data.mat', 'combined_data');
