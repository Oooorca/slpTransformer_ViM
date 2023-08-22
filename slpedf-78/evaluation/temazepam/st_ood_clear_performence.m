clc; clear all;
% Load yt and yh
load('labels_yt.mat');
load('labels_yh.mat');

% List of excluded rows files
excluded_row_files = {'f50_ood_cleared_row.csv', 'f45_ood_cleared_row.csv', ...
                      'f43_ood_cleared_row.csv', 'f40_ood_cleared_row.csv', ...
                      'f38_ood_cleared_row.csv', 'f35_ood_cleared_row.csv'};

% Initialize arrays to store mismatched indices and counts
mismatched_indices = cell(1, length(excluded_row_files));
counts = zeros(5, length(excluded_row_files));

% Add a case for the original mismatched indices (no rows excluded)
original_mismatched_indices = find(yh ~= yt);
mismatched_indices{1} = original_mismatched_indices;
counts(:, 1) = histcounts(yt(mismatched_indices{1}), 1:6);

for i = 2:length(excluded_row_files)
    excluded_rows = [];
    fileID = fopen(excluded_row_files{i}, 'r');
    while ~feof(fileID)
        line = fgetl(fileID);
        excluded_rows = [excluded_rows; str2double(line)];
    end
    fclose(fileID);

    % Remove excluded rows from yt and yh
    yt_temp = yt;
    yh_temp = yh;
    yt_temp(excluded_rows) = [];
    yh_temp(excluded_rows) = [];

    % Save mismatched predictions
    mismatched_indices{i} = find(yh_temp ~= yt_temp);

    % Calculate occurrences of each number
    counts(:, i) = histcounts(yt_temp(mismatched_indices{i}), 1:6);
end

% Calculate accuracy
acc = sum(yh == yt) / numel(yt);

% Create legend labels
legend_labels = {'Ori ACC:0.6463', '50 ACC:0.6751', '45 ACC:0.6922', '43 ACC:0.7104', '40 ACC:0.7632', '38 ACC:0.7819', '35 ACC:0.7862'};

% Plot the histogram
figure;
bar(1:5, counts, 'barwidth', 0.3);
xlabel('Class');
ylabel('Count');
title('Histogram of Mismatched Class Labels (ST-Temazepam)');
xticks(1:5);
xticklabels({'Wake', 'N1', 'N2', 'N3', 'REM'});
legend(legend_labels, 'Location', 'Best');
grid on;
