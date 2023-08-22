clc; clear all;
% Load yt and yh
load('labels_yt.mat');
load('labels_yh.mat');

% List of excluded rows files
excluded_row_files = {'f0_origin.csv','f50_ood_cleared_row.csv', 'f45_ood_cleared_row.csv', ...
                      'f40_ood_cleared_row.csv','f35_ood_cleared_row.csv', ...
                      'f30_ood_cleared_row.csv'};
                  
% Initialize arrays to store mismatched indices and counts
mismatched_indices = cell(1, length(excluded_row_files));
counts = zeros(5, length(excluded_row_files));

% Create legend labels
legend_labels = {'Ori','f50', 'f45',  'f40', 'f35', 'f30'};

for i = 1:length(excluded_row_files)
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
    
    % Calculate accuracy
    acc = sum(yh_temp == yt_temp) / numel(yt_temp);

    % Calculate occurrences of each number
    counts(:, i) = histcounts(yt_temp(mismatched_indices{i}), 1:6);
    
    % Append accuracy to the legend label
    legend_labels{i} = [legend_labels{i}, sprintf(' ACC:%.4f', acc)];
end

% Plot the histogram
figure;
bar(1:5, counts, 'barwidth', 0.3);
xlabel('Class');
ylabel('Count');
title('Histogram of Mismatched Class Labels (SC)');
xticks(1:5);
xticklabels({'Wake', 'N1', 'N2', 'N3', 'REM'});
legend(legend_labels, 'Location', 'Best');
grid on;
