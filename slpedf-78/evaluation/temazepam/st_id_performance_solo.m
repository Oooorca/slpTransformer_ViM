clc; clear all;
% Load yt and yh
load('labels_yt.mat');
load('labels_yh.mat');

% Load the excluded rows from ood_cleared_row.csv
excluded_rows = [];
fileID = fopen('f0_origin.csv', 'r');
while ~feof(fileID)
    line = fgetl(fileID);
    excluded_rows = [excluded_rows; str2double(line)];
end
fclose(fileID);

% Remove excluded rows from yt and yh
yt(excluded_rows) = [];
yh(excluded_rows) = [];

% Save mismatched predictions
mismatched_indices = find(yh ~= yt);
mismatched_yt = yt(mismatched_indices);  

% Save mismatched_yt to unmatch.mat
%save('unmatch.mat', 'mismatched_yt');

% Calculate accuracy
acc = sum(yh == yt) / numel(yt)

% Count occurrences of each number
counts = histcounts(mismatched_yt, 1:6);

% Plot the histogram
figure;
bar(1:5, counts, 'barwidth', 0.5, 'facecolor', 'b', 'edgecolor', 'black');
xlabel('Class');
ylabel('Count');
title('Histogram of Mismatched Class Labels');
xticks(1:5);
xticklabels({'Wake', 'N1', 'N2', 'N3', 'REM'});
ylim([0 max(counts)+1]);
grid on;