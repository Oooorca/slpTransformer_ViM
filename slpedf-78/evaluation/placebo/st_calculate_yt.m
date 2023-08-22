clc; clear all;

% Load yt
load('labels_yh.mat');

% Count occurrences of each class in yt
class_counts = histcounts(yh, 1:6);

% Plot the histogram
figure;
bar(1:5, class_counts, 'barwidth', 0.3);
xlabel('Class');
ylabel('Count');
title('Histogram of Class Labels in yh');
xticks(1:5);
xticklabels({'Wake', 'N1', 'N2', 'N3', 'REM'});
grid on;
