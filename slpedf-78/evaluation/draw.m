% Load mismatched_yt from unmatch.mat
%load('unmatch.mat', 'mismatched_yt');
load('labels_yt.mat', 'yt');
% Count occurrences of each number
%counts = histcounts(mismatched_yt, 1:6);
counts = histcounts(yt, 1:6);
% Plot the histogram
figure;
bar(1:5, counts, 'barwidth', 0.5, 'facecolor', 'b', 'edgecolor', 'black');
xlabel('Class');
ylabel('Count');
title('Histogram of Mismatched Class Labels');
xticks(1:5);
xticklabels({'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5'});
ylim([0 max(counts)+1]);
grid on;
