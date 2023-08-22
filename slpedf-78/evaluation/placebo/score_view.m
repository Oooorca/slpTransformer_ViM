clc; clear all;

% Load the MATLAB .mat file
mat_data = load('temazepam_ret.mat');

% Access the 'score' data
score_data = mat_data.score;
yhat_data = mat_data.yhat;

% Extract the scores and yhat for the 465th sample
sample_465_scores = squeeze(score_data(465, :, :));
sample_465_yhat = squeeze(yhat_data(465, :));

% Print the scores and yhat for each time step
for i = 1:size(sample_465_scores, 1)
    fprintf("Time step %d scores: %s\n", i, mat2str(sample_465_scores(i, :)));
    fprintf("Time step %d yhat: %s\n", i, mat2str(sample_465_yhat(i)));
end
