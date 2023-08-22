clc; 

% Load the MATLAB .mat file
mat_data = load('test_ret.mat');

% Access the 'score' data
score_data = mat_data.score;
yhat_data = mat_data.yhat;

sample_scores = squeeze(score_data(10, :, :));
sample_yhat = squeeze(yhat_data(10, :));

% Print the scores and yhat for each time step
for i = 1:size(sample_scores, 1)
    fprintf("Time step %d scores: %s\n", i, mat2str(sample_scores(i, :)));
    fprintf("Time step %d yhat: %s\n", i, mat2str(sample_yhat(i)));
end
