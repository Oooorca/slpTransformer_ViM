clc; clear all;
% 设置nchan参数
nchan = 1; % 假设为16

% 调用函数并获取实验结果
result = aggregate_sleeptransformer(nchan);

% 将结果保存到MAT文件
save('experiment_results.mat', 'result');
