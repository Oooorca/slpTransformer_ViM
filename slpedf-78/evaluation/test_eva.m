clc; clear all;
% 设置nchan参数
nchan = 1; % 假设为16

% 调用函数并获取实验结果
%result = aggregate_sleeptransformer(nchan);

result = aggregate_performance(nchan);

%[acc, kappa, f1, sens, spec, classwise_sens, classwise_sel, C] = aggregate_performance('sleeptransformer', 1)

% 将结果保存到MAT文件
save('test_results.mat', 'result');
