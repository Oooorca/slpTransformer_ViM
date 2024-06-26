clear all
close all
clc


rng(10); % for repeatable

% divide subjects into training, evaluation, and test sets for consistency
% between various networks

Nsub = 200;

subjects = randperm(Nsub);

test = 0.3;
train = 0.7;

test_sub = sort(subjects(1 : round(test*Nsub)));
rest = setdiff(subjects, test_sub);
perm_list = randperm(numel(rest));

% 50 subjects as eval set
eval_sub = sort(rest(perm_list(1:50)));
train_check_sub = sort(rest(perm_list(51:60)));
train_sub = sort(rest(perm_list(51:end)));

save('./data_split_eval.mat', 'train_sub','test_sub','eval_sub','train_check_sub')
%save('./data_split_eval.mat', 'train_sub','test_sub','eval_sub')