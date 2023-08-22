clear all
close all
clc

addpath('edf_reader');

xml_path = '../../SleepEDF/annotation';
edf_path = '../../SleepEDF/PSG';
mat_path = './mat/';

if(~exist(mat_path,'dir'))
    mkdir(mat_path);
end

dirlist = dir([edf_path, '*.edf']);
N = numel(dirlist)


parfor n = 1 : N
    filename = dirlist(n).name
    disp(filename)
    process_and_save_1file(filename, n)
end