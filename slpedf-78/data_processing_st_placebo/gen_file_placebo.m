clear all
close all
clc

rng(10); % for repeatable

mat_path = './mat/';
Nfold = 1;
load('../data_split_sleepedf_st.mat');

tf_path = './file_placebo/eeg/';
if(~exist(tf_path, 'dir'))
    mkdir(tf_path);
end

for s = 1 : Nfold
    disp(['Fold: ', num2str(s),'/',num2str(Nfold)]);
    
	train_s = train_sub{s};
    %train_check_s = train_check_sub{s};
    eval_s = eval_sub{s};
    test_s = test_sub{s};
    
    train_filename = [tf_path, 'train_list','.txt'];
    fid = fopen(train_filename,'wt');
    for i = 1 : numel(train_s)
        for night = 1 : 2
            sname = ['n', num2str(train_s(i),'%02d'), '_',num2str(night),'_eeg.mat'];
            if(exist([mat_path,sname], 'file'))
                load([mat_path,sname], 'label');
                num_sample = numel(label);
                file_path = ['../../matst_placebo/',sname];
                fprintf(fid, '%s\t%d\n', file_path, num_sample);
            end
        end
    end
    fclose(fid);
    clear fid file_path

    
    eval_filename = [tf_path, 'eval_list','.txt'];
    fid = fopen(eval_filename,'wt');
    for i = 1 : numel(eval_s)
        for night = 1 : 2
            sname = ['n', num2str(eval_s(i),'%02d'), '_',num2str(night),'_eeg.mat'];
            if(exist([mat_path,sname], 'file'))
                load([mat_path,sname], 'label');
                num_sample = numel(label);
                file_path = ['../../matst_placebo/',sname];
                fprintf(fid, '%s\t%d\n', file_path, num_sample);
            end
        end
    end
    fclose(fid);
    clear fid file_path
    
    test_filename = [tf_path, 'test_list','.txt'];
    fid = fopen(test_filename,'wt');
    for i = 1 : numel(test_s)
        for night = 1 : 2
            sname = ['n', num2str(test_s(i),'%02d'), '_',num2str(night),'_eeg.mat'];
            if(exist([mat_path,sname], 'file'))
                load([mat_path,sname], 'label');
                num_sample = numel(label);
                file_path = ['../../matst_placebo/',sname];
                fprintf(fid, '%s\t%d\n', file_path, num_sample);
            end
        end
    end
    fclose(fid);
    clear fid file_path
end








