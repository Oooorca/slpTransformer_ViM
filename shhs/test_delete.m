clear all
close all
clc

data_dir = './mat/';
file_list = dir(fullfile(data_dir, '*.mat'));

for i = 1:numel(file_list)
    file_name = file_list(i).name;
    [~, name, ext] = fileparts(file_name);
    
    if endsWith(name, '_eeg')
        % Keep the file
    else
        % Delete the file
        file_path = fullfile(data_dir, file_name);
        delete(file_path);
        disp(['Deleted file: ', file_name]);
    end
end

disp('Operation completed.');
