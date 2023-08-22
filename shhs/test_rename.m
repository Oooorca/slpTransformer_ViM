folder = './mat/';   % 文件夹路径
prefix = 'n';        % 文件名前缀
extension = '.mat';  % 文件扩展名
totalFiles = 199;    % 文件总数

startNumber = 1;     % 起始数字
endNumber = totalFiles;  % 结束数字
missingFiles = [200:210];  % 缺失的文件

% 计算需要移动的步数
numMissing = length(missingFiles);
numToMove = totalFiles + numMissing - endNumber;

% 遍历每个文件并重命名
for i = endNumber:-1:startNumber
    % 构建原始文件名和目标文件名
    oldFileName = sprintf('%s%04d_eeg%s', prefix, i, extension);
    newNumber = i + numToMove;
    newFileName = sprintf('%s%04d_eeg%s', prefix, newNumber, extension);
    
    % 构建完整的文件路径
    oldFilePath = fullfile(folder, oldFileName);
    newFilePath = fullfile(folder, newFileName);
    
    % 重命名文件
    movefile(oldFilePath, newFilePath);
    
    disp(['重命名文件：' oldFileName ' -> ' newFileName]);
end

% 处理缺失的文件
for i = 1:numMissing
    % 构建原始文件名和目标文件名
    oldNumber = missingFiles(i);
    newNumber = startNumber + i - 1;
    oldFileName = sprintf('%s%04d_eeg%s', prefix, oldNumber, extension);
    newFileName = sprintf('%s%04d_eeg%s', prefix, newNumber, extension);
    
    % 构建完整的文件路径
    oldFilePath = fullfile(folder, oldFileName);
    newFilePath = fullfile(folder, newFileName);
    
    % 重命名文件
    movefile(oldFilePath, newFilePath);
    
    disp(['重命名文件：' oldFileName ' -> ' newFileName]);
end
