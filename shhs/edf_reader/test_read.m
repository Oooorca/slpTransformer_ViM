% 设置文件路径和文件名
%dataPath = 'D:\NC Program\sleep-edf-database-expanded-1.0.0\sleep-cassette';
dataPath = '.\';
psgFile = 'SC4001E0-PSG.edf';
hypnogramFile = 'SC4001EC-Hypnogram.edf';

% 读取脑电信号数据
psgFilePath = fullfile(dataPath, psgFile);
[psgHeader, psgRecord] = edfread16(psgFilePath);

% 读取注释信息数据
hypnogramFilePath = fullfile(dataPath, hypnogramFile);
[hypnoHeader, hypnoRecord] = edfread16(hypnogramFilePath);

% 打印脑电信号和注释信息的头信息
disp('脑电信号头信息：');
disp(psgHeader);

disp('注释信息头信息：');
disp(hypnoHeader);

% 打印脑电信号和注释信息的波形数据大小
disp('脑电信号数据大小：');
disp(size(psgRecord));

disp('注释信息数据大小：');
disp(size(hypnoRecord));

% 进行进一步的数据处理和分析...
