function [fscore, sensitivity, specificity] = litis_class_wise_f1(yt, yh)
    % yt: 真实的类别标签
    % yh: 预测的类别标签
    
    % 初始化变量
    num_classes = max(max(yt), max(yh));
    fscore = zeros(num_classes, 1);
    sensitivity = zeros(num_classes, 1);
    specificity = zeros(num_classes, 1);
    
    for c = 1:num_classes
        true_positive = sum(yt == c & yh == c);
        false_positive = sum(yt ~= c & yh == c);
        false_negative = sum(yt == c & yh ~= c);
        true_negative = sum(yt ~= c & yh ~= c);
        
        % 计算精确度（Precision）
        precision = true_positive / (true_positive + false_positive);
        
        % 计算敏感度（Sensitivity）
        sensitivity(c) = true_positive / (true_positive + false_negative);
        
        % 计算特异度（Specificity）
        specificity(c) = true_negative / (true_negative + false_positive);
        
        % 计算 F1 分数
        fscore(c) = 2 * precision * sensitivity(c) / (precision + sensitivity(c));
    end
end
