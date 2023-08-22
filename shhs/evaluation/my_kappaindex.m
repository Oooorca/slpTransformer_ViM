function kappa = kappaindex(yh, yt, num_classes)
    N = numel(yt);  % 样本数量

    % 计算混淆矩阵
    C = confusionmat(yt, yh);

    % 计算预测的概率分布
    p_pred = sum(C, 1) / N;

    % 计算真实的概率分布
    p_true = sum(C, 2) / N;

    % 计算预期的概率分布
    p_expected = p_pred .* p_true';

    % 计算观察到的一致性（observed agreement）
    p_o = sum(diag(C)) / N;

    % 计算预期的一致性（expected agreement）
    p_e = sum(diag(p_expected));

    % 计算 Cohen's Kappa 系数
    kappa = (p_o - p_e) / (1 - p_e);
end
