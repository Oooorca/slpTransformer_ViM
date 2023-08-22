function collection = st_aggregate_performance(nchan)

    seq_len = 21;
    Nfold = 1;
    yh = cell(Nfold,1);
    yt = cell(Nfold,1);
    mat_path = '../matst_temazepam/';
    load('../data_split_sleepedf_st.mat');

    acc_novote = [];
    actual_labels = [];
    pred_labels = [];
    for fold = 1 : Nfold
        test_s = train_sub{fold};
        sample_size = [];
        for i = 1 : numel(test_s)
            for night = 1 : 2
                sname = ['n', num2str(test_s(i),'%02d'), '_', num2str(night), '_eeg.mat'];
                if(~exist([mat_path, sname], 'file'))
                    continue
                end
                load([mat_path,sname], 'label');
                sample_size = [sample_size; numel(label) -  (seq_len - 1)];
                yt{fold} = [yt{fold}; double(label)];
                actual_labels = [actual_labels; double(label)];
            end
        end
        
    %save('labels.mat', 'actual_labels', '-v7.3');
    %disp('All labels saved to labels.mat');
        
        if(seq_len < 100)
	    load(['./test_ret.mat']);
	else
	    load(['./test_ret.mat']);
    end
    
        acc_novote = [acc_novote; acc];    
        
        score_ = cell(1,seq_len);
        for n = 1 : seq_len
            score_{n} = softmax(squeeze(score(:,n,:)));
        end
        score = score_;
        clear score_;

        count = 0;
        for i = 1 : numel(test_s)
            for night = 1 : 2
                sname = ['n', num2str(test_s(i),'%02d'), '_', num2str(night), '_eeg.mat'];
                if(~exist([mat_path, sname], 'file'))
                    continue
                end
                count = count + 1;
                start_pos = sum(sample_size(1:count-1)) + 1;
                end_pos = sum(sample_size(1:count-1)) + sample_size(count);
                score_i = cell(1,seq_len);
                for n = 1 : seq_len
                    score_i{n} = score{n}(start_pos:end_pos, :);
                    N = size(score_i{n},1);
                    score_i{n} = [ones(seq_len-1,5); score{n}(start_pos:end_pos, :)];
                    score_i{n} = circshift(score_i{n}, -(seq_len - n), 1);
                end

                fused_score = log(score_i{1});
                for n = 2 : seq_len
                    fused_score = fused_score + log(score_i{n});
                end

                yhat = zeros(1,size(fused_score,1));
                for k = 1 : size(fused_score,1)
                    [~, yhat(k)] = max(fused_score(k,:));
                end

                yh{fold} = [yh{fold}; double(yhat')];
                pred_labels = [pred_labels; double(yhat')];
            end
        end
    end

    %save('labels.mat', 'pred_labels', '-v7.3');
    %disp('All labels saved to labels.mat');
    
    yh = cell2mat(yh);
    yt = cell2mat(yt);
    
    save('labels_yt.mat', 'yt', '-v7.3');
    save('labels_yh.mat', 'yh', '-v7.3');
    disp('All labels saved to labels.mat');    
    
    acc = sum(yh == yt)/numel(yt)
    
    % Save mismatched predictions
    mismatched_indices = find(yh ~= yt);
    mismatched_yt = yt(mismatched_indices);  

    % Save mismatched_yt to unmatch.mat
    save('unmatch.mat', 'mismatched_yt');
    
    C = confusionmat(yt, yh);
    
    [fscore, sensitivity, specificity] = litis_class_wise_f1(yt, yh);
    mean_fscore = mean(fscore)
    mean_sensitivity = mean(sensitivity)
    mean_specificity = mean(specificity)
    kappa = kappaindex(yh,yt,5)
    
    
    str = '';
    % acc
    str = [str, '$', num2str(acc*100, '%.1f'), '$', ' & '];
    % kappa
    str = [str, '$', num2str(kappa, '%.3f'), '$', ' & '];
    % fscore
    str = [str, '$', num2str(mean_fscore*100, '%.1f'), '$', ' & '];
    % mean_sensitivity
    str = [str, '$', num2str(mean_sensitivity*100, '%.1f'), '$', ' & '];
    % mean_specificity
    str = [str, '$', num2str(mean_specificity*100, '%.1f'), '$', ' & '];
    
    % class-wise MF1
    for i = 1 : 5
        str = [str, '$', num2str(fscore(i)*100,'%.1f'), '$ & '];
    end
    str;
    
    collection = [acc, mean_fscore, kappa, mean_sensitivity, mean_specificity];


end

