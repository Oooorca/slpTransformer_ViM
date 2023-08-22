function process_sleep_edf_dataset(data_path, output_path)
    psg_files = dir(fullfile(data_path, '*-PSG.edf'));
    annotation_files = dir(fullfile(data_path, '*-Hypnogram.edf'));

    for i = 1:numel(psg_files)
        psg_filename = fullfile(data_path, psg_files(i).name);
        annotation_filename = fullfile(data_path, annotation_files(i).name);

        [~, name, ~] = fileparts(psg_filename);
        file_prefix = name(1:7);

        [~, stages] = read_sleep_edf_annotation(annotation_filename);
        [eeg, fs] = read_sleep_edf_records(psg_filename, {'EEG'});

        % Perform additional preprocessing and feature extraction if needed
        % ...

        % Save the processed data as .mat file
        mat_filename = fullfile(output_path, [file_prefix, '.mat']);
        save(mat_filename, 'eeg', 'stages', 'fs');
        disp(['Saved processed data to: ', mat_filename]);
    end
end


