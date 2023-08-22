function stages = read_sleep_edf_annotation(filename)
    [~, record] = edfread(filename);
    stages = record;
end
