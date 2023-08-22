function [signal, fs] = read_sleep_edf_records(filename, channels)
    [hdr, edf] = edfread(filename, 'desiredSignals', channels);
    assert(length(hdr.samples) == numel(channels));
    
    fs = unique(hdr.samples);
    assert(length(fs) == 1);
    
    if length(channels) > 1
        signal = edf';
    else
        signal = edf;
    end
end
