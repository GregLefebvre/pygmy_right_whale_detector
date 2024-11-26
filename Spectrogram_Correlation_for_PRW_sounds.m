%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  SPECTROGRAM CORRELATION FOR PRW SOUNDS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear vars
clc

% Define input paths
input_folder = 'D:\3501\flac\';
known_sounds = {
    'C:\Users\297661C\Documents\WAV\20150918T123001.flac', [21.380595148, 22.878479654, 31.6, 115.0]; 
    'C:\Users\297661C\Documents\WAV\20150918T123001.flac', [288.5807735, 290.1508946, 36, 113.9]; 
    'C:\Users\297661C\Documents\WAV\20150926T224501.flac', [38.46967227, 39.69009847, 36.8, 109.7]; 
};

% Parameters for spectrogram analysis and correlation threshold
window = hamming(2048);  % Hamming window of size 1024
nfft = 2048;             % FFT size
noverlap = 1843;         % 90% overlap
seuil = 0.6;             % Correlation threshold
freq_min = 20;           % Min frequency of target window
freq_max = 120;          % Max frequency of target window

% Loop through each known sound
for n = 1:length(known_sounds)
    known_file = known_sounds{n, 1};
    time_freq_coords = known_sounds{n, 2};
    
    % Load the known sound audio
    [known_audio_data, known_fs] = audioread(known_file);

    % Calculate the spectrogram of the known sound
    [known_S, known_F, known_T, known_P] = spectrogram(known_audio_data, window, noverlap, nfft, known_fs);

    % Extract the coordinates
    t_start = time_freq_coords(1);
    t_end = time_freq_coords(2);
    f_start = time_freq_coords(3);
    f_end = time_freq_coords(4);

    % Find indices for time and frequency coordinates
    [~, t_start_idx] = min(abs(known_T - t_start));
    [~, t_end_idx] = min(abs(known_T - t_end));
    [~, f_start_idx] = min(abs(known_F - f_start));
    [~, f_end_idx] = min(abs(known_F - f_end));

    % Verify and adjust indices to avoid out-of-bounds errors
    t_start_idx = max(1, t_start_idx);
    t_end_idx = min(length(known_T), t_end_idx);
    f_start_idx = max(1, f_start_idx);
    f_end_idx = min(length(known_F), f_end_idx);

    % Extract the known sound spectrogram portion
    known_sound = known_P(f_start_idx:f_end_idx, t_start_idx:t_end_idx);

    % Define output file for each known sound
    output_file = sprintf('detection_results_son_%d_3501.txt', n);
    
    % Open the output file for writing
    fileID = fopen(output_file, 'w');
    
    % Initialize total detections counter
    total_detections = 0;

    % Process each FLAC file in the input folder
    files = dir(fullfile(input_folder, '*.flac'));
    num_files = length(files);
    
    for k = 1:num_files
        % Current file name
        current_file = files(k).name;
        fprintf('Processing file %d of %d: %s\n', k, num_files, current_file);
        
        % Load the target audio
        [audio_data, fs] = audioread(fullfile(input_folder, current_file));
        
        % Calculate the spectrogram of the target audio
        [S, F, T, P] = spectrogram(audio_data, window, noverlap, nfft, fs);

        % Filter frequencies to keep only 20-120 Hz
        f_indices = F >= freq_min & F <= freq_max;
        P = P(f_indices, :);    
        F = F(f_indices);       

        % Perform 2D cross-correlation
        correlation_matrix = normxcorr2(known_sound, P);

        % Find indices above the threshold
        above_seuil_indices = find(correlation_matrix > seuil);

        % If no correlations are above threshold, skip
        if isempty(above_seuil_indices)
            fprintf(fileID, 'Processed file: %s\n', current_file);
            fprintf(fileID, 'Number of detections: 0\n\n');
            fprintf('File %d of %d processed: %s\n', k, num_files, current_file);
            fprintf('Detections in file: 0\n');
            continue;
        end
        
        % Extract correlation values and coordinates
        corr_values = correlation_matrix(above_seuil_indices);
        [f_offsets, t_offsets] = ind2sub(size(correlation_matrix), above_seuil_indices);
        
        % Adjust indices for correlation matrix size difference
        adjust_row = size(known_sound, 1) - 1;
        adjust_col = size(known_sound, 2) - 1;

        t_offsets = t_offsets - adjust_col;
        f_offsets = f_offsets - adjust_row;

        % Filter valid indices
        valid_t_indices = t_offsets > 0 & t_offsets <= length(T);
        valid_f_indices = f_offsets > 0 & f_offsets <= length(F);
        valid_indices = valid_t_indices & valid_f_indices;

        % Keep only valid indices
        t_offsets = t_offsets(valid_indices);
        f_offsets = f_offsets(valid_indices);
        corr_values = corr_values(valid_indices);

        % Convert indices to time and frequency
        t_coords = T(t_offsets);
        f_coords = F(f_offsets);

        % Remove duplicates, rounding time to 2 decimals
        rounded_t_coords = round(t_coords, 2);
        [unique_t_coords, ~, ic] = unique(rounded_t_coords);
        
        % Initialize max correlation values and frequencies
        max_corr_values = zeros(size(unique_t_coords));
        max_frequencies = zeros(size(unique_t_coords));

        % Max correlation and frequency per unique time
        for i = 1:length(unique_t_coords)
            valid_indices = (ic == i);
            if any(valid_indices)
                max_corr_values(i) = max(corr_values(valid_indices));
                max_frequencies(i) = mean(f_coords(valid_indices));
            else
                max_corr_values(i) = 0;
                max_frequencies(i) = NaN;
            end
        end

        % Filter to retain maximum correlation detections within 2 seconds
        filtered_t_coords = [];
        filtered_max_corr_values = [];
        filtered_frequencies = [];

        i = 1;
        while i <= length(unique_t_coords)
            current_time = unique_t_coords(i);
            within_2sec_indices = find(unique_t_coords > current_time & unique_t_coords <= current_time + 2);
            within_2sec_indices = [i, within_2sec_indices];
            
            [~, max_idx] = max(max_corr_values(within_2sec_indices));
            filtered_t_coords(end + 1) = unique_t_coords(within_2sec_indices(max_idx));
            filtered_max_corr_values(end + 1) = max_corr_values(within_2sec_indices(max_idx));
            filtered_frequencies(end + 1) = max_frequencies(within_2sec_indices(max_idx));
            
            i = within_2sec_indices(end) + 1;
        end

        % Update total detections count
        total_detections = total_detections + length(filtered_t_coords);

        % Write results to output file
        fprintf(fileID, 'Processed file: %s\n', current_file);
        fprintf(fileID, 'Number of detections: %d\n', length(filtered_t_coords));
        fprintf(fileID, 'Detections:\n');
        for i = 1:length(filtered_t_coords)
            fprintf(fileID, 'Detection %d: Time = %.2f s, Frequency = %.2f Hz, Correlation = %.2f\n', ...
                i, filtered_t_coords(i), filtered_frequencies(i), filtered_max_corr_values(i));
        end
        fprintf(fileID, '\n');
    end

    % Final output for each known sound
    fprintf(fileID, 'Total detections for known sound %d: %d\n', n, total_detections);
    fclose(fileID);
    fprintf('Total detections for known sound %d in all files: %d\n', n, total_detections);
end 