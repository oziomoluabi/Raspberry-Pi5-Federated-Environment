function [processed_data, stats] = env_preprocess(raw_data, options)
% ENV_PREPROCESS Preprocess environmental sensor data
%
% Inputs:
%   raw_data - Matrix of environmental data [time x features]
%              Features: [temperature, humidity, pressure]
%   options  - Struct with preprocessing options (optional)
%
% Outputs:
%   processed_data - Preprocessed data matrix
%   stats         - Statistics struct with preprocessing info
%
% Example:
%   data = randn(100, 3);  % Simulate 100 samples, 3 features
%   [clean_data, stats] = env_preprocess(data);

% Default options
if nargin < 2
    options = struct();
end

% Set default values
if ~isfield(options, 'remove_outliers')
    options.remove_outliers = true;
end
if ~isfield(options, 'outlier_threshold')
    options.outlier_threshold = 3.0;
end
if ~isfield(options, 'normalize')
    options.normalize = true;
end
if ~isfield(options, 'smooth_data')
    options.smooth_data = true;
end
if ~isfield(options, 'smooth_window')
    options.smooth_window = 5;
end

fprintf('Starting environmental data preprocessing...\n');
fprintf('Input data shape: %d x %d\n', size(raw_data, 1), size(raw_data, 2));

% Initialize output
processed_data = raw_data;
stats = struct();

% Store original statistics
stats.original_mean = mean(raw_data, 1);
stats.original_std = std(raw_data, 1);
stats.original_min = min(raw_data, [], 1);
stats.original_max = max(raw_data, [], 1);

% Remove outliers using z-score method
if options.remove_outliers
    fprintf('Removing outliers (threshold: %.1f std)...\n', options.outlier_threshold);
    
    outlier_mask = false(size(processed_data));
    for col = 1:size(processed_data, 2)
        z_scores = abs(zscore(processed_data(:, col)));
        outlier_mask(:, col) = z_scores > options.outlier_threshold;
    end
    
    % Replace outliers with median values
    for col = 1:size(processed_data, 2)
        median_val = median(processed_data(~outlier_mask(:, col), col));
        processed_data(outlier_mask(:, col), col) = median_val;
    end
    
    stats.outliers_removed = sum(any(outlier_mask, 2));
    fprintf('Removed %d outlier samples\n', stats.outliers_removed);
end

% Smooth data using moving average
if options.smooth_data
    fprintf('Smoothing data (window size: %d)...\n', options.smooth_window);
    
    for col = 1:size(processed_data, 2)
        processed_data(:, col) = smooth(processed_data(:, col), options.smooth_window);
    end
end

% Normalize data (z-score normalization)
if options.normalize
    fprintf('Normalizing data...\n');
    
    stats.norm_mean = mean(processed_data, 1);
    stats.norm_std = std(processed_data, 1);
    
    processed_data = (processed_data - stats.norm_mean) ./ stats.norm_std;
    
    % Handle zero standard deviation
    zero_std_cols = stats.norm_std == 0;
    if any(zero_std_cols)
        processed_data(:, zero_std_cols) = 0;
        fprintf('Warning: Zero standard deviation in columns: %s\n', ...
                mat2str(find(zero_std_cols)));
    end
end

% Calculate final statistics
stats.final_mean = mean(processed_data, 1);
stats.final_std = std(processed_data, 1);
stats.final_min = min(processed_data, [], 1);
stats.final_max = max(processed_data, [], 1);

% Data quality metrics
stats.data_quality.completeness = sum(~isnan(processed_data(:))) / numel(processed_data);
stats.data_quality.num_samples = size(processed_data, 1);
stats.data_quality.num_features = size(processed_data, 2);

fprintf('Preprocessing completed successfully!\n');
fprintf('Final data shape: %d x %d\n', size(processed_data, 1), size(processed_data, 2));
fprintf('Data completeness: %.2f%%\n', stats.data_quality.completeness * 100);

% Optional: Create visualization
if nargout == 0
    figure('Name', 'Environmental Data Preprocessing Results');
    
    subplot(2, 2, 1);
    plot(raw_data);
    title('Original Data');
    xlabel('Time');
    ylabel('Value');
    legend({'Temperature', 'Humidity', 'Pressure'}, 'Location', 'best');
    grid on;
    
    subplot(2, 2, 2);
    plot(processed_data);
    title('Processed Data');
    xlabel('Time');
    ylabel('Normalized Value');
    legend({'Temperature', 'Humidity', 'Pressure'}, 'Location', 'best');
    grid on;
    
    subplot(2, 2, 3);
    histogram(raw_data(:, 1), 20, 'Alpha', 0.7);
    hold on;
    histogram(processed_data(:, 1), 20, 'Alpha', 0.7);
    title('Temperature Distribution');
    xlabel('Value');
    ylabel('Frequency');
    legend({'Original', 'Processed'}, 'Location', 'best');
    grid on;
    
    subplot(2, 2, 4);
    boxplot([raw_data, processed_data], ...
            'Labels', {'Temp (orig)', 'Hum (orig)', 'Press (orig)', ...
                      'Temp (proc)', 'Hum (proc)', 'Press (proc)'});
    title('Data Distribution Comparison');
    ylabel('Value');
    grid on;
    
    sgtitle('Environmental Data Preprocessing Analysis');
end

end
