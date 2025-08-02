function [processed_data, stats, forecast] = env_preprocess(raw_data, options)
% ENV_PREPROCESS Environmental data preprocessing and analysis
% Raspberry Pi 5 Federated Environmental Monitoring Network
%
% Inputs:
%   raw_data - Structure with fields: temperature, humidity, timestamp
%   options  - Structure with preprocessing options (optional)
%
% Outputs:
%   processed_data - Cleaned and processed data
%   stats         - Statistical analysis results
%   forecast      - Short-term forecast predictions

% Default options
if nargin < 2
    options = struct();
end

% Set default parameters
default_options = struct(...
    'window_size', 24, ...           % Hours for moving average
    'outlier_threshold', 3, ...      % Standard deviations for outlier detection
    'forecast_horizon', 6, ...       % Hours to forecast ahead
    'sampling_rate', 1, ...          % Samples per hour
    'enable_plots', false ...        % Generate diagnostic plots
);

% Merge with user options
field_names = fieldnames(default_options);
for i = 1:length(field_names)
    if ~isfield(options, field_names{i})
        options.(field_names{i}) = default_options.(field_names{i});
    end
end

fprintf('Starting environmental data preprocessing...\n');
fprintf('Data points: %d\n', length(raw_data.temperature));
fprintf('Time range: %.2f hours\n', (raw_data.timestamp(end) - raw_data.timestamp(1)) / 3600);

%% Data Validation and Cleaning
fprintf('Step 1: Data validation and cleaning\n');

% Extract data vectors
temp = raw_data.temperature(:);
humidity = raw_data.humidity(:);
timestamps = raw_data.timestamp(:);

% Remove NaN values
valid_idx = ~(isnan(temp) | isnan(humidity) | isnan(timestamps));
temp = temp(valid_idx);
humidity = humidity(valid_idx);
timestamps = timestamps(valid_idx);

fprintf('  Removed %d invalid data points\n', sum(~valid_idx));

% Outlier detection using modified Z-score
temp_outliers = abs(temp - median(temp)) > options.outlier_threshold * mad(temp, 1);
humidity_outliers = abs(humidity - median(humidity)) > options.outlier_threshold * mad(humidity, 1);

% Replace outliers with interpolated values
if sum(temp_outliers) > 0
    temp(temp_outliers) = interp1(find(~temp_outliers), temp(~temp_outliers), find(temp_outliers), 'linear', 'extrap');
    fprintf('  Corrected %d temperature outliers\n', sum(temp_outliers));
end

if sum(humidity_outliers) > 0
    humidity(humidity_outliers) = interp1(find(~humidity_outliers), humidity(~humidity_outliers), find(humidity_outliers), 'linear', 'extrap');
    fprintf('  Corrected %d humidity outliers\n', sum(humidity_outliers));
end

%% Signal Processing and Filtering
fprintf('Step 2: Signal processing and filtering\n');

% Design low-pass filter to remove high-frequency noise
fs = 1 / (median(diff(timestamps)) / 3600); % Sampling frequency in Hz
cutoff_freq = 1 / (2 * 3600); % Cut-off at 0.5 cycles per hour
[b, a] = butter(2, cutoff_freq / (fs/2), 'low');

% Apply filter
temp_filtered = filtfilt(b, a, temp);
humidity_filtered = filtfilt(b, a, humidity);

fprintf('  Applied low-pass filtering (cutoff: %.4f Hz)\n', cutoff_freq);

%% Feature Extraction
fprintf('Step 3: Feature extraction\n');

% Moving averages
window_samples = round(options.window_size * options.sampling_rate);
temp_ma = movmean(temp_filtered, window_samples);
humidity_ma = movmean(humidity_filtered, window_samples);

% Trend analysis using linear regression
time_hours = (timestamps - timestamps(1)) / 3600;
temp_trend = polyfit(time_hours, temp_filtered, 1);
humidity_trend = polyfit(time_hours, humidity_filtered, 1);

% Seasonal decomposition (simplified)
% Assume daily cycle (24 hours)
if length(temp_filtered) >= 48 % At least 2 days of data
    daily_samples = round(24 * options.sampling_rate);
    
    % Extract daily patterns
    n_complete_days = floor(length(temp_filtered) / daily_samples);
    if n_complete_days >= 2
        temp_daily = reshape(temp_filtered(1:n_complete_days*daily_samples), daily_samples, n_complete_days);
        humidity_daily = reshape(humidity_filtered(1:n_complete_days*daily_samples), daily_samples, n_complete_days);
        
        temp_seasonal = mean(temp_daily, 2);
        humidity_seasonal = mean(humidity_daily, 2);
    else
        temp_seasonal = [];
        humidity_seasonal = [];
    end
else
    temp_seasonal = [];
    humidity_seasonal = [];
end

%% Statistical Analysis
fprintf('Step 4: Statistical analysis\n');

stats = struct();

% Basic statistics
stats.temperature = struct(...
    'mean', mean(temp_filtered), ...
    'std', std(temp_filtered), ...
    'min', min(temp_filtered), ...
    'max', max(temp_filtered), ...
    'median', median(temp_filtered), ...
    'trend_slope', temp_trend(1) ...
);

stats.humidity = struct(...
    'mean', mean(humidity_filtered), ...
    'std', std(humidity_filtered), ...
    'min', min(humidity_filtered), ...
    'max', max(humidity_filtered), ...
    'median', median(humidity_filtered), ...
    'trend_slope', humidity_trend(1) ...
);

% Cross-correlation between temperature and humidity
[xcorr_vals, lags] = xcorr(temp_filtered - mean(temp_filtered), ...
                          humidity_filtered - mean(humidity_filtered), ...
                          min(50, floor(length(temp_filtered)/4)), 'coeff');
[max_corr, max_idx] = max(abs(xcorr_vals));
stats.cross_correlation = struct(...
    'max_correlation', max_corr, ...
    'lag_samples', lags(max_idx), ...
    'lag_hours', lags(max_idx) / options.sampling_rate ...
);

% Data quality metrics
stats.quality = struct(...
    'completeness', sum(valid_idx) / length(raw_data.temperature), ...
    'outlier_rate_temp', sum(temp_outliers) / length(temp), ...
    'outlier_rate_humidity', sum(humidity_outliers) / length(humidity), ...
    'sampling_regularity', std(diff(timestamps)) / mean(diff(timestamps)) ...
);

fprintf('  Temperature: %.2f°C ± %.2f°C (trend: %.3f°C/hour)\n', ...
        stats.temperature.mean, stats.temperature.std, stats.temperature.trend_slope);
fprintf('  Humidity: %.1f%% ± %.1f%% (trend: %.3f%%/hour)\n', ...
        stats.humidity.mean, stats.humidity.std, stats.humidity.trend_slope);
fprintf('  Cross-correlation: %.3f at lag %.1f hours\n', ...
        stats.cross_correlation.max_correlation, stats.cross_correlation.lag_hours);

%% Forecasting
fprintf('Step 5: Short-term forecasting\n');

forecast = struct();

if length(temp_filtered) >= 24 % Need at least 24 points for forecasting
    % Simple ARIMA-like forecasting using recent trend and seasonal patterns
    recent_window = min(72, length(temp_filtered)); % Last 72 hours or available data
    recent_temp = temp_filtered(end-recent_window+1:end);
    recent_humidity = humidity_filtered(end-recent_window+1:end);
    recent_time = time_hours(end-recent_window+1:end);
    
    % Fit polynomial trend to recent data
    temp_recent_trend = polyfit(recent_time - recent_time(1), recent_temp, 1);
    humidity_recent_trend = polyfit(recent_time - recent_time(1), recent_humidity, 1);
    
    % Generate forecast timestamps
    forecast_hours = (1:options.forecast_horizon)';
    forecast_timestamps = timestamps(end) + forecast_hours * 3600;
    
    % Basic trend extrapolation
    temp_forecast = recent_temp(end) + temp_recent_trend(1) * forecast_hours;
    humidity_forecast = recent_humidity(end) + humidity_recent_trend(1) * forecast_hours;
    
    % Add seasonal component if available
    if ~isempty(temp_seasonal)
        % Find corresponding seasonal indices
        current_hour_of_day = mod(time_hours(end), 24);
        forecast_hours_of_day = mod(current_hour_of_day + forecast_hours, 24);
        seasonal_indices = round(forecast_hours_of_day * length(temp_seasonal) / 24) + 1;
        seasonal_indices(seasonal_indices > length(temp_seasonal)) = length(temp_seasonal);
        seasonal_indices(seasonal_indices < 1) = 1;
        
        % Apply seasonal adjustment
        temp_seasonal_adj = temp_seasonal(seasonal_indices) - mean(temp_seasonal);
        humidity_seasonal_adj = humidity_seasonal(seasonal_indices) - mean(humidity_seasonal);
        
        temp_forecast = temp_forecast + 0.3 * temp_seasonal_adj; % Weighted seasonal component
        humidity_forecast = humidity_forecast + 0.3 * humidity_seasonal_adj;
    end
    
    % Add uncertainty bounds (simple approach)
    temp_std_recent = std(recent_temp);
    humidity_std_recent = std(recent_humidity);
    
    forecast.timestamps = forecast_timestamps;
    forecast.temperature = struct(...
        'mean', temp_forecast, ...
        'upper_bound', temp_forecast + 1.96 * temp_std_recent, ...
        'lower_bound', temp_forecast - 1.96 * temp_std_recent ...
    );
    forecast.humidity = struct(...
        'mean', humidity_forecast, ...
        'upper_bound', humidity_forecast + 1.96 * humidity_std_recent, ...
        'lower_bound', humidity_forecast - 1.96 * humidity_std_recent ...
    );
    
    fprintf('  Generated %d-hour forecast\n', options.forecast_horizon);
    fprintf('  Predicted temperature range: %.1f - %.1f°C\n', ...
            min(forecast.temperature.lower_bound), max(forecast.temperature.upper_bound));
    fprintf('  Predicted humidity range: %.1f - %.1f%%\n', ...
            min(forecast.humidity.lower_bound), max(forecast.humidity.upper_bound));
else
    fprintf('  Insufficient data for forecasting (need ≥24 points)\n');
    forecast.timestamps = [];
    forecast.temperature = struct('mean', [], 'upper_bound', [], 'lower_bound', []);
    forecast.humidity = struct('mean', [], 'upper_bound', [], 'lower_bound', []);
end

%% Prepare Output
processed_data = struct(...
    'timestamps', timestamps, ...
    'temperature', struct(...
        'raw', temp, ...
        'filtered', temp_filtered, ...
        'moving_average', temp_ma, ...
        'seasonal', temp_seasonal ...
    ), ...
    'humidity', struct(...
        'raw', humidity, ...
        'filtered', humidity_filtered, ...
        'moving_average', humidity_ma, ...
        'seasonal', humidity_seasonal ...
    ), ...
    'time_hours', time_hours ...
);

%% Generate Diagnostic Plots (if requested)
if options.enable_plots
    fprintf('Step 6: Generating diagnostic plots\n');
    
    figure('Name', 'Environmental Data Analysis', 'Position', [100, 100, 1200, 800]);
    
    % Time series plot
    subplot(2, 2, 1);
    plot(time_hours, temp, 'b.', 'MarkerSize', 4, 'DisplayName', 'Raw Temperature');
    hold on;
    plot(time_hours, temp_filtered, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Filtered Temperature');
    plot(time_hours, temp_ma, 'g-', 'LineWidth', 1, 'DisplayName', 'Moving Average');
    xlabel('Time (hours)');
    ylabel('Temperature (°C)');
    title('Temperature Analysis');
    legend('Location', 'best');
    grid on;
    
    subplot(2, 2, 2);
    plot(time_hours, humidity, 'b.', 'MarkerSize', 4, 'DisplayName', 'Raw Humidity');
    hold on;
    plot(time_hours, humidity_filtered, 'r-', 'LineWidth', 1.5, 'DisplayName', 'Filtered Humidity');
    plot(time_hours, humidity_ma, 'g-', 'LineWidth', 1, 'DisplayName', 'Moving Average');
    xlabel('Time (hours)');
    ylabel('Humidity (%)');
    title('Humidity Analysis');
    legend('Location', 'best');
    grid on;
    
    % Cross-correlation plot
    subplot(2, 2, 3);
    plot(lags / options.sampling_rate, xcorr_vals, 'b-', 'LineWidth', 1);
    xlabel('Lag (hours)');
    ylabel('Cross-correlation');
    title('Temperature-Humidity Cross-correlation');
    grid on;
    
    % Seasonal patterns (if available)
    subplot(2, 2, 4);
    if ~isempty(temp_seasonal)
        hour_of_day = (0:length(temp_seasonal)-1) * 24 / length(temp_seasonal);
        yyaxis left;
        plot(hour_of_day, temp_seasonal, 'r-', 'LineWidth', 2);
        ylabel('Temperature (°C)');
        yyaxis right;
        plot(hour_of_day, humidity_seasonal, 'b-', 'LineWidth', 2);
        ylabel('Humidity (%)');
        xlabel('Hour of Day');
        title('Daily Seasonal Patterns');
        grid on;
    else
        text(0.5, 0.5, 'Insufficient data for seasonal analysis', ...
             'HorizontalAlignment', 'center', 'Units', 'normalized');
        title('Seasonal Analysis');
    end
    
    % Add forecast if available
    if ~isempty(forecast.timestamps)
        subplot(2, 2, 1);
        hold on;
        forecast_hours = (forecast.timestamps - timestamps(1)) / 3600;
        plot(forecast_hours, forecast.temperature.mean, 'k--', 'LineWidth', 2, 'DisplayName', 'Forecast');
        fill([forecast_hours; flipud(forecast_hours)], ...
             [forecast.temperature.upper_bound; flipud(forecast.temperature.lower_bound)], ...
             'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', '95% CI');
        legend('Location', 'best');
        
        subplot(2, 2, 2);
        hold on;
        plot(forecast_hours, forecast.humidity.mean, 'k--', 'LineWidth', 2, 'DisplayName', 'Forecast');
        fill([forecast_hours; flipud(forecast_hours)], ...
             [forecast.humidity.upper_bound; flipud(forecast.humidity.lower_bound)], ...
             'k', 'FaceAlpha', 0.1, 'EdgeColor', 'none', 'DisplayName', '95% CI');
        legend('Location', 'best');
    end
    
    fprintf('  Diagnostic plots generated\n');
end

fprintf('Environmental data preprocessing completed successfully!\n');
fprintf('Processed %d data points over %.2f hours\n', length(temp_filtered), time_hours(end));

end
