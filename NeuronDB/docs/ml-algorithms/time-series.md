# Time Series

Forecast and analyze time series data.

## Time Series Forecasting

Predict future values:

```sql
-- ARIMA forecasting
SELECT arima_forecast(
    'time_series_table',
    'value_column',
    'timestamp_column',
    10  -- forecast horizon
);
```

## Moving Average

Calculate moving averages:

```sql
-- Simple moving average
SELECT timestamp,
       value,
       moving_average(value, 7) OVER (ORDER BY timestamp) AS ma_7
FROM time_series;
```

## Exponential Smoothing

```sql
-- Exponential smoothing forecast
SELECT exponential_smoothing_forecast(
    'time_series_table',
    'value',
    'timestamp',
    10,
    0.3  -- alpha (smoothing factor)
);
```

## Trend Detection

```sql
-- Detect trends
SELECT detect_trend(
    'time_series_table',
    'value',
    'timestamp'
) AS trend_direction;  -- 'up', 'down', 'stable'
```

## Learn More

For detailed documentation on time series analysis, forecasting models, seasonality detection, and anomaly detection in time series, visit:

**[Time Series Documentation](https://neurondb.ai/docs/ml/timeseries/)**

## Related Topics

- [Outlier Detection](outlier-detection.md) - Detect anomalies in time series
- [Drift Detection](drift-detection.md) - Detect distribution changes

