def kolmogorov_smirnov_test(historical_data, current_data):
    from scipy import stats

    ks_statistic, p_value = stats.ks_2samp(historical_data, current_data)
    alpha = 0.05
    drift_detected = p_value < alpha
    return {
        "ks_statistic": ks_statistic,
        "p_value": p_value,
        "drift_detected": drift_detected,
    }
