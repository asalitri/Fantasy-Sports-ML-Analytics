def z_score_scale(values):
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    stddev = variance ** 0.5
    if stddev == 0:  # prevents division by zero
        return [0.0 for _ in values]
    return [(v - mean_val) / stddev for v in values]

def min_max_scale(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5 for _ in values]  # avoid divide-by-zero
    return [(v - min_val) / (max_val - min_val) for v in values]

def scaled_metric(values):
    z_score_scaled = z_score_scale(values)
    return min_max_scale(z_score_scaled)