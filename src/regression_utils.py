def z_score_scale(values):
    """
    Applies z-score normalization to a list of numeric values.

    Args:
        values (list[float]): List of values to normalize.

    Returns:
        list[float]: Z-score normalized values, with mean 0 and std deviation 1.
                     If standard deviation is 0, returns all zeros.
    """
    mean_val = sum(values) / len(values)
    variance = sum((v - mean_val) ** 2 for v in values) / len(values)
    stddev = variance ** 0.5
    if stddev == 0:  # prevents division by zero
        return [0.0 for _ in values]
    return [(v - mean_val) / stddev for v in values]

def min_max_scale(values):
    """
    Applies min-max scaling to a list of numeric values to map them into [0, 1].

    Args:
        values (list[float]): List of values to scale.

    Returns:
        list[float]: Min-max scaled values between 0 and 1.
                     If all values are the same, returns 0.5 for all entries.
    """
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.5 for _ in values]  # avoid divide-by-zero
    return [(v - min_val) / (max_val - min_val) for v in values]

def scaled_metric(values):
    """
    Applies z-score normalization followed by min-max scaling to input values.

    Args:
        values (list[float]): List of numeric values.

    Returns:
        list[float]: Final scaled values in [0, 1] range, combining z-score and min-max.
    """
    z_score_scaled = z_score_scale(values)
    return min_max_scale(z_score_scaled)