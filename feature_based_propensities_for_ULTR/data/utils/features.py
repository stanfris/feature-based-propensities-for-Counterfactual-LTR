def parse_feature_selection(features, total_columns):
    """
    Parses a feature description into a list of column ids:
    features="all"    # Selects all features
    features="1,3,5"  # Selects columns 1, 3, and 5
    features="2-6"    # Selects columns 2 to 6 (inclusive start and end)
    features="1-4,6,8-10" # Mixtures of specific columns and ranges are possible
    """
    if features == "all":
        return list(range(total_columns))

    indices = set()

    for part in features.split(","):
        if "-" in part:
            start, end = map(int, part.split("-"))
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))

    return sorted(indices)
