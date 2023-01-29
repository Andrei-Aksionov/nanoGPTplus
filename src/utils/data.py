def train_test_split(data, train_fraction):
    test_split = int(len(data) * train_fraction)
    return data[:test_split], data[test_split:]
