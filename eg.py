class DataFilter:
    def filter_data(self, data, feature, targeted_feature_value):
        filtered_data = {key: [] for key in data.keys() if key != feature}

        target_feature = data.get(feature, [])

        for i, value in enumerate(target_feature):
            if value == targeted_feature_value:
                for key in filtered_data:
                    filtered_data[key].append(i)

        return filtered_data

# Example usage
if __name__ == "__main__":
    data = {
        'feature1': [1, 2, 3, 2, 1],
        'feature2': ['a', 'b', 'c', 'd', 'e']
    }

    filter_obj = DataFilter()
    target_value = 2
    filtered_result = filter_obj.filter_data(data, 'feature1', target_value)
    print(filtered_result)