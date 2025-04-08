class ValuePrefixer:
    """
        Add colname to values like a prefix
    """

    @staticmethod
    def transform(data, col_name:str):
        return data[col_name].apply(lambda x: f"{col_name}_{x}")

    @staticmethod
    def inverse_transform(data, col_name):
        return data[col_name].apply(lambda x: x.replace(f"{col_name}_", ""))

    @staticmethod
    def transform_all(data, exclude=[], include=[]):
        for col in data.columns:
            if col not in exclude and (len(include) == 0 or col in include):
                data[col] = ValuePrefixer.transform(col)
        return data

    @staticmethod
    def inverse_transform_all(data, exclude=[], include=[]):
        for col in data.columns:
            if col not in exclude and (len(include) == 0 or col in include):
                data[col] = ValuePrefixer.inverse_transform(col)
        return data

