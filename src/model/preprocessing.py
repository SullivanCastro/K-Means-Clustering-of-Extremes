import numpy as np


class Preprocessing():

    @staticmethod
    def transform_to_extreme_values(data):
        extreme_data = (data[:, None] >= data).mean(axis=1)
        return 1/(extreme_data)
    

    @staticmethod
    def filter_largest(data, threshold):
        pass


    @staticmethod
    def project_onto_unit_sphere(data):
        pass


    @staticmethod
    def process(data, threshold):
        data = Preprocessing.transform_to_extreme_values(data)
        data = Preprocessing.filter_largest(data)
        data = Preprocessing.project_onto_unit_sphere(data)
        return data

    

if __name__ == "__main__":
    data = np.array([1.1, 1.2, 1.3, 9, 3, 4, 5])
    extreme_data = Preprocessing.transform_to_extreme_values(data)
    print(extreme_data)