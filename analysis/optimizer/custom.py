from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pickle
import itertools
import numpy as np 
from icecream import ic
from .optimizer import BaseOptimizer


instance_types = ['m5', 'm5a',  'm6g' , 'c5', 'c5a', 'c6g']
cpu_limits = [str(limit) for limit in range(250, 2250, 250)]
mem_limits = [str(limit) for limit in range(256, 1280, 256)]
base_configs = [instance_types, cpu_limits, mem_limits]

# TODO: There should be multiple models available for the optimizer
class Models(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.models = dict()
        self.search_spaces = {}
        self.tested_configs = {}
        self.tested_performance = {}
        self.data_dependence = {}
        self.configs = {}
        # Per function best configurations. Each function can have multiple best configurations
        # TODO: Configurations with ranking when slightly suboptimal cases are acceptable?
        self.best_configs = {}
    
    def _config_to_X(self, function, config):
        # print(config)
        X = [
            instance_types.index(config["instance_type"]),
            cpu_limits.index(config["cpu"]),
            mem_limits.index(config["memory"])
            ]
        if self.data_dependence[function]:
            bucket = self.configs[function][-1]
            X.append(bucket.index(config["marker"]))

        return X 

    def create_model(self, function, data_size_buckets=None):
        clf = linear_model.LinearRegression()
        self.models[function] = clf
        self.tested_configs[function] = []
        self.tested_performance[function] = []
        self.configs[function] = base_configs
        if data_size_buckets != None:
            self.configs[function].extend(data_size_buckets)
            self.data_dependence[function] = True
        else:
            self.data_dependence[function] = False

        # ic(self.configs[function])

        self.search_spaces[function] = list(itertools.product(*self.configs[function]))
        with open('models.pkl', 'wb') as fp:
            pickle.dump(self.models, fp)

    def update(self, config, result, function, success):
        self.tested_configs[function].append(config)
        self.tested_performance[function].append(result)

        poly = PolynomialFeatures(degree=2)
        
        configs = [self._config_to_X(function, config) for config in self.tested_configs[function]]
        X_ = poly.fit_transform(configs)

        clf = self.models[function]
        clf.fit(X_, self.tested_performance[function])
        self.models[function] = clf
        
        pickle.dump(self.models, open('models.pkl', 'wb'))

    def get_best_config(self, function) -> list:
        best_configs = self.best_configs[function]
        return best_configs

    def get_next_config(self, function, bucket=""):
        # self.models = pickle.load(open('models.pkl', 'rb'))
        return self.find_best(function, bucket)


    def find_best(self, function, bucket):
        model = self.models[function]
        poly = PolynomialFeatures(degree=2)
        minY = np.Inf
        minConfig = {}
        for config in self.search_spaces[function]:

            if self.data_dependence[function]:
                # If the configuration doesn't correspond 
                if config[3] != bucket:
                    continue

                conf = {'instance_type': config[0], 'cpu': config[1],'memory': config[2], 'marker': config[3]}
            else:
                conf = {'instance_type': config[0], 'cpu': config[1],'memory': config[2]}

            predict_ = poly.fit_transform([self._config_to_X(function, conf)])
            Y = model.predict(predict_)

            if Y[0] < minY:
                minConfig = conf
                minY = Y[0]

        return minConfig 