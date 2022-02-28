import numpy as np
import os
import json
import sys
from skopt import gp_minimize, gbrt_minimize, forest_minimize, Optimizer
from skopt.space import Real, Integer, Categorical
import uuid
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import pickle
import itertools
import numpy as np 
from icecream import ic
from .optimizer import BaseOptimizer
import operator

instance_types = ['m5', 'm5a',  'm6g' , 'c5', 'c5a', 'c6g']
cpu_limits = [str(limit) for limit in range(250, 2250, 250)]
# mem_limits = [str(limit) for limit in range(256, 1280, 256)]
mem_limits = [ '128', '256', '512', '768', '1024', '2048' ]

# base_configs = [instance_types, cpu_limits, mem_limits]

"""
For the Bayesian Optimizer we have a model for each of the data size bucket for a function
"""
class Models(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.models = dict()
        self.trim_on_fail = True
        self.search_spaces = {}
        self.tested_configs = {}
        self.tested_performance = {}
        self.data_dependence = {}
        self.configs = {}
        self.data_size_buckets = {}
        self.random_state = {}
        self.base_estimator = 'GP'
        # Per function best configurations. Each function can have multiple best configurations
        # TODO: Configurations with ranking when slightly suboptimal cases are acceptable?
        self.best_configs = {}

    def _transform_config(self, config):
        # print(config)
        config = (config[0], cpu_limits[config[1]-1], config[2])
        return config

    def _transform_config_to_X(self, config):
        # print(config)
        config[1] = cpu_limits.index(config[1])+1
        return config

    """
    Modify the model to cut the search space. 
    This also means retraining the new model on previous samples
    """
    def _modify_model(self, function, search_space, marker='all'):
        # data_size_buckets = self.data_size_buckets[function]
        if marker != 'all':
            self.configs[function][marker] = search_space
            model = Optimizer(self.configs[function][marker], base_estimator=self.base_estimator,
            n_random_starts=3, acq_optimizer="sampling",
            acq_func='EI', acq_func_kwargs={},
            random_state=self.random_state[function]
            #acq_optimizer_kwargs={'n_points': 100}
            )
            self.models[function][marker] = model
            for conf, result in zip(self.tested_configs[function][marker], self.tested_performance[function][marker]):
                conf = [conf['instance_type'], conf['cpu'], conf['memory']]
                conf = self._transform_config_to_X(conf)
                try:
                    model.tell(conf, result)
                # This exception is related to the peculiar behavior of transcode benchmark
                except ValueError:
                    # print('Point not in the search space')
                    pass

            self.models[function][marker] = model

            # print('Search Space Trimmed')
            # print(self.configs[function][marker])

        else:
            # marker = 'all'
            self.configs[function][marker] = search_space
            model = Optimizer(self.configs[function][marker], base_estimator=self.base_estimator,
            n_random_starts=3, acq_optimizer="sampling",
            acq_func='EI', acq_func_kwargs={},
            random_state=self.random_state[function]
            #acq_optimizer_kwargs={'n_points': 100}
            )
            self.models[function][marker] = model

            for conf, result in zip(self.tested_configs[function][marker], self.tested_performance[function][marker]):
                
                conf = [conf['instance_type'], conf['cpu'], conf['memory']]
                conf = self._transform_config_to_X(conf)
                try:
                    model.tell(conf, result)
                # This exception is related to the peculiar behavior of transcode benchmark
                except ValueError:
                    # print('Point not in the search space')
                    pass
                    
            self.models[function][marker] = model

            # print('Search Space Trimmed')
            # print(self.configs[function][marker])


        # ic(self.configs[function])
        # self.search_spaces[function] = list(itertools.product(*self.configs[function]))
        with open('models.pkl', 'wb') as fp:
            pickle.dump(self.models, fp)   


    def _create_model(self, function, data_size_buckets, tested_configs, tested_performance, configs, models, best_configs, search_space, seed=None):
        self.data_size_buckets[function] = data_size_buckets
        self.tested_configs[function] = tested_configs
        self.tested_performance[function] = tested_performance
        self.configs[function] = configs
        self.models[function] = models
        self.best_configs[function] = best_configs
        self.random_state[function] = seed

        if data_size_buckets != None:
            for marker in data_size_buckets:
                self.configs[function][marker] = search_space
                model = Optimizer(self.configs[function][marker], base_estimator=self.base_estimator,
                n_random_starts=3, acq_optimizer="sampling",
                acq_func='EI', acq_func_kwargs={},
                random_state=self.random_state[function]
                #acq_optimizer_kwargs={'n_points': 100}
                )
                self.models[function][marker] = model
                self.tested_configs[function][marker] = []
                self.tested_performance[function][marker] = []
                self.best_configs[function][marker] = {} 

            self.data_dependence[function] = True

        else:
            self.configs[function]['all'] = search_space
            model = Optimizer(self.configs[function]['all'], base_estimator=self.base_estimator,
            n_random_starts=3, acq_optimizer="sampling",
            acq_func='EI', acq_func_kwargs={},
            random_state=self.random_state[function]
            #acq_optimizer_kwargs={'n_points': 100}
            )
            self.models[function]['all'] = model
            self.tested_configs[function]['all'] = []
            self.tested_performance[function]['all'] = []
            self.best_configs[function]['all'] = {} 

            self.data_dependence[function] = False

        # ic(self.configs[function])

        # self.search_spaces[function] = list(itertools.product(*self.configs[function]))
        with open('models.pkl', 'wb') as fp:
            pickle.dump(self.models, fp)

    def create_model(self, function, data_size_buckets=None, base_estimator='GP', trim_on_fail = True, seed=None):
        self.base_estimator = base_estimator
        self.trim_on_fail = trim_on_fail
        search_space = [
                    Categorical(instance_types),
                    Integer(1, len(cpu_limits)),
                    Categorical(mem_limits),
                ]
        self._create_model(function, data_size_buckets, {}, {}, {}, {}, {}, search_space, seed=seed)

    def update(self, config, result, function, marker='all', success=True):
        # If the result was because of a failure of function rather than timeout or success
        # Update the search space to exclude configurations 
        if result == 10000.0 and self.trim_on_fail:
            # print(config)
            index = mem_limits.index(config['memory'])
            search_space = [
                    Categorical(instance_types),
                    Integer(1, len(cpu_limits)),
                    Categorical(mem_limits[index+1:]),
                ]
            # print(search_space)
            # self._create_model(function, self.data_size_buckets[function], self.tested_configs[function],
            #          self.tested_performance[function], self.configs[function], self.models[function], 
            #          self.best_configs[function], search_space)
            self._modify_model(function, search_space, marker)

        else:
            self.tested_configs[function][marker].append(config)
            self.tested_performance[function][marker].append(result)

            # ic(config)
            # ic(result)

            config = [config['instance_type'], config['cpu'], config['memory']]
            config = self._transform_config_to_X(config)
            # ic(config)    

            model = self.models[function][marker]
            try:
                model.tell(config, result)
            except ValueError:
                print('Value Error')
                pass 
            self.models[function][marker] = model
            
            with open('models.pkl', 'wb') as fp:
                pickle.dump(self.models, fp)

    def get_best_config(self, function) -> list:
        best_configs = self.best_configs[function]
        return best_configs

    def get_next_config(self, function, marker='all'):
        # self.models = pickle.load(open('models.pkl', 'rb'))
        return self.find_best(function, marker)


    def find_best(self, function, marker):
        model = self.models[function][marker]
        # print(self.configs[function][marker])

        next_x = model.ask()
        
        next_x = self._transform_config(next_x)
        # ic(next_x)
        return  {'instance_type': next_x[0], 'cpu': next_x[1],'memory': next_x[2]}

    def get_top_across_types(self, function, marker='all'):
        model = self.models[function][marker]
        _instance_types = self.configs[function][marker][0].categories
        # _cpu =  self.configs[function][marker][1].bounds
        _cpu = list(range(1, len(cpu_limits)+1))
        _memory =  self.configs[function][marker][2].categories

        k = len(_instance_types)
        _results = []

        for i in range(0, k):
            # print(_memory[0])
            # This picks the smallest memory size in the search space
            _configs = list(itertools.product(*[ [_instance_types[i]], _cpu, [_memory[0]] ]))
            # print(_configs)
            _instance_results = []
            for config in _configs:
                transformed_conf = model.space.transform([config])
                Y = model.models[-1].predict(transformed_conf)


                _instance_results.append((Y[0], self._transform_config(config)))

            _instance_results.sort(key=operator.itemgetter(0))

            _results.append(_instance_results[0])
        
        _results.sort(key=operator.itemgetter(0))
        
        return _results

    def get_top_k_configs(self, function, marker='all', k=5):
        model = self.models[function][marker]
        _instance_types = self.configs[function][marker][0].categories
        # _cpu =  self.configs[function][marker][1].categories
        _cpu = list(range(1, len(cpu_limits)+1))
        _memory =  self.configs[function][marker][2].categories
        # print(_memory[0])
        # This picks the smallest memory size in the search space
        _configs = list(itertools.product(*[_instance_types, _cpu, [_memory[0]] ]))
        # print(_configs)
        _results = []
        for config in _configs:
            transformed_conf = model.space.transform([config])
            Y = model.models[-1].predict(transformed_conf)

            _results.append((Y[0], self._transform_config(config)))

        _results.sort(key=operator.itemgetter(0))
        return _results[:k]

    def get_all_config_predictions(self, function, marker='all'):
        model = self.models[function][marker]
        _instance_types = self.configs[function][marker][0].categories
        # _cpu =  self.configs[function][marker][1].categories
        _cpu = list(range(1, len(cpu_limits)+1))
        _memory =  self.configs[function][marker][2].categories
        # print(_memory[0])
        # This picks the smallest memory size in the search space
        _configs = list(itertools.product(*[_instance_types, _cpu, _memory ]))
        # print(_configs)
        _results = []
        for config in _configs:
            transformed_conf = model.space.transform([config])
            Y = model.models[-1].predict(transformed_conf)

            _results.append((Y[0], self._transform_config(config)))

        # _results.sort(key=operator.itemgetter(0))
        return _results        

