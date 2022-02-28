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

glob_instance_types = ['m5', 'm5a',  'm6g' , 'c5', 'c5a', 'c6g']
glob_cpu_limits = [str(limit) for limit in range(250, 2250, 250)]
# mem_limits = [str(limit) for limit in range(256, 1280, 256)]
glob_mem_limits = [ '128', '256', '512', '768', '1024', '2048' ]

# base_configs = [instance_types, cpu_limits, mem_limits]

"""
For the Bayesian Optimizer we have a model for each of the data size bucket for a function
"""
class Models(BaseOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.models = dict()
        self.trim_on_fail = True
        self.n_initial_points = {}
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

    # def _config_to_X(self, function, config):
    #     # print(config)
    #     X = [
    #         instance_types.index(config["instance_type"]),
    #         cpu_limits.index(config["cpu"]),
    #         mem_limits.index(config["memory"])
    #         ]

    #     return X 

    """
    Modify the model to cut the search space. 
    This also means retraining the new model on previous samples
    """
    def _modify_model(self, function, search_space, marker='all'):
        # data_size_buckets = self.data_size_buckets[function]
        if marker != 'all':
            self.configs[function][marker] = search_space
            model = Optimizer(self.configs[function][marker], base_estimator=self.base_estimator,
            n_initial_points=self.n_initial_points[function], acq_optimizer="sampling",
            acq_func='EI', acq_func_kwargs={},
            random_state=self.random_state[function]
            #acq_optimizer_kwargs={'n_points': 100}
            )
            self.models[function][marker] = model
            for conf, result in zip(self.tested_configs[function][marker], self.tested_performance[function][marker]):
                conf = [conf['instance_type'], conf['cpu'], conf['memory']]
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
            n_initial_points=self.n_initial_points[function], acq_optimizer="sampling",
            acq_func='EI', acq_func_kwargs={},
            random_state=self.random_state[function]
            #acq_optimizer_kwargs={'n_points': 100}
            )
            self.models[function][marker] = model

            for conf, result in zip(self.tested_configs[function][marker], self.tested_performance[function][marker]):
                
                conf = [conf['instance_type'], conf['cpu'], conf['memory']]
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
  
        # with open('models.pkl', 'wb') as fp:
        #     pickle.dump(self.models, fp)   


    def _create_model(self, function, data_size_buckets, tested_configs, tested_performance, configs, models, best_configs, search_space, seed=None, n_initial_points=3):
        self.data_size_buckets[function] = data_size_buckets
        self.tested_configs[function] = tested_configs
        self.tested_performance[function] = tested_performance
        self.configs[function] = configs
        self.models[function] = models
        self.best_configs[function] = best_configs
        self.random_state[function] = seed
        self.n_initial_points[function] = n_initial_points

        if data_size_buckets != None:
            for marker in data_size_buckets:
                self.configs[function][marker] = search_space
                model = Optimizer(self.configs[function][marker], base_estimator=self.base_estimator,
                n_initial_points=n_initial_points, acq_optimizer="sampling",
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
            n_initial_points=n_initial_points, acq_optimizer="sampling",
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
        # with open('models.pkl', 'wb') as fp:
        #     pickle.dump(self.models, fp)

    def create_model(self, function, data_size_buckets=None, base_estimator='GP', trim_on_fail = True, seed=None,
                     n_initial_points=3, instance_types=glob_instance_types, cpu_limits=glob_cpu_limits, mem_limits=glob_mem_limits):
        self.base_estimator = base_estimator
        self.trim_on_fail = trim_on_fail
        cpu_limits = [x.strip('m') for x in  cpu_limits]
        mem_limits = [x.strip('Mi') for x in  mem_limits]
        search_space = [
                    Categorical(instance_types),
                    Categorical(cpu_limits),
                    Categorical(mem_limits),
                ]
        self._create_model(function, data_size_buckets, {}, {}, {}, {}, {}, search_space, seed=seed, n_initial_points=n_initial_points)

    def update(self, config, result, function, marker='all', success=True):
        # If the result was because of a failure of function rather than timeout or success
        # Update the search space to exclude configurations 
        if result == 10000.0 and self.trim_on_fail:
            # print(config)
            instance_types = self.configs[function][marker][0].categories
            cpu_limits = self.configs[function][marker][1].categories
            mem_limits = self.configs[function][marker][2].categories
            # print(instance_types, cpu_limits, mem_limits)
            # print(config['memory'])
            if config['memory'] not in mem_limits:
                return 
            index = mem_limits.index(config['memory'])
            search_space = [
                    Categorical(instance_types), # Instance types
                    Categorical(cpu_limits), # Cpus
                    Categorical(mem_limits[index+1:]), # Memory
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

            # ic(config)

            model = self.models[function][marker]
            try:
                model.tell(config, result)
            except ValueError:
                print('Value Error')
                pass 
            self.models[function][marker] = model
            
            # with open('models.pkl', 'wb') as fp:
            #     pickle.dump(self.models, fp)

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

        # ic(next_x)
        return  {'instance_type': next_x[0], 'cpu': next_x[1],'memory': next_x[2]}

    def get_top_across_types(self, function, marker='all'):
        model = self.models[function][marker]
        _instance_types = self.configs[function][marker][0].categories
        _cpu =  self.configs[function][marker][1].categories
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

                _instance_results.append((Y[0], config))

            _instance_results.sort(key=operator.itemgetter(0))

            _results.append(_instance_results[0])
        
        _results.sort(key=operator.itemgetter(0))
        
        return _results

    def get_top_k_configs(self, function, marker='all', k=5):
        model = self.models[function][marker]
        _instance_types = self.configs[function][marker][0].categories
        _cpu =  self.configs[function][marker][1].categories
        _memory =  self.configs[function][marker][2].categories
        # print(_memory[0])
        # This picks the smallest memory size in the search space
        _configs = list(itertools.product(*[_instance_types, _cpu, [_memory[0]] ]))
        # print(_configs)
        _results = []
        for config in _configs:
            transformed_conf = model.space.transform([config])
            Y = model.models[-1].predict(transformed_conf)

            _results.append((Y[0], config))

        _results.sort(key=operator.itemgetter(0))
        return _results[:k]

    def get_all_config_predictions(self, function, marker='all'):
        model = self.models[function][marker]
        _instance_types = self.configs[function][marker][0].categories
        _cpu =  self.configs[function][marker][1].categories
        _memory =  self.configs[function][marker][2].categories
        # print(_memory[0])
        # This picks the smallest memory size in the search space
        _configs = list(itertools.product(*[_instance_types, _cpu, _memory ]))
        # print(_configs)
        _results = []
        for config in _configs:
            transformed_conf = model.space.transform([config])
            Y = model.models[-1].predict(transformed_conf)

            _results.append((Y[0], config))

        # _results.sort(key=operator.itemgetter(0))
        return _results        

######################################
## THE FOLLOWING CODE IS DEPRECATED ##
class boSkOpt():
    def __init__(self, function_name, types, cpu_allocs,
                            memory_allocs, objective_function, points_to_evaluate=[], optimizer='GP', acquisition_method='EI',
                                                initial_samples=3, seed=1, acq_kwargs={}):
        # super(boSkOpt, self).__init__(app, system, datasize, budget, parent_dir, types, sizes, number_of_nodes, objective_function, points_to_evaluate)

        self.cpu_allocs = cpu_allocs
        self.memory_allocs = memory_allocs
        self.function_name = function_name 
        self.types = types 
        self.objective_function = objective_function
        self.points_to_evaluate = points_to_evaluate

        self.domain = [
                Categorical(self.types),
                Categorical(self.cpu_allocs),
                Categorical(self.memory_allocs)
                # Integer(0, len(self.cpu_allocs)-1),
                # Integer(0, len(self.memory_allocs)-1)
        ]
        self.optimizer = optimizer
        self.seed = seed
        self.initial_samples = initial_samples
        self.acquisition_method = acquisition_method
        self.uuid = uuid.uuid4().hex
        self.trialsFile = 'trials-'+self.uuid+'.pickle'
        self.acq_kwargs = acq_kwargs
        

    def convertToConfig(self, x):
        # x = bounds(x)
        type = x[0]
        size = self.sizes[int(round(x[1]))]
        index = int(round(x[2])) % len(self.number_of_nodes[size])
        num = self.number_of_nodes[size][index]
        return type, size, num

    def getObjectiveValue(self, x):
        # print(x)
        type, size, num = self.convertToConfig(x)
        dir = self.parent_dir + str(num) + '_'+ type+'.'+size+ '_'+ self.app + "_" +self.system + "_" + self.datasize + "_1/"
        jsonName= dir + 'report.json'
        objective_value = self.objective_function(jsonName, type, size, num)
        t = {'params': {'type': type,'size': size,'num': num}, 'value': objective_value}
        updatePickle(t, filename=self.trialsFile)
        return objective_value

    def convert_points(self, points_to_evaluate):
        pte = list()
        for point in points_to_evaluate:
            p = list()
            p.append(point["type"])
            p.append(self.sizes.index(point["size"]))
            p.append(self.number_of_nodes[point["size"]].index(point["num"]))
            pte.append(p)
        return pte

    def runOptimizer(self):
        # if self.optimizer=='gp':
        #     res = gp_minimize(self.getRuntime, self.domain, n_calls=self.budget,
        #                 n_random_starts=self.initial_samples)
        # elif self.optimizer=='gbrt':
        #     res = gbrt_minimize(self.getRuntime, self.domain, n_calls=self.budget,
        #             n_random_starts=self.initial_samples)
        # elif self.optimizer=='forest':
        #     res = forest_minimize(self.getRuntime, self.domain, n_calls=self.budget,
        #             n_random_starts=self.initial_samples)
        opt = Optimizer(self.domain, base_estimator=self.optimizer,
                n_random_starts=self.initial_samples, acq_optimizer="sampling",
                acq_func=self.acquisition_method, acq_func_kwargs=self.acq_kwargs
                #acq_optimizer_kwargs={'n_points': 100}
                )
        count = 0
        trails = list()
        results = list()
        min_x = list()
        min_val = 10000
        
        pte = self.convert_points(self.points_to_evaluate)
        # print("Evaluating initial points")
        for point in pte:
            f_val = self.getObjectiveValue(point)
            count +=1 
            if f_val < min_val:
                min_val = f_val
                min_x = point
            trails.append(point)
            results.append(f_val)
            opt.tell(point, f_val)

        # print("Doing optimization runs")
        while count < self.budget:
            next_x = opt.ask()
            if next_x not in trails:
                f_val = self.getObjectiveValue(next_x)
                count +=1
                if f_val < min_val:
                    min_val = f_val
                    min_x = next_x
                trails.append(next_x)
                results.append(f_val)
            else:
                f_val = results[trails.index(next_x)]
            opt.tell(next_x, f_val)

        best_parameters = dict()
        best_parameters['type'], best_parameters['size'], best_parameters['num'] = self.convertToConfig(min_x)
        print(min_val, best_parameters)
        trials = pickleRead(self.trialsFile)
        return trials
