from abc import ABC, abstractmethod


# instance_types = ['m5', 'm5a',  'm6g' , 'c5', 'c5a', 'c6g']
# cpu_limits = [str(limit) for limit in range(128, 2048, 128)]
# mem_limits = [str(limit) for limit in range(128, 2048, 128)]
# base_configs = [instance_types, cpu_limits, mem_limits]

# TODO: There should be multiple models available for the optimizer
class BaseOptimizer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def create_model(self, function, data_size_buckets=None):
        pass

    @abstractmethod
    def update(self, config, result, function, marker, success):
        pass
    
    @abstractmethod
    def get_best_config(self, function) -> list:
        return []

    @abstractmethod
    def get_next_config(self, function, marker=[]):
        pass

    @abstractmethod
    def find_best(self, function, marker):
        pass

    @abstractmethod
    def get_top_across_types(self, function, marker):
        pass

    @abstractmethod
    def get_top_k_configs(self, function, marker, k=5):
        pass

    @abstractmethod
    def get_all_config_predictions(self, function, marker='all'):
        pass
