import random

from skopt import Optimizer

from analysis.utils import get_result_from_data 
from analysis.optimizer.boskopt import Models

def get_random_samples(cpu_limits, memory_limits, instance_types, seed=1, n_samples=3):
    
    random_samples = []
    random.seed(seed)

    for i in range(0, n_samples):
        
        cpu = random.choice(cpu_limits)
        mem = random.choice(memory_limits)
        inst_type = random.choice(instance_types)
        random_samples.append({"instance_type": inst_type, "cpu": cpu.replace("m", ""), "memory": mem.replace("Mi", "")})

    return random_samples

def create_model(instance_types, cpu_limits, memory_limits, function_name, data_name, n_trials, base_estimator="GP", seed=1, n_init_samples=3):
    """
    Args:
        instance_types: list
            List of instance types to be used for optimization
        cpu_limits: list
            List of cpu limits to be used for optimization
        memory_limits: list
            List of memory limits to be used for optimization
        function_name: str
            Name of the function to be optimized
        data_name: str
            Name of the data to be used for optimization
        n_trials: int
            Number of optimization trial i.e. number of configurations to build the model with
        base_estimator: str
            Name of the base estimator to be used for optimization: GP, RF, ET, GBRT
        seed: int
            Seed for random number generator for reproducibility and generation of random samples
    """
    minimum = 600.0

    init_samples = get_random_samples(cpu_limits, memory_limits, instance_types, seed=seed)

    default_config = init_samples[0]
    config = default_config
    
    model = Models()
    model.create_model(function=function_name, base_estimator=base_estimator, seed=seed)
    for i in range(0, n_trials):
        result = get_result_from_data(function_name, data_name, config, memory_limits)

        if result < minimum:
            minimum = result

        model.update(config, result, function_name, success=True)
        
        if i < n_init_samples-1:
            config = init_samples[i+1]
        else:
            config = model.get_next_config(function_name)
        
    return model


if __name__ == "__main__":
    CPU_LIMITS= [ '250m', '500m', '750m', '1000m', '1250m', '1500m', '1750m', '2000m' ]
    MEMORY_LIMITS=[ '128Mi', '256Mi', '512Mi', '768Mi', '1024Mi', '2048Mi' ]
    instance_types = ['m5', 'm5a', 'c5', 'c5a', 'c6g', 'm6g']

    # This assumes that there are data files corresponding to these configuration, function and data in data directory in the root of the project
    model: Models = create_model(instance_types=instance_types, cpu_limits=CPU_LIMITS, memory_limits=MEMORY_LIMITS, function_name="transcode", data_name="cats-1", n_trials=10, base_estimator="GP", seed=1, n_init_samples=3)
    predictions = model.get_all_config_predictions(function="transcode")