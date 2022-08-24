import pandas as pd 
import numpy as np 
import os 
import random 
import sympy as sym
from pyDOE import *
import math 

# cost_intel = {}
# costs_arm = {}
# cost_amd = {}
mem_cost = {}
cpu_cost = {}
# ondemand_costs_intel = 0.0
# ondemand_costs_arm = 0.0
# ondemand_costs_amd = 0.0 

# spot_costs_intel = 0.0 
# spot_costs_arm = 0.0
# spot_costs_amd = 0.0 

# TODO: Use a class for a Cost object instead of using global vars
# Separates the memory cost and cpu cost from the total cost of instance. 
def separate_costs(tc1, tc2, tc3):
    cpu1 = sym.Symbol('cpu1')
    cpu2 = sym.Symbol('cpu2')
    m = sym.Symbol('m')
    solution = sym.solve(
        (2 * cpu1 + 4 * m - tc1 , 2 * cpu2 + 8 * m - tc2 , 2 * cpu2 + 16 * m - tc3 ),
        (cpu1, cpu2, m)
    )
    return solution

def calculate_cpu_mem_costs(pricing_model='ondemand', discount_factor=0.8, instance_types=['m6g', 'c6g', 'm5', 'c5', 'm5a', 'c5a']):
    if pricing_model == 'ondemand':
        cost_intel, cost_arm, cost_amd = ondemand_pricing() 
    else:
        cost_intel, cost_arm, cost_amd = spot_pricing(discount_factor=discount_factor) 

    cpu1 = sym.Symbol('cpu1')
    cpu2 = sym.Symbol('cpu2')
    m = sym.Symbol('m')
    # print(cost_intel, cost_amd, cost_arm)

    global mem_cost 
    global cpu_cost

    for t in instance_types:
        if t == 'm6g':
            mem_cost['m6g'] = cost_arm[m]/(1024.0*3600.0)
            cpu_cost['m6g'] = cost_arm[cpu2]/3600.0 
        elif t == 'c6g':
            mem_cost['c6g'] = cost_arm[m]/(1024.0*3600.0)
            cpu_cost['c6g'] = cost_arm[cpu1]/3600.0 
        elif t == 'm5':
            mem_cost['m5'] =  cost_intel[m]/(1024.0*3600.0)
            cpu_cost['m5'] =  cost_intel[cpu2]/3600.0 
        elif t == 'c5':
            mem_cost['c5'] = cost_intel[m]/(1024.0*3600.0)
            cpu_cost['c5'] = cost_intel[cpu1]/3600.0
        elif t == 'm5a':
            mem_cost['m5a'] =  cost_amd[m]/(1024.0*3600.0)  
            cpu_cost['m5a'] =  cost_amd[cpu2]/3600.0
        else:
            mem_cost['c5a'] = cost_amd[m]/(1024.0*3600.0)
            cpu_cost['c5a']  = cost_amd[cpu1]/3600.0
    
    # print(mem_cost)
    # print(cpu_cost)

def ondemand_pricing():
    c5 = 0.085
    m5 = 0.096 
    r5 = 0.126
    # global ondemand_costs_intel 
    ondemand_costs_intel = separate_costs(c5, m5, r5)

    c6g = 0.068
    m6g = 0.077 
    r6g = 0.1008
    # global ondemand_costs_arm
    ondemand_costs_arm = separate_costs(c6g, m6g, r6g)

    c5a = 0.077
    m5a = 0.086
    r5a = 0.113
    # global ondemand_costs_amd
    ondemand_costs_amd = separate_costs(c5a, m5a, r5a)
    return ondemand_costs_intel, ondemand_costs_arm, ondemand_costs_amd
    # calculate_cpu_mem_costs(ondemand_costs_intel, ondemand_costs_arm, ondemand_costs_amd)

def spot_pricing(discount_factor=0.8):
    c5 = 0.085*discount_factor
    m5 = 0.096*discount_factor 
    r5 = 0.126*discount_factor
    # global spot_costs_intel 
    spot_costs_intel = separate_costs(c5, m5, r5)

    c6g = 0.068*discount_factor
    m6g = 0.077*discount_factor 
    r6g = 0.1008*discount_factor
    # global spot_costs_arm 
    spot_costs_arm = separate_costs(c6g, m6g, r6g)

    c5a = 0.077*discount_factor
    m5a = 0.086*discount_factor
    r5a = 0.113*discount_factor
    # global spot_costs_amd 
    spot_costs_amd = separate_costs(c5a, m5a, r5a)
    return spot_costs_intel, spot_costs_arm, spot_costs_amd
    # calculate_cpu_mem_costs(spot_costs_intel, spot_costs_arm, spot_costs_amd)

def calculate_cost(cpu, memory, time, instance_type):
    cpu = float(cpu.strip('m'))/1000.0
    memory = float(memory.strip('Mi'))

    instance_family = instance_type.split('.')[0]
    exec_cost = cpu_cost[instance_family]*cpu*time + mem_cost[instance_family]*memory*time 

    return float(exec_cost)

def calculate_runtime(cpu, memory, exec_cost, instance_type):
    cpu = float(cpu.strip('m'))/1000.0
    memory = float(memory.strip('Mi'))
    instance_family = instance_type.split('.')[0]
    time = exec_cost/(cpu_cost[instance_family]*cpu + mem_cost[instance_family]*memory) 
    return float(time)

def get_init_samples(cpu_limits, memory_limits, instance_types, seed=1, n_samples=3, var=False):
    
    init_samples = []
    random.seed(seed)

    cpus = ["500", "1000", "2000"]
    mem = str(max([int(m.strip("Mi")) for m in memory_limits]))
    inst_type = "m5"
    for i in range(0, n_samples):
        
        cpu = cpus[i]
        if var:
            inst_type = random.choice(instance_types)
            
        init_samples.append({"instance_type": inst_type, "cpu": cpu.replace("m", ""), "memory": mem.replace("Mi", "")})

    return init_samples

def get_random_samples(cpu_limits, memory_limits, instance_types, seed=1, n_samples=3):
    
    random_samples = []
    random.seed(seed)

    for i in range(0, n_samples):
        
        cpu = random.choice(cpu_limits)
        mem = random.choice(memory_limits)
        inst_type = random.choice(instance_types)
        random_samples.append({"instance_type": inst_type, "cpu": cpu.replace("m", ""), "memory": mem.replace("Mi", "")})

    return random_samples

def get_lhs_samples(cpu_limits, memory_limits, instance_types, seed=1, n_samples=3):
    
    lhs_samples = []
    random.seed(seed)

    lhd = lhs(3, n_samples, criterion="maximin")

    # print(lhd)

    for i in range(0, n_samples):
        sample = lhd[i]

        cpu_index =  math.floor(len(cpu_limits) * sample[0])
        cpu = cpu_limits[cpu_index]

        mem_index= math.floor(len(memory_limits) * sample[1])
        mem = memory_limits[mem_index]


        inst_index = math.floor(len(instance_types) * sample[2])
        inst_type = instance_types[inst_index]

        lhs_samples.append({"instance_type": inst_type, "cpu": cpu.replace("m", ""), "memory": mem.replace("Mi", "")})

    # print(lhs_samples)
    return lhs_samples

def find_filename(directory, benchmark, data_name, mem, cpu, instance_type, return_if_exists=True):
    filename1 = directory + benchmark + "_" + data_name + "_" + mem + "Mi_" + cpu + "m_" + instance_type + ".large.csv"
    filename2 = directory + benchmark + "_" + data_name + "_" + mem + "Mi_" + cpu + "m_" + instance_type + ".xlarge.csv" 
    # print(filename1, filename2)
    filename = ''
    if os.path.exists(filename1): 
        filename = filename1
    elif os.path.exists(filename2):
        filename = filename2 
    else:
        return None
    
    # NOTE: This is to make sure that configuration is working 
    df = pd.read_csv(filename)
    if df['status-code'].iloc[-1] == 200:
        return filename
    elif df['status-code'].iloc[0] == 502:
        return filename
    elif return_if_exists:
        return filename
    else:
        return None


def get_result_from_data(benchmark, data_name, config, MEMORY_LIMITS, metric='runtime'):
    directory = '../data/' + benchmark + '/'
    mem = config["memory"]
    cpu = config["cpu"]
    instance_type = config["instance_type"]

    if MEMORY_LIMITS != None:
        return_if_exists = False
    else:
        return_if_exists = True
    
    filename = find_filename(directory, benchmark, data_name, mem, cpu, instance_type, return_if_exists=return_if_exists)
    if filename == None:
        for m in MEMORY_LIMITS:
            m = m.strip('Mi')
            if int(m) > int(mem):
                return 10000.0 
            
            filename = find_filename(directory, benchmark, data_name, m, cpu, instance_type, return_if_exists=return_if_exists)
            if filename != None:
                # print(filename)
                break
        # return np.nan
    # print(filename)
    if filename == None:
        print(filename)
        print(benchmark, data_name, mem, cpu, instance_type)

    if os.stat(filename).st_size == 0:
        print("No data for this")
        return np.nan

    df = pd.read_csv(filename)
    r, c = df.shape
    if r == 0:
        print("No data for this")
        return np.nan

    # print(filename)
    if df['status-code'].iloc[-1] == 200:
        median = df[df['status-code']==200]['Response-delay'].median()
    elif df['status-code'].iloc[0] == 502:
        median = 600.0
    else:
        median = 10000.0

    result = median 
    if metric != 'runtime':
        if result != 10000.0:
            result = calculate_cost(cpu, mem, median, instance_type)
    return result 

def get_all_runtimes_costs_from_data(instance_types, CPU_LIMITS, MEMORY_LIMITS, benchmark, data_name, return_configs = False):    
    runtimes= []
    costs = []
    configs = []
    # exec_cost_df =  pd.DataFrame(columns = ['cost', 'memory', 'cpu'])
    for instance_type in instance_types:
        for cpu in CPU_LIMITS:
            for mem in MEMORY_LIMITS:
                # directory = "../data/" + benchmark + "/"
                config = {'instance_type': instance_type, 'memory': mem.strip('Mi'), 'cpu': cpu.strip('m')}
                median = get_result_from_data(benchmark, data_name, config, MEMORY_LIMITS)
                
                cost = calculate_cost(cpu, mem, median, instance_type)
                runtimes.append(median)
                costs.append(cost)
                configs.append(config)

    if return_configs:
        return runtimes, costs, configs 
    else:
        return runtimes, costs

def get_best_configuration(instance_types, CPU_LIMITS, MEMORY_LIMITS, benchmark, data_name, metric='runtime'):    
    dataframe = pd.DataFrame(columns = ['time', 'cost', 'memory', 'cpu'])
    # exec_cost_df =  pd.DataFrame(columns = ['cost', 'memory', 'cpu'])
    for instance_type in instance_types:
        for cpu in CPU_LIMITS:
            for mem in MEMORY_LIMITS:
                config = {'instance_type': instance_type, 'memory': mem.strip('Mi'), 'cpu': cpu.strip('m')}
                median = get_result_from_data(benchmark, data_name, config, MEMORY_LIMITS)
                
                cost = calculate_cost(cpu, mem, median, instance_type)
                dataframe = dataframe.append({'time': median, 'cost': cost, 'memory': mem, 
                                                'cpu': instance_type+ '_'+ cpu}, ignore_index=True)

    # dataframe["cpu"] = pd.to_numeric(dataframe["cpu"])
    if metric=='runtime':
        dataframe.sort_values(by=["time"], inplace=True)
        best_config = dataframe.iloc[0]
        # print("Best configuration")
        # print(best_config)
        return best_config["time"], best_config["memory"], best_config["cpu"]
    elif metric=='cost':
        dataframe.sort_values(by=["cost"], inplace=True)
        best_config = dataframe.iloc[0]
        # print("Best configuration")
        # print(best_config)
        return best_config["cost"], best_config["memory"], best_config["cpu"]      
    
    elif '/' in metric:
        minimum_runtime,_,_ = get_best_configuration(instance_types, CPU_LIMITS, MEMORY_LIMITS, benchmark, data_name, metric='runtime')
        minimum_cost,_,_ = get_best_configuration(instance_types, CPU_LIMITS, MEMORY_LIMITS, benchmark, data_name, metric='cost')
        runtime_weight = float(metric.split('/')[0])
        cost_weight = float(metric.split('/')[1])
        dataframe['metric'] = ((dataframe['time']/minimum_runtime) * runtime_weight) + ((dataframe['cost']/minimum_cost) * cost_weight)
        dataframe.sort_values(by=["metric"], inplace=True)
        best_config = dataframe.iloc[0]
        # print("Best configuration")
        # print(best_config)
        return best_config["time"], best_config["cost"], best_config["memory"], best_config["cpu"]



def get_best_configuration_from_defaults(confs, benchmark, data_name, metric='runtime'):    
    if metric=='runtime':
        dataframe = pd.DataFrame(columns = ['time', 'memory', 'cpu'])
    else:
        dataframe = pd.DataFrame(columns = ['cost', 'memory', 'cpu'])
    # exec_cost_df =  pd.DataFrame(columns = ['cost', 'memory', 'cpu'])

    for conf in confs:
        # print(conf)
        cpu = conf[0]
        mem = conf[1]
        instance_type = 'm5'

        config = {'instance_type': instance_type, 'memory': mem.strip('Mi'), 'cpu': cpu.strip('m')}
        # TODO: Check where this is being used.
        median = get_result_from_data(benchmark, data_name, config, MEMORY_LIMITS=None)
        
        cost = calculate_cost(cpu, mem, median, instance_type)
        if metric=='runtime':
            dataframe = dataframe.append({'time': median, 'memory': mem, 
                                        'cpu': instance_type+ '_'+ cpu}, ignore_index=True)
        else:
            dataframe = dataframe.append({'cost': cost, 'memory': mem, 
                                        'cpu': instance_type+ '_'+ cpu}, ignore_index=True)

    # dataframe["cpu"] = pd.to_numeric(dataframe["cpu"])
    if metric=='runtime':
        # print(dataframe)
        dataframe.sort_values(by=["time"], inplace=True)
        best_config = dataframe.iloc[0]
        # print("Best configuration")
        # print(best_config)
        return best_config["time"], best_config["memory"], best_config["cpu"]
    else:
        dataframe.sort_values(by=["cost"], inplace=True)
        best_config = dataframe.iloc[0]
        # print("Best configuration")
        # print(best_config)
        return best_config["cost"], best_config["memory"], best_config["cpu"] 

def get_top_k_configs_from_data(instance_types, CPU_LIMITS, MEMORY_LIMITS, benchmark, data_name, metric='runtime', k=5):    
    if metric=='runtime':
        dataframe = pd.DataFrame(columns = ['time', 'memory', 'cpu'])
    else:
        dataframe = pd.DataFrame(columns = ['cost', 'memory', 'cpu'])
    # exec_cost_df =  pd.DataFrame(columns = ['cost', 'memory', 'cpu'])
    for instance_type in instance_types:
        for cpu in CPU_LIMITS:
            for mem in MEMORY_LIMITS:
                config = {'instance_type': instance_type, 'memory': mem.strip('Mi'), 'cpu': cpu.strip('m')}
                median = get_result_from_data(benchmark, data_name, config, MEMORY_LIMITS)
                
                cost = calculate_cost(cpu, mem, median, instance_type)
                if metric=='runtime':
                    dataframe = dataframe.append({'time': median, 'memory': mem, 
                                                'cpu': instance_type+ '_'+ cpu}, ignore_index=True)
                else:
                    dataframe = dataframe.append({'cost': cost, 'memory': mem, 
                                                'cpu': instance_type+ '_'+ cpu}, ignore_index=True)

    # dataframe["cpu"] = pd.to_numeric(dataframe["cpu"])
    if metric=='runtime':
        dataframe.sort_values(by=["time"], inplace=True)
        best_k_configs = []
        for i in range(0, k):
            conf = dataframe.iloc[i]
            best_k_configs.append((conf['time'], (conf['cpu'].split('_')[0], 
                        conf['cpu'].split('_')[1].strip('m'), conf['memory'].strip('Mi'))
                ))        
        # print("Best configuration")
        # print(best_config)
        # return best_config["time"], best_config["memory"], best_config["cpu"]
        return best_k_configs
    else:
        dataframe.sort_values(by=["cost"], inplace=True)
        best_config = dataframe.iloc[0]
        # print("Best configuration")
        # print(best_config)
        return best_config["cost"], best_config["memory"], best_config["cpu"]   

def get_top_per_type_configs_from_data(instance_types, CPU_LIMITS, MEMORY_LIMITS, benchmark, data_name, metric='runtime'):    
    best_k_configs = []
    # exec_cost_df =  pd.DataFrame(columns = ['cost', 'memory', 'cpu'])
    for instance_type in instance_types:
        if metric=='runtime':
            dataframe = pd.DataFrame(columns = ['time', 'memory', 'cpu'])
        else:
            dataframe = pd.DataFrame(columns = ['cost', 'memory', 'cpu'])

        for cpu in CPU_LIMITS:
            for mem in MEMORY_LIMITS:
                config = {'instance_type': instance_type, 'memory': mem.strip('Mi'), 'cpu': cpu.strip('m')}
                median = get_result_from_data(benchmark, data_name, config, MEMORY_LIMITS)
                
                cost = calculate_cost(cpu, mem, median, instance_type)
                if metric=='runtime':
                    dataframe = dataframe.append({'time': median, 'memory': mem, 
                                                'cpu': instance_type+ '_'+ cpu}, ignore_index=True)
                else:
                    dataframe = dataframe.append({'cost': cost, 'memory': mem, 
                                                'cpu': instance_type+ '_'+ cpu}, ignore_index=True)

        # dataframe["cpu"] = pd.to_numeric(dataframe["cpu"])
        if metric=='runtime':
            dataframe.sort_values(by=["time"], inplace=True)
            
            conf = dataframe.iloc[0]
            best_k_configs.append((conf['time'], (conf['cpu'].split('_')[0], 
                        conf['cpu'].split('_')[1].strip('m'), conf['memory'].strip('Mi'))
                ))        
            # print("Best configuration")
            # print(best_config)
            # return best_config["time"], best_config["memory"], best_config["cpu"]
            
        else:
            dataframe.sort_values(by=["cost"], inplace=True)
            conf = dataframe.iloc[0]
            # print("Best configuration")
            # print(best_config)
            best_k_configs.append((conf['cost'], (conf['cpu'].split('_')[0], 
                        conf['cpu'].split('_')[1].strip('m'), conf['memory'].strip('Mi'))
                ))  
    
    
    return best_k_configs


