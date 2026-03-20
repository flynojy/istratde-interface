from concurrent.futures import ProcessPoolExecutor
from cec2013lsgo.cec2013 import Benchmark
# from env.Overlapping_Benchmark.OB import OB
import random
import numpy as np
import math
import time
# from MMES_change.mmes import MMES
from MMES.istratde_optimizer import IStratDEOptimizer
import os
from utils import plot_evaluation_curve_best_so_far, result_record, combine

bench = Benchmark()
ALGORITHM_NAME = os.environ.get("DEMO_OPTIMIZER", "ISTRATDE").upper()
ISTRATDE_BACKEND = os.environ.get("ISTRATDE_BACKEND", "torch")
POP_SIZE = int(os.environ.get("POP_SIZE", "1000"))
MAX_FES = float(os.environ.get("MAX_FES", "1E6"))
CYCLE_NUM = int(os.environ.get("CYCLE_NUM", "10"))
FUN_ID_START = int(os.environ.get("FUN_ID_START", "15"))
FUN_ID_END = int(os.environ.get("FUN_ID_END", "15"))
USE_PROCESS_POOL = os.environ.get("USE_PROCESS_POOL", "1") == "1"
VERBOSE_EVERY = int(os.environ.get("VERBOSE_EVERY", "1000"))


def build_optimizer(problem, options):
    if ALGORITHM_NAME == "MMES":
        from MMES.mmes import MMES
        return MMES(problem, options)
    if ALGORITHM_NAME == "ISTRATDE":
        return IStratDEOptimizer(problem, options)
    raise ValueError(f"Unsupported optimizer: {ALGORITHM_NAME}")


def print_profiling(results):
    profiling = results.get("profiling")
    if not profiling:
        return

    print("[profiling]")
    print(f"  generations: {profiling['generations']}")
    print(f"  evaluation calls: {profiling['evaluation_calls']}")
    print(f"  total step time: {profiling['step_time']:.6f}s")
    print(
        f"  evaluation pipeline: {profiling['evaluation_time']:.6f}s "
        f"({profiling['evaluation_share'] * 100:.2f}%)"
    )
    print(
        f"  algorithm/framework: {profiling['algorithm_time']:.6f}s "
        f"({profiling['algorithm_share'] * 100:.2f}%)"
    )

class fun_record():
    def __init__(self, fun):
        self.fun = fun
        self.fitness_record = []
    def __call__(self, x):
        fitness = self.fun(x)
        self.fitness_record.extend(fitness)
        return fitness

def combine(small_vec, background_vec, location):
    if location is None:
        return small_vec
    else:
        combination = np.tile(background_vec, (len(small_vec), 1))
        combination[:, location] = small_vec
        return combination
    
def optimization_task(fun_id, best_individual, MaxFEs, grouping_result, info, cycle_num_index):
    time_start = time.time()
    fun = bench.get_function(fun_id)
    fun_ = fun_record(fun)

    problem_ = {'fitness_function': fun_,  # fitness function
    'ndim_problem': info['dimension'],  # dimension
    'lower_boundary': info['lower'] * np.ones((info['dimension'],)),  # lower search boundary
    'upper_boundary': info["upper"]* np.ones((info['dimension'],))}

    options_ = {'max_function_evaluations': MaxFEs,  # to set optimizer options
        'mean': (best_individual,) ,
        'sigma': 0.05,
        'is_restart': True,
        'verbose': VERBOSE_EVERY,
        'seed_rng': 42+ cycle_num_index,
        'backend': ISTRATDE_BACKEND,
        'n_individuals': POP_SIZE} 
    optimizer = build_optimizer(problem_, options_)
    results_ = optimizer.optimize()
    if cycle_num_index == 0:
        print_profiling(results_)

    time_end = time.time()
    return  (time_end - time_start),fun_.fitness_record

def parallel_optimization(fun_id, best_individual, MaxFEs, cycle_num, grouping_result, info):
    if (not USE_PROCESS_POOL) or cycle_num <= 1:
        average_time, fitness_record = optimization_task(
            fun_id, best_individual, MaxFEs, grouping_result, info, 0
        )
        return average_time, [fitness_record]

    with ProcessPoolExecutor() as executor:
        futures = []
        for _ in range(cycle_num):
            futures.append(executor.submit(optimization_task, fun_id, best_individual, MaxFEs, grouping_result, info, _))
        
        Algorithm = ALGORITHM_NAME
        average_time = 0
        fitness_record = []
        for future in futures:
            result = future.result()
            fitness_record.append(result[1])
            average_time += result[0]
            
        
        return average_time / cycle_num , fitness_record
    
output_data = {ALGORITHM_NAME:[[]], f'{ALGORITHM_NAME}_time':[]}


def print_runtime_info():
    print(f'Optimizer: {ALGORITHM_NAME}')
    if ALGORITHM_NAME != "ISTRATDE":
        return

    print(f'iStratDE backend: {ISTRATDE_BACKEND}')
    if ISTRATDE_BACKEND == "torch":
        import torch

        print(f'torch version: {torch.__version__}')
        print(f'cuda available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'cuda device count: {torch.cuda.device_count()}')
            print(f'active device: {torch.cuda.current_device()}')
            print(f'device name: {torch.cuda.get_device_name(torch.cuda.current_device())}')
        else:
            print('running on CPU because CUDA is unavailable')


print_runtime_info()

for fun_id in range(FUN_ID_START, FUN_ID_END + 1):
    MaxFEs = MAX_FES
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    fes_tag = f'{int(MaxFEs):.0E}'.replace('+0', '').replace('+', '')
    output_path = f'save_dir/baseline/{ALGORITHM_NAME}_{fes_tag}/F{fun_id}_{timestamp}/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    bench = Benchmark()

    info = bench.get_info(fun_id)
    fun = bench.get_function(fun_id)

    best_individual = np.zeros(info['dimension'])
    grouping_result = None
    average_time, fitness_record = parallel_optimization(fun_id, best_individual, MaxFEs, CYCLE_NUM, grouping_result, info)

    output_data[ALGORITHM_NAME] = fitness_record
    output_data[f'{ALGORITHM_NAME}_time'].append(average_time)

    result_record(output_data, output_path, record_FEs_list=[1.2E5, 2E5, 1.5E6, 2E6, 3E6])
    plot_evaluation_curve_best_so_far(output_data, output_path, font_size=12, maxfes=MaxFEs+100,log_scale=True,show_variance=True)
