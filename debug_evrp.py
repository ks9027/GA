import argparse
import os

import numpy as np
from algorithms.debug_GSGA import GSGA
from algorithms.Origin_GSGA import Origin_GSGA
from algorithms.ox_GSGA import ox_GSGA
from algorithms.random_GA import random_GA
from objects.problem import Problem
from algorithms.debug_GreedySearch import GreedySearch
from src.utils import get_problem_name, logger
import random
import time

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem-path', type=str, default='./benchmarks/evrp-2019/E-n22-k41.evrp')
    parser.add_argument('-a', '--algorithm', type=str, default='GSGA')
    parser.add_argument('-o', '--result-path', type=str, default='./results/debug_GSGA/')
    parser.add_argument('-n', '--nruns', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1234)
    args = parser.parse_args()
    return args

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    

if __name__ == "__main__":
    start_time = time.time() 
    args = argparser()
    set_random_seed(args.seed)
    problem_name = get_problem_name(args.problem_path)
    problem = Problem(args.problem_path)
    
    if args.algorithm == 'GreedySearch':
        algorithm = GreedySearch()
        
        kwargs = {
            'problem': problem,
            'verbose': True
        }
        
    elif args.algorithm == 'GSGA':
        algorithm = GSGA(population_size=4, generations=200, 
                          crossover_prob=0.85, mutation_prob=0.95, elite_rate=0.2)
        
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(args.result_path, problem_name, 'fitness_history.png')
        }
    elif args.algorithm == 'Origin_GSGA':
        algorithm = Origin_GSGA(population_size=4, generations=200, 
                          crossover_prob=0.85, mutation_prob=0.95, elite_rate=0.2)
        
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(args.result_path, problem_name, 'fitness_history.png')
        }    
    elif args.algorithm == 'random_GA':
        algorithm = random_GA(population_size=4, generations=200, 
                          crossover_prob=0.85, mutation_prob=0.95, elite_rate=0.2)
        
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(args.result_path, problem_name, 'fitness_history.png')
        }
    elif args.algorithm == 'ox_GSGA':
        algorithm = ox_GSGA(population_size=4, generations=200, 
                          crossover_prob=0.85, mutation_prob=0.95, elite_rate=0.2)
        
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(args.result_path, problem_name, 'fitness_history.png')
        }    
    

    else:
        raise ValueError(f'Invalid algorithm {args.algorithm}')

        
    results = []
    
    for i in range(args.nruns):
        result_path = os.path.join(args.result_path, problem_name)
        result_file = os.path.join(result_path, f"run_{i}.txt")
        log_file = os.path.join(result_path, f"run_{i}_log.txt")  # ✅ 로그 저장
        figure_file = os.path.join(result_path, f"run_{i}.png")

        if not os.path.exists(result_path):
            os.makedirs(result_path)

        # ✅ Loguru 설정 (파일에 저장)
        logger.remove()  # 기존 핸들러 제거
        logger.add(log_file, level="DEBUG")  # ✅ 로그 저장 활성화

        solution = algorithm.solve(**kwargs)

        if problem.check_valid_solution(solution, verbose=True):
            tour_length = solution.get_tour_length()
            with open(result_file, 'w') as f:
                f.write(f"{tour_length}\n")
                
            results.append(tour_length)
            algorithm.free()
            problem.plot(solution, figure_file)
            logger.info(f"Solution valid: {solution}")
        else:
            logger.error('Invalid solution')
            results.append(np.inf)
            with open(result_file, 'w') as f:
                f.write(f"{np.inf}\n")
                    
    end_time = time.time()  # 프로그램 종료 시간 기록
    elapsed_time = end_time - start_time  # 전체 실행 시간 계산
    print(f"Total execution time: {elapsed_time:.2f} seconds")  # 결과 출력
            
            