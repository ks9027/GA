import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
from objects.problem import Problem
from algorithms.GSGA import GSGA
from algorithms.Origin_GSGA import Origin_GSGA
from algorithms.ox_GSGA import ox_GSGA
from algorithms.random_GA import random_GA
from algorithms.GreedySearch import GreedySearch
from src.utils import get_problem_name, logger


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--problem-path', type=str,
                        default='./benchmarks/evrp-POMO/test_results_20.evrp')
    parser.add_argument('-a', '--algorithm', type=str, default='Origin_GSGA')
    parser.add_argument('-o', '--result-path', type=str,
                        default='./results/Origin_GSGA/')  
    parser.add_argument('-n', '--nruns', type=int, default=10)
    parser.add_argument('--seed', type=int, default=1232)
    args = parser.parse_args()
    return args
def make_result_folder(base_path:str, algo_name:str):
    from datetime import datetime
    now_str = datetime.now().strftime('%Y%m%d_%H%M%S')  
    # 예) 20250211_140945
    folder_name = f"{algo_name}_{now_str}"
    result_dir = os.path.join(base_path, folder_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir
def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    start_time = time.time()
    args = argparser()
    set_random_seed(args.seed)

    # -------------------------
    # (A) 결과 폴더 생성
    # -------------------------
    result_path = make_result_folder(args.result_path, args.algorithm)

    # -------------------------
    # (B) logger 설정
    # -------------------------
    # main_log.txt에 모든 정보를 기록할 수 있도록
    main_log_file = os.path.join(result_path, "main_log.txt")
    
    # 기존 로거 제거, 새로 추가
    logger.remove()
    logger.add(main_log_file, level="DEBUG", format="{time} | {message}")
    
    # print() 함수 출력도 log에 남기려면 아래처럼 Hooking
    # 간단히는 "print(...)" 대신 logger.info(...)를 쓰는 방법도 있음
    all_console_lines = []
    def log_and_print(msg):
        print(msg)
        logger.info(msg)       # logger로도 기록
        all_console_lines.append(msg)

    # -------------------------
    # (C) argparser 인자 초기 로그 출력
    # -------------------------
    log_and_print("=== Program Start ===")
    log_and_print(f"problem-path = {args.problem_path}")
    log_and_print(f"algorithm    = {args.algorithm}")
    log_and_print(f"result-path  = {args.result_path}")
    log_and_print(f"nruns        = {args.nruns}")
    log_and_print(f"seed         = {args.seed}")

    # -------------------------
    # 문제 불러오기, 알고리즘 생성
    # -------------------------
    problem_name = get_problem_name(args.problem_path)
    problem = Problem(args.problem_path)

    if args.algorithm == 'GreedySearch':
        algorithm = GreedySearch()
        kwargs = {
            'problem': problem,
            'verbose': True,
            # 그림 저장 경로: fitness_history.png
            'plot_path': os.path.join(result_path, 'fitness_history.png')
        }
        
    elif args.algorithm == 'GSGA':
        algorithm = GSGA(population_size=150, generations=200,   
                         crossover_prob=0.85, mutation_prob=0.95, elite_rate=0.2)
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(result_path, 'fitness_history.png')
        }

    elif args.algorithm == 'Origin_GSGA':
        algorithm = Origin_GSGA(population_size=150, generations=200,
                                crossover_prob=0.85, mutation_prob=0.95, elite_rate=0.2)
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(result_path, 'fitness_history.png')
        }

    elif args.algorithm == 'random_GA':
        algorithm = random_GA(population_size=150, generations=200,
                              crossover_prob=0.85, mutation_prob=0.95, elite_rate=0.2)
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(result_path, 'fitness_history.png')
        }

    elif args.algorithm == 'ox_GSGA':
        algorithm = ox_GSGA(population_size=150, generations=200,
                            crossover_prob=0.85, mutation_prob=0.95, elite_rate=0.2)
        kwargs = {
            'problem': problem,
            'verbose': True,
            'plot_path': os.path.join(result_path, 'fitness_history.png')
        }
    else:
        log_and_print(f"Invalid algorithm {args.algorithm}")
        raise ValueError(f'Invalid algorithm {args.algorithm}')
    
    # -------------------------
    # (D) 여러 run 수행
    # -------------------------
    results = []
    for run_i in range(args.nruns):
        log_and_print(f"=== Run {run_i}/{args.nruns} ===")
        solution = algorithm.solve(**kwargs)  # 여기서 세대별 로그가 콘솔+logger로 찍힘

        # run_i 결과 파일
        run_result_file = os.path.join(result_path, f"run_{run_i}.txt")
        run_figure_file = os.path.join(result_path, f"run_{run_i}.png")

        # 유효성 검사
        if problem.check_valid_solution(solution, verbose=True):
            cost_val = solution.get_tour_length()
            log_and_print(f"[Run {run_i}] valid solution => cost={cost_val}")
            results.append(cost_val)

            # 최종 솔루션 그림 저장
            problem.plot(solution, run_figure_file)
            log_and_print(f"[Run {run_i}] final solution figure => {run_figure_file}")

            # 텍스트 파일 기록
            with open(run_result_file, "w") as f:
                f.write(str(cost_val)+"\n")
        else:
            log_and_print(f"[Run {run_i}] invalid solution => cost=inf")
            results.append(np.inf)
            with open(run_result_file, "w") as f:
                f.write(str(np.inf)+"\n")

        # run 끝날 때 algorithm.free()로 초기화(히스토리 등)
        algorithm.free()

    # -------------------------
    # (E) 모든 run 끝나고 종료 처리
    # -------------------------
    end_time = time.time()
    total_sec = end_time - start_time
    log_and_print(f"Program finished. Elapsed Time: {total_sec:.3f} seconds.")
    log_and_print(f"Results = {results}")
    log_and_print("=== Program End ===")

