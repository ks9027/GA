from bisect import bisect_right
from copy import deepcopy
from time import time
from typing import List
from random import shuffle, randint, uniform, random, choice
from loguru import logger
import numpy as np
from datetime import datetime
import random as rnd
import logging
from objects.node import Node
from objects.solution import Solution
from objects.problem import Problem
from algorithms.debug_GreedySearch import GreedySearch

import pandas as pd
import matplotlib.pyplot as plt

class ox_GSGA():
    """
    Here is a basic implementation of a GA class in Python for solving the VRP problem. 
    It takes a `Problem` object, population size, generations, crossover probability, mutation probability, and elite size as input arguments. 
    It has a `run` method which returns the best solution found by the GA after the specified number of generations. 
    It also has several helper functions for initializing the population, obtaining the elite, tournament selection, crossover, and mutation. 
    Note that this implementation only provides a basic framework for a GA and may need to be modified or extended depending on the specific VRP problem you are attempting to solve.

    """
    def __init__(self, population_size: int, generations: int, crossover_prob: float,
                 mutation_prob: float, elite_rate: int):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.elite_size = int(population_size * elite_rate)
        self.history = {
            'Mean Pop Fitness': [],
            'Best Pop Fitness': [],
             'Gen Time': []   
        }
        self.total_exec_time = 0.0 
        self.ranks = []
        self.population = []
        self.gs = GreedySearch()
    
    def set_problem(self, problem: Problem):
        self.problem = problem
        self.gs.set_problem(problem)
        self.population = self._initial_population()

    def free(self):
        self.history = {
            'Mean Pop Fitness': [],
            'Best Pop Fitness': [],
            'Gen Time': []
        }
        self.ranks = []
        self.population = self._initial_population()

    def selection(self, population: List[Solution], num: int) -> List[Solution]:
        #logger.debug(f"[Selection] Start. Population size: {len(population)}, Target size: {num}")
        # Step 1: Sort population
        population = sorted(population)
        #logger.debug(f"[Selection] Sorted population. Min Tour Length: {population[0].get_tour_length()}, "
        #            f"Max Tour Length: {population[-1].get_tour_length()}")

        # Step 2: Drop duplicates
        unique_pop = [population[0]]
        duplicates_removed = 0

        for i in range(1, len(population)):
            if population[i].get_presentation() != population[i - 1].get_presentation():
                unique_pop.append(population[i])
            else:
                duplicates_removed += 1

        #logger.debug(f"[Selection] Unique population size: {len(unique_pop)}, Duplicates removed: {duplicates_removed}")
        
        # Step 3: Select elite individuals
        new_pop = unique_pop[:self.elite_size]
        #logger.debug(f"[Selection] Elite size: {len(new_pop)}, "
        #            f"Elite Tour Lengths: {', '.join(map(str, [sol.get_tour_length() for sol in new_pop]))}")

        # Step 4: Select remaining individuals using probabilities
        remaining_count = num - len(new_pop)
        if remaining_count > 0:
        #    logger.debug(f"[Selection] Selecting {remaining_count} individuals using probabilities.")
            selected_by_probs = self.choose_by_probs(unique_pop[len(new_pop):], remaining_count)
        #    logger.debug(f"[Selection] Selected by probabilities: {[sol.get_tour_length() for sol in selected_by_probs]}")
            new_pop.extend(selected_by_probs)

        #logger.debug(f"[Selection] Final population size: {len(new_pop)}")
        return new_pop


    def solve(self, problem: Problem, verbose=False, plot_path=None) -> Solution:
        total_start_time = time()
        self.set_problem(problem)
        total_elapsed_time = 0  # 누적 시간 초기화
        visualization_times = []
        for i in range(self.generations):
            start_time = time()  # 세대 시작 시간 기록
            alpha = np.cos(np.pi / 3 * (i + 1) / self.generations) ** 2
            new_population = []
            self.compute_rank(self.population)
            while len(new_population) < self.population_size:
                id_1 = self.choose_by_rank(self.population)
                id_2 = self.choose_by_rank(self.population)
                
                while id_1 == id_2:
                    id_2 = self.choose_by_rank(self.population)
                    
                child_1, child_2 = self.population[id_1], self.population[id_2]
                

                child_1, child_2 = self.aox_crossover(child_1, child_2)
                child_1 = self.gs.optimize(child_1)
                child_2 = self.gs.optimize(child_2)
                new_population.append(child_1)
                new_population.append(child_2)
            
            n_news = int(self.population_size * alpha * 0.2)
            logger.debug(f" [Solve] Generating {n_news} new individuals using GreedySearch")

            new_indvs = [self.gs.optimize(self.gs.init_solution()) for _ in range(n_news)]
            logger.debug(f" [Solve] {len(new_indvs)} new individuals generated")

            # 기존 개체군 + 새로운 개체군을 합친 후 선택
            combined_population = self.population + new_population
            logger.debug(f" [Solve] Population before selection: {len(combined_population)} individuals")

            selected_population = self.selection(combined_population, self.population_size - n_news)
            logger.debug(f" [Solve] Population after selection: {len(selected_population)} individuals")

            self.population = selected_population + new_indvs
            logger.debug(f" [Solve] Final population size: {len(self.population)} (After adding new individuals)")

            valids = []
            depot = self.problem.get_depot()  # Depot 노드 가져오기

            logger.debug(f" [Solve] Depot Node ID: {depot.get_id()}")

            # 유효한 투어 확인 (depot 추가 후 검사)
            for idx, solution in enumerate(self.population):
                logger.debug(f" Checking solution {idx}: {solution.get_presentation()}")

                updated_tours = []
                for tour in solution.get_basic_tours():
                    updated_tour = [depot] + tour + [depot]
                    updated_tours.append(updated_tour)

                # Solution 업데이트 (Depot이 추가된 상태)
                solution_with_depot = Solution(tours=updated_tours)

                # Valid 검사 수행
                is_valid = self.problem.check_valid_solution(solution_with_depot)
                valids.append(is_valid)

                if is_valid:
                    logger.debug(f" Solution {idx} is VALID after adding depots.")
                    for tour_idx, tour in enumerate(updated_tours):
                        tour_node_ids = [node.get_id() for node in tour]
                        logger.debug(f" Updated Tour {tour_idx}: {tour_node_ids}")
                else:
                    logger.warning(f" Solution {idx} is INVALID after adding depots.")

                
                # 디버깅 및 출력
                if is_valid:
                    logger.debug(f"Solution {idx} is valid after adding depots.")
                    for tour_idx, tour in enumerate(updated_tours):
                        tour_node_ids = [node.get_id() for node in tour]
                        logger.debug(f"  Updated Tour {tour_idx}: Node IDs: {tour_node_ids}")
                else:
                    logger.debug(f"Solution {idx} is invalid after adding depots.")

            # 기존 valid logic
            valid_tour_lengths = [indv.get_tour_length() for i, indv in enumerate(self.population) if valids[i]]
            if valid_tour_lengths:  # 유효한 투어가 존재할 경우
                mean_fit = np.mean(valid_tour_lengths)
                current_best_fit = np.min(valid_tour_lengths)
            else:  # 유효한 투어가 없는 경우
                mean_fit = float('inf')  # 평균을 무한대로 설정
                current_best_fit = float('inf')  # 최소값도 무한대로 설정

            # mean_fit와 best_fit 확인
                # 추가: 최소 투어 길이에 대한 상세 정보 출력
            if current_best_fit == np.min(valid_tour_lengths):
                min_index = valid_tour_lengths.index(current_best_fit)
                #logger.debug(f"valid_tour_lengths: {valid_tour_lengths}")
                best_indv = self.population[min_index]
                
                min_tour_info = []
                for route in best_indv.tours:
                    current_load = self.problem.get_capacity()
                    current_battery = self.problem.get_battery_capacity()
                    route_info = []
                        # 각 투어의 시작 시 Depot과 연결
                    depot = self.problem.get_depot()
                    prev_node = depot
                    for idx, node in enumerate(route):
                        demand = node.get_demand()
                        previous_load = current_load
                        # 배터리 소비량 계산은 도착 전 적재량을 기준으로 계산
                        battery_consumption = self.problem.get_energy_consumption(prev_node, node, previous_load)
                        current_battery -= battery_consumption

                        if not node.is_depot():
                            # 적재량 업데이트 (Depot 제외)
                            current_load -= demand


                        route_info.append({
                            "node": node,
                            "node_id": node.get_id(),
                            "position": (node.get_x(), node.get_y()),
                            "demand": demand,
                            "current_load": current_load,
                            "battery_consumption":battery_consumption,
                            "current_battery": current_battery,
                        })
                            # 현재 노드를 이전 노드로 갱신
                        prev_node = node

                            # Depot 복귀 처리
                    battery_consumption_to_depot = self.problem.get_energy_consumption(prev_node, depot, current_load)
                    current_battery -= battery_consumption_to_depot

                    route_info.append({
                        "node": depot,
                        "node_id": depot.get_id(),
                        "position": (depot.get_x(), depot.get_y()),
                        "demand": 0,
                        "current_load": current_load,
                        "previous_load": current_load,  # Depot 복귀 시 적재량
                        "battery_consumption": battery_consumption_to_depot,
                        "current_battery": current_battery,
                    })
                    min_tour_info.append(route_info)

                # 디버깅 출력 (verbose=True일 때만)
                if verbose:
                    print(f"Minimum Tour Length: {current_best_fit}")
                    for route_idx, route in enumerate(min_tour_info):
                        print(f"  Route {route_idx + 1}:")
                        for step in route:
                            print(f"    Node ID: {step['node_id']}, "
                                f"Position: {step['position']}, "
                                f"Demand: {step['demand']}, "
                                f"Load: {step['current_load']}, "
                                f"battery consumption: {step['battery_consumption']}, "
                                f"Battery: {step['current_battery']}")

            end_time = time()  # 세대 종료 시간 기록
            elapsed_time = end_time - start_time  # 경과 시간 계산
            total_elapsed_time += elapsed_time  # 누적 시간 갱신
            current_time = datetime.now()
            formatted_time = f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}.{current_time.microsecond:03d}" 
            if verbose and i % 5 == 0:
                print(f"{formatted_time} | Generation: {i}, mean fit: {np.round(mean_fit, 3)}, "
                f"min fit: {np.round(current_best_fit, 3)}, alpha: {np.round(alpha, 3)}, "
                f"elapsed time: {np.round(elapsed_time, 3)}s, "
                f"total elapsed time: {np.round(total_elapsed_time, 3)}s")
                
            self.history['Mean Pop Fitness'].append(mean_fit)
            self.history['Best Pop Fitness'].append(current_best_fit)
            self.history['Gen Time'].append(elapsed_time)
            # 유효성 검사 결과 확인
            best_fit = float('inf')  # 초기값 설정: 매우 큰 값
            if current_best_fit < best_fit:  # 최적 솔루션 갱신
                best_fit = current_best_fit
                best_sol = self.population[np.argmin([indv.get_tour_length() for indv in self.population])]
            if plot_path is not None:
                self.problem.plot(best_sol, plot_path.replace('.png', '_solution.png'))
                plt.close()  # 메모리 정리
        total_end_time = time()
        self.total_exec_time = total_end_time - total_start_time        
        return self.population[np.argmin([indv.get_tour_length() for indv in self.population])]
    
    def aox_crossover(self, parent_1: Solution, parent_2: Solution):
        """
        Adjusted Order Crossover (AOX), 단일 라우트 전제:
        1) 부모 1의 라우트에서 임의의 [start, end] 구간을 segment로 복사
        2) 만약 이 segment가 부모 2에도 완전히 같은 형태로 발견되면,
            부모2의 절단점(= [start2, end2])을 무작위로 골라 다양성 확보
        3) 부모 2에서 '마지막으로 복사된 고객'이 등장하는 직후부터 순환하며
            child에 들어있지 않은 노드를 순서대로 채워넣기
        4) child_2도 같은 방식(부모1/2 역전)으로 생성
        5) Depot은 맨 앞/뒤 제거 후 작업, 끝나면 다시 삽입
        """

        # 1) 부모 1,2 각각에서 라우트(노드 ID 순열) 추출
        #    - 앞뒤 depot 제거 (존재한다면)
        p1_array = parent_1.to_array().tolist()  # numpy -> list
        p2_array = parent_2.to_array().tolist()
        logger.debug(f"Parent 1 Route: {p1_array}")
        logger.debug(f"Parent 2 Route: {p2_array}")
        # 혹시 첫/마지막이 depot이면 제거
        p1_array_stripped = self._strip_all_depots(p1_array)
        p2_array_stripped = self._strip_all_depots(p2_array)
        logger.debug(f"Parent 1 Route (Stripped): {p1_array_stripped}")
        logger.debug(f"Parent 2 Route (Stripped): {p2_array_stripped}")
        size1 = len(p1_array_stripped)
        size2 = len(p2_array_stripped)
        if size1 == 0 or size2 == 0:
            logger.warning("One of the parents has an empty route after depot stripping! Returning parents as is.")
            # 극단 상황이면 그냥 부모 반환
            return deepcopy(parent_1), deepcopy(parent_2)

        # 2) 무작위로 start, end 절단점 결정 (부모1 기준)
        start = rnd.randint(0, size1 - 1)
        end   = rnd.randint(0, size1 - 1)
        if start > end:
            start, end = end, start

        # segment: 부모1에서 [start:end] 구간
        segment = p1_array_stripped[start:end + 1]
        logger.debug(f"Selected segment from Parent 1: {segment} (Start: {start}, End: {end})")

        # 3) segment가 부모2에도 정확히 연속으로 존재하는지 확인
        #    - 만약 있다면, 부모2의 절단점을 무작위로 새로 선택
        #    (논문에서 "부모2 구간이 동일하면 무작위 절단점" 제안)
        if self._subarray_in_array(segment, p2_array_stripped):
            start2 = rnd.randint(0, size2 - 1)
            end2   = rnd.randint(0, size2 - 1)
            if start2 > end2:
                start2, end2 = end2, start2
            logger.debug(f"Segment found in Parent 2, selecting new random crossover points: Start2={start2}, End2={end2}")
        else:
            # 전통 OX에서는 보통 '동일한 start,end'를 쓰거나,
            # start2, end2를 따로 고르기도 함
            # 여기서는 간단히 동일 구간으로 처리(혹은 다른 로직)
            start2, end2 = start, end
            if end2 >= size2:  # 혹시 인덱스 초과 방지
                end2 = size2 - 1
            logger.debug(f"Segment not found in Parent 2, using default crossover points: Start2={start2}, End2={end2}")

        # ------------ child_1 생성 ------------
        child_1_arr = [None] * size1
        # (A) segment 복사
        child_1_arr[start:end+1] = segment

        # (B) '마지막 복사된 고객' in segment => segment[-1]
        last_copied = segment[-1]
        # 이 고객이 부모2 내에서 등장하는 index 찾기
        # => p2_array_stripped에서
        idx_in_p2 = p2_array_stripped.index(last_copied)
        logger.debug(f"Child 1 initial array (with segment inserted): {child_1_arr}")
        # (C) 나머지 노드 채우기
        used = set(segment)
        fill_idx = (end + 1) % size1

        # p2를 (idx_in_p2+1)부터 순환
        scan_idx = (idx_in_p2 + 1) % size2
        for _ in range(size2):
            candidate = p2_array_stripped[scan_idx]
            if candidate not in used:
                child_1_arr[fill_idx] = candidate
                used.add(candidate)
                fill_idx = (fill_idx + 1) % size1
            scan_idx = (scan_idx + 1) % size2
        logger.debug(f"Child 1 final array: {child_1_arr}")
        # ------------ child_2 생성 (반대 방향) ------------
        child_2_arr = [None] * size2

        # parent2의 [start2, end2] 구간 복사
        seg2 = p2_array_stripped[start2:end2+1]
        child_2_arr[start2:end2+1] = seg2

        last_cp2 = seg2[-1]
        idx_in_p1 = p1_array_stripped.index(last_cp2) if last_cp2 in p1_array_stripped else 0
        fill_idx2 = (end2 + 1) % size2

        used2 = set(seg2)
        scan_idx2 = (idx_in_p1 + 1) % size1
        for _ in range(size1):
            cand = p1_array_stripped[scan_idx2]
            if cand not in used2:
                child_2_arr[fill_idx2] = cand
                used2.add(cand)
                fill_idx2 = (fill_idx2 + 1) % size2
            scan_idx2 = (scan_idx2 + 1) % size1
        logger.debug(f"Child 2 final array: {child_2_arr}")
        # 4) depot 다시 삽입 (앞뒤)
        #    => child_1_arr, child_2_arr 에 대해

        sub_tours_1 = self.split_into_subtours(child_1_arr, self.problem)
        sub_tours_2 = self.split_into_subtours(child_2_arr, self.problem)
        logger.debug(f"Child 1 sub-tour After Filling: {sub_tours_1}")
        logger.debug(f"Child 2 sub-tour After Filling: {sub_tours_2}")
        # 5) Solution 객체로 변환
        child_1 = sub_tours_1
        child_2 = sub_tours_2

        # 🔹 ID 리스트 확인 (변환 전)
        logger.debug(f"Child 1 (ID list before conversion): {sub_tours_1}")
        logger.debug(f"Child 2 (ID list before conversion): {sub_tours_2}")

        # 🔹 서브 투어를 유지하며 노드 객체로 변환
        child_1 = [[self.problem.get_node_from_id(node_id) for node_id in sub_tour] for sub_tour in child_1]
        child_2 = [[self.problem.get_node_from_id(node_id) for node_id in sub_tour] for sub_tour in child_2]

        # 🔹 변환된 노드 객체 리스트 확인
        logger.debug(f"Child 1 (node IDs as sub-tours): {[[node.id for node in sub_tour] for sub_tour in child_1]}")
        logger.debug(f"Child 2 (node IDs as sub-tours): {[[node.id for node in sub_tour] for sub_tour in child_2]}")

        # Solution 객체 생성
        # Solution 객체 생성 후 단일 투어로 변환하여 반환
        return self._merge_into_single_route_with_depot(child_1), self._merge_into_single_route_with_depot(child_2)





    def _strip_all_depots(self, route_ids):
        """
        route_ids: [node_id, node_id, ...]
        모든 depot ID를 제거하여 반환.
        앞뒤뿐만 아니라 중간에 있는 depot도 포함해서 제거함.
        """
        depot_id = self.problem.get_depot().get_id()
        return [node_id for node_id in route_ids if node_id != depot_id]


    def _merge_into_single_route_with_depot(self, tours: List[List[Node]]) -> Solution:
        """
        예) 
        tours = [
            [node9, node7, node6, node1],
            [node8, node4, node3, node2, node5]
        ]
        => 최종 라우트 (하나의 리스트):
        [0,9,7,6,1,0,8,4,3,2,5,0]
        => => Solution([[0,9,7,6,1,0,8,4,3,2,5,0]])
        """
        depot_node = self.problem.get_depot()

        final_route = []
        for i, sub_route in enumerate(tours):
            # 첫 sub-route에 앞서 depot 추가
            if i == 0:
                final_route.append(depot_node)

            # sub-route의 고객들 추가
            for node in sub_route:
                final_route.append(node)

            # sub-route 끝날 때 depot 추가
            final_route.append(depot_node)

        # 이제 final_route는 하나의 시퀀스에 [Depot, ..., Depot, ..., Depot] 식으로
        # sub-route 경계마다 Depot이 들어가 있음

        # Solution 객체로 만들기 (투어가 1개)
        single_tour = [final_route]
        new_sol = Solution(single_tour)
        return new_sol


    def _subarray_in_array(self, subarr, arr):
        """
        subarr가 arr 내부에 '연속 부분 수열'로 존재하는지 간단 검사
        """
        s_len = len(subarr)
        a_len = len(arr)
        if s_len > a_len or s_len == 0:
            return False

        for start_idx in range(a_len - s_len + 1):
            if arr[start_idx:start_idx + s_len] == subarr:
                return True
        return False

    def _merge_into_single_route_with_depot(self, tours: List[List[Node]]) -> Solution:
        """
        예) 
        tours = [
            [node9, node7, node6, node1],
            [node8, node4, node3, node2, node5]
        ]
        => 최종 라우트 (하나의 리스트):
        [0,9,7,6,1,0,8,4,3,2,5,0]
        => => Solution([[0,9,7,6,1,0,8,4,3,2,5,0]])
        """
        depot_node = self.problem.get_depot()

        final_route = []
        for i, sub_route in enumerate(tours):
            # 첫 sub-route에 앞서 depot 추가
            if i == 0:
                final_route.append(depot_node)

            # sub-route의 고객들 추가
            for node in sub_route:
                final_route.append(node)

            # sub-route 끝날 때 depot 추가
            final_route.append(depot_node)

        # 이제 final_route는 하나의 시퀀스에 [Depot, ..., Depot, ..., Depot] 식으로
        # sub-route 경계마다 Depot이 들어가 있음

        # Solution 객체로 만들기 (투어가 1개)
        single_tour = [final_route]
        new_sol = Solution(single_tour)
        return new_sol


    def plot_history(self, path):
        df = pd.DataFrame(self.history)
        df.plot()
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title('Convergence trending ({})'.format(self.problem.get_name()))
        plt.legend()
        plt.grid()
        plt.savefig(path)
        plt.close()

    def _initial_population(self) -> List[Solution]:
        initial_pop = []
        for i in range(self.population_size):
            raw_sol = self.gs.init_solution()


            opt_sol = self.gs.optimize(raw_sol)


            initial_pop.append(opt_sol)
        return initial_pop

    def _get_elite(self, population: List[Solution]) -> List[Solution]:
        return sorted(population)[:self.elite_size]

    def _tournament_selection(self):
        return self.population[randint(0, len(self.population) - 1)]
    
    def sep_mutation(self, solution: Solution) -> Solution:
        solution.set_tour_index()
        tours = solution.get_basic_tours()

        if len(tours) == 1:
            return solution
        
        tours = solution.get_basic_tours()
        rd_tour_idx = choice(range(len(tours)))
        if len(tours[rd_tour_idx]) == 0:
            return solution
        rd_customer_idx = choice(range(len(tours[rd_tour_idx])))
        rd_customer = tours[rd_tour_idx][rd_customer_idx]

        tour_idx = solution.tour_index[rd_customer.get_id()]
        mm_customer_list = []
        for customer_id in self.gs.nearest_dist_customer_matrix[rd_customer.get_id()]:
            if solution.tour_index[customer_id] != tour_idx:
                mm_customer_list.append(self.problem.get_node_from_id(customer_id))
                if len(mm_customer_list) > 5:
                    break

        probs = [(len(mm_customer_list) - i + 1) ** 2 for i in range(len(mm_customer_list))]
        sum_probs = sum(probs)
        probs = [p / sum_probs for p in probs]
        mm_customer = np.random.choice(mm_customer_list, p=probs)
        mm_customer_tour_idx = solution.tour_index[mm_customer.get_id()]
        
        have = np.zeros(self.problem.get_num_dimensions())
        alens = set()
        
        for node in tours[rd_tour_idx]:
            alens.add(node.get_id())
            have[node.get_id()] = 1
            
        for node in tours[mm_customer_tour_idx]:
            alens.add(node.get_id())
            have[node.get_id()] = 1
            
        common_set = []
        
        alen_list = list(alens)
        shuffle(alen_list)
        
        for node_id in alen_list:
            if random() < 0.5 or len(common_set) < 2:
                common_set.append(self.problem.get_node_from_id(node_id))
                alens.remove(node_id)
                have[node_id] = 0
                
        distances = []
        depot_node = self.problem.get_depot()
        
        for node_1 in common_set:
            for node_2 in common_set:
                if node_1.get_id() != node_2.get_id():
                    vec_1 = np.array([node_1.get_x() - depot_node.get_x(), node_1.get_y() - depot_node.get_y()])
                    vec_2 = np.array([node_2.get_x() - depot_node.get_x(), node_2.get_y() - depot_node.get_y()])
                    dot_product = np.dot(vec_1, vec_2)
                    norm_product = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
                    if norm_product != 0:
                        cosine = dot_product / norm_product
                        cosine = np.clip(cosine, -1, 1)  # Ensure cosine is within valid range
                        angle = np.arccos(cosine)
                        angle = np.abs(angle)
                        distances.append((node_1, node_2, angle))
                    
        distances = sorted(distances, key=lambda x: x[2], reverse=True)
        center_1, center_2 = distances[0][:2]
        set_1 = []
        set_2 = []
        
        for node_id in alen_list:
            if have[node_id] == 1:
                if self.problem.get_distance(self.problem.get_node_from_id(node_id), center_1) < \
                    self.problem.get_distance(self.problem.get_node_from_id(node_id), center_2):
                    set_1.append(self.problem.get_node_from_id(node_id))
                else:
                    set_2.append(self.problem.get_node_from_id(node_id))
              
        for node in tours[rd_tour_idx]:
            if have[node.get_id()] == 0:
                set_1.append(node)
                
        for node in tours[mm_customer_tour_idx]:
            if have[node.get_id()] == 0:
                set_2.append(node)
                
        tours[rd_tour_idx] = set_1
        tours[mm_customer_tour_idx] = set_2
                
        return Solution(tours)

    def split_into_subtours(self, route, problem):
        """
        route: 예) [4,3,6,8,9,2,5,7,1]  (노드 ID들)
        self:  Problem 객체 자신 (배터리/적재/거리 계산 로직 보유)

        return: sub_tours (예: [ [4,3], [6,8,9], [2,5], [7], [1] ])
                - 각 subTour는 (출발-depot ~ 도착-depot) 사이의 고객 노드 ID들만 담음
                - 여기서는 'depot 노드'는 리스트에 포함 안 하고, 구간만 분리
        """

        sub_tours = []
        current_sub_tour = []

        # 차량이 처음 출발할 때는 "가득 실은 상태"
        load = problem.get_capacity()               # ex) 200.0
        battery = self.problem.get_battery_capacity()    # ex) 300.0

        #logger.debug(f"Initial Load: {load}, Battery: {battery}, Route: {route}")

        i = 0
        while i < len(route):
            node_i_id = route[i]
            node_i_obj = self.problem.get_node_from_id(node_i_id)

            #logger.debug(f"Processing Node ID: {node_i_id}, Load: {load}, Battery: {battery}")

            # --- (A) 만약 sub-tour 시작점(predecessor)이라면 ---
            if not current_sub_tour:
                # sub-tour 시작 노드
                current_sub_tour.append(node_i_id)

                # 첫 노드를 방문하므로, 해당 노드의 수요(demand) 처리
                demand_i = node_i_obj.get_demand()
                load -= demand_i
                # depot -> node_i 이동 배터리 소모
                battery_needed_i = self.problem.get_energy_consumption(self.problem.get_depot(), node_i_obj, self.problem.get_capacity())  
                #logger.debug(f"battery_needed ID: {battery_needed_i}")
                #   ↑ 배달형 로직에서, 첫 출발 시의 load=capacity로 계산 가능
                battery -= battery_needed_i

                #logger.debug(f"Starting new sub-tour: {current_sub_tour}, Load after update: {load}, Battery after update: {battery}")

                i += 1
                continue

            # --- (B) predecessor가 아닌 일반 노드 접근 ---
            node_next_id = route[i]
            node_next_obj = self.problem.get_node_from_id(node_next_id)

            # 이번 이동에 필요한 배터리
            battery_needed = self.problem.get_energy_consumption(node_i_obj, node_next_obj, load)
            demand_next = node_next_obj.get_demand()

            #logger.debug(f"Checking Node ID: {node_next_id}, Battery Needed: {battery_needed}, Next Demand: {demand_next}")

            # (1) 배터리 혹은 적재 초과 검사
            if (battery < battery_needed) or (load < demand_next):
                # sub-tour 종료 (depot 복귀)
                sub_tours.append(current_sub_tour)
                #logger.debug(f"Sub-tour ended: {current_sub_tour}, restarting from depot.")
                current_sub_tour = []

                # depot에서 다시 출발: load, battery 리셋
                load = self.problem.get_capacity()
                battery = self.problem.get_battery_capacity()
                #logger.debug(f"Reset Load: {load}, Battery: {battery}")
                # i는 증가시키지 않음(같은 노드를 다시 predecessor로 시도)
                continue

            # (2) comparing_way 로직 (i+1 노드가 있는지 확인 후, way1 vs way2 거리 비교)
            if i+1 < len(route):
                # i+1 노드
                node_i1_id = route[i]      # == node_next_id
                node_i2_id = route[i+1]    # 그다음 노드
                node_i1_obj = self.problem.get_node_from_id(node_i1_id)
                node_i2_obj = self.problem.get_node_from_id(node_i2_id)

                # way1: (i-> depot) + (i+1-> i+2)
                dist_way1 = self.problem.get_distance(node_i_obj, self.problem.get_depot()) \
                            + self.problem.get_distance(node_i1_obj, node_i2_obj)

                # way2: (i-> i+1) + (0-> i+2)
                dist_way2 = self.problem.get_distance(node_i_obj, node_i1_obj) \
                            + self.problem.get_distance(self.problem.get_depot(), node_i2_obj)

                #logger.debug(f"Way1 Distance: {dist_way1}, Way2 Distance: {dist_way2}")

                if dist_way1 < dist_way2:
                    # depot 복귀가 이득 => sub-tour 종료
                    sub_tours.append(current_sub_tour)
                    #logger.debug(f"Sub-tour ended (depot return): {current_sub_tour}")
                    current_sub_tour = []
                    # depot 리셋
                    load = self.problem.get_capacity()
                    battery = self.problem.get_battery_capacity()
                    # i는 증가 X
                    continue
                else:
                    # 그냥 next_node로 직행
                    current_sub_tour.append(node_next_id)
                    battery -= battery_needed
                    load -= demand_next
                    #logger.debug(f"Continuing sub-tour: {current_sub_tour}, Load: {load}, Battery: {battery}")
                    i += 1
            else:
                # i+1이 없다 => 마지막 노드
                current_sub_tour.append(node_next_id)
                battery -= battery_needed
                load -= demand_next
                #logger.debug(f"Adding final node to sub-tour: {current_sub_tour}, Load: {load}, Battery: {battery}")
                i += 1

        # --- (C) 마지막 sub-tour가 남았다면 마무리 ---
        if current_sub_tour:
            sub_tours.append(current_sub_tour)
            #logger.debug(f"Final sub-tour added: {current_sub_tour}")

        #logger.debug(f"All Sub-tours: {sub_tours}")
        return sub_tours



    
    def compute_rank(self, pop):
        _sum = 0
        self.ranks = []
        fit_min = min([pop[i].get_tour_length() for i in range(len(pop))])
        fit_max = max([pop[i].get_tour_length() for i in range(len(pop))])
        for i in range(len(pop)):
            temp_fit = ((fit_max - pop[i].get_tour_length()) / (fit_max - fit_min + 1e-6)) ** np.e
            _sum += temp_fit
            self.ranks.append(temp_fit)
            
        if _sum == 0:
            self.ranks = [1 / len(pop) for _ in range(len(pop))]
        else:
            for i in range(len(pop)):
                self.ranks[i] /= _sum
                if i > 0:
                    self.ranks[i] += self.ranks[i - 1]
                    
    
    def choose_by_rank(self, population: List[Solution]) -> int:
        prob = random()
        return bisect_right(self.ranks, prob, hi=len(population)) - 1
    
    def choose_by_probs(self, pop: List[Solution], k: int) -> List[Solution]:
        fit_min = min([pop[i].get_tour_length() for i in range(len(pop))])
        fit_max = max([pop[i].get_tour_length() for i in range(len(pop))])
        probs = []
        _sum = 0
        
        for i in range(len(pop)):
            temp_fit = ((fit_max - pop[i].get_tour_length()) / (fit_max - fit_min + 1e-6)) ** np.e
            _sum += temp_fit
            probs.append(temp_fit)
            
        if _sum == 0:
            probs = [1 / len(pop) for _ in range(len(pop))]
        else:
            probs /= _sum
        
        choices = sorted(np.random.choice(pop, k, p=probs, replace=False))
        return choices
        
