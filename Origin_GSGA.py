from bisect import bisect_right
from copy import deepcopy
import random
from time import time
from typing import List
from random import shuffle, randint, uniform, random, choice
from loguru import logger
import numpy as np
from datetime import datetime

from objects.node import Node
from objects.solution import Solution
from objects.problem import Problem
from algorithms.GreedySearch import GreedySearch

import pandas as pd
import matplotlib.pyplot as plt

class Origin_GSGA():
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
            'Best Pop Fitness': []
        }
        self.ranks = []
        self.population = []
        self.gs = GreedySearch()
        self.all_console_lines = []  # 로그 저장용 리스트 추가

    def log_and_print(self, msg):
        """ print() + logger.info() 동시에 실행 """
        print(msg)         # 콘솔에 출력
        logger.info(msg)   # 로그 파일에도 저장
        self.all_console_lines.append(msg)  # 필요하면 리스트에도 저장
    
    def set_problem(self, problem: Problem):
        self.problem = problem
        self.gs.set_problem(problem)
        self.population = self._initial_population()

    def free(self):
        self.history = {
            'Mean Pop Fitness': [],
            'Best Pop Fitness': []
        }
        self.ranks = []
        self.population = self._initial_population()

    def selection(self, population: List[Solution], num: int) -> List[Solution]:
        
        population = sorted(population)
        unique_pop = [population[0]]
        duplicates_removed = 0

        for i in range(1, len(population)):
            if population[i].get_presentation() != population[i - 1].get_presentation():
                unique_pop.append(population[i])
            else:
                duplicates_removed += 1
        new_pop = unique_pop[:self.elite_size]
        remaining_count = num - len(new_pop)
        if remaining_count > 0:
            selected_by_probs = self.choose_by_probs(unique_pop[len(new_pop):], remaining_count)
            new_pop.extend(selected_by_probs)

        return new_pop


    def solve(self, problem: Problem, verbose=False, plot_path=None) -> Solution:
        
        self.set_problem(problem)
        total_start_time = time()  # 전체 시작 시간
        total_elapsed_time = 0  # 누적 시간 초기화
        best_so_far = float('inf')
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
                
                # 기존 GA 루프 일부 (crossover_prob, mutation_prob 등 그대로 유지)
                if random() < self.crossover_prob:
                    # 1) 교차(crossover)
                    child_1, child_2 = self.distribute_crossover(child_1, child_2)

                    # -------------------------
                    # (A) child_1 변이 or '강제' 단일 라우트
                    # -------------------------
                    if random() < self.mutation_prob:
                        # mutation_prob 조건 통과 시 => 기존 로직 (random_swap_mutation / hsm)
                        if random() < (0.2 + (0.6 - 0.2) * (1 - alpha)):
                            child_1 = self.random_swap_mutation(child_1)
                        else:
                            child_1 = self.hsm(child_1)
                    else:
                        # mutation_prob 실패 시 => 반드시 단일 라우트로
                        child_1 = self._merge_into_single_route_with_depot(child_1.tours)

                    # -------------------------
                    # (B) child_2 변이 or '강제' 단일 라우트
                    # -------------------------
                    if random() < self.mutation_prob:
                        if random() < (0.2 + (0.6 - 0.2) * (1 - alpha)):
                            child_2 = self.random_swap_mutation(child_2)
                        else:
                            child_2 = self.hsm(child_2)
                    else:
                        child_2 = self._merge_into_single_route_with_depot(child_2.tours)

                    # 2) 최적화(optimize)
                    #logger.debug(
                        #f"Before Optimization - Child 1: Tour Length: {child_1.get_tour_length()}, "
                        #f"Tours: {[[(node.id) for node in tour] for tour in child_1.tours]}"
                    #)
                    #logger.debug(
                        #f"Before Optimization - Child 2: Tour Length: {child_2.get_tour_length()}, "
                        #f"Tours: {[[(node.id) for node in tour] for tour in child_2.tours]}"
                    #)
                    child_1 = self.gs.optimize(child_1)
                    child_2 = self.gs.optimize(child_2)

                    # 3) 두 자식 중 더 좋은 쪽만 new_population에 추가
                    if child_1.get_tour_length() < child_2.get_tour_length():
                        new_population.append(child_1)
                    else:
                        new_population.append(child_2)

                else:
                    # crossover에 실패한 경우 => child_1만 변이·최적화·추가
                    if random() < (0.2 + (0.6 - 0.2) * (1 - alpha)):
                        child_1 = self.random_swap_mutation(child_1)
                    else:
                        child_1 = self.hsm(child_1)
                    child_1 = self.gs.optimize(child_1)
                    new_population.append(child_1)

            
            n_news = int(self.population_size * alpha * 0.2)
            #logger.debug(f"n_news (new individuals): {n_news}, alpha: {alpha}")
            
            new_indvs = [self.gs.optimize(self.gs.init_solution()) for _ in range(n_news)]
            #logger.debug(f"New Individuals (n_news): {[indv.get_tour_length() for indv in new_indvs]}")
            
            self.population = self.selection(self.population + new_population, self.population_size - n_news) + new_indvs
                        # self.population 업데이트 확인
            #ogger.debug(f"Population before selection: {[indv.get_tour_length() for indv in self.population]}")
            #logger.debug(f"New Population: {[indv.get_tour_length() for indv in new_population]}")
            #logger.debug(f"Combined Population Length: {len(self.population + new_population)}")
                        # self.selection 확인
            selected_population = self.selection(self.population + new_population, self.population_size - n_news)
            #logger.debug(f"Selected Population: {[indv.get_tour_length() for indv in selected_population]}")

            valids = [self.problem.check_valid_solution(indv) for indv in self.population]
            valid_tour_lengths = [indv.get_tour_length() for i, indv in enumerate(self.population) if valids[i]]

            if valid_tour_lengths:  # 유효한 투어가 존재할 경우
                mean_fit = np.mean(valid_tour_lengths)
                current_best_fit = np.min(valid_tour_lengths)
            else:  # 유효한 투어가 없는 경우
                mean_fit = float('inf')  # 평균을 무한대로 설정 (문제 상황 명확히 하기 위해)
                current_best_fit = np.min(valid_tour_lengths)
                #logger.warning("No valid solutions found in the current population.")
            # 최종 population 확인
            final_population = selected_population + new_indvs
            #logger.debug(f"Final Population: {[indv.get_tour_length() for indv in final_population]}")

            # 유효성 검사 결과 확인
            #logger.debug(f"Valid Solutions: {valids}")
            #logger.debug(f"Valid Solution Count: {sum(valids)} / {len(valids)}")

            # mean_fit와 best_fit 확인
            #logger.debug(f"Mean Fitness: {mean_fit}")
            #logger.debug(f"Best Fitness: {best_fit}")
            end_time = time()  # 세대 종료 시간 기록
            elapsed_time = end_time - start_time  # 경과 시간 계산
            total_elapsed_time += elapsed_time  # 누적 시간 갱신
            current_time = datetime.now()
            formatted_time = f"{current_time.strftime('%Y-%m-%d %H:%M:%S')}.{current_time.microsecond//1000:3d}" 
            if verbose:
                self.log_and_print(f"{formatted_time} | Generation: {i}, mean fit: {np.round(mean_fit, 3)}, "
                f"min fit: {np.round(current_best_fit, 3)}, alpha: {np.round(alpha, 3)}, "
                f"elapsed time: {np.round(elapsed_time, 3)}s, "
                f"total elapsed time: {np.round(total_elapsed_time, 3)}s")
                
            self.history['Mean Pop Fitness'].append(mean_fit)
            self.history['Best Pop Fitness'].append(current_best_fit)
            if current_best_fit < best_so_far:
                best_so_far = current_best_fit
                # population 중 최적해 찾아서 best_sol 업데이트
                best_idx = np.argmin([indv.get_tour_length() for indv in self.population])
                best_sol = self.population[best_idx]

                # 그림 업데이트
                if plot_path is not None:
                    solution_png = plot_path.replace('.png', '_solution.png')
                    self.problem.plot(best_sol, solution_png)

            ### [중요] 20세대마다만 history plot 갱신
            if plot_path is not None and (i % 1== 0):
                self.plot_history(plot_path)

        total_end_time = time()
        self.total_exec_time = total_end_time - total_start_time

        ### [중요] 루프가 끝난 후, 지금까지 찾은 best_sol 반환
        if best_sol is not None:
            return best_sol
        else:
            # valid 솔루션이 없다면, 그냥 가장 짧은 tour_length를 가진 해 or 첫 해
            return self.population[np.argmin([p.get_tour_length() for p in self.population])]
    
    def distribute_crossover(self, parent_1: Solution, parent_2: Solution) -> (Solution, Solution):

        #logger.debug(f"[Crossover] Parent 1 Full Tours: {[[node.get_id() for node in tour] for tour in parent_1.tours]}")
        #logger.debug(f"[Crossover] Parent 2 Full Tours: {[[node.get_id() for node in tour] for tour in parent_2.tours]}")

        # 1. 무작위 고객(crd) 선택
        crd = choice(self.problem.get_all_customers())
        crd_id = crd.get_id()
        #logger.debug(f"[Crossover] Selected random customer ID: {crd_id}")

        # 2. 선택된 고객이 포함된 부모의 투어 추출
        id1 = parent_1.tour_index[crd_id]
        id2 = parent_2.tour_index[crd_id]
        V1_nodes = parent_1.get_basic_tours()[id1]
        V2_nodes = parent_2.get_basic_tours()[id2]
        V1 = [node.get_id() for node in V1_nodes]
        V2 = [node.get_id() for node in V2_nodes]

        #logger.debug(f"[Crossover] Parent 1 tour index: {id1}, Tour Nodes: {V1}")
        #logger.debug(f"[Crossover] Parent 2 tour index: {id2}, Tour Nodes: {V2}")

        # 3. 부모1의 고객 순서 복사
        sub1 = V1[:]
        #logger.debug(f"[Crossover] sub1 (from Parent 1): {sub1}")

        # 4. 부모2에서 새로운 고객 추가
        sub2 = [node_id for node_id in V2 if node_id not in V1]
        #logger.debug(f"[Crossover] sub2 (from Parent 2, new nodes): {sub2}")

        # 5. 두 개의 crossover sequence 생성
        SC1 = sub2 + sub1
        SC2 = list(reversed(sub1)) + list(reversed(sub2))
        #logger.debug(f"[Crossover] SC1: {SC1}")
        #logger.debug(f"[Crossover] SC2: {SC2}")

        # 6. 전체 교차 고객 집합
        union_set = set(V1) | set(V2)
        #logger.debug(f"[Crossover] Union set (V1 ∪ V2): {union_set}")

        # 7. 부모1의 전체 고객 리스트 (순서 유지)
        parent1_basic = []
        for tour in parent_1.get_basic_tours():
            for node in tour:
                parent1_basic.append(node.get_id())

        parent2_basic = []
        for tour in parent_2.get_basic_tours():
            for node in tour:
                parent2_basic.append(node.get_id())

        # 중복 제거 (순서 유지)
        seen = set()
        parent1_basic_ordered = []
        for node_id in parent1_basic:
            if node_id not in seen:
                seen.add(node_id)
                parent1_basic_ordered.append(node_id)

        seen = set()
        parent2_basic_ordered = []
        for node_id in parent2_basic:
            if node_id not in seen:
                seen.add(node_id)
                parent2_basic_ordered.append(node_id)

        # 8. 나머지 노드 (SC1, SC2에 없는 노드)
        remainder1 = [node_id for node_id in parent1_basic_ordered if node_id not in union_set]
        remainder2 = [node_id for node_id in parent2_basic_ordered if node_id not in union_set]
        #logger.debug(f"[Crossover] Remainder Parent1 (not in union_set): {remainder1}")
        #logger.debug(f"[Crossover] Remainder Parent2 (not in union_set): {remainder2}")

        # **SC1은 Parent1의 순서를 유지하며 배치**
        final_sc1 = []
        for node_id in parent1_basic_ordered:
            if node_id in SC1:
                final_sc1.append(node_id)  # SC1 원래 순서 유지
            elif node_id in remainder1:
                final_sc1.append(node_id)  # 나머지 노드 추가

        # **SC2는 Parent2의 순서를 유지하며 배치**
        final_sc2 = []
        for node_id in parent2_basic_ordered:
            if node_id in SC2:
                final_sc2.append(node_id)  # SC2 원래 순서 유지
            elif node_id in remainder2:
                final_sc2.append(node_id)  # 나머지 노드 추가

        #logger.debug(f"[Crossover] Final SC1 with ordered remainder: {final_sc1}")
        #logger.debug(f"[Crossover] Final SC2 with ordered remainder: {final_sc2}")

        #  SC1, SC2를 Node 객체 리스트로 변환
        child1_nodes = [self.problem.get_node_from_id(node_id) for node_id in final_sc1]
        child2_nodes = [self.problem.get_node_from_id(node_id) for node_id in final_sc2]

        #  Debug 로그: 변환된 Node 객체 확인
        #logger.debug(f"Child 1 Nodes (as objects): {[node.id for node in child1_nodes]}")
        #logger.debug(f"Child 2 Nodes (as objects): {[node.id for node in child2_nodes]}")

        #  sub_tours_1, sub_tours_2는 ID 리스트(int) 상태
        sub_tours_1 = self.split_into_subtours([node.get_id() for node in child1_nodes], self.problem)
        sub_tours_2 = self.split_into_subtours([node.get_id() for node in child2_nodes], self.problem)

        #  Debug 로그: 변환 전 (ID 리스트 상태)
        #logger.debug(f"Child 1 sub-tour Before Conversion: {sub_tours_1}")
        #logger.debug(f"Child 2 sub-tour Before Conversion: {sub_tours_2}")

        #  **ID 리스트 → Node 객체 리스트 변환**
        sub_tours_1 = [[self.problem.get_node_from_id(node_id) for node_id in sub_tour] for sub_tour in sub_tours_1]
        sub_tours_2 = [[self.problem.get_node_from_id(node_id) for node_id in sub_tour] for sub_tour in sub_tours_2]

        #  Debug 로그: 변환 후 (Node 객체 리스트 상태)
        #logger.debug(f"Child 1 sub-tour After Filling: {[[node.id for node in sub_tour] for sub_tour in sub_tours_1]}")
        #logger.debug(f"Child 2 sub-tour After Filling: {[[node.id for node in sub_tour] for sub_tour in sub_tours_2]}")

        #  Solution 객체 생성
        return Solution(tours=sub_tours_1), Solution(tours=sub_tours_2)





    def hsm(self, solution: Solution) -> Solution:
        solution.set_tour_index()
        tours = solution.get_basic_tours()
        #logger.debug(f"HSM: Initial basic tours (node IDs): {[[node.id for node in tour] for tour in tours]}")

        if len(tours) == 1:
            return solution

        tours = solution.get_basic_tours()
        #logger.debug(f"HSM: Basic tours after get_basic_tours() (node IDs): {[[node.id for node in tour] for tour in tours]}") 
        
        rd_tour_idx = choice(range(len(tours)))

        #logger.debug(f"HSM: Selected random tour index: {rd_tour_idx}")
        if len(tours[rd_tour_idx]) == 0:
            return solution
        
        rd_customer_idx = choice(range(len(tours[rd_tour_idx])))
        rd_customer = tours[rd_tour_idx][rd_customer_idx]
        #logger.debug(f"HSM: Selected random customer: {rd_customer}")

        tour_idx = solution.tour_index[rd_customer.get_id()]
        mm_customer_list = []
        for customer_id in self.gs.nearest_dist_customer_matrix[rd_customer.get_id()]:
            if solution.tour_index[customer_id] != tour_idx:
                mm_customer_list.append(self.problem.get_node_from_id(customer_id))
                if len(mm_customer_list) > 3:
                    break
        #logger.debug(f"HSM: Candidates for mutation (IDs): {[node.id for node in mm_customer_list]}")
        
        probs = [(len(mm_customer_list) - i + 1) ** 2 for i in range(len(mm_customer_list))]
        sum_probs = sum(probs)
        probs = [p / sum_probs for p in probs]
        mm_customer = np.random.choice(mm_customer_list, p=probs)
        #logger.debug(f"HSM: Selected customer for mutation: {mm_customer}")
        mm_customer_tour_idx = solution.tour_index[mm_customer.get_id()]
        mm_customer_idx = -1
        for idx in range(len(tours[mm_customer_tour_idx])):
            if tours[mm_customer_tour_idx][idx].get_id() == mm_customer.get_id():
                mm_customer_idx = idx
                break
        #logger.debug(f"HSM: Mutation target index in its tour: {mm_customer_idx}")
        #logger.debug(f"HSM: Swapping {rd_customer} and {mm_customer}.")
        tours[tour_idx][rd_customer_idx], tours[mm_customer_tour_idx][mm_customer_idx] = \
            tours[mm_customer_tour_idx][mm_customer_idx], tours[tour_idx][rd_customer_idx]
        #logger.debug(f"HSM: Updated tours (node IDs): {[[node.id for node in tour] for tour in tours]}")
        return self._merge_into_single_route_with_depot(tours)
    
    def random_swap_mutation(self, solution: Solution) -> Solution:
        tours = solution.get_basic_tours()
        #logger.debug(f"Random Swap Mutation: Initial basic tours:  {[[node.id for node in tour] for tour in tours]}")
        if len(tours) == 1:
            return solution
        
        tours = solution.get_basic_tours()
        #logger.debug("Random Swap Mutation: Only one tour, returning original solution.")
        rd_tour_idx = choice(range(len(tours)))
        #logger.debug(f"Random Swap Mutation: Selected random tour index: {rd_tour_idx}")
        if len(tours[rd_tour_idx]) == 0:
            return solution
        
        rd_customer_idx_1 = choice(range(len(tours[rd_tour_idx])))
        rd_customer_idx_2 = choice(range(len(tours[rd_tour_idx])))
        #logger.debug(f"Random Swap Mutation: Selected indices for swap: {rd_customer_idx_1}, {rd_customer_idx_2}")
        #logger.debug(f"Random Swap Mutation: Swapping {tours[rd_tour_idx][rd_customer_idx_1]} and {tours[rd_tour_idx][rd_customer_idx_2]}.")
        tours[rd_tour_idx][rd_customer_idx_1], tours[rd_tour_idx][rd_customer_idx_2] = \
            tours[rd_tour_idx][rd_customer_idx_2], tours[rd_tour_idx][rd_customer_idx_1]
        #logger.debug(f"Random Swap Mutation: Updated tours: {[[node.id for node in tour] for tour in tours]}")
        return self._merge_into_single_route_with_depot(tours)
    
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
        
