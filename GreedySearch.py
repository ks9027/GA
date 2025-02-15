
from copy import deepcopy
from random import shuffle
import time
from random import choice
from loguru import logger
import numpy as np
from objects.problem import Problem
from objects.solution import Solution


class GreedySearch():
    """
    Algorithm for insert energy stations into all tours for each vehicle.
    
    """
    def __init__(self) -> None:
        pass

    def set_problem(self, problem: Problem):
        self.problem = problem
        self.nearest_dist_customer_matrix = {}
        self.calc_nearest_dist_customer_matrix()
        

    def free(self):
        pass
    
    def init_solution(self) -> Solution:
        solution = self.stochastic_greedy_solution()
        return solution
    
    def optimize(self, solution: Solution) -> Solution:
        solution = self.local_search(solution)
        solution = self.optimize_depots(solution)
        solution.set_tour_length(self.problem.calculate_tour_length(solution))
        #logger.debug(f"[Optimize] Final => length={solution.get_tour_length()}, route:\n{solution}")
        return solution

    def solve(self, problem, verbose=False) -> Solution:
        self.set_problem(problem)
        self.verbose = verbose
        solution = self.init_solution()
        solution = self.optimize(solution)
        return solution
    
    def calc_nearest_dist_customer_matrix(self):
        all_customers = self.problem.get_all_customers()
        self.nearest_dist_customer_matrix = {}
        
        for i in range(len(all_customers)):
            distances = []
            for j in range(len(all_customers)):
                distances.append(all_customers[i].distance(all_customers[j]))
            argsort_dist = np.argsort(distances)
            self.nearest_dist_customer_matrix[all_customers[i].get_id()] = \
                [all_customers[j].get_id() for j in argsort_dist if i != j]
                
                
            
    def stochastic_greedy_solution(self, k=10) -> Solution:
            """
            - stochastic greedy로 초기 해 생성
            - k-nearest customer를 무작위로 선택해 투어 구성
            """
            solution = Solution()
            #logger.debug(f"[InitSolution] before_stochastic_greedy => {solution}") #완전 그냥 빈값 solution = []
            cust_ids = self.problem.get_customer_ids()
            depot = self.problem.get_depot()

            current_load = self.problem.get_capacity()
            current_energy = self.problem.get_battery_capacity()
            tour = [depot]
            #logger.debug(f"[InitSolution] before_stochastic_greedy => {tour}") #depot만 들어간 상태 solution = [0]
            if cust_ids:
                first_customer_id = choice(cust_ids)
                first_customer = self.problem.get_node_from_id(first_customer_id)
                needed_to_first = self.problem.get_energy_consumption(depot, first_customer, current_load)
                #logger.debug(f"first_customer_id => {first_customer_id}")
                #logger.debug(f"first_customer => {first_customer}")
                #logger.debug(f"needed_to_first => {needed_to_first}")
                distance_to_first = self.problem.get_distance(depot, first_customer)
                #logger.debug(f"Distance from depot to first customer: {distance_to_first}")
                tour.append(first_customer)
                cust_ids.remove(first_customer_id)
                current_load -= first_customer.get_demand()
                current_energy -= needed_to_first
                #logger.debug(f"current_load => {current_load}")
                #logger.debug(f"current_energy => {current_energy}")
            while cust_ids:
                #logger.debug(f"Remaining customers: {cust_ids}")  # 현재 남은 고객 ID들
                current_node = tour[-1]
                #logger.debug(f"Current node: {current_node.get_id()}")  # 현재 노드 ID
                distances = [(cid, self.problem.get_distance(current_node, self.problem.get_node_from_id(cid))) for cid in cust_ids]
                #logger.debug(f"Distances from current node: {[(cid, dist) for cid, dist in distances]}")  # 거리 출력
                distances.sort(key=lambda x: x[1])
                k_nearest = [cid for cid, _ in distances[:k]]
                #logger.debug(f"k-nearest customers: {k_nearest}")  # k개 고객 출력

                found_next = False
                shuffle(k_nearest)
                #logger.debug(f"k-nearest customers(shuffle): {k_nearest}")
                for next_customer_id in k_nearest:
                    next_customer = self.problem.get_node_from_id(next_customer_id)
                    demand = next_customer.get_demand()
                    #logger.debug(f"Checking next customer ID: {next_customer_id}")  # 다음 고객 ID
                    #logger.debug(f"Next customer demand: {demand}")  # 다음 고객의 수요
                    distance_to_next = self.problem.get_distance(current_node, next_customer)
                    distance_to_depot = self.problem.get_distance(next_customer, depot)
                    #logger.debug(f"Distance from current node {current_node.get_id()} to next customer {next_customer_id}: {distance_to_next}")
                    #logger.debug(f"Distance from next customer {next_customer_id} to depot: {distance_to_depot}")
                    needed_to_next = self.problem.get_energy_consumption(current_node, next_customer, current_load)
                    load_after_next = current_load - demand  # next_customer를 방문한 후의 남은 적재량
                    #logger.debug(f"load after next custmoer: {load_after_next}")  # 다음 고객의 수요
                    needed_to_depot = self.problem.get_energy_consumption(next_customer, depot, load_after_next)
                    total_needed = needed_to_next + needed_to_depot
                    #logger.debug(f"Energy needed to next customer: {needed_to_next}, to depot: {needed_to_depot}, total: {total_needed}")
                    if demand <= current_load and total_needed <= current_energy:
                        #logger.debug(f"Adding customer {next_customer_id} to tour")  # 고객 추가
                        tour.append(next_customer)
                        cust_ids.remove(next_customer_id)
                        current_load -= demand
                        current_energy -= needed_to_next
                        #logger.debug(f"Updated current load: {current_load}, current energy: {current_energy}")  # 상태 업데이트
                        found_next = True
                        break

                if not found_next:
                    tour.append(depot)
                    current_load = self.problem.get_capacity()
                    current_energy = self.problem.get_battery_capacity()
                    #logger.debug(f"Reset load: {current_load}, energy: {current_energy}. Starting new route.")

                    if cust_ids:
                        next_start_customer_id = choice(cust_ids)
                        next_start_customer = self.problem.get_node_from_id(next_start_customer_id)
                        #logger.debug(f"Starting new route with customer {next_start_customer_id}")  # 새로운 고객 ID
                        tour.append(next_start_customer)
                        cust_ids.remove(next_start_customer_id)
                        current_load -= next_start_customer.get_demand()
                        current_energy -= self.problem.get_energy_consumption(depot, next_start_customer)
    
            if tour[-1] != depot:
                tour.append(depot)

            solution.add_tour(tour)
            solution.set_tour_index()
            return solution



    def local_search(self, solution: Solution) -> Solution:
        """
        예시:
        - solution.tours 안에 여러 sub-route가 있음.
        - 각 sub-route마다 depot(0)은 앞뒤에 있다고 가정 [0, c1, c2, ..., 0].
        - 이 sub-route에서 '고객 부분'만 추출 -> 2-opt 수행 -> 다시 depot 붙여서 갱신.
        - 최종적으로 solution.tours는 여러 라우트 형태를 그대로 보존.
        """
        depot = self.problem.get_depot()  # Depot 노드 가져오기

        # 1) depot을 기준으로 tours 분리
        all_nodes = solution.tours[0]  # 단일 라우트로 저장된 전체 노드 리스트
        multi_tours = []
        current_tour = []

        for node in all_nodes:
            if node.is_depot():
                if current_tour:
                    # 현재 투어가 비어있지 않으면 저장
                    multi_tours.append([depot] + current_tour + [depot])
                    current_tour = []
            else:
                current_tour.append(node)

        # 마지막 투어 처리
        if current_tour:
            multi_tours.append([depot] + current_tour + [depot])

        #logger.debug(f"Split multi_tours: {[[node.get_id() for node in tour] for tour in multi_tours]}")

        # 2) 각 sub-route별로 로컬서치(2-opt) 적용
        for route_idx, route in enumerate(multi_tours):
            #logger.debug(f"Processing route {route_idx}: {[node.get_id() for node in route]}")
            # route 예: [0, c1, c2, c3, 0]
            if len(route) <= 2:
              #  logger.debug(f"Route {route_idx} skipped (too short): {[node.get_id() for node in route]}")
                # 고객이 0~1명인 sub-route면 2-opt 할게 없음
                continue

            # A) depot을 임시 제거:  route[0], route[-1]이 depot이라고 가정
            #    고객 부분만 추출
            inner_customers = route[1:-1]  # ex: [c1, c2, c3]
            #logger.debug(f"Inner customers (before 2-opt) for route {route_idx}: {[node.get_id() for node in inner_customers]}")

            # B) 2-opt 수행
            improved_customers = self.local_search_2opt(inner_customers)
            #logger.debug(f"Improved customers (after 2-opt) for route {route_idx}: {[node.get_id() for node in improved_customers]}")
            # C) depot 다시 붙임
            new_subroute = [route[0]] + improved_customers + [route[-1]]
            #logger.debug(f"New subroute for route {route_idx}: {[node.get_id() for node in new_subroute]}")

            # D) 기존 sub-route 갱신
            multi_tours[route_idx] = new_subroute

        # 3) solution에 다시 반영
        solution.tours = multi_tours
        #logger.debug(f"Before set_tour_index: {solution}")
        solution.set_tour_index()
        #logger.debug(f"After set_tour_index: {solution}")
        return solution




    def local_search_2opt(self, customer_list):
        """
        간단 2-opt 예시:
        - customers_list는 depot 제외된 고객들의 순서
        - 2-opt로 더 짧아질 경우 swap.
        - 예시 구현이므로, 실제론 while 루프 돌며 improvement 없을 때까지 반복하는 방식 등 사용.
        """
        #logger.debug(f"Starting 2-opt optimization on customer list: {[node.get_id() for node in customer_list]}")

        improved = True
        best_route = customer_list[:]
        best_distance = self.route_distance(best_route)
        #logger.debug(f"Initial best route: {[node.get_id() for node in best_route]} with distance: {best_distance}")

        while improved:
            improved = False
            for i in range(len(best_route) - 2):
                for j in range(i + 2, len(best_route)):
                    if j - i == 1: 
                        continue

                    new_route = best_route[:]
                    new_route[i:j] = reversed(new_route[i:j])

                    new_dist = self.route_distance(new_route)
                    #logger.debug(f"Testing swap between indices {i} and {j}: New route: {[node.get_id() for node in new_route]} with distance: {new_dist}")

                    if new_dist < best_distance:
                        #logger.debug(f"Improved route found: {[node.get_id() for node in new_route]} with distance: {new_dist}")
                        best_route = new_route
                        best_distance = new_dist
                        improved = True
                        break

                if improved:
                    break

        #logger.debug(f"Final optimized route: {[node.get_id() for node in best_route]} with distance: {best_distance}")
        return best_route

    def route_distance(self, customers):
        """
        depot 제외한 순서의 거리 계산:
        - 만약 실제로는 depot~customer도 필요하다면 별도 인자로 depot ID/Node를 붙여서 계산해야 함.
        - 여기서는 "customers만"의 내부 거리합을 예시로 계산.
        """
        dist = 0.0
        #logger.debug(f"Calculating distance for customer list: {[node.get_id() for node in customers]}")

        for i in range(len(customers) - 1):
            d = customers[i].distance(customers[i + 1])
            #logger.debug(f"Distance between {customers[i].get_id()} and {customers[i + 1].get_id()}: {d}")
            dist += d

        #logger.debug(f"Total route distance: {dist}")
        return dist

        
    def optimize_depots(self, solution: Solution) -> Solution:

        # (1) Merged Route 생성
        merged_route = []
        for sub_route in solution.tours:
            if len(sub_route) > 2:
                merged_route.extend(sub_route[1:-1])
            elif len(sub_route) == 3:
                merged_route.append(sub_route[1])
        #logger.debug(f"Merged route (customers only): {[n.id for n in merged_route]}")

        original_route = self.flatten_original_routes(solution)
        original_distance = self.problem.calculate_route_distance(original_route)

        depot = self.problem.get_depot()
        CAPACITY = self.problem.get_capacity()
        BATTERY = self.problem.get_battery_capacity()

        LOAD_THRESHOLD = 0.4 * CAPACITY
        ENERGY_THRESHOLD = 0.4 * BATTERY

        new_tours = []
        i = 0

        while i < len(merged_route):
            segment_nodes = []
            current_load = CAPACITY
            current_energy = BATTERY

            threshold_list = []

            # Segment 확장
            while i < len(merged_route):
                next_node = merged_route[i]
                demand = next_node.get_demand()

                if len(segment_nodes) == 0:
                    prev_node = depot
                    load_for_battery = current_load
                else:
                    prev_node = segment_nodes[-1]
                    load_for_battery = current_load

                battery_needed_1 = self.problem.get_energy_consumption(prev_node, next_node, load_for_battery)
                battery_needed_2 = self.problem.get_energy_consumption(next_node, depot, (load_for_battery - demand))
                battery_needed = battery_needed_1 + battery_needed_2

                #logger.debug(f"Evaluating node {next_node.id}: Demand={demand}, "
                #            f"Battery Needed={battery_needed}, Current Load={current_load}, "
                #            f"Current Energy={current_energy}")

                if (demand > current_load) or (battery_needed > current_energy):
                    #logger.debug(f"Node {next_node.id} cannot be added. Breaking segment.")
                    break

                segment_nodes.append(next_node)
                current_load -= demand
                current_energy -= battery_needed_1

                if current_load <= LOAD_THRESHOLD or current_energy <= ENERGY_THRESHOLD:
                    threshold_list.append(next_node)
                    #logger.debug(f"Threshold reached at node {next_node.id}: "
                    #            f"Load={current_load}, Energy={current_energy}")

                i += 1

            #logger.debug(f"Current Segment Nodes: {[n.id for n in segment_nodes]}")

            # Delta L로 Segment 끝 조정
            chosen_end_idx = len(segment_nodes) - 1
            if threshold_list:
                best_idx = chosen_end_idx
                min_delta = float('inf')

                for thr_node in threshold_list:
                    thr_pos = segment_nodes.index(thr_node)
                    if thr_pos == 0:
                        prev_node = depot
                    else:
                        prev_node = segment_nodes[thr_pos - 1]

                    if thr_pos == len(segment_nodes) - 1:
                        next_node = depot
                    else:
                        next_node = segment_nodes[thr_pos + 1]

                    dist_prev_thr = self.problem.get_distance(prev_node, thr_node)
                    dist_thr_next = self.problem.get_distance(thr_node, next_node)
                    dist_prev_next = self.problem.get_distance(prev_node, next_node)

                    delta_l = dist_prev_thr + dist_thr_next - dist_prev_next
                    #logger.debug(f"Delta L at node {thr_node.id}: {delta_l}")

                    if delta_l < min_delta:
                        min_delta = delta_l
                        best_idx = thr_pos

                chosen_end_idx = best_idx

            chosen_subroute = [depot] + segment_nodes[:(chosen_end_idx + 1)] + [depot]
            new_tours.append(chosen_subroute)

            remaining_nodes = segment_nodes[(chosen_end_idx + 1):]
            leftover = remaining_nodes + merged_route[i:]
            merged_route = leftover
            i = 0

            if len(merged_route) == 0:
                break

        new_solution = Solution(tours=new_tours)
        new_sol_distance = self.problem.calculate_route_distance(
            [node for subr in new_tours for node in subr]
        )

        #logger.debug(f"Original Distance={original_distance}, New Distance={new_sol_distance}")

        if new_sol_distance < original_distance:
            new_solution.set_tour_length(new_sol_distance)
            return new_solution
        else:
            return solution


    def flatten_original_routes(self, solution: Solution):
        """
        solution에 들어있는 모든 sub-route를 하나로 이어붙여(Depot 포함) 
        단일 리스트(예: [0,2,5,6,0,7,9,1,0,8,3,4,0]) 형태로 반환.
        """
        flat_route = []
        for sub_route in solution.tours:
            flat_route.extend(sub_route)
        return flat_route



    def _insert_depot_with_min_delta(self, segment):
        """
        Delta L 기법으로 segment 내부에 depot을 삽입하여 
        거리 증가분(Delta L)이 가장 작은 위치를 찾는다.
        segment 예: [Depot, c1, c2, c3, ..., (Depot 미포함)]
        - 첫 노드는 이미 Depot이므로 일반적으로 [0]은 제외
        - 마지막에 Depot이 없다고 가정한 상태에서 삽입 위치를 찾는다.
        """

        depot = self.problem.get_depot()
        if len(segment) <= 2:
            return  # 고객이 없거나 1명인 경우 굳이 삽입할 필요가 없을 수 있음
        
        best_insertion_index = -1
        min_delta = float('inf')

        # segment 내 각 위치를 순회하며 delta L 계산
        for i in range(1, len(segment)):  # 0번(Depot)은 제외
            prev_node = segment[i - 1]
            curr_node = segment[i]
            # Delta L: prev->Depot->curr - (prev->curr)
            dist_prev_curr = self.problem.get_distance(prev_node, curr_node)
            dist_prev_depot = self.problem.get_distance(prev_node, depot)
            dist_depot_curr = self.problem.get_distance(depot, curr_node)

            delta_l = dist_prev_depot + dist_depot_curr - dist_prev_curr
            if delta_l < min_delta:
                min_delta = delta_l
                best_insertion_index = i

        # 삽입
        if best_insertion_index > 0:
            segment.insert(best_insertion_index, depot)



