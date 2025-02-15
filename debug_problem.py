from collections import OrderedDict
from copy import deepcopy
import os
from random import shuffle, choice
import numpy as np
from matplotlib import pyplot as plt

from objects.node import Node
from objects.debug_solution import Solution

from src.utils import get_problem_name, logger

class Problem(): #문제 데이터를 읽어와서 problem 객체로 초기화
    
    def __init__(self, problem_path=None):
        """
        Initializes an instance of the class with the given parameters.

        :param problem_name: A string representing the name of the problem to be solved (e.g. E-n22-k4)
        :type problem_name: str
        :param dataset_path: A string representing the path of the dataset to be used (default is ./EVRP/benchmark/)
        :type dataset_path: str
        """
        self.problem_path = problem_path # 문제 인스턴스 초기화
        self.problem_name = get_problem_name(problem_path)
        if not os.path.isfile(problem_path):
            raise ValueError(f"Problem file not found: {problem_path}. Please input a valid problem name.")

        #self.max_num_vehicles = None
        self.energy_capacity = None
        self.capacity = None
        #self.num_stations = None
        self.num_dimensions = None
        self.optimal_value = None
        self.energy_consumption = None
        self.nodes = []
        self.node_dict = dict()
        self.customers = []
        self.customer_ids = []
        #self.stations = []
        #self.station_ids = []
        self.demands = []
        self.depot = None
        self.depot_id = None

        self.problem = self.__read_problem(problem_path)
        
    def get_name(self):
        return self.problem_name
            
    def get_problem_size(self):
        return len(self.nodes)
    
    def get_depot(self):
        return self.depot
    
    def get_num_customers(self):
        return self.num_customers
    
    #def get_num_stations(self):
    #    return self.num_stations
    
    def get_num_dimensions(self):
        return self.num_dimensions
    
    #def get_max_num_vehicles(self):
    #    return self.max_num_vehicles
    
    def get_customer_demand(self, node):
        return node.get_demand()
    
    def get_energy_consumption(self, from_node, to_node, current_load=0.0):
        distance = from_node.distance(to_node)
        #logger.debug(f"from_node_to_node distance => {distance}")
        base = self.energy_consumption
        lf = self.load_factor
        #logger.debug(f"current_load => {current_load}")
        return distance * (base + lf * current_load)

    
    def get_depot_id(self):
        return self.depot_id
    
    def get_customer_ids(self):
        return deepcopy(self.customer_ids)
    
    def get_battery_capacity(self):
        return self.energy_capacity
    
    def get_capacity(self):
        return self.capacity
    
    def get_all_customers(self):
        return self.customers
    
    def get_node_from_id(self, id):
        return self.node_dict[id]
    
    def get_distance(self, from_node, to_node):
        return from_node.distance(to_node)
        
    def __read_problem(self, problem_file_path):
        with open(problem_file_path, 'r') as f:
            lines = f.readlines()
            
            """ Read metadata """
            logger.info(f"Read problem file: {problem_file_path}")
            for i in range(10):
                logger.debug(f"Metadata Line {i + 1}: {lines[i].strip()}")
            
            self.num_dimensions = int(lines[4].split()[-1])
            self.num_customers = self.num_dimensions - 1
            self.capacity = float(lines[5].split()[-1])
            self.energy_capacity = float(lines[6].split()[-1])
            self.energy_consumption = float(lines[7].split()[-1])
            self.load_factor = float(lines[8].split()[-1])
            edge_weight_type = lines[9].split()[-1] 
            
            logger.info(f"Problem dimensions: {self.num_dimensions}")
            logger.info(f"Capacity: {self.capacity}, Energy Capacity: {self.energy_capacity}")
            logger.info(f"Energy Consumption: {self.energy_consumption}, Load Factor: {self.load_factor}")
            logger.info(f"Edge Weight Type: {edge_weight_type}")
            
            """ Read NODES """
            if edge_weight_type == 'EUC_2D':
                start_line = 11
                end_line = start_line + self.num_dimensions
                logger.debug(f"Reading nodes from line {start_line} to {end_line - 1}")

                for i in range(start_line, end_line):
                    id, x, y = lines[i].split()
                    id = int(id) - 1
                    self.nodes.append(Node(int(id), float(x), float(y)))
                    self.node_dict[id] = self.nodes[-1]
                    logger.debug(f"Node {id} added at ({x}, {y}).")

                start_line = end_line + 1
                end_line = start_line + self.num_customers +1
                logger.debug(f"Reading customer demands from line {start_line} to {end_line - 1}")
                
                for i in range(start_line, end_line):
                    _id, demand = lines[i].split()[-2:]
                    _id = int(_id) - 1
                    demand = float(demand)
                    self.demands.append(demand)
                    self.nodes[_id].set_type('C')
                    self.nodes[_id].set_demand(demand)
                    self.customer_ids.append(_id)
                    self.customers.append(self.nodes[_id])
                    logger.debug(f"Customer node {_id} with demand {demand} added.")

                # Depot 설정
                self.depot_id = 0
                self.nodes[self.depot_id].set_type('D')
                self.depot = self.nodes[self.depot_id]
                logger.info(f"Depot set at Node {self.depot_id}.")

                # 고객 목록에서 depot 제거
                self.customer_ids.remove(self.depot_id)
                for i in range(len(self.customers)):
                    if self.customers[i].is_depot():
                        logger.debug(f"Depot found in customer list at index {i}. Removing it.")
                        self.customers.pop(i)
                        break

                # **고객 수에서 depot 제거 반영 (빠진 부분 추가)**  
                self.num_customers -= 1
                logger.info(f"Total customers after depot removal: {len(self.customers)}")
                logger.info(f"Total nodes: {len(self.nodes)}")

            else:
                logger.error(f"Unsupported edge weight type: {edge_weight_type}")
                raise ValueError(f"Invalid benchmark, edge weight type: {edge_weight_type} not supported.")


    def check_valid_solution(self, solution, verbose=False):
        """
        동적 배터리 & 배달형 로직을 고려:
        - depot에서 출발할 때 load = capacity, battery = energy_capacity
        - 이동 전 배터리 계산: get_energy_consumption(..., current_load)
        - 방문 시 demand만큼 load 감소
        - depot으로 돌아오면 load, battery 리셋
        """
        is_valid = solution.set_tour_index()
        if not is_valid:
            if verbose:
                logger.warning("The vehicle has visited a customer more than once.")
            return False

        tours = solution.get_tours()
        visited = {}

        for tour in tours:
            current_energy = self.get_battery_capacity()
            current_load   = self.get_capacity()

            for i in range(len(tour) - 1):
                from_node = tour[i]
                to_node   = tour[i + 1]

                # 동적 배터리 소모
                energy_needed = self.get_energy_consumption(from_node, to_node, current_load)
                current_energy -= energy_needed

                if current_energy < 0.0:
                    if verbose:
                        logger.warning(f"Battery exceeded at node {to_node.get_id()}")
                    return False

                # 배달형: 방문 시, load 감소
                if to_node.is_customer():
                    # 한 고객을 두 번 방문?
                    if to_node.get_id() in visited:
                        if verbose:
                            logger.warning(f"Customer {to_node.get_id()} visited more than once.")
                        return False
                    visited[to_node.get_id()] = 1

                    demand = to_node.get_demand()
                    current_load -= demand
                    if current_load < 0.0:
                        if verbose:
                            logger.warning(f"Capacity exceeded at node {to_node.get_id()}")
                        return False

                # depot이면 load & battery 리셋
                if to_node.is_depot():
                    current_load   = self.get_capacity()
                    current_energy = self.get_battery_capacity()

        return True
    def calculate_route_distance(self, route):
        """
        주어진 경로(route)의 총 거리를 계산합니다.
        """
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += route[i].distance(route[i + 1])  # 각 노드 간의 거리 계산
        return total_distance
    def random_solution(self):
        """
        Returns:
            solution (Solution): a randomly generated solution for the EVRP problem instance
            
            Example:
            Basic presentation: [0, 1, 2, 3, 0, 4, 5, 0, 6, 0]
            Vehicle tours: 
                Vehicle 1: 0 -> 1 -> 2 -> 3 -> 0
                Vehicle 2: 0 -> 4 -> 5 -> 0
                Vehicle 3: 0 -> 6 -> 0
                
        (*) Note:   The solution generated by the algorithm is not guaranteed to be valid 
                    in terms of capacity and energy constraints.
                    Your task is to modify the solution to a valid one that has the shortest tour length.
        """
        solution = Solution()

        cust_ids = self.get_customer_ids()
        shuffle(cust_ids)

        depot = self.get_depot()

        # 첫 출발
        current_load   = self.get_capacity()        # 가득 실음(배달형)
        current_energy = self.get_battery_capacity()
        tour = [depot]

        for cid in cust_ids:
            next_node = self.get_node_from_id(cid)
            demand    = next_node.get_demand()

            # "현재노드 -> next_node" 에 필요한 배터리
            needed_to_next = self.get_energy_consumption(tour[-1], next_node, current_load)
            # "next_node -> depot" 에는 방문 후 (current_load - demand)일 때 배터리 소모
            needed_to_depot = self.get_energy_consumption(next_node, depot, current_load - demand)

            total_needed = needed_to_next + needed_to_depot

            # 1) 수요가 현재 load보다 큰 경우?
            # 2) (다음 노드 + depot)까지 갈 배터리가 부족한 경우?
            if demand > current_load or total_needed > current_energy:
                # depot 복귀 (단, 이미 depot이면 생략)
                if tour[-1] != depot:
                    tour.append(depot)

                # load, battery 리셋
                current_load   = self.get_capacity()
                current_energy = self.get_battery_capacity()

                # 다시 계산
                needed_to_next = self.get_energy_consumption(tour[-1], next_node, current_load)
                needed_to_depot= self.get_energy_consumption(next_node, depot, current_load - demand)
                total_needed   = needed_to_next + needed_to_depot

                # 그래도 안 되면 스킵
                if demand > current_load or total_needed > current_energy:
                    continue

            # 충분하면 이동
            tour.append(next_node)
            # 배터리 소모
            current_energy -= needed_to_next
            # 배달 -> load -= demand
            current_load   -= demand

        # 마무리로 depot 복귀
        if tour[-1] != depot:
            tour.append(depot)

        solution.add_tour(tour)
        solution.set_tour_index()
        return solution
    
    def stochastic_greedy_solution(self, k=3) -> Solution:
        """
        Generates a stochastic greedy solution for the EVRP problem without using visited_set.
        
        Returns:
            Solution: A solution generated by stochastically selecting k-nearest customers.
        """
        solution = Solution()

        cust_ids = self.get_customer_ids()
        depot = self.get_depot()

        current_load = self.get_capacity()  # 초기 적재량
        current_energy = self.get_battery_capacity()  # 초기 배터리
        tour = [depot]  # depot에서 출발

        # 1. 첫 고객은 랜덤으로 선택
        if cust_ids:
            first_customer_id = choice(cust_ids)
            first_customer = self.get_node_from_id(first_customer_id)
            logger.debug(f"Starting stochastic greedy solution with first customer {first_customer_id}")

            needed_to_first = self.get_energy_consumption(depot, first_customer, current_load)

            tour.append(first_customer)
            cust_ids.remove(first_customer_id)  # 방문한 고객 제거
            current_load -= first_customer.get_demand()
            current_energy -= needed_to_first


        # 2. 나머지 고객은 탐욕적 방식으로 선택
        while cust_ids:
            current_node = tour[-1]

            # k-nearest customer 찾기
            distances = [(cid, self.get_distance(current_node, self.get_node_from_id(cid))) for cid in cust_ids]
            distances.sort(key=lambda x: x[1])  # 거리순 정렬
            k_nearest = [cid for cid, _ in distances[:k]]  # 가장 가까운 k명 선택

            if not k_nearest:
                break

            found_next_customer = False  # 고객을 찾았는지 플래그

            # k명 중에서 무작위 선택 및 조건 확인
            for next_customer_id in k_nearest:
                next_customer = self.get_node_from_id(next_customer_id)
                demand = next_customer.get_demand()

                needed_to_next = self.get_energy_consumption(current_node, next_customer, current_load)
                needed_to_depot = self.get_energy_consumption(next_customer, depot, current_load - demand)
                total_needed = needed_to_next + needed_to_depot

                # 적재량 및 배터리 조건 확인
                if demand <= current_load and total_needed <= current_energy:
                    # 고객 방문
                    tour.append(next_customer)
                    cust_ids.remove(next_customer_id)  # 방문한 고객 제거
                    current_load -= demand
                    current_energy -= needed_to_next
                    found_next_customer = True
                    break  # 고객을 찾았으면 루프 종료

            # k명 모두 조건 미충족 시 depot 복귀 및 재출발
            if not found_next_customer:
                tour.append(depot)
                current_load = self.get_capacity()
                current_energy = self.get_battery_capacity()

                # depot에서 다시 출발할 고객 랜덤 선택
                if cust_ids:
                    next_start_customer_id = choice(cust_ids)
                    next_start_customer = self.get_node_from_id(next_start_customer_id)
                    tour.append(next_start_customer)
                    cust_ids.remove(next_start_customer_id)  # 방문한 고객 제거
                    current_load -= next_start_customer.get_demand()
                    current_energy -= self.get_energy_consumption(depot, next_start_customer, current_load)

        # 마지막으로 depot 복귀
        if tour[-1] != depot:
            tour.append(depot)

        solution.add_tour(tour)
        solution.set_tour_index()
        logger.debug(f"Final tour_index: {solution.tour_index}")
        return solution

    
    def get_tour_length(self, tour):
        tour_length = 0
        for i in range(len(tour) - 1):
            tour_length += tour[i].distance(tour[i + 1])
        return tour_length
    
    def calculate_tour_length(self, solution: Solution):
        tour_length = 0
        # 전체 투어를 하나의 리스트로 결합
        full_tour = [self.get_depot()] + [node for tour in solution.get_tours() for node in tour] + [self.get_depot()]
        
        # 거리 계산
        for i in range(len(full_tour) - 1):
            tour_length += full_tour[i].distance(full_tour[i + 1])
        
        # 유효성 검사
        if self.check_valid_solution(solution):
            return tour_length
        else:
            # 유효하지 않으면 패널티 부여
            return tour_length * 2

    
    def plot(self, solution=None, path=None):
        """
        Plot the solution of the vehicle routing problem on a scatter plot.

        Args:
            solution (Solution): A `Solution` object containing the solution of the vehicle routing problem.

        Returns:
            None.
        """

        _, ax = plt.subplots()

        for node in self.nodes:
            if node.is_customer():
                ax.scatter(node.x, node.y, c='green', marker='o',
                        s=30, alpha=0.5, label="Customer Node")
            elif node.is_depot():
                ax.scatter(node.x, node.y, c='red', marker='s',
                        s=30, alpha=0.5, label="Depot Node")
            else:
                raise ValueError("Invalid node type")

        # Set title and labels
        ax.set_title(f"Problem {self.problem_name}")

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                loc='upper right',
                prop={'size': 6})

        # 솔루션 플롯
        if solution is not None:
            full_tour = []
            capacity_temp = self.get_capacity()
            energy_temp = self.get_battery_capacity()

            # 전체 투어 생성 (연속적인 하나의 경로)
            for tour in solution.get_tours():
                for node in tour:
                    full_tour.append(node)

                    # 용량 및 에너지 체크
                    capacity_temp -= node.get_demand()
                    if len(full_tour) > 1:
                        energy_temp -= self.get_energy_consumption(full_tour[-2], node)

                    # 용량 또는 에너지가 부족하면 depot 복귀
                    if capacity_temp < 0 or energy_temp < 0:
                        full_tour.append(self.get_depot())  # depot 복귀
                        capacity_temp = self.get_capacity()
                        energy_temp = self.get_battery_capacity()

            # depot에서 시작하고, 마지막에 depot 추가
            full_tour = [self.get_depot()] + full_tour + [self.get_depot()]

            # 경로 시각화
            for i in range(len(full_tour) - 1):
                first_node = full_tour[i]
                second_node = full_tour[i + 1]
                plt.plot([first_node.x, second_node.x],
                        [first_node.y, second_node.y],
                        c='black', linewidth=0.5, linestyle='--')

        # 결과 출력 또는 저장
        if path is None:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()

if __name__ == "__main__":
    # evrp = EVRP('X-n1006-k43-s5', dataset_path='./EVRP/benchmark-2022/')
    evrp = Problem('E-n22-k4', dataset_path='./EVRP/benchmark-2019/')
    solution = evrp.random_solution()
    logger.info("Random solution is {}".format("valid" if evrp.check_valid_solution(solution, verbose=True) else "invalid"))
    print(solution)
    evrp.plot(solution)
        
    
    