from copy import deepcopy
from hashlib import md5
import numpy as np
from src.utils import logger


class Solution():
    def __init__(self, tours=None):
        self.tour_index = {}
        self.tour_length = np.inf
        if tours:
            self.tours = tours  # tours는 List[List[Node]] 형태라고 가정
            self.set_tour_index()
        else:
            self.tours = []

    def add_tour(self, tour):
        self.tours.append(tour)

    def get_num_tours(self):
        return len(self.tours)

    def set_tour_index(self):
        self.tour_index = {}
        for idx, tour in enumerate(self.tours):
            for node in tour:
                if node.is_customer():
                    if node.id not in self.tour_index:
                        self.tour_index[node.id] = idx
                    else:
                        logger.warning('Node {} already in tour {}'.format(node.id, idx))
                        return 0
        return 1

    def get_tour_index_by_node(self, node_id):
        return self.tour_index[node_id]

    def get_presentation(self):
        list_node = [[x.get_id() for x in tour] for tour in self.tours]
        return md5(str(list_node).encode()).hexdigest()

    def __ge__(self, other):
        return self.tour_length >= other.tour_length

    def __gt__(self, other):
        return self.tour_length > other.tour_length

    def __le__(self, other):
        return self.tour_length <= other.tour_length

    def __lt__(self, other):
        return self.tour_length < other.tour_length

    def __repr__(self) -> str:
        if self.tour_length < np.inf:
            presentation = f"Tour length: {self.tour_length}\n"
        else:
            presentation = "Tour length: (not calculated)\n"

        for i, tour in enumerate(self.tours):
            presentation += f"Tour {i}: " + " -> ".join(str(node.id) for node in tour) + "\n"
        return presentation

    def get_tours(self):
        return deepcopy(self.tours)

    def get_basic_tours(self):

        tours = []
        for tour in self.tours:
            _tour = [node for node in tour if node.is_customer()]
            tours.append(_tour)
        return tours

    def get_tour_length(self):
        return self.tour_length

    def set_tour_length(self, tour_length):
        self.tour_length = tour_length


    def to_array(self):
        """단일 라우트만 있을 경우를 가정하고, 해당 라우트의 노드 ID를 numpy 배열로 리턴"""
        # 만약 tours가 여러 개 있다면, 이를 어떻게 펼칠지 정의 필요
        if len(self.tours) == 1:
            return np.array([node.id for node in self.tours[0]])
        else:
            # 여러 라우트를 일렬로 펼치는 경우
            arr = []
            for t in self.tours:
                arr.extend([node.id for node in t])
            return np.array(arr)
    
    def split_tours_by_depot(self, Problem):
        """
        현재 self.tours가 이미 여러 개의 분리된 라우트인 경우 그대로 반환.
        병합된 단일 라우트라면, depot 기준으로 분리하여 반환.
        """
        depot = Problem.get_depot()
        new_tours = []

        if len(self.tours) > 1:
            # 이미 분리된 상태
            return deepcopy(self.tours)

        # 병합된 상태라고 가정하고 처리
        current_tour = []
        for node in self.tours[0]:  # self.tours는 하나의 라우트로 병합된 상태라고 가정
            if node.is_depot():
                if current_tour:
                    new_tours.append([depot] + current_tour + [depot])
                    current_tour = []
            else:
                current_tour.append(node)

        if current_tour:
            new_tours.append([depot] + current_tour + [depot])

        return new_tours

    # ------------------------------------------
    # 새로 추가: 여러 라우트를 '하나로' 이어 붙이는 함수
    # ------------------------------------------
    def merge_all_routes_into_one(self):
        """
        self.tours에 들어있는 여러 라우트를 하나의 라우트로 이어 붙인다.
        예:
          Tour0: [0,1,2,0]
          Tour1: [0,3,0]
        => [0,1,2,0,3,0]
        단, 
          - 중간에 depot(0) 중복이 있다면 필요에 따라 제거 가능.
          - 여기서는 '앞 라우트 끝'과 '뒤 라우트 시작'이 동일 depot이면 뒤 라우트 시작 depot 생략
        """
        if len(self.tours) == 0:
            return  # 아무것도 없음

        # 일단 첫 번째 라우트를 복사
        merged_route = self.tours[0][:]  # deepcopy

        # 두 번째 라우트부터 순서대로 붙임
        for i in range(1, len(self.tours)):
            curr_route = self.tours[i]

            if not curr_route:
                continue

            # 만약 merged_route의 마지막 노드와 curr_route의 첫 노드가 같은 depot이면, 
            # curr_route의 첫 노드를 스킵
            if merged_route[-1].id == curr_route[0].id:
                # 둘 다 depot일 가능성
                # curr_route를 curr_route[1:]부터 이어 붙인다
                merged_route.extend(curr_route[1:])
            else:
                # 그냥 전부 이어 붙임
                merged_route.extend(curr_route)

        # 이제 self.tours를 [merged_route] 하나만 남기도록 세팅
        self.tours = [merged_route]
        self.set_tour_index()

    def set_tour(self, tour):
        """
        외부에서 직접 하나의 투어(노드 리스트)를 넘겨받아 self.tours=[new_tour]로 설정
        """
        self.tours = [tour]
        self.set_tour_index()


    def set_tours(self, tours):
        """ 여러 개의 서브 투어를 유지하도록 강제로 설정 """
        self.tours = tours
        self.set_tour_index()