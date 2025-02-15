import numpy as np

class Node():
    def __init__(self, id, x, y):
        self.id = id #노드의 고유번호
        self.x = x #노드의 x좌표
        self.y = y #노드의 y좌표
        self.type = None #node type 초기는 none
        self.demand = 0 #해당 고객의 노드 수요
    
    def __str__(self):
        return str(self.id)
    
    def get_x(self):
        return self.x
    
    def get_y(self):
        return self.y
    
    def get_xy(self):
        return (self.x, self.y)
    
    def get_type(self):
        return self.type
    
    def get_id(self): 
        return self.id #get x부터 get id 까지 노드의 기본 정보를 반환 ex) 1 23 34 C ID
    
    def set_type(self, type):
        if type not in ['C', 'D']:
            raise ValueError(f"Invalid type: {type}, must be 'C', or 'D'.")
        self.type = type #노드 타임 설정 C 커스토머  D depot
    
    def get_demand(self):
        return self.demand
    
    def set_demand(self, demand):
        self.demand = demand
    
    def distance(self, P):
        return np.sqrt((self.x - P.x)**2 + (self.y - P.y)**2)
    
    def is_customer(self):
        return self.type == 'C'
    
    def is_depot(self):
        return self.type == 'D'
    def __repr__(self):
        return f"Node(id={self.id}, x={self.x}, y={self.y}, type={self.type}, demand={self.demand})"
    
#이 파일의 역할:
#노드의 전반적인 설명을 위해 존재하는 것이 아닙니다.
#노드의 정보를 세분화하고, 노드 객체를 효율적으로 관리 및 사용하기 위한 구조입니다.
#VRP에서 출발지, 고객, 충전소의 속성과 동작을 통합적으로 다루기 위한 객체 지향적 설계라고 보면 됩니다.

# 출발지(Depot)와 고객(Customer) 노드 생성
#node1 = Node(1, 100, 100)  # ID 1, 위치 (100, 100)
#node2 = Node(2, 200, 200)  # ID 2, 위치 (200, 200)

# 노드 유형 설정
#node1.set_type('D')  # 출발지(Depot)로 설정
#node2.set_type('C')  # 고객(Customer)으로 설정

# 고객 노드에 수요 설정
#node2.set_demand(500)

# 노드 정보 출력
#print(node1)  # 출력: 1 (ID)
#print(node1.get_type())  # 출력: 'D' (Depot)
#print(node2.get_demand())  # 출력: 500 (고객의 수요)
#print(node1.distance(node2))  # 두 노드 간 거리 계산