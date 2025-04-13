import csv
import random
import torch
import torch.nn as nn
from math import sqrt, log
import Env
import numpy as np
import time
MCTS_ITERATIONS = 100

def create_feature(board, current_player):
    directions = [(-1,-1), (-1,0), (-1,1),
                (0,-1),          (0,1),
                (1,-1),  (1,0),  (1,1)]
    def get_valid(board):
        valid = np.zeros((8,8),dtype=np.float32)
        for x in range(8):
            for y in range(8):
                if is_valid(board, x, y):
                    valid[x][y] = 1
        return valid
    def is_valid(board, x, y):
        if board[x][y] != 0:
            return False
        for dx, dy in directions:
            if check_direction(board, x, y, dx, dy):
                return True
        return False

    def check_direction(board, x, y, dx, dy):
        nx, ny = x + dx, y + dy
        found_opponent = False
        while 0 <= nx < 8 and 0 <= ny < 8:
            if board[nx][ny] == -1:
                found_opponent = True
                nx += dx
                ny += dy
            elif board[nx][ny] == 1:
                return found_opponent
            else:
                break
        return False
    # 初始化特征数组
    features = np.zeros((5, 8, 8), dtype=np.float32)
    

    # 基础特征
    np_board = np.array(board)*current_player
    features[0, :, :] = (np_board == 1).astype(np.float32)  # 我方位置
    features[1, :, :] = (np_board == -1).astype(np.float32)  # 敌方位置
    features[2, :, :] = (np_board == 0).astype(np.float32)   # 空白位置
    
    # 动态特征
    features[3, :, :] = get_valid(np_board)  # 需要实现get_legal_moves

    # 战略位置
    features[4,:,0] = 1  # 上边缘
    features[4,:,7] = 1 # 下边缘
    features[4,0,:] = 1  # 左边缘
    features[4,7,:] = 1 # 右边缘
    return features

class MCTSNode:
    def __init__(self, env, parent=None, action_taken=None, prior_prob=1.0, vision = 1):
        #self.env = copy.deepcopy(env)
        self.env = env
        self.parent = parent
        self.action_taken = action_taken
        self.vision = vision
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions = self.env.get_valid_moves()
        self.prior_prob = prior_prob

    def print_tree(self, depth=0):
        # 根节点无动作，子节点显示动作坐标
        action_str = f"Root" if depth == 0 else f"Action: {self.action_taken}"
        value_avg = self.value / self.visits if self.visits != 0 else 0.0
        node_info = (
            f"{'  ' * depth}|-- {action_str}"
            f"\tVisits: {self.visits}"
            f"\tValue: {value_avg:.2f}"
            f"\tPrior: {self.prior_prob:.2f}"
        )
        print(node_info)
        for child in self.children:
            child.print_tree(depth + 1)

    def select_child(self):
        best_score = -float('inf')
        best_child = None
        for child in self.children:
            if self.prior_prob == -1:
                ucb = child.value/child.visits + sqrt(2*np.log(self.visits)/child.visits)
            else:
                ucb = child.value / child.visits + 2 * child.prior_prob * sqrt(self.visits) / child.visits
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child

class MCTSAgent:
    def __init__(self, iterations=None, time_limit = 3, policy_model = None, value_model = None):
        self.time_limit = time_limit
        self.iterations = iterations    
        self.policy_model = policy_model
        self.value_model = value_model
        self.data = []
    def selection(self, node):
        while node.untried_actions == [] and node.children:
            node = node.select_child()   
        return node 
    def expansion(self, node):
        if node.untried_actions and not node.env.is_game_over():
                action = random.choice(node.untried_actions)
                if self.policy_model == None:
                    new_env = node.env.fast_copy()
                    vision=new_env.current_player
                    new_env.step(action)
                    child = MCTSNode(new_env, parent=node, action_taken=action,prior_prob=-1,vision=vision)
                else:
                    feature = create_feature(node.env.board, node.env.current_player)
                    tensor_feature = torch.from_numpy(feature).to("cuda").unsqueeze(0)
                    with torch.no_grad():
                        logits = self.policy_model(tensor_feature)
                        prior_prob = nn.functional.softmax(logits, dim=1)[0]
                    new_env = node.env.fast_copy()
                    vision = node.env.current_player
                    new_env.step(action)
                    child = MCTSNode(new_env, parent=node, action_taken=action, prior_prob=prior_prob[8*action[0]+action[1]].item(), vision=vision)
                node.children.append(child)
                node.untried_actions.remove(action)
                node = child
        return node
    def backpropagation(self, node,result):
        while node is not None:
            node.visits += 1
            node.value += result*node.vision
            node = node.parent       
    def simulation(self, node):
        if self.value_model == None:
                sim_env = node.env.fast_copy()
                while not sim_env.is_game_over():
                    actions = sim_env.get_valid_moves()
                    if actions:
                        action = random.choice(actions)
                        sim_env.step(action)
                    else:
                        sim_env.current_player *= -1
                result = sim_env.get_score()
        else:
            feature = create_feature(node.env.board, node.env.current_player)
            tensor_feature = torch.from_numpy(feature).to("cuda").unsqueeze(0)
            with torch.no_grad():
                result = self.value_model(tensor_feature)*node.env.current_player
        return result
    def steps(self, root):
        node = root
        # Selection
        node = self.selection(node)
        # Expansion
        node = self.expansion(node)
        # Simulation
        result = self.simulation(node)   
        # Backpropagation    
        self.backpropagation(node,result)
    def search(self, env):
        root_env = env.fast_copy()
        root = MCTSNode(root_env)
        start = time.time()
        if self.iterations == None:
            while time.time() - start < self.time_limit:
                self.steps(root)
        else:
            for _ in range(self.iterations):
                self.steps(root)
        #********mcts的data，用於顯示棋面勝率************#
        mcts_data = {}
        if root.children:
            for child in root.children:
                key = child.action_taken
                mcts_data[key] = (
                    child.visits,
                    child.value / child.visits if child.visits else 0,
                    child.prior_prob
                )
        #**************************************************#

        #root.print_tree()   #打印mcts

        best_child = max(root.children, key=lambda c: c.visits) if root.children else None #選擇最優子節點
        #*************************************儲存data analysis的data*************************************#
        if self.value_model != None:
            best_child.value = best_child.value.item()
        self.data.append([best_child.prior_prob, best_child.value, time.time()-start])
        #*****************************************************************************#

        return best_child.action_taken if best_child else None, mcts_data

