#%%
from gomoku.gomoku_env import GomokuEnv, GomokuEnvSimple
from gomoku.policy import ZeroPolicy
import math
import torch
import numpy as np
import random

class ZeroTreeNode:
    def __init__(self, env: GomokuEnv, parent=None, prior_prob=None): 
        self.env = env
        self.visits = 0
        self.value_sum = 0  
        self.children = {}
        self.parent = parent
        self.action_prob = None
        self.prior_prob = prior_prob if prior_prob is not None else 0.0  

    def add_child(self, action, child_node):
        self.children[action] = child_node


    def update(self, value):
        self.visits += 1
        self.value_sum += value
    

    @property
    def q_value(self):
        if self.visits == 0:
            return 0
        return self.value_sum / self.visits

    def is_fully_expanded(self):
        """Check if the node is fully expanded"""
        valid_actions = self.env.get_valid_actions()
        return len(self.children) == len(valid_actions)
    

class ZeroMCTS:
    def __init__(self, policy: ZeroPolicy, 
                puct=2, 
                device='cpu',
                dirichlet_alpha = 0.3,
                dirichlet_epsilon=0.25):

        self.policy = policy
        self.puct = puct 

        self.device = device
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

        self.env2node = {}

    def step(self, action):
        pass
    
    def hash(self, env: GomokuEnv): 
        return env.board.tobytes()
    
    def find_root(self, env): 
        return self.env2node.get(self.hash(env), None)
    
    def run(self, env, iterations, use_dirichlet=True):

        hash_key = self.hash(env)
        # print(hash_key)
        if hash_key not in self.env2node: 
            self.env2node[hash_key] = ZeroTreeNode(env.clone())
        self.root = self.env2node[hash_key]
        self.root.parent = None

        # print(self.root.)
        # print(self.root.__dict__)

        # print(self.root.env.move_size)

        if use_dirichlet and self.dirichlet_alpha > 0 and self.root.env.move_size <= 1:
            print('use dirichlet')
            self._apply_dirichlet_noise_to_root()
        
        for _ in range(iterations):
            leaf_node = self._select(self.root)
            value = self._expand_and_evaluate(leaf_node)
            self._backpropagation(leaf_node, value)
        return self._best_action()
    
    def _apply_dirichlet_noise_to_root(self):

        if self.root.action_prob is None:
 
            self._compute_policy_and_value(self.root)

        valid_actions = self.root.env.get_valid_actions()
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_actions))
        
        for i, action in enumerate(valid_actions):
            self.root.action_prob[action] = \
                (1 - self.dirichlet_epsilon) * self.root.action_prob[action] + \
                self.dirichlet_epsilon * noise[i]
    
    def _compute_policy_and_value(self, node: ZeroTreeNode):

        if node.action_prob is not None:
            return node.action_prob, node.q_value

        obs = node.env._get_observation()
        torch_x = torch.from_numpy(obs).unsqueeze(0).float().to(self.device)
        valid_actions_tensor = torch.tensor(node.env.get_valid_actions())
        
        with torch.no_grad():
            policy_logits, value_tensor = self.policy(torch_x)
            

        mask = torch.full_like(policy_logits, -1e8)
        try:
            mask[0, valid_actions_tensor] = 0.0
            
        except Exception:
            print(f"Error applying mask: {valid_actions_tensor}, logits shape: {policy_logits.shape}")
        masked_logits = policy_logits + mask
        

        action_prob_tensor = torch.softmax(masked_logits, dim=1)[0]
        node.action_prob = {
            action: action_prob_tensor[action].item() for action in valid_actions_tensor.tolist()
        }
        
        return node.action_prob, value_tensor.item()
    
    def _select(self, node: ZeroTreeNode):
        current = node
        while current.children:
            current = max(current.children.values(), key=lambda child: self._puct_value(child))
        return current
    
    def _expand_and_evaluate(self, node: ZeroTreeNode):
        if node.env._is_terminal():
            winner = node.env.winner
            if winner == 0:
                return 0.0
            if winner == (3-node.env.current_player):
                return -1.0
            else:
                return 1.0

        policy_probs, value = self._compute_policy_and_value(node) 

        valid_actions = node.env.get_valid_actions()

        for action in valid_actions:
            if action not in node.children:
                child_env = node.env.clone()
                child_env.step(action)
                child_node = ZeroTreeNode(child_env, parent=node, prior_prob=policy_probs[action])
                node.add_child(action, child_node)
                self.env2node[self.hash(child_env)] = child_node

        return value
    
    def _backpropagation(self, node: ZeroTreeNode, value: float):
        current = node
        while current is not None:
            current.update(value)
            value = -value
            current = current.parent
    
    def _puct_value(self, child: ZeroTreeNode):
        q_value = -child.q_value
        prior_prob = child.prior_prob
        
        exploration_term = self.puct * prior_prob * \
                           (math.sqrt(child.parent.visits) / (1 + child.visits))
        
        return q_value + exploration_term
    
    def _best_action(self):

        return self.select_action_with_temperature(temperature=0, top_k=1)
    
    def select_action_with_temperature(
        self, temperature=1.0, top_k=None, forbidden_actions=None
    ):
        if not self.root.children:
 
            return random.choice(self.root.env.get_valid_actions()), {}

        forbidden_set = set(forbidden_actions or [])
        child_visits = sorted(
            [(action, child, child.visits) for action, child in self.root.children.items()],
            key = lambda x: x[2], 
            reverse=True,
        )

        if forbidden_set:
            filtered = [visit for visit in child_visits if visit[0] not in forbidden_set]
            if filtered:
                child_visits = filtered
    
        if temperature == 0:

            action = max(child_visits, key=lambda x: x[2])[0]
        else:
   
            if top_k is not None:
                if top_k > len(child_visits):
                    top_k = len(child_visits)
                child_visits = child_visits[:top_k]
            
            visit_counts = np.array([visit[2] for visit in child_visits])
            visit_probs = visit_counts**(1 / temperature)
            if np.sum(visit_probs) == 0:
                visit_probs = np.ones_like(visit_probs)
            visit_probs /= np.sum(visit_probs) 
  
            action_index = np.random.choice(len(visit_counts), p=visit_probs)
            action = child_visits[action_index][0]
            
  
        pi_distribution = {a: node.visits for a, node in self.root.children.items()}
        total_visits = sum(pi_distribution.values())
        total_visits = total_visits if total_visits > 0 else 1  
        probs_for_training = [
            pi_distribution.get(i, 0) / total_visits
            for i in range(self.root.env.board_size**2)
        ]
        if forbidden_set:
            for act in forbidden_set:
                if 0 <= act < len(probs_for_training):
                    probs_for_training[act] = 0.0
            prob_sum = sum(probs_for_training)
            if prob_sum > 0:
                probs_for_training = [p / prob_sum for p in probs_for_training]
    
        return action, probs_for_training

    
#%%
if __name__ == '__main__':
    policy = ZeroPolicy(board_size=9)
    game = GomokuEnv(board_size=9)

    game.board[0, 1:4] = 1
    game.board[1, 1:4] = 2

    game.render()
    zero_player = ZeroMCTS(game, policy)
    # %%
    action = zero_player.run(800)
    print(f"Best action: {action}")
    # %%
    # zero_player.root.
    for child_action, child_node in zero_player.root.children.items():
        print(f"Action: {child_action}, Visits: {child_node.visits}, Wins: {child_node.value_sum}, Action Prob: {child_node.action_prob}")
    # %%

    game.get_valid_actions()
