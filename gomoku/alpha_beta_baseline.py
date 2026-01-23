import time
import math
import numpy as np
from gomoku.gomoku_env import GomokuEnv 

class AlphaBeta:

    def get_action(self, game: GomokuEnv):  # in here, we define the action function in the class
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            return None

        best_action = valid_actions[0]
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        original_player = game.current_player  
        self._root_player = original_player

        for action in valid_actions:
            child_env = game.clone()
            child_env.step(action)

            score = self.minimax(
                game=child_env,
                depth=self.search_depth - 1,
                alpha=alpha,
                beta=beta,
                is_maximizing_player=False
                )
            if score > best_score:
                best_score = score
                best_action = action

            alpha = max(alpha, best_score)

        return best_action

    def __init__(self, search_depth=4, evaluator=None):
        self.policy = None
        self.device = "cpu"
        self._root_player = 1   
        self.mcts = None       
        self.name = f"AlphaBeta Player (Depth {search_depth})"
        if search_depth < 2:
            raise ValueError("Search depth must be at least 2.")
        self.search_depth = search_depth
        self.evaluator = evaluator if evaluator is not None else self.stronger_evaluator

    def play(self, game: GomokuEnv, *args, **kwargs):
        self._root_player = game.current_player
        start_time = time.time()
        
        valid_actions = game.get_valid_actions()
        if not valid_actions:
            return {'action': None} 

        best_action = -1
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in valid_actions:
            child_env = game.clone()
            child_env.step(action)
            
            score = self.minimax(child_env, self.search_depth - 1, alpha, beta, is_maximizing_player=False)
            
            if score > best_score:
                best_score = score
                best_action = action
            
            alpha = max(alpha, best_score)
            

        end_time = time.time()
        print(f"[{self.name}] Chose action {best_action} with score {best_score:.2f}. Time taken: {end_time - start_time:.2f}s")
        
        game.step(best_action)
        return {'action': best_action,
                'state': np.zeros((3, game.board_size, game.board_size)), 
                'probs': np.zeros(game.board_size * game.board_size) }      


    def minimax(self, game: GomokuEnv, depth: int, alpha: float, beta: float, is_maximizing_player: bool):
        if depth == 0 or game._is_terminal():
            score = self.evaluator(game, original_player=self._root_player)
            return score

        valid_actions = game.get_valid_actions()

        if is_maximizing_player:
            max_eval = -math.inf
            for action in valid_actions:
                child_env = game.clone()
                child_env.step(action)
                eval = self.minimax(child_env, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break 
            return max_eval
        else: 
            min_eval = math.inf
            for action in valid_actions:
                child_env = game.clone()
                child_env.step(action)
                eval = self.minimax(child_env, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break 
            return min_eval

    def stronger_evaluator(self, game: GomokuEnv, original_player: int):
        if game._is_terminal():
            winner = game.winner
            if winner == original_player: return 100000
            if winner == (3 - original_player): return -100000
            return 0 

        my_player = original_player
        opponent_player = 3 - original_player
        pattern_scores = {
            "FIVE": 100000,
            "LIVE_FOUR": 10000,
            "DEAD_FOUR": 1000,
            "LIVE_THREE": 1000,
            "DEAD_THREE": 100,
            "LIVE_TWO": 10,
            "DEAD_TWO": 1,
        }

        my_score = self.calculate_patterns(game.board, my_player, pattern_scores)
        opponent_score = self.calculate_patterns(game.board, opponent_player, pattern_scores)

        return my_score - opponent_score * 1.1

    def calculate_patterns(self, board, player, scores):
        total_score = 0
        board_size = len(board)

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] == player:
                    for dr, dc in directions:
                        
                        count = 0
                        for i in range(1, 5):
                            nr, nc = r + i * dr, c + i * dc
                            if 0 <= nr < board_size and 0 <= nc < board_size and board[nr, nc] == player:
                                count += 1
                            else:
                                break
                        

                        if count == 1: 
                            total_score += scores["DEAD_TWO"]
                        elif count == 2: 
                            total_score += scores["DEAD_THREE"]
                        elif count == 3: 
                            total_score += scores["DEAD_FOUR"]
                        elif count == 4:
                            total_score += scores["FIVE"]
        return total_score

    def simple_evaluator(self, game: GomokuEnv, original_player: int):
        if game._is_terminal():
            winner = game.winner
            if winner == original_player:
                return 100000  
            elif winner == (3 - original_player):
                return -100000 
            else:
                return 0 

        score = 0
        board = game.board
        
        patterns = {
            (original_player,) * 5: 100000,
            (0,) + (original_player,) * 4 + (0,): 5000, 
            (original_player,) * 4: 500, 
            (0,) + (original_player,) * 3 + (0,): 200, 
            (original_player,) * 3: 50, 
            (0,) + (original_player,) * 2 + (0,): 10, 
            ((3 - original_player),) * 5: -100000,
            (0,) + ((3 - original_player),) * 4 + (0,): -10000, 
            ((3 - original_player),) * 4: -1000,
            (0,) + ((3 - original_player),) * 3 + (0,): -800,
            ((3 - original_player),) * 3: -80,
            (0,) + ((3 - original_player),) * 2 + (0,): -15,
        }
        
        board_size = game.board_size
        for r in range(board_size):
            for c in range(board_size - 4):
                segment = tuple(board[r, c:c+5])
                if segment in patterns:
                    score += patterns[segment]
                    
        return score

if __name__ == '__main__':

    board_size = 9

    player2 = AlphaBetaPlayer(search_depth=5)

    game = GomokuEnv(board_size)
    players = [player2, player2] 
    current_idx = 0
    while not game._is_terminal():
        game.render()
        player_to_move = players[current_idx]
        player_to_move.play(game)
        current_idx = 1 - current_idx
    
    game.render()
    print(f"Game Over! Winner is: Player {game.winner}")