import numpy as np
BOARD_SIZE = 8
PLAYER1 = 1
PLAYER2 = -1
CELL_SIZE = 100
DIRECTIONS = [(-1,-1), (-1,0), (-1,1),
              (0,-1),          (0,1),
              (1,-1),  (1,0),  (1,1)]
class ReversiEnv:
    def __init__(self):
        self.board = None
        self.current_player = None
        self.reset()

    def fast_copy(self):
        new_env = ReversiEnv()
        new_env.board = [row.copy() for row in self.board]  
        new_env.current_player = self.current_player
        return new_env    

    def reset(self):
        self.board = [[0, 0, 0, 0, 0, 0, 0, 0],
                     [ 0, 0, 0, 0, 0, 0, 0, 0],
                     [ 0, 0, 0, 0, 0, 0, 0, 0],
                     [ 0, 0, 0,-1, 1, 0, 0, 0],
                     [ 0, 0, 0, 1,-1, 0, 0, 0],
                     [ 0, 0, 0, 0, 0, 0, 0, 0],
                     [ 0, 0, 0, 0, 0, 0, 0, 0],
                     [ 0, 0, 0, 0, 0, 0, 0, 0]]
        self.current_player = PLAYER1

    def get_valid_moves(self, player=None):
        if player is None:
            player = self.current_player
        valid_moves = []
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                if self._is_valid_move(x, y, player):
                    valid_moves.append((x, y))
        return valid_moves

    def _is_valid_move(self, x, y, player):
        if self.board[x][y] != 0:
            return False
        for dx, dy in DIRECTIONS:
            if self._check_direction(x, y, dx, dy, player):
                return True
        return False

    def _check_direction(self, x, y, dx, dy, player):
        nx, ny = x + dx, y + dy
        found_opponent = False
        while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
            if self.board[nx][ny] == -player:
                found_opponent = True
                nx += dx
                ny += dy
            elif self.board[nx][ny] == player:
                return found_opponent
            else:
                break
        return False
    def step(self, action):
        x, y = action
        self.board[x][y] = self.current_player
        for dx, dy in DIRECTIONS:
            tmp_flip = []
            nx, ny = x+dx, y+dy
            while 0<=nx<8 and 0<=ny<8:
                if self.board[nx][ny] == -self.current_player:
                    tmp_flip.append((nx, ny))
                    nx += dx
                    ny += dy
                elif self.board[nx][ny] == self.current_player:
                    for (fx, fy) in tmp_flip:
                        self.board[fx][fy] = self.current_player
                    break
                else:
                    break
        self.current_player *= -1
        if len(self.get_valid_moves()) == 0:
            self.current_player *= -1

    def is_game_over(self):
        return (len(self.get_valid_moves(PLAYER1)) == 0 and 
                len(self.get_valid_moves(PLAYER2)) == 0)

    def get_score(self):
        count = np.sum(self.board)
        if count > 0:
            score = 1
        elif count < 1: 
            score = -1
        else: 
            score = 0
        return score