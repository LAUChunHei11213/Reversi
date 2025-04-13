import Env
from Env import ReversiEnv
import MCTS
from MCTS import MCTSAgent
import tkinter as tk
import pickle
import time
from models import SimpleCNN,ValueCNN
import torch
import torch.nn as nn
import csv

win_rate = 0
policy_model_v1 = torch.load('policy_net_final.pth',weights_only=False)
policy_model_v1.eval()
value_model_v1 = torch.load('value_net_final.pth',weights_only=False)
value_model_v1.eval()
class Human_agent():
    pass
class ReversiGUI:
    def __init__(self, master):
        self.mcts_data = {}  # 格式: {(x,y): (visits, value, prob)}
        self.master = master
        self.master.title("Reversi")
        self.env = ReversiEnv()
        self.agents = {Env.PLAYER1:Human_agent(),
                      Env.PLAYER2: Human_agent()}
        self.last_move = None
        self.game_move = []
        self.canvas = tk.Canvas(master, width=Env.BOARD_SIZE*Env.CELL_SIZE, height=Env.BOARD_SIZE*Env.CELL_SIZE)
        self.canvas.pack()
        self.status_label = tk.Label(master, text="Game Starting...", font=("Arial", 14))
        self.status_label.pack()
        self.canvas.bind("<Button-1>", self.handle_click)
        self.draw_board()
        self.start_game()

    def draw_board(self):
        self.canvas.delete("all")
        valid_moves = self.env.get_valid_moves()
        
        for x in range(Env.BOARD_SIZE):
            for y in range(Env.BOARD_SIZE):
                # 绘制棋盘格子
                self.canvas.create_rectangle(y*Env.CELL_SIZE, x*Env.CELL_SIZE,
                                           (y+1)*Env.CELL_SIZE, (x+1)*Env.CELL_SIZE,
                                           fill="#228B22", outline="black")
                
                # 绘制合法移动提示（半透明绿色）
                if (x, y) in valid_moves:
                    self.canvas.create_oval(y*Env.CELL_SIZE+15, x*Env.CELL_SIZE+15,
                                           (y+1)*Env.CELL_SIZE-15, (x+1)*Env.CELL_SIZE-15,
                                           fill="#00FF00", outline="")
       
        for x in range(Env.BOARD_SIZE):
            for y in range(Env.BOARD_SIZE):         
                # 绘制最新落子标记（红色边框）
                if self.last_move and (x, y) == self.last_move:
                    self.canvas.create_rectangle(y*Env.CELL_SIZE+2, x*Env.CELL_SIZE+2,
                                               (y+1)*Env.CELL_SIZE-2, (x+1)*Env.CELL_SIZE-2,
                                               outline="red", width=3)
                
                # 绘制棋子
                if self.env.board[x][y] == Env.PLAYER1:
                    self.canvas.create_oval(y*Env.CELL_SIZE+5, x*Env.CELL_SIZE+5,
                                          (y+1)*Env.CELL_SIZE-5, (x+1)*Env.CELL_SIZE-5,
                                          fill="black")
                elif self.env.board[x][y] == Env.PLAYER2:
                    self.canvas.create_oval(y*Env.CELL_SIZE+5, x*Env.CELL_SIZE+5,
                                          (y+1)*Env.CELL_SIZE-5, (x+1)*Env.CELL_SIZE-5,
                                          fill="white")
        for (x, y), (visits, value, prob) in self.mcts_data.items():
            text = f"V:{visits}\nQ:{value:.2f}\nP:{prob:.5f}"
            self.canvas.create_text(
                y*Env.CELL_SIZE + Env.CELL_SIZE/2,
                x*Env.CELL_SIZE + Env.CELL_SIZE/2,
                text=text,
                fill="black",
                font=("Arial", 12),
                anchor="center"
            )
    def handle_click(self, event):
        if isinstance(self.agents[self.env.current_player], Human_agent):
            x = event.y // Env.CELL_SIZE
            y = event.x // Env.CELL_SIZE
            valid_moves = self.env.get_valid_moves(self.env.current_player)
            if (x, y) in valid_moves:
                self.action = (x, y)

    def start_game(self):
        self.master.after(1, self.play_turn)

    def write_file(self):
            with open('mcts_1s.csv', 'a', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(self.agents[Env.PLAYER2].data)
            #with open('data_100_value_new.csv', 'a', newline='') as file:
            #    writer = csv.writer(file)
            #    writer.writerows(self.agents[Env.PLAYER1].data)
            with open('data_1s_no_value.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerows(self.agents[Env.PLAYER2].data)
    def play_turn(self):
        if self.env.is_game_over():
            score = self.env.get_score()
            global win_rate
            win_rate += score
            self.write_file()
            result = "Black Wins!" if score > 0 else "White Wins!" if score < 0 else "Draw!"
            self.status_label.config(text=f"Game Over! {result}")
            self.master.destroy()
            return
        
        current_agent = self.agents[self.env.current_player]
        if isinstance(current_agent, MCTSAgent):
            action, mcts_data = current_agent.search(self.env)
            self.mcts_data = mcts_data
            #action = current_agent.search(self.env)
        elif isinstance(current_agent, Human_agent):
            if hasattr(self, 'action') and self.action is not None:
                valid_moves = self.env.get_valid_moves(self.env.current_player)
                if self.action in valid_moves:
                    action = self.action
                else:
                    del self.action 
                    action = None
            else:
                action = None
        if action:
            self.last_move = action  # 更新最新落子位置
            self.game_move.append(action)
            self.env.step(action)
            self.draw_board()
        self.master.after(1, self.play_turn)


def main():
    root = tk.Tk()
    gui = ReversiGUI(root)
    root.mainloop()

if __name__ == "__main__":
    for i in range(99):
        main()
        print(win_rate)