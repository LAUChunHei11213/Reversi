import MCTS as rvs
import tkinter as Tk
import time
import tkinter.messagebox

total=[]
class ReversiBoard(Tk.Canvas):
    cell_size = 46
    margin = 5
    board = rvs.getInitialBoard()
    validBoard = True
    isPayerTurn = True
    step = []
    def __init__(self, master, mode=1):
        self.mode = mode
        cwidth = rvs.BOARD_SIZE * self.cell_size
        Tk.Canvas.__init__(self, master, relief=Tk.RAISED, bd=4, bg='white', width=cwidth, height=cwidth,cursor="cross")
        self.bind("<1>", self.put_stones)
        for i in range(rvs.BOARD_SIZE):
            for j in range(rvs.BOARD_SIZE):
                bcolor = "#808000"
                x0 = i * self.cell_size + self.margin
                y0 = j * self.cell_size + self.margin
                self.create_rectangle(x0, y0, x0 + self.cell_size, y0 + self.cell_size, fill=bcolor, width=1)
        # 1: PLAYER vs MCTS
        # 2: MYAI vs MCTS
        if self.mode == 1:
            rvs.PLAYER1 = 2
        else:
            rvs.PLAYER1 = 3
        self.refresh()

    def put_stones(self, event):  # 放置棋子
        # 是否游戏结束
        if self.validBoard == False:
            self.validBoard = True
            self.board = rvs.getInitialBoard()
            self.isPayerTurn = True

            for numid in self.step:
                self.delete(numid)
            self.step = []
            self.refresh()
            return

        # 电脑轮次
        if not (self.isPayerTurn):
            return
        # 玩家轮次
        if self.mode == 1:
            x = self.canvasx(event.x)
            y = self.canvasy(event.y)
        # 获得坐标
        i = int(x / self.cell_size)
        j = int(y / self.cell_size)
        if self.board[i][j] != 0 or rvs.updateBoard(self.board, rvs.PLAYER1, i, j, self.mode, checkonly=True) == 0:
            return

        rvs.updateBoard(self.board, rvs.PLAYER1, i, j, self.mode)
        self.refresh()
        self.isPayerTurn = False
        self.after(10, self.AI_move)

    def AI_move(self):
        while True:
            player_possibility = len(rvs.possible_positions(self.board, rvs.PLAYER1, self.mode))
            mcts_possibility = len(rvs.possible_positions(self.board, rvs.PLAYER2, self.mode))
            if mcts_possibility == 0:
                break
            start= time.time()
            stone_pos = rvs.mctsNextPosition(self.board, self.mode)
            end =time.time()
            one_time=end-start
            print("Computer position:", stone_pos)
            print("Step time:",format(one_time, '.4f'),"s")
            total.append(one_time)
            rvs.updateBoard(self.board, rvs.PLAYER2, stone_pos[0], stone_pos[1], self.mode)
            self.refresh()

            player_possibility = len(rvs.possible_positions(self.board, rvs.PLAYER1, self.mode))
            mcts_possibility = len(rvs.possible_positions(self.board, rvs.PLAYER2, self.mode))

            if mcts_possibility == 0 or player_possibility > 0:
                break

        if player_possibility == 0 and mcts_possibility == 0:
            self.showResult()
            self.validBoard = False

        self.isPayerTurn = True

    def showResult(self):
        player_stone = rvs.countTile(self.board, rvs.PLAYER1)
        mcts_stone = rvs.countTile(self.board, rvs.PLAYER2)

        if player_stone > mcts_stone:
            tkinter.messagebox.showinfo('Game Over', "You won")

        elif player_stone == mcts_stone:
            tkinter.messagebox.showinfo('Game Over', "Draw")

        else:
            tkinter.messagebox.showinfo('Game Over', "You lose")
        print(sum(total))

    def refresh(self):
        player_possible_position = rvs.possible_positions(self.board, rvs.PLAYER1, self.mode)
        for i in range(rvs.BOARD_SIZE):
            for j in range(rvs.BOARD_SIZE):
                x0 = i * self.cell_size + self.margin
                y0 = j * self.cell_size + self.margin
                if (i, j) in player_possible_position:
                    bcolor = "#92a86d"
                elif self.board[i][j] == 0:
                    bcolor = "#808000"
                if self.board[i][j] == rvs.PLAYER1:
                    bcolor = "#000000"
                if self.board[i][j] == rvs.PLAYER2:
                    bcolor = "#ffffff"
                self.create_oval(x0 + 2, y0 + 2, x0 + self.cell_size - 2, y0 + self.cell_size - 2, fill=bcolor,
                                         width=0)


class Reversi(Tk.Frame):
    def __init__(self, master=None, mode=1):
        Tk.Frame.__init__(self, master)
        self.master.title("打败人工智障")
        l_title = Tk.Label(self, text='Reversi_AI', font=('Times', '24', ('italic', 'bold')), fg='#191970', bg='#EEE8AA',
                           width=12)
        l_title.pack(padx=10, pady=10)
        self.f_board = ReversiBoard(self, mode)
        self.f_board.pack(padx=10, pady=10)


if __name__ == '__main__':
    app = Reversi(mode=1)
    app.pack()
    app.mainloop()
