import webbrowser
import gym
import time
from itertools import count
import random
import numpy as np
from gym_backgammon.envs.backgammon import WHITE, BLACK, COLORS, TOKEN, assert_board
from abc import ABC, abstractmethod
from dice import dices
from net.model import TDGammon
from bg_board_cv import BackgammonCV, Status
import copy
import cv2

backgammon_net = TDGammon(hidden_units=40, lr=0.1, lamda=0.7, init_weights=False, seed=None)
# net_black=TDGammon(hidden_units=40,lr=0.1,lamda=0.7,init_weights=False,seed=None)
backgammon_net.load(checkpoint_path="./net/saved_models/exp1/exp1_20210218_2302_13_205025_100000.tar"
                    , optimizer=None, eligibility_traces=True)
# net_black.load(checkpoint_path="./net/saved_models/exp1/exp1_20210218_2302_13_205025_100000.tar"
#                ,optimizer=None,eligibility_traces=True)

env = gym.make('gym_backgammon:backgammon-v0')

# env = gym.make('gym_backgammon:backgammon-pixel-v0')

# random.seed(0)
# np.random.seed(0)

animation_speed_factor = 1


# class Agent:
#     def __init__(self, color):
#         self.color = color
#         self.name = 'Agent({})'.format(COLORS[color])
#
#     def roll_dice(self):
#         return (-randint(1, 6), -randint(1, 6)) if self.color == WHITE else (randint(1, 6), randint(1, 6))
#
#     def choose_best_action(self, actions, env):
#         raise NotImplementedError


class Agent(ABC):
    def __init__(self, color, name):
        self.color = color
        self.name = name  # 'AgentExample({})'.format(self.color)

    def roll_dice(self):
        return (-random.randint(1, 6), -random.randint(1, 6)) if self.color == WHITE else (
            random.randint(1, 6), random.randint(1, 6))

    @abstractmethod
    def choose_best_action(self, actions, env):
        raise NotImplemented


class AiAgent(Agent):
    def __init__(self, color, name):
        super().__init__(color, name)


class RandomAgent(AiAgent):
    def __init__(self, color, name):
        super().__init__(color, name)
        # self.color = color
        # self.name = 'AgentExample({})'.format(self.color)

    def choose_best_action(self, actions, env):
        return random.choice(list(actions)) if actions else None


# HUMAN AGENT =======================================================================================


class HumanAgent(Agent):
    def __init__(self, color, name):
        super().__init__(color, name)
        # self.name = 'HumanAgent({})'.format(COLORS[color])
        self.name = name

    def choose_best_action(self, actions=None, env=None):
        pass


# TD-GAMMON AGENT =====================================================================================


class TDAgent(AiAgent):
    def __init__(self, color, name, net):
        super().__init__(color, name)
        self.net = net
        self.name = 'TDAgent({})'.format(COLORS[color])

    def choose_best_action(self, actions, env):
        best_action = None

        if actions:
            values = [0.0] * len(actions)
            tmp_counter = env.counter
            env.counter = 0
            state = env.game.save_state()

            # Iterate over all the legal moves and pick the best action
            for i, action in enumerate(actions):
                observation, reward, done, info = env.step(action)
                values[i] = self.net(observation)

                # restore the board and other variables (undo the action)
                env.game.restore_state(state)

            # practical-issues-in-temporal-difference-learning, pag.3
            # ... the network's output P_t is an estimate of White's probability of winning from board position x_t.
            # ... the move which is selected at each time step is the move which maximizes P_t when White is to play and minimizes P_t when Black is to play.
            best_action_index = int(np.argmax(values)) if self.color == WHITE else int(np.argmin(values))
            best_action = list(actions)[best_action_index]
            env.counter = tmp_counter

        return best_action


# if __name__ == '__main__':
# make_plays()


from tkinter import *
import tkinter as tk
from tkinter import ttk
import sys
from PIL import Image, ImageTk
from tkinter import messagebox
import os

root = Tk()
root.title("Title")
root.geometry("600x600")
root.tk.call('wm', 'iconphoto', root._w, tk.PhotoImage(file="resources/ICBV211-123395-logo.png"))
root.title("Backgammon")
root.configure(background="black")


class Example(Frame):
    def __init__(self, master, *pargs):
        Frame.__init__(self, master, *pargs)
        # self.can_play = False
        self.dice_rolled = False
        self.mid_turn = False
        self.actions = 0  # None
        self.actions_todo = None
        self.try_play_id = None
        self.roll = None
        # PARAMS RESET for game
        self.image = Image.open("./resources/ICBV211-123395-img.png")
        self.img_copy = self.image.copy()

        self.background_image = ImageTk.PhotoImage(self.image)

        self.background = Label(self, image=self.background_image)
        self.background.pack(fill=BOTH, expand=YES)
        self.background.bind('<Configure>', self._resize_image)

        self.tutorial_btn = Button(self.background, text="Toturial", font='Helvetica 16 bold', bg="blue",
                                   command=self.on_click_faq)
        # self.tutorial_btn.place(relx=0.8,rely=0.05,anchor=CENTER)
        self.tutorial_btn.pack(side=TOP, anchor=NE)

        self.wins = {WHITE: 0, BLACK: 0}

        self.close_btn = Button(self.background, text="Quit", font='Helvetica 16 bold', command=master.quit)
        # self.close_btn.pack(side=TOP,anchor=NW)
        self.close_btn.place(y=0)
        self.master_ = master

        self.white_agent_label = LabelFrame(self.background, text='select the WHITE agent type',
                                            font='Helvetica 13 bold')
        self.white_agent_LB = Listbox(self.white_agent_label, height=3, exportselection=False)
        self.white_agent_LB.insert(1, "PC random agent")
        self.white_agent_LB.insert(2, "PC smart agent")
        self.white_agent_LB.insert(3, "Human player")

        self.white_agent_label.place(relx=0.725, rely=0.7, anchor=CENTER)
        self.white_agent_LB.pack()

        self.black_agent_label = LabelFrame(self.background, text='select the BLACK agent type',
                                            font='Helvetica 13 bold', bg='#000000', fg='#ffffff')
        self.black_agent_LB = Listbox(self.black_agent_label, height=3, exportselection=False)
        self.black_agent_LB.insert(1, "PC random agent")
        self.black_agent_LB.insert(2, "PC smart agent")
        self.black_agent_LB.insert(3, "Human player")

        self.black_agent_label.place(relx=0.725, rely=0.2, anchor=CENTER)
        self.black_agent_LB.pack()

        self.start_btn = Button(self.background, text="START", bg="#00ff00", font='Helvetica 16 bold')
        self.start_btn['command'] = lambda binst=self.start_btn: self.start(binst)
        self.start_btn.place(relx=0.5, rely=0.5, anchor=CENTER)
        self.backggamonCV = BackgammonCV()

    def on_click_faq(self):
        filename = "./resources/ICBV211-123395-index.html"
        webbrowser.open('file://' + os.path.realpath(filename))

    def start(self, binst):
        # TODO
        # cv2 start camera

        print("starting calibration", binst)
        white_agent_id = self.white_agent_LB.curselection()
        black_agent_id = self.black_agent_LB.curselection()
        if white_agent_id == ():
            messagebox.showerror(title="white agent selection", message="Please select white agent")
            return
        elif white_agent_id == (0,):
            white_agent = RandomAgent(WHITE, "White random agent")
        elif white_agent_id == (1,):
            white_agent = TDAgent(WHITE, "white ai agent", backgammon_net)
        elif white_agent_id == (2,):
            white_agent = HumanAgent(WHITE, "white_human agent={}".format(white_agent_id))
        else:
            raise Exception("bad value for white agent")
        if black_agent_id == ():
            messagebox.showerror(title="black agent selection", message="Please select black agent")
            return
        elif black_agent_id == (0,):
            black_agent = RandomAgent(BLACK, "black random agent")
        elif black_agent_id == (1,):
            black_agent = TDAgent(BLACK, "black ai agent", backgammon_net)
        elif black_agent_id == (2,):
            black_agent = HumanAgent(BLACK, "black human agent")
        else:
            raise Exception("bad value for black agent={}".format(black_agent_id))
        self.agents = {WHITE: white_agent, BLACK: black_agent}
        binst.destroy()  # start button
        self.tutorial_btn.destroy()
        self.black_agent_LB.destroy()
        self.black_agent_label.destroy()
        # black agent selector
        self.white_agent_LB.destroy()
        self.white_agent_label.destroy()
        # white agent selector
        self.close_btn.destroy()
        # close_btn selector
        self.calibration_btn = Button(self.background, text="Press when ready for calibration", font='Helvetica 16 bold', )
        self.calibration_btn['command'] = lambda: self.calibration()
        self.calibration_btn.place(relx=0.5, rely=0.5, anchor=CENTER)
        # self.canvas=None
        self.backup_tk_image = None
        self.backup_calibration_img = None
        self.backup_copy_img = None

    def calibration(self):

        self.backup_tk_image = self.background_image
        self.backup_calibration_img = self.image
        self.backup_copy_img = self.img_copy
        # todo calibration
        # cv2 stuff here to calibrate state
        calibration_status = self.backggamonCV.board_init()
        if not calibration_status.return_value:
            # cv2.imshow("calibration_error",calibration_status.output_image)
            img = Image.fromarray(cv2.cvtColor(calibration_status.output_image, cv2.COLOR_BGR2RGB)).resize(
                (self.image.size[0], self.image.size[1]))
            self.image = img
            self.img_copy = self.image.copy()
            self.background_image = ImageTk.PhotoImage(self.image)
            self.background["image"] = self.background_image

            messagebox.showerror(title="calibration_error",
                                 message=calibration_status.error_message)
            return
        self.background_image = self.backup_tk_image
        self.image = self.backup_calibration_img
        self.img_copy = self.backup_copy_img

        messagebox.showinfo("succesfull calibration", "calibration was sucessfull")
        self.calibration_btn.destroy()
        self.init_game()

    def init_game(self):
        agent_color, first_roll, observation = env.reset()
        self.dice_rolled = False
        self.cur_agent = agent_color
        self.roll = first_roll
        self.observation = observation
        self.game_over = False
        # self.can_play = False
        self.mid_turn = False
        img = env.render(mode='rgb_array')
        img = Image.fromarray(img).resize((self.image.size[0], self.image.size[1]))
        self.image = img
        self.img_copy = self.image.copy()
        self.background_image = ImageTk.PhotoImage(self.image)
        self.background["image"] = self.background_image
        self.init_btn = Button(self.background, text="Setup the Board to look like This", font='Helvetica 15 bold',
                               bg="yellow")
        # self.init_btn.after
        self.init_btn.place(relx=0.5, rely=0.04, anchor=CENTER)
        self.init_btn.bind("<Enter>", self.on_enter_init_btn)
        self.init_btn.bind("<Leave>", self.on_leave_init_btn)
        # self.init_btn_clicked=False
        self.init_btn['command'] = lambda binst=self.calibration_btn: self.on_init_btn_clicked(binst)
        self.dices = dices(self.master)
        self.dices.hide_dices()

    def on_enter_init_btn(self, event):
        pass
        # if self.init_btn_clicked:
        #     return
        # self.init_btn["bg"]="orange"

    def on_leave_init_btn(self, event):
        pass
        # if self.init_btn_clicked:
        #     return
        # self.init_btn["bg"]="yellow"

    def on_init_btn_clicked(self, binst):
        realboard_init = True
        # TODO opencv was game setup correctly
        game = env.game
        print("on init")
        current_status, currentcv_board, current_cv_bar, current_cv_off = self.backggamonCV.get_current_board_status()
        if current_status.return_value:
            print("succesfully readed value of board?")
            if (game.board == currentcv_board and game.bar == current_cv_bar and game.off == current_cv_off):
                self.init_btn["bg"] = "red"
                messagebox.showerror(title="bad starting position",
                                     message="Please set the board pieces like the shown configuration and try again")
            else:
                self.init_btn.destroy()
                # self.can_play = True
                self.mid_turn = False
                self.try_play_id = self.after(100, self.try_play)
        else:
            print("failed to read board state")
            print(current_status)
            messagebox.showerror(title="failed to recognize pieces", message=current_status.error_message)

    def render_bg(self, cur_action):
        img = env.render(mode='rgb_array', action=cur_action)
        img = Image.fromarray(img).resize((self.image.size[0], self.image.size[1]))
        self.image = img
        self.img_copy = self.image.copy()
        self.background_image = ImageTk.PhotoImage(self.image)
        self.background["image"] = self.background_image

    def get_roll(self):
        if self.roll is not None:  # first move
            roll = self.roll
            self.roll = None
        else:
            roll = self.agents[self.cur_agent].roll_dice()
        return roll

    def after_dice_rolled(self):
        print("dice rolled")
        self.dice_rolled = True
        # self.try_play_id=self.after(1000,self.try_play)
        self.try_play()

    def try_play(self):
        # print("try play",self.mid_turn,"self.midturn")
        global env
        # if self.can_play:  # if not in middle of move on the real board
        if self.game_over:  # TODO game over show other screen,restart btn
            if self.try_play_id is not None:
                self.after_cancel(self.try_play_id)
                print("game over")
            return
        cur_agent = self.agents[self.cur_agent]
        if not self.mid_turn:
            if isinstance(cur_agent, AiAgent):  # PC agent
                # cur_agent.try_play(self)
                if self.actions_todo is None:
                    if not self.dice_rolled:
                        self.render_bg(None)
                        roll = self.get_roll()
                        self.roll_debug = roll
                        self.dices = dices(self.master)
                        self.dices.roll_dices(n1=roll[0], n2=roll[1], callback_after_all=self.after_dice_rolled)
                        print("Current player={} ({} - {}) | Roll={}".format(cur_agent.color, TOKEN[cur_agent.color],
                                                                             COLORS[cur_agent.color],
                                                                             roll))
                        actions = env.get_valid_actions(roll)
                        # self.actions = actions
                        self.actions += 1
                        action = cur_agent.choose_best_action(actions, env)
                        self.actions_todo = list(action) if action is not None else None
                        self.after_cancel(self.try_play_id)
                        return
                self.mid_turn = True
                cur_action = None
                if self.actions_todo is None:
                    # print("no actions avilable?")#
                    cur_player_color = self.cur_agent
                    assert cur_player_color in [BLACK, WHITE]

                    if cur_player_color == BLACK:
                        print("cur dice={} has no use for black agent".format(self.roll_debug))
                        messagebox.showinfo(title="no actions avilable",
                                            message="no actions avilable for the {} player".format("BLACK"))
                    else:
                        print("cur dice={} has no use for white agent".format(self.roll_debug))
                        messagebox.showinfo(title="no actions avilable",
                                            message="no actions avilable for the {} player".format("WHITE"))
                    self.render_bg(None)
                    self.dices.d1.destroy()
                    self.dices.d2.destroy()
                    self.dice_rolled = False
                    observation_next, reward, done, winner = env.step(None)
                    if done:
                        self.game_over = True
                else:
                    # execute_single_move(self, current_player, move):
                    cur_action = self.actions_todo.pop(0)  # note here we get part of the move
                    self.render_bg(cur_action)
                    self.end_turn_btn = Button(self.background, text="Next move...",
                                               font='Helvetica 15 bold',
                                               bg="cyan")
                    self.end_turn_btn.place(relx=0.0, rely=0.0)  # Todo check that placement
                    self.end_turn_btn['command'] = lambda: self.end_turn()
                    if len(self.actions_todo) == 0:
                        self.actions_todo = None
                        observation_next, reward, done, winner = env.step((cur_action,))
                        self.dices.d1.destroy()
                        self.dices.d2.destroy()
                        self.dices.destroy()
                        self.dice_rolled = False
                        if done:
                            self.render_bg(None)
                            self.game_over = True
                    else:
                        env.step((cur_action,))
                        # env.game.execute_single_move(env.current_agent, cur_action)to
                    return
            else:  # TODO human agent spwan buttom? to end turn if sucessfull destroy the buttom else keep it
                assert isinstance(cur_agent, HumanAgent)
                if self.actions_todo is None:
                    if not self.dice_rolled:
                        self.render_bg(None)
                        roll = self.get_roll()
                        self.roll_debug = roll
                        self.dices = dices(self.master)
                        self.dices.roll_dices(n1=roll[0], n2=roll[1], callback_after_all=self.after_dice_rolled)
                        print("Current player={} ({} - {}) | Roll={}".format(cur_agent.color, TOKEN[cur_agent.color],
                                                                             COLORS[cur_agent.color],
                                                                             roll))
                        actions = env.get_valid_actions(roll)
                        self.actions = actions
                        action = cur_agent.choose_best_action(actions, env)
                        assert action is None
                        self.actions_todo = None
                        # self.after_cancel(self.try_play_id)
                        self.end_turn_btn = Button(self.background, text="Next move...",
                                                   font='Helvetica 15 bold',
                                                   bg="cyan")
                        self.end_turn_btn.place(relx=0.0, rely=0.0)  # Todo check that placement
                        self.end_turn_btn['command'] = lambda: self.end_turn()
                        return
        else:  # check the condition on the real board(depend on agent)#todo
            pass  # moved this logic to button for now
        self.try_play_id = self.after(1000, self.try_play)

    # def make_plays(self):
    #     # wins = {WHITE: 0, BLACK: 0}
    #
    #     # agents = {WHITE: RandomAgent(WHITE), BLACK: RandomAgent(BLACK)}
    #
    #     # agent_color, first_roll, observation = env.reset()
    #
    #     # agent = agents[agent_color]
    #
    #     # t = time.time()
    #
    #     # env.render(mode='rgb_array')
    #
    #     for i in count():
    #         if first_roll:
    #             roll = first_roll
    #             first_roll = None
    #         else:
    #             roll = agent.roll_dice()
    #
    #         print("Current player={} ({} - {}) | Roll={}".format(agent.color, TOKEN[agent.color], COLORS[agent.color],
    #                                                              roll))
    #
    #         actions = env.get_valid_actions(roll)
    #         #
    #         # print("\nLegal Actions:")
    #         # for a in actions:
    #         #     print(a)
    #         # detect if human move was legal
    #         # detect if pc move was done
    #         action = agent.choose_best_action(actions, env)
    #         print("action=", action)
    #
    #         observation_next, reward, done, winner = env.step(action)
    #
    #         img = env.render(mode='rgb_array')
    #         print(img.shape)
    #         if done:
    #             if winner is not None:
    #                 wins[winner] += 1
    #
    #             tot = wins[WHITE] + wins[BLACK]
    #             tot = tot if tot > 0 else 1
    #
    #             print(
    #                 "Game={} | Winner={} after {:<4} plays || Wins: {}={:<6}({:<5.1f}%) | {}={:<6}({:<5.1f}%) | Duration={:<.3f} sec".format(
    #                     1, winner, i,
    #                     agents[WHITE].name, wins[WHITE], (wins[WHITE] / tot) * 100,
    #                     agents[BLACK].name, wins[BLACK], (wins[BLACK] / tot) * 100, time.time() - t))
    #
    #             break
    #
    #         agent_color = env.get_opponent_agent()
    #         agent = agents[agent_color]
    #         observation = observation_next
    #
    #     env.close()

    def _resize_image(self, event):

        new_width = event.width
        new_height = event.height

        self.image = self.img_copy.resize((new_width, new_height))

        self.background_image = ImageTk.PhotoImage(self.image)
        self.background.configure(image=self.background_image)

    def _exit(self):
        sys.exit

    def end_turn(self):
        global env
        cur_agent = self.agents[self.cur_agent]
        if isinstance(cur_agent, AiAgent):
            game = env.game
            current_status, currentcv_board, current_cv_bar, current_cv_off = self.backggamonCV.get_current_board_status()
            if current_status.return_value:
                if game.board == currentcv_board and game.bar == current_cv_bar and game.off == current_cv_off:
                    self.mid_turn = False
                    self.end_turn_btn.destroy()
            else:
                messagebox.showerror("Board capture failure", "Failed capturing board status!")
            # self.try_play_id = self.after(1000, self.try_play)
        else:
            current_status, currentcv_board, current_cv_bar, current_cv_off = self.backggamonCV.get_current_board_status()
            if current_status.return_value:
                for action in self.actions_todo:
                    env_copy = copy.deepcopy(env)
                    env_copy.step(action)
                    game = env_copy.game
                    if game.board == currentcv_board and game.bar == current_cv_bar and game.off == current_cv_off:
                        self.mid_turn = False
                        env = env_copy
                        self.end_turn_btn.destroy()
                        # self.try_play_id=self.after(1000,self.try_play)
        if self.actions_todo is None:
            self.cur_agent = env.get_opponent_agent()
            # self.cur_agent=
            self.agent = self.agents[self.cur_agent]
            # self.observation = observation_next

            # self.dice_rolled=False
            # self.dices.destroy()#.hide_dices()
            # self.dices.d1.destroy()
            # self.dices.d2.destroy()
            # self.can_play = True
            # if isinstance(self.cur_agent,AiAgent):
            #     self.can_play=True
        if self.game_over:  
            if self.try_play_id is not None:
                self.after_cancel(self.try_play_id)
            return
        else:
            self.try_play_id = self.after(1000 * animation_speed_factor, self.try_play)


e = Example(root)
e.pack(fill=BOTH, expand=YES)

root.mainloop()
