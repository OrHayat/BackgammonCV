import random
import tkinter as tk

animation_speed = 1


def create_dice(root):
    """
    create the dice canvas list as dice[0] to dice[6]
    """
    dice = []
    dice.append(draw_dice(root, 'dot0'))  # empty
    dice.append(draw_dice(root, 'dot5'))  # center dot --> 1
    dice.append(draw_dice(root, 'dot4', 'dot6'))
    dice.append(draw_dice(root, 'dot3', 'dot5', 'dot7'))
    dice.append(draw_dice(root, 'dot1', 'dot3', 'dot7', 'dot9'))
    dice.append(draw_dice(root, 'dot1', 'dot3', 'dot5', 'dot7', 'dot9'))
    dice.append(draw_dice(root, 'dot1', 'dot3', 'dot4', 'dot6', 'dot7', 'dot9'))
    return dice


def draw_dice(root, *arg):
    """
    draws the 7 different dice dots on the canvas
    """
    w = 20
    h = 20
    c = tk.Canvas(root, width=w + 3, height=h + 3, bg='blue')
    # set the dot specs
    x = 2
    y = 2
    r = 5
    bg = c.create_rectangle(x, y, x + w + 3, y + w + 3, fill='#aaaaaa', outline='red', width=2)
    # print(bg)
    if 'dot1' in arg:
        dot1 = c.create_oval(x, y, x + r, y + r, fill='black')
    x = w / 2
    x = 19
    if 'dot3' in arg:
        dot3 = c.create_oval(x, y, x + r, y + r, fill='black')
    x = 2
    y = h / 2
    if 'dot4' in arg:
        dot4 = c.create_oval(x, y, x + r, y + r, fill='black')
    x = w / 2
    if 'dot5' in arg:
        dot5 = c.create_oval(x, y, x + r, y + r, fill='black')
    x = 19
    if 'dot6' in arg:
        dot6 = c.create_oval(x, y, x + r, y + r, fill='black')
    x = 2
    y = 19
    if 'dot7' in arg:
        dot7 = c.create_oval(x, y, x + r, y + r, fill='black')
    x = w / 2
    x = 19
    if 'dot9' in arg:
        dot9 = c.create_oval(x, y, x + r, y + r, fill='black')
    if 'dot0' in arg:
        # defult=c.create_text(x,y,text="\u2684", font="Helvetica 20 bold")
        c.delete(bg)
    return c


t0 = 150 * animation_speed


class dice(tk.Frame):

    def __init__(self, parent, n, ):
        tk.Frame.__init__(self, parent)
        self.dice_list = create_dice(self)
        self.index = n
        # self.master = parent
        self.dice_list[self.index].pack()  # .place(relx=0,rely=0)#grid(row=3, column=0, columnspan=3)

    def roll_dice(self, target_number, num_times, t=t0, callback=None):
        assert num_times >= 0
        # if num_times >= 0:
        #     if callback:
        #         callback()
        #     return
        if num_times <= 1:
            # self.dice_list[self.index].pack_forget()
            self.hide_dice()
            self.dice_list[abs(target_number)].pack()
            # print(callback)
            print("dice rolled",target_number)
            if callback:
                callback()
        else:
            # self.dice_list[self.index].pack_forget()
            self.hide_dice()
            self.index = random.randint(1, 6)
            self.dice_list[self.index].pack()  # place(relx=0,rely=0)#.grid(row=3, column=0, columnspan=3)

            self.master.after(t,
                              lambda target=target_number, num_times=num_times - 1,
                                     t=t + 10, callback=callback: self.roll_dice(target_number=target,
                                                                                 num_times=num_times,
                                                                                 t=t,
                                                                                 callback=callback))

    def hide_dice(self):
        self.dice_list[self.index].pack_forget()


#
# def click():
#     # button1['state']='disabled'
#     """
#     display a randomly selected dice value
#     """
#     # start with a time delay of 100 ms and increase it as the dice rolls
#     t = 100
#     stop = random.randint(13, 18)
#     for x in range(stop):
#         dice_index = x % 6 + 1
#         root.title(str(dice_index))  # test
#         dice_list[dice_index].grid(row=1, column=0, pady=5)
#         root.update()
#         if x == stop - 1:
#             # final result available via var1.get()
#             var1.set(str(x % 6 + 1))
#             break
#         root.after(t, dice_list[dice_index].grid_forget())
#         t += 25


class dices(tk.Frame):
    def __init__(self, parent, ):
        tk.Frame.__init__(self, parent)
        self.d1 = dice(parent, n=0)
        self.d2 = dice(parent, n=0)
        self.d1.place(relx=0.5, rely=0.3)
        self.d2.place(relx=0.2, rely=0.8)
        self.rolling = 0

    def roll_dices(self, n1=None, n2=None, callback_after_each=None, callback_after_all=None):
        def after_roll():
            self.rolling -= 1
            if callback_after_each:
                callback_after_each()
            if self.rolling == 0:
                if callback_after_all:
                    callback_after_all()
                    # self.after(100, callback_after_all)
                    # callback_after_all()

        self.rolling = 2
        self.d1.place_forget()
        self.d2.place_forget()
        self.d1.place(relx=random.uniform(0.5, 0.85), rely=random.uniform(0.5, 0.85))
        self.d2.place(relx=random.uniform(0.2, 0.4), rely=random.uniform(0.2, 0.4))
        n1 = n1 if n1 else random.randint(1, 6)
        n2 = n2 if n2 else random.randint(1, 6)
        # target, num_times, t=t0, callback=None
        self.d1.roll_dice(n1, random.randint(10, 20), t0, after_roll)
        self.d2.roll_dice(n2, random.randint(10, 20), t0, after_roll)

    def hide_dices(self):
        self.d1.place_forget()
        self.d2.place_forget()

#
# def cb1():
#     print("all dices done")
#
#
# def cb2():
#     print("done dice")
#
# #
# root = tk.Tk()
# d = dices(root)
# d.place()
#
# d.roll_dices(callback_after_each = cb2, callback_after_all = cb1)
# # d1 = dice(root, n=1)
# # d1.place(relx=random.uniform(0.5,0.85), rely=random.uniform(0.5,0.85))
# # # d1.grid(row=3, column=2, columnspan=2)#place(relx=0.5,rely=0,anchor=CENTER)#pack(side=tk.LEFT)
# # d2 = dice(root, n=2)
# # d2.place(relx=random.uniform(0.2,0.4), rely=random.uniform(0.2,0.4))
# # d1.roll_dice(random.randint(1,6),random.randint(10,20),random.randint(50,200))
# # d2.grid(row=3, column=0, columnspan=2)#place(relx=0.5,rely=0.5,anchor=CENTER)#pack(side=tk.RIGHT)
# # d2.place_forget()
# # # StringVar() updates result label automatically
# # var1 = tk.StringVar()
# # # set initial value
# # var1.set("")
# # # create the result label
# result = tk.Label(root, textvariable=var1, fg='red')
# # result.grid(row=3, column=0, columnspan=3)
#
# # dice_list = create_dice(root)
# # # start with an empty canvas
# # dice_list[0].grid(row=1, column=0, pady=5)
#
# # button1 = tk.Button(root, text="Press me", command=click,)
# # button1.grid(row=2, column=0, pady=3)
#
# # start of program event loop
# root.mainloop()
# import tkinter as tk

# class ScrolledText(tk.Frame):
#     def __init__(self, parent, *args, **kwargs):
#         tk.Frame.__init__(self, parent)
#         self.text = tk.Text(self, *args, **kwargs)
#         self.vsb = tk.Scrollbar(self, orient="vertical", command=self.text.yview)
#         self.text.configure(yscrollcommand=self.vsb.set)
#         self.vsb.pack(side="right", fill="y")
#         self.text.pack(side="left", fill="both", expand=True)

# class Example(tk.Frame):
#     def __init__(self, parent):
#         tk.Frame.__init__(self, parent)
#         self.scrolled_text = ScrolledText(self)
#         self.scrolled_text.pack(side="top", fill="both", expand=True)
#         with open(__file__, "r") as f:
#             self.scrolled_text.text.insert("1.0", f.read())

# root = tk.Tk()
# Example(root).pack(side="right")
# Example(root).pack(side="left")
# root.mainloop()
