import random
import webbrowser
from datetime import datetime
from enum import Enum, auto
from tkinter import messagebox
import numpy as np
from asd import *
from score import *

# EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.53.3\bin\gswin64c'
model = load_model_from_dict('files/resnet18_v3.pt')


class Board(Frame):
    def __init__(self, train_mode, master=None, row=0, column=0):
        super().__init__(master)
        self.master = master
        self.train_mode = train_mode
        self.time_start = datetime.now()

        self.var = StringVar()
        self.number = 47
        self.var.set(f'draw "{get_label(self.number)}"')
        self.correct = 0
        self.counter = 0

        if self.train_mode:
            self.sample = Label(self)
            self.sample_pred = Label(self)
            self.sample_button = Button(self, text="hint", command=self.give_hint)
        self.label = Label(self, textvariable=self.var)
        self.mark = Mark(self)
        self.sketch = Sketchpad(self, bg="white", height=300, width=300)
        self.button_check = Button(self, text="check", command=self.evaluate)
        self.button_clear = Button(self, text="clear", command=self.clear)
        self.button_next = Button(self, text="next", command=self.next_letter)
        self.button_back = Button(self, text="back", command=self.back)
        if not train_mode:
            self.game_length = 5
            self.score = Score(self)

        self.column = column
        self.row = row

        self.create_widgets()
        self.grid()

    def give_hint(self):
        self.asd = ImageTk.PhotoImage(get_sample(self.number))
        self.sample.configure(image=self.asd)

    def back(self):
        if not self.train_mode:
            if self.counter < self.game_length:
                if not messagebox.askokcancel('Warning', 'Are you sure?'):
                    return
        self.grid_remove()
        save_stats(time_played=(datetime.now() - self.time_start).total_seconds())
        Menu(self.master)

    def create_widgets(self):
        self.label.grid(column=0, row=0)
        if self.train_mode:
            self.mark.grid(row=0, column=4, sticky='ne')
            self.sample_button.grid(row=0, column=2)
            self.sample.grid(row=0, column=1)
            self.sample_pred.grid(row=0, column=3)
        self.sketch.grid(column=self.column, row= 2, sticky=(N, W, E, S), columnspan=6)
        self.button_check.grid(column=self.column, row=3, sticky='se')
        self.button_clear.grid(column=self.column + 1, row=3, sticky='sw')
        self.button_next.grid(column=self.column + 3, row=3, sticky='se')
        self.button_back.grid(column=5, row=0, sticky='se')
        if not self.train_mode:
            self.disable_sketch(False)
            self.score.grid(row=0, column=2)

    def give_random_num(self):
        return random.randint(0, 48)

    def classify(self):
        index, probability = classify(model, self.sketch.save_canvas())
        print(f'{self.number} {index} {get_label(index)} {probability:.2f}')
        save_stats(result={"number": self.number, "id": index, 'prob': probability,
                           'train_mode': self.train_mode})
        self.asd2 = ImageTk.PhotoImage(get_sample(index))
        if self.train_mode:self.sample_pred.configure(image=self.asd2)
        if probability <= 0.1:
            return Result.idk
        if index == self.number:
            return Result.correct
        return Result.wrong

    def evaluate(self):
        result = self.classify()
        self.counter += 1
        if result == Result.correct:
            self.correct += 1
            self.mark.correct()
            if not self.train_mode: self.score.correct()
        elif result == Result.idk:
            self.mark.idk()
            if not self.train_mode: self.score.idk()
        else:
            self.mark.wrong()
            if not self.train_mode:  self.score.wrong()

        if not self.train_mode:
            self.disable_sketch(True)
            if self.counter == self.game_length:
                messagebox.showinfo(title='Score', message=f'Your score is {self.correct}/{self.counter}')
                self.button_next['text'] = 'finish'
                self.button_next['command'] = self.back

    def disable_sketch(self, bool):
        if bool:
            self.button_check['state'] = DISABLED
            self.button_clear['state'] = DISABLED
            self.button_next['state'] = NORMAL
            self.sketch.enabled = False
        else:
            self.button_check['state'] = NORMAL
            self.button_clear['state'] = NORMAL
            self.button_next['state'] = DISABLED
            self.sketch.enabled = True

    def next_letter(self):
        self.number = self.give_random_num()
        self.var.set(f'draw "{get_label(self.number)}"')
        self.sketch.reset_canvas()
        self.mark.configure(image='')
        if not self.train_mode:
            self.disable_sketch(False)
        if self.train_mode:
            self.sample.configure(image='')
            self.sample_pred.configure(image='')

    def clear(self):
        self.sketch.reset_canvas()
        self.mark.reset()


class Result(Enum):
    correct = auto()
    wrong = auto()
    idk = auto()


class Menu(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.color = 'white'
        self.master = master

        self.item_list = [
            Button(self, text='Train', command=lambda: self.start_game(True), padx=50, pady=10, bg=self.color),
            Button(self, text='Test', command=lambda: self.start_game(False), padx=53, pady=10, bg=self.color),
            Button(self, text='Settings', padx=42, pady=10, bg=self.color),
            Button(self, text='Statistics', padx=40, pady=10, bg=self.color),
            Button(self, text='About', padx=46, pady=10, bg=self.color,
                   command=lambda: webbrowser.open('https://google.com', new=0, autoraise=True))]
        for item in self.item_list:
            item.grid(pady=10, padx=20)

        self.grid(pady=10, padx=20)

    def start_game(self, train_mode):
        Board(train_mode=train_mode, master=self.master)
        self.grid_remove()


root = Tk()
root.title('LearnHiragana')
root.geometry('+800+300')
root.resizable(height=False, width=False)
app = Menu(root)

root.mainloop()
