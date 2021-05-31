import random
import webbrowser
from datetime import datetime
from enum import Enum, auto
from tkinter import messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui_classes import *
from model_utils import *
from stats_utils import create_plots

# EpsImagePlugin.gs_windows_binary = r'C:\Program Files\gs\gs9.53.3\bin\gswin64c'
model = load_model_from_dict('files/resnet18_v3.pt')


def read_settings(path='settings.json'):
    with open(path) as f:
        s = dict(json.load(f))
    return s


class Board(Frame):
    def __init__(self, train_mode, master=None, row=0, column=0):
        super().__init__(master)
        self.master = master
        self.train_mode = train_mode
        self.time_start = datetime.now()

        self.var = StringVar()
        self.number = self.give_random_num()
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
        if not self.train_mode:
            self.game_length = 5
            self.score = Score(self)

        self.column = column
        self.row = row

        self.create_widgets()
        self.grid()
        self.master.eval('tk::PlaceWindow . center')

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
        self.sketch.grid(column=self.column, row=2, sticky=(N, W, E, S), columnspan=6)
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
        if self.train_mode: self.sample_pred.configure(image=self.asd2)
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

    def disable_sketch(self, switch):
        if switch:
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
            Button(self, text='Settings', padx=42, pady=10, bg=self.color, command=self.show_settings),
            Button(self, text='Statistics', command=self.show_stats, padx=40, pady=10, bg=self.color),
            Button(self, text='About', padx=46, pady=10, bg=self.color,
                   command=lambda: webbrowser.open('https://github.com/lukasztroc/LearnHiragana', new=0,
                                                   autoraise=True))]
        for item in self.item_list:
            item.grid(pady=10, padx=20)

        self.grid(pady=10, padx=20)
        self.master.eval('tk::PlaceWindow . center')

    def start_game(self, train_mode):
        Board(train_mode=train_mode, master=self.master)
        self.grid_remove()

    def show_stats(self):
        Statistics(master=self.master)
        self.grid_remove()

    def show_settings(self):
        Settings(master=self.master)
        self.grid_remove()


class Statistics(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.time_played = StringVar()
        self.average_accuracy = StringVar()
        self.get_total_time()
        self.calculate_accuracy()

        self.canvas = FigureCanvasTkAgg(create_plots(), master=self)  # A tk.DrawingArea.
        self.button = Button(self, text='back', command=self.back)

        self.label_list = [
            Label(self, text='Total time played:'),
            Label(self, textvariable=self.time_played),
            Label(self, text='Overall Accuracy:'),
            Label(self, textvariable=self.average_accuracy)
        ]
        for item in self.label_list:
            item.configure(font=("Courier", 15))
            item.grid(pady=10, padx=20)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(pady=10, padx=20)
        self.button.grid(pady=10, padx=20)
        self.grid(pady=10, padx=20)

    def get_total_time(self):
        with open('stats.json') as f:
            y = json.load(f)["time_played"]
        hours = y // 3600
        minutes = y // 60
        seconds = y % 60
        self.time_played.set(f'{int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds')

    def calculate_accuracy(self):
        with open('stats.json') as f:
            y = json.load(f)["results"]
        df = pd.DataFrame(y)
        num_correct = 0
        for index, row in df.iterrows():
            if row['number'] == row['id']:
                num_correct += 1
        self.average_accuracy.set(f'{num_correct / len(df) * 100:.2f}%')

    def back(self):
        self.grid_remove()
        Menu(self.master)


class Settings(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.master = master

        self.var_scale = DoubleVar()
        self.var_check = IntVar()

        self.label_scale = Label(self, text='Weighted Polling rate:')
        self.scale = Scale(self, variable=self.var_scale, from_=0, to=1, orient=HORIZONTAL, resolution=0.01)
        self.label_checkbutton = Label(self, text='Wighted polling:')
        self.checkbutton = Checkbutton(self, variable=self.var_check, onvalue=1,
                                       offvalue=0,
                                       height=2,
                                       width=10)
        self.back_button = Button(self, text='back', command=self.back)
        self.save_button = Button(self, text='save')
        self.create_widgets()
        self.grid()
        self.master.eval('tk::PlaceWindow . center')

    def create_widgets(self):
        self.label_checkbutton.grid(row=0, column=0, padx=5)
        self.checkbutton.grid(row=1, column=0, padx=5)
        self.label_scale.grid(row=0, column=1, padx=5)
        self.scale.grid(row=1, column=1, padx=20)
        self.back_button.grid(row=2, column=0, padx=5, pady=10, sticky='SW')
        self.save_button.grid(row=2, column=1, padx=5, pady=10, sticky='SE')

    def back(self):
        self.grid_remove()
        Menu(self.master)


def main():
    root = Tk()
    root.title('LearnHiragana')
    root.resizable(height=False, width=False)
    Menu(root)
    root.mainloop()


if __name__ == '__main__':
    main()
