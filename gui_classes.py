import io
import json
from tkinter import *

from PIL import Image, ImageTk


class Mark(Label):
    def __init__(self, master):
        super().__init__(master)
        self.img_wrong = ImageTk.PhotoImage(Image.open(fp='files/wrong_image.png').resize((22, 22)))
        self.img_correct = ImageTk.PhotoImage(Image.open(fp='files/correct_image.png').resize((22, 22)))
        self.img_idk = ImageTk.PhotoImage(Image.open(fp='files/idk_image.png').resize((22, 22)))
        self.img_basic = ImageTk.PhotoImage(Image.open(fp='files/basic_img.png').resize((22, 22)))

    def correct(self):
        self.configure(image=self.img_correct)

    def wrong(self):
        self.configure(image=self.img_wrong)

    def idk(self):
        self.configure(image=self.img_idk)

    def reset(self):
        self.configure(image='')

    def basic(self):
        self.configure(image=self.img_basic)


class Score(Frame):
    def __init__(self, master):
        super().__init__(master)
        self.icons = [Mark(self) for i in range(5)]
        self.create_widgets()
        self.counter = 0

    def create_widgets(self):
        for i, label in enumerate(self.icons):
            label.grid(row=5, column=i)
            label.basic()

    def correct(self):
        self.icons[self.counter].correct()
        self.counter += 1

    def wrong(self):
        self.icons[self.counter].wrong()
        self.counter += 1

    def idk(self):
        self.icons[self.counter].idk()
        self.counter += 1

    def basic(self):
        self.icons[self.counter].basic()
        self.counter += 1

    def reset(self):
        self.counter = 0
        for icon in self.icons:
            icon.basic()


def save_stats(*, time_played=None, result=None):
    try:
        with open('stats.json', 'r') as f:
            content = json.loads(f.read())
    except FileNotFoundError:
        content = {}
    if result:
        results = content.get('results', [])
        results.append(result)
        content['results'] = results
    if time_played:
        content['time_played'] = int(content.get('time_played', 0)) + time_played
    with open('stats.json', 'w') as f:
        f.write(json.dumps(content, indent=2))


class Sketchpad(Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind("<Button-1>", self.save_posn)
        self.bind("<B1-Motion>", self.add_line)
        self.enabled = True
        self.lastx, self.lasty = 0, 0

    def save_posn(self, event):
        self.lastx, self.lasty = event.x, event.y

    def add_line(self, event):
        if self.enabled:
            self.create_line((self.lastx, self.lasty, event.x, event.y), width=7, tags="drawing")
            self.save_posn(event)

    def save_canvas(self):
        self.update()
        ps = self.postscript(colormode='color')
        img = Image.open(io.BytesIO(ps.encode('utf-8')))
        return img

    def reset_canvas(self): self.delete("all")
