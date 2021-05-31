# LearnHiragana
Is a language learning tool created to assist in learning Japanese phonetic lettering system Hiragana in a form of drawing game. It uses neural networks to classify user input and provide feedback accordingly.
# Gameplay
LearnHiragana has 2 modes of gameplay available: Train and Test. 
## Train mode

![base_train](https://github.com/lukasztroc/LearnHiragana/blob/master/files/readme_files/train_base.png) 

In the left-top corner we can see english phonetic of the symbol that program asks user to draw, white rectangular area in the center of the window is the place to draw our letter.
"Clear" button - resets drawing area. "Next" button - proceed to next letter.

### Hint


![hint](https://github.com/lukasztroc/LearnHiragana/blob/master/files/readme_files/hint.png) 

Program has a feature in "Train" mode called "hint". It shows pixel average of a given symbol over the dataset.

### Evaluation

Once we drew the symbol and clicked "check" button, program gives us feedback in the form of the predicted letter and "Right" or "Wrong" symbol shown at the top of the window.


![right](https://github.com/lukasztroc/LearnHiragana/blob/master/files/readme_files/train_check.png) ![trainbad](https://github.com/lukasztroc/LearnHiragana/blob/master/files/readme_files/train_bad.png)

## Test mode


The difference between test and train mode is that user has no access to hints. The gameplay consists of 5 questions after which user is presented their final score.


![start](https://github.com/lukasztroc/LearnHiragana/blob/master/files/readme_files/test_mode.png) ![finish](https://github.com/lukasztroc/LearnHiragana/blob/master/files/readme_files/test_mode_finished.png)


## User feedback


User can check their progress using the "Statistics" window as shown below:


![statistics](https://github.com/lukasztroc/LearnHiragana/blob/master/files/readme_files/statistics.png)


# Technical overview

This section provides short overview of technologies and the way they were used in the project. Entire code is written in **Python 3.8**.

## Data processing and classification

The main feature of the project ResNet-18 neural network trained from scratch using **PyTorch** and **PyTorch lightning**. It was trained on [Kuzushiji-49 dataset](http://github.com/rois-codh/kmnist) using Google Colab.
Model achieved weighted accuracy of over 95% on the test set, which is comparable to PreActResNet-18 model listed in the benchmark on the dataset github page.
Input preprocessing is done using **NumPy** and **Pillow**. 

### The training process and evaluation:

See: [ResnetHiragana](https://github.com/lukasztroc/LearnHiragana/blob/master/dev/ResnetHiragana.ipynb).

## Statistics and plots

Statistics were calculated using **Pandas** and some of them plotted using **Matplotlib**.
See: [stats_utils](https://github.com/lukasztroc/LearnHiragana/blob/master/stats_utils.py)

## GUI

Entire user interface was made using **tkinter** library. Main windows are implemented in [main](https://github.com/lukasztroc/LearnHiragana/blob/master/main.py) 
and less important classes are shown in : [gui_classes](https://github.com/lukasztroc/LearnHiragana/blob/master/gui_classes.py)
