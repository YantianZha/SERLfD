# keyboardAgents.py
# -----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The common projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from pacman_src.game import Agent
from pacman_src.game import Directions
import tkinter as tk
import random

keyboard_root = None
last_key = None


def key_event(event):
    global last_key
    if str(event.char) == 'w':
        last_key = KeyboardAgent.NORTH_KEY
    elif str(event.char) == 's':
        last_key = KeyboardAgent.SOUTH_KEY
    elif str(event.char) == 'd':
        last_key = KeyboardAgent.EAST_KEY
    elif str(event.char) == 'a':
        last_key = KeyboardAgent.WEST_KEY


def sleep_tk(secs):
    global keyboard_root
    keyboard_root.update_idletasks()
    keyboard_root.after(int(1000 * secs), keyboard_root.quit)
    keyboard_root.mainloop()


# noinspection PyMissingConstructor,PySimplifyBooleanCheck
class KeyboardAgent(Agent):
    """
    An agent controlled by the keyboard.
    """
    # NOTE: Arrow keys also work.
    WEST_KEY = 'a'
    EAST_KEY = 'd'
    NORTH_KEY = 'w'
    SOUTH_KEY = 's'
    STOP_KEY = 'q'

    def __init__(self, index=0):
        global keyboard_root
        keyboard_root = tk.Tk()

        super().__init__(index)
        self.lastMove = Directions.STOP
        self.index = index
        self.keys = []

        keyboard_root.geometry('200x200+300+300')
        keyboard_root.bind("<Key>", lambda a: key_event(a))
        sleep_tk(0)

    def getAction(self, state):
        global last_key
        keys = []
        if last_key is not None:
            keys = [last_key]
            last_key = None
        if keys != []:
            self.keys = keys

        legal = state.getLegalActions(self.index)
        move = self.getMove(legal)

        if move == Directions.STOP:
            # Try to move in the same direction as before
            if self.lastMove in legal:
                move = self.lastMove

        if (self.STOP_KEY in self.keys) and Directions.STOP in legal:
            move = Directions.STOP

        if move not in legal:
            move = random.choice(legal)

        self.lastMove = move
        return move

    def getLastMove(self):
        return self.lastMove

    def getMove(self, legal):
        move = Directions.STOP
        if (self.WEST_KEY in self.keys or 'Left' in self.keys) and Directions.WEST in legal:
            move = Directions.WEST
        if (self.EAST_KEY in self.keys or 'Right' in self.keys) and Directions.EAST in legal:
            move = Directions.EAST
        if (self.NORTH_KEY in self.keys or 'Up' in self.keys) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (self.SOUTH_KEY in self.keys or 'Down' in self.keys) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move


class KeyboardAgent2(KeyboardAgent):
    """
    A second agent controlled by the keyboard.
    """
    # NOTE: Arrow keys also work.
    WEST_KEY = 'j'
    EAST_KEY = "l"
    NORTH_KEY = 'i'
    SOUTH_KEY = 'k'
    STOP_KEY = 'u'

    def __init__(self, index=0):
        super().__init__(index)

    def getMove(self, legal):
        move = Directions.STOP
        if (self.WEST_KEY in self.keys) and Directions.WEST in legal:
            move = Directions.WEST
        if (self.EAST_KEY in self.keys) and Directions.EAST in legal:
            move = Directions.EAST
        if (self.NORTH_KEY in self.keys) and Directions.NORTH in legal:
            move = Directions.NORTH
        if (self.SOUTH_KEY in self.keys) and Directions.SOUTH in legal:
            move = Directions.SOUTH
        return move
