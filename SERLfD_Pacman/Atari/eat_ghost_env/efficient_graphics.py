import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.cm as cm
import random
import time
from skimage.draw import polygon, circle, rectangle

from eat_ghost_env.eatGhostPacmanGame import PREDICATES


def formatColor(r, g, b):
    return int(r * 255), int(g * 255), int(b * 255)


SCALE = 30.0    # will be recomputed later
DEFAULT_GRID_SIZE = 30
#########################################################
###################### Ghost Config #####################
#########################################################
GHOST_SIZE = 0.65

GHOST_COLORS = []
GHOST_COLORS.append(formatColor(0, .3, .9))  # Blue
GHOST_COLORS.append(formatColor(.98, .41, .07))  # Orange
GHOST_COLORS.append(formatColor(.9, 0, 0))  # Red1
GHOST_COLORS.append(formatColor(.1, .75, .7))  # Green
GHOST_COLORS.append(formatColor(1.0, 0.6, 0.0))  # Yellow
GHOST_COLORS.append(formatColor(.4, 0.13, 0.91))  # Purple

GHOST_SHAPE = [
    (0,     0.3),
    (0.25,  0.75),
    (0.5,   0.3),
    (0.75,  0.75),
    (0.75,  -0.5),
    (0.5,   -0.75),
    (-0.5,  -0.75),
    (-0.75, -0.5),
    (-0.75, 0.75),
    (-0.5,  0.3),
    (-0.25, 0.75)
]

GHOST_Y = np.array([0, 0.25, 0.5, 0.75, 0.75, 0.5, -0.5, -0.75, -0.75, -0.5, -0.25])
GHOST_X = np.array([0.3, 0.75, 0.3, 0.75, -0.5, -0.75, -0.75, -0.5, 0.75, 0.3, 0.75])
GHOST_EYE_RIGHT = np.array([-0.3, 0.3])
GHOST_EYE_LEFT = np.array([-0.3, -0.3])
GHOST_EYE_RADIUS = 0.25
GHOST_PUPIL_RADIUS = 0.15
GHOST_EYES_COLOR = (255, 255, 255)
GHOST_PUPIL_COLOR = (0, 0, 0)

GHOST_SCARED_COLOR = formatColor(1, 1, 1)
IS_CHANGE_SCARED_COLOR = False

#########################################################
###################### Pacman Config ####################
#########################################################

PACMAN_COLOR = formatColor(255.0/255.0, 255.0/255.0, 61.0/255)
PACMAN_RADIUS = 0.6
PACMAN_SCALE = 0.7

#########################################################
################ Game Objects Config ####################
#########################################################

# Food
FOOD_COLOR = formatColor(1, 1, 1)
FOOD_SIZE = 0.1

# Laser
LASER_COLOR = formatColor(1, 0, 0)
LASER_SIZE = 0.02

# Capsule graphics
CAPSULE_COLOR = formatColor(1, 1, 1)
CAPSULE_SIZE = 0.25

# Drawing walls
WALL_RADIUS = 0.6
WALL_COLOR = (0, 0, 255)

#########################################################
#################  Helper Functions  ####################
#########################################################


def get_utility_color(color_mapper, predicate, utility_values):
    if utility_values is None or predicate not in utility_values:
        return formatColor(1.0, 1.0, 1.0)
    else:
        rgba_color = color_mapper.to_rgba(utility_values[predicate])
        return formatColor(rgba_color[0], rgba_color[1], rgba_color[2])


def get_screen_pos(pos_x, pos_y, original_height, original_width):
    """
    Map the coordinate in original game into the coordinate of the screen
    Note: The pacman game (Berkeley project) has different coordinate system, in which
        the (0,0) is the left-bottom corner
    """
    # swap pos_x and pos_y
    tmp = pos_x
    pos_x = pos_y
    pos_y = tmp

    pos_x = original_height - pos_x - 1
    return int(pos_x*SCALE), int(pos_y*SCALE)

#########################################################
############### Ghost Graphic Functions  ################
#########################################################


def get_ghost_color(ghost_index, is_scared=False):
    if is_scared and IS_CHANGE_SCARED_COLOR:
        return GHOST_SCARED_COLOR
    else:
        return GHOST_COLORS[ghost_index]


def get_ghost_poly(pos_x, pos_y):
    x = GHOST_X*GHOST_SIZE*SCALE + pos_x
    y = GHOST_Y*GHOST_SIZE*SCALE + pos_y
    rr_ghost, cc_ghost = polygon(x, y)
    return rr_ghost, cc_ghost


def draw_ghost(img_array, ghost_index, pos_x, pos_y, is_scared=False):
    body_rr, body_cc = get_ghost_poly(pos_x, pos_y)

    left_eye_coords = GHOST_SIZE*GHOST_EYE_RIGHT*SCALE
    right_eye_coords = GHOST_SIZE*GHOST_EYE_LEFT*SCALE
    eye_radius = GHOST_EYE_RADIUS*GHOST_SIZE*SCALE
    left_eye_rr, left_eye_cc = circle(pos_x+left_eye_coords[0], pos_y+left_eye_coords[1], eye_radius)
    right_eye_rr, right_eye_cc = circle(pos_x+right_eye_coords[0], pos_y+right_eye_coords[1], eye_radius)

    pupil_radius = GHOST_PUPIL_RADIUS*GHOST_SIZE*SCALE
    left_pupil_rr, left_pupil_cc = circle(pos_x+left_eye_coords[0], pos_y+left_eye_coords[1], pupil_radius)
    right_pupil_rr, right_pupil_cc = circle(pos_x+right_eye_coords[0], pos_y+right_eye_coords[1], pupil_radius)

    img_array[body_rr, body_cc, :] = get_ghost_color(ghost_index, is_scared=is_scared)
    img_array[left_eye_rr, left_eye_cc, :] = GHOST_EYES_COLOR
    img_array[right_eye_rr, right_eye_cc, :] = GHOST_EYES_COLOR
    img_array[left_pupil_rr, left_pupil_cc, :] = GHOST_PUPIL_COLOR
    img_array[right_pupil_rr, right_pupil_cc, :] = GHOST_PUPIL_COLOR


def draw_ghost_utility(img_array, color_mapper, utility_values, pos_x, pos_y, utility_mask=None):
    body_rr, body_cc = get_ghost_poly(pos_x, pos_y)

    ghost_predicate = PREDICATES.GHOST_PREDICATE.value
    color = get_utility_color(color_mapper, ghost_predicate, utility_values)
    img_array[body_rr, body_cc, :] = color

    if utility_mask is not None:
        utility_mask[body_rr, body_cc] = utility_values[ghost_predicate]

#########################################################
############### Pacman Graphic Functions  ###############
#########################################################


def draw_pacman(img_array, pos_x, pos_y, background_color=(0, 0, 0), direction=None):
    pacman_radius = PACMAN_RADIUS*SCALE*PACMAN_SCALE
    body_rr, body_cc = circle(pos_x, pos_y, pacman_radius)
    img_array[body_rr, body_cc, :] = PACMAN_COLOR


#########################################################
############### Game Objects Functions  #################
#########################################################


def draw_wall(img_array, pos_x, pos_y):
    start = (pos_x, pos_y)
    end = (int(pos_x + WALL_RADIUS*SCALE), int(pos_y + WALL_RADIUS*SCALE))

    rr, cc = rectangle(start, end)
    img_array[rr, cc, :] = WALL_COLOR


def draw_food(img_array, pos_x, pos_y):
    food_radius = FOOD_SIZE * SCALE
    rr, cc = circle(pos_x, pos_y, food_radius)
    img_array[rr, cc, :] = FOOD_COLOR


def draw_capsule(img_array, pos_x, pos_y):
    capsule_radius = CAPSULE_SIZE * SCALE
    rr, cc = circle(pos_x, pos_y, capsule_radius)
    img_array[rr, cc, :] = CAPSULE_COLOR


def draw_capsule_utility(img_array, color_mapper, utility_values, pos_x, pos_y, utility_mask=None):
    capsule_predicate = PREDICATES.CAPSULE_PREDICATE.value
    color = get_utility_color(color_mapper, capsule_predicate, utility_values)

    capsule_radius = CAPSULE_SIZE * SCALE
    rr, cc = circle(pos_x, pos_y, capsule_radius)
    img_array[rr, cc, :] = color

    if utility_mask is not None:
        utility_mask[rr, cc] = utility_values[capsule_predicate]

#########################################################
#################### The Graphics  ######################
#########################################################


_main_window = None
_canvas = None


def sleep(secs):
    global _main_window
    if _main_window is None:
        time.sleep(secs)
    else:
        _main_window.update_idletasks()
        _main_window.after(int(1000 * secs), _main_window.quit)
        _main_window.mainloop()


class PacmanEfficientGraphics:
    def __init__(self, zoom=1.0, frameTime=0.0, show_utility=False, is_render=False):
        self.zoom = zoom
        self.frameTime = frameTime
        self.show_utility = show_utility
        self.grid_size = float(int(DEFAULT_GRID_SIZE*zoom))

        util_color_norm = mpl_colors.Normalize(vmin=-20, vmax=1, clip=True)
        self.util_color_mapper = cm.ScalarMappable(norm=util_color_norm, cmap=plt.get_cmap('bwr'))

    def initialize(self, state):
        global SCALE
        layout = state.layout
        self.last_state = state
        self.width = layout.width
        self.height = layout.height

        # compute window size
        self.screen_width = int((self.width+1)*self.grid_size)
        self.screen_height = int((self.height+1)*self.grid_size)
        # in sklearn, height is x, width is y
        self.background_img = np.zeros(shape=(self.screen_height, self.screen_width, 3)).astype(np.uint8)
        self.rgb_observation = np.zeros_like(self.background_img)
        self.utility_map = np.zeros_like(self.background_img)
        self.utility_mask = np.zeros(shape=(self.utility_map.shape[0], self.utility_map.shape[1]))

        self.display_height = self.screen_height
        if self.show_utility:
            self.display_height = self.display_height*2

        SCALE = self.grid_size
        # init background
        self.draw_walls(self.background_img, layout.walls)

    def render(self):
        global _main_window, _canvas

        if _main_window is None:
            _main_window = tk.Tk()
            _main_window.geometry('%dx%d+%d+%d' % (self.screen_width, self.display_height, 10, 0))
            _main_window.resizable(False, False)

            _canvas = tk.Canvas(_main_window, width=self.screen_width-1, height=self.display_height-1)
            _canvas.pack()

        img = ImageTk.PhotoImage(master=_canvas, image=Image.fromarray(self.rgb_observation))
        _canvas.create_image((0, 0), image=img, anchor="nw")
        img_utility = ImageTk.PhotoImage(master=_canvas, image=Image.fromarray(self.utility_map))
        _canvas.create_image((0, self.screen_height), image=img_utility, anchor="nw")
        sleep(self.frameTime)

    def draw_walls(self, img, walls):
        for x in range(self.width):
            for y in range(self.height):
                if walls[x][y]:
                    pos_x, pos_y = get_screen_pos(x, y, self.height, self.width)
                    draw_wall(img, pos_x, pos_y)

    def draw_foods(self, img, foods):
        for x in range(self.width):
            for y in range(self.height):
                if foods[x][y]:
                    pos_x, pos_y = get_screen_pos(x, y, self.height, self.width)
                    pos_x = pos_x + int(self.grid_size/2.0)
                    pos_y = pos_y + int(self.grid_size/2.0)
                    draw_food(img, pos_x, pos_y)

    def update(self, state_data):
        self.last_state = state_data
        # update graphics
        rgb_observation = self.background_img.copy()
        # draw foods
        self.draw_foods(rgb_observation, state_data.food)

        for agent_idx in range(len(state_data.is_agent_dead)):
            if not state_data.is_agent_dead[agent_idx]:
                x, y = state_data.agentStates[agent_idx].configuration.pos
                # draw agents
                if agent_idx == 0:
                    # in sk-image, row is x, column is y
                    pos_x, pos_y = get_screen_pos(x, y, self.height, self.width)
                    pos_x = pos_x + int(self.grid_size / 2.0)
                    pos_y = pos_y + int(self.grid_size / 2.0)
                    draw_pacman(rgb_observation, pos_x, pos_y)
                # draw ghost
                else:
                    # in sk-image, row is x, column is y
                    pos_x, pos_y = get_screen_pos(x, y, self.height, self.width)
                    pos_x = pos_x + int(self.grid_size / 2.0)
                    pos_y = pos_y + int(self.grid_size / 2.0)
                    draw_ghost(rgb_observation, agent_idx, pos_x, pos_y)

        # draw capsules
        for capsule in state_data.capsules:
            x, y = capsule
            pos_x, pos_y = get_screen_pos(x, y, self.height, self.width)
            pos_x = pos_x + int(self.grid_size / 2.0)
            pos_y = pos_y + int(self.grid_size / 2.0)
            draw_capsule(rgb_observation, pos_x, pos_y)

        self.rgb_observation = rgb_observation
        return rgb_observation

    def update_utility_map(self, utility_values):
        self.utility_map = np.zeros_like(self.rgb_observation)
        self.utility_mask = np.zeros(shape=(self.utility_map.shape[0], self.utility_map.shape[1]))
        state_data = self.last_state
        # draw ghost utility
        for agent_idx in range(len(state_data.is_agent_dead)):
            if not state_data.is_agent_dead[agent_idx]:
                x, y = state_data.agentStates[agent_idx].configuration.pos
                # draw agents
                if agent_idx != 0:
                    # in sk-image, row is x, column is y
                    pos_x, pos_y = get_screen_pos(x, y, self.height, self.width)
                    pos_x = pos_x + int(self.grid_size / 2.0)
                    pos_y = pos_y + int(self.grid_size / 2.0)
                    draw_ghost_utility(self.utility_map, self.util_color_mapper, utility_values, pos_x, pos_y, self.utility_mask)
        # draw capsule utility
        for capsule in state_data.capsules:
            x, y = capsule
            pos_x, pos_y = get_screen_pos(x, y, self.height, self.width)
            pos_x = pos_x + int(self.grid_size / 2.0)
            pos_y = pos_y + int(self.grid_size / 2.0)
            draw_capsule_utility(self.utility_map, self.util_color_mapper, utility_values, pos_x, pos_y, self.utility_mask)

    def getRgbObservation(self):
        return self.rgb_observation.copy()

    def getRgbUtilityMap(self, utilities_values, get_mask=False):
        self.update_utility_map(utilities_values)
        if get_mask:
            return self.utility_mask.copy()
        else:
            return self.utility_map.copy()

    def finish(self):
        self.rgb_observation = np.zeros_like(self.rgb_observation)
        self.utility_map = np.zeros_like(self.utility_map)


if __name__ == '__main__':
    pass


