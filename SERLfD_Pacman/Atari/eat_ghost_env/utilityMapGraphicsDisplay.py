# utilityMapGraphicsDisplay.py
# ------------------

from pacman_src import graphicsUtils, graphicsDisplay
from eat_ghost_env import utilityGraphicsUtils
import torchvision.transforms as transforms
from eat_ghost_env.eatGhostPacmanGame import PREDICATES
from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mpl_colors
import matplotlib.cm as cm
import numpy as np
import io

###########################
#  GRAPHICS DISPLAY CODE  #
###########################

REFRESH_UTILITY_MAP_ON_DEMAND = True


# Most code by Dan Klein and John Denero written or rewritten for cs188, UC Berkeley.
# Some code from a Pacman implementation by LiveWires, and used / modified with permission.

formatColor = utilityGraphicsUtils.formatColor
colorToVector = utilityGraphicsUtils.colorToVector
DEFAULT_GRID_SIZE = 30.0
INFO_PANE_HEIGHT = 12
BACKGROUND_COLOR = formatColor(255, 255, 255)
WALL_COLOR = formatColor(0.0/255.0, 51.0/255.0, 255.0/255.0)
INFO_PANE_COLOR = formatColor(.4, .4, 0)
SCORE_COLOR = formatColor(.9, .9, .9)
PACMAN_OUTLINE_WIDTH = 2
PACMAN_CAPTURE_OUTLINE_WIDTH = 4

GHOST_COLORS = []
GHOST_COLORS.append(formatColor(.9, 0, 0))  # Red
GHOST_COLORS.append(formatColor(0, .3, .9))  # Blue
GHOST_COLORS.append(formatColor(.98, .41, .07))  # Orange
GHOST_COLORS.append(formatColor(.1, .75, .7))  # Green
GHOST_COLORS.append(formatColor(1.0, 0.6, 0.0))  # Yellow
GHOST_COLORS.append(formatColor(.4, 0.13, 0.91))  # Purple

TEAM_COLORS = GHOST_COLORS[:2]

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
GHOST_SIZE = 0.65
SCARED_COLOR = formatColor(1, 1, 1)

GHOST_VEC_COLORS = list(map(colorToVector, GHOST_COLORS))

PACMAN_COLOR = formatColor(255.0/255.0, 255.0/255.0, 61.0/255)
PACMAN_SCALE = 0.5

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
WALL_RADIUS = 0.15


# noinspection DuplicatedCode,PyPep8Naming,PyAttributeOutsideInit
class UtilityPacmanGraphics(graphicsDisplay.PacmanGraphics):
    def __init__(self, zoom=1.0, frameTime=0.0, capture=False):
        super().__init__(zoom=1.0, frameTime=0.0, capture=False)
        self.have_window = 0
        self.currentGhostImages = {}
        self.pacmanImage = None
        self.zoom = zoom
        self.gridSize = float(int(DEFAULT_GRID_SIZE * zoom))
        self.capture = capture
        self.frameTime = frameTime
        self.latestState = None
        self.pil_to_tensor = transforms.ToTensor()

        # utility images
        self.capsules_utility_img = {}
        self.ghosts_utility_img = {}

        # init colour map
        util_color_norm = mpl_colors.Normalize(vmin=-1, vmax=1, clip=True)
        self.util_color_mapper = cm.ScalarMappable(norm=util_color_norm, cmap=plt.get_cmap('bwr'))

    def initialize(self, state, isBlue=False):
        self.isBlue = isBlue
        self.startGraphics(state)

        self.drawStaticObjects(state)
        self.drawAgentObjects(state)
        self.init_utility_window(state)

        # Information
        self.previousState = state
        self.latestState = state

    def startGraphics(self, state):
        # create game play window
        self.layout = state.layout
        layout = self.layout
        self.width = layout.width
        self.height = layout.height

        # create utility map window
        self.make_utility_window(self.width, self.height)

        # make the root window
        self.make_window(self.width, self.height)
        self.infoPane = graphicsDisplay.InfoPane(layout, self.gridSize)
        self.currentState = layout

    def init_utility_window(self, state, utility_values=None):
        # draw capsule utility
        self.capsules_utility_img = self.draw_capsules_utility(state, utility_values)
        # draw ghost utility
        ghosts_utility_img = {}
        for index, agentState in enumerate(state.agentStates):
            if not agentState.isPacman:
                ghosts_utility_img[index] = self.draw_ghost_utility(agentState, utility_values)
        self.ghosts_utility_img = ghosts_utility_img
        # refresh
        utilityGraphicsUtils.refresh()

    def make_utility_window(self, width, height):
        grid_width = (width-1) * self.gridSize
        grid_height = (height-1) * self.gridSize
        screen_width = 2*self.gridSize + grid_width
        screen_height = 2*self.gridSize + grid_height + INFO_PANE_HEIGHT

        utilityGraphicsUtils.begin_graphics(screen_width, screen_height, BACKGROUND_COLOR, "Pacman Utility")

    def update(self, newState):
        agentIndex = newState._agentMoved
        is_agent_dead = newState.is_agent_dead
        if is_agent_dead is not None and is_agent_dead[agentIndex]:
            return

        agentState = newState.agentStates[agentIndex]
        if self.agentImages[agentIndex][0].isPacman != agentState.isPacman:
            self.swapImages(agentIndex, agentState)
        prevState, prevImage = self.agentImages[agentIndex]
        if agentState.isPacman:
            self.animatePacman(agentState, prevState, prevImage)
        else:
            self.moveGhost(agentState, agentIndex, prevState, prevImage)
        self.agentImages[agentIndex] = (agentState, prevImage)

        if newState._foodEaten != None:
            self.removeFood(newState._foodEaten, self.food)
        if newState._capsuleEaten != None:
            # remove capsule utility
            self.remove_capsule_utility(newState._capsuleEaten)
            # remove capsule
            self.removeCapsule(newState._capsuleEaten, self.capsules)
        self.infoPane.updateScore(newState.score)

        # move the objects (not update the utility values)
        self.latestState = newState
        self.update_utility(newState, None)

    def removeGhost(self, state, ghostIndex):
        _, ghost_image_parts = self.agentImages[ghostIndex]
        for ghost_image_part in ghost_image_parts:
            graphicsUtils.remove_from_screen(ghost_image_part)
        self.remove_ghost_utility(ghostIndex)

    def update_utility(self, newState, utility_values=None):
        agentIndex = newState._agentMoved
        agentState = newState.agentStates[agentIndex]

        for capsule_pos in newState.capsules:
            self.update_capsule_utility(capsule_pos, utility_values)
        if not agentState.isPacman:
            self.update_ghost_utility(agentIndex, agentState, utility_values)
        # refresh
        if not REFRESH_UTILITY_MAP_ON_DEMAND:
            utilityGraphicsUtils.refresh()

    def get_utility_color(self, predicate, utility_values):
        if utility_values is None or predicate not in utility_values:
            WHITE = formatColor(1.0, 1.0, 1.0)
            return WHITE
        else:
            rgba_color = self.util_color_mapper.to_rgba(utility_values[predicate])
            return formatColor(rgba_color[0], rgba_color[1], rgba_color[2])

    def draw_ghost_utility(self, ghostState, utility_values=None):
        pos = self.getPosition(ghostState)
        (screen_x, screen_y) = (self.to_screen(pos))
        coords = []
        for (x, y) in GHOST_SHAPE:
            coords.append((x * self.gridSize * GHOST_SIZE + screen_x,
                           y * self.gridSize * GHOST_SIZE + screen_y))

        ghost_predicate = PREDICATES.GHOST_PREDICATE.value
        ghost_color = self.get_utility_color(ghost_predicate, utility_values)
        body_img = utilityGraphicsUtils.polygon(coords, ghost_color, filled=1)
        return body_img

    def draw_capsules_utility(self, state, utility_values=None):
        capsule_predicate = PREDICATES.CAPSULE_PREDICATE.value
        capsules_color = self.get_utility_color(capsule_predicate, utility_values)

        capsule_utility_images = {}
        capsule_positions = state.capsules
        for capsule_pos in capsule_positions:
            (screen_x, screen_y) = self.to_screen(capsule_pos)
            dot = utilityGraphicsUtils.circle((screen_x, screen_y), CAPSULE_SIZE * self.gridSize
                                              , outlineColor=capsules_color,
                                              fillColor=capsules_color,
                                              width=1)
            capsule_utility_images[capsule_pos] = dot
        return capsule_utility_images

    def update_ghost_utility(self, ghostIndex, ghostState, utility_values=None):
        ghost_predicate = PREDICATES.GHOST_PREDICATE.value
        ghost_color = self.get_utility_color(ghost_predicate, utility_values)

        new_x, new_y = self.to_screen(self.getPosition(ghostState))
        utilityGraphicsUtils.edit(self.ghosts_utility_img[ghostIndex], ('fill', ghost_color), ('outline', ghost_color))
        utilityGraphicsUtils.move_to(self.ghosts_utility_img[ghostIndex], x=new_x, y=new_y)

    def update_capsule_utility(self, capsulePos, utility_values=None):
        capsule_predicate = PREDICATES.CAPSULE_PREDICATE.value
        capsules_color = self.get_utility_color(capsule_predicate, utility_values)

        utilityGraphicsUtils.edit(self.capsules_utility_img[capsulePos], ('fill', capsules_color), ('outline', capsules_color))

    def remove_ghost_utility(self, agent_index):
        utilityGraphicsUtils.remove_from_screen(self.ghosts_utility_img[agent_index])

    def remove_capsule_utility(self, capsulePos):
        utilityGraphicsUtils.remove_from_screen(self.capsules_utility_img[capsulePos])

    # noinspection PyMethodMayBeStatic
    def getRgbObservation(self):
        """
        get the rbg observation of current state
        :return: return numpy 3d array, and PIL image
        """
        rgb_img = Image.open(io.BytesIO(graphicsUtils.getPostscript().encode('utf-8')))
        if graphicsDisplay.OBSERVATION_RETURN_TENSOR:
            return np.asarray(rgb_img)
        else:
            return np.asarray(rgb_img)

    # noinspection PyMethodMayBeStatic
    def getRgbUtilityMap(self, utility_values, get_mask=False):
        """
        The utility map is generated by the utility values used in the update function
        :return: return numpy 3d array, and PIL image
        """
        state = self.latestState
        for agent_idx, agent_state in enumerate(state.agentStates):
            if agent_idx == 0 or self.latestState.is_agent_dead[agent_idx]:
                continue
            self.update_ghost_utility(agent_idx, agent_state, utility_values)
        for capsule_pos in state.capsules:
            self.update_capsule_utility(capsule_pos, utility_values)

        rgb_utility_img = Image.open(io.BytesIO(utilityGraphicsUtils.getPostscript().encode('utf-8')))
        if graphicsDisplay.OBSERVATION_RETURN_TENSOR:
            return np.asarray(rgb_utility_img)
        else:
            return np.asarray(rgb_utility_img)


# noinspection DuplicatedCode
def add(x, y):
    return (x[0] + y[0], x[1] + y[1])


# Saving graphical output
# -----------------------
# Note: to make an animated gif from this postscript output, try the command:
# convert -delay 7 -loop 1 -compress lzw -layers optimize frame* out.gif
# convert is part of imagemagick (freeware)

SAVE_POSTSCRIPT = False
POSTSCRIPT_OUTPUT_DIR = 'frames'
FRAME_NUMBER = 0
import os


def saveFrame():
    "Saves the current graphical output as a postscript file"
    global SAVE_POSTSCRIPT, FRAME_NUMBER, POSTSCRIPT_OUTPUT_DIR
    if not SAVE_POSTSCRIPT:
        return
    if not os.path.exists(POSTSCRIPT_OUTPUT_DIR):
        os.mkdir(POSTSCRIPT_OUTPUT_DIR)
    name = os.path.join(POSTSCRIPT_OUTPUT_DIR, 'frame_%08d.ps' % FRAME_NUMBER)
    FRAME_NUMBER += 1
    graphicsUtils.writePostscript(name)  # writes the current canvas
