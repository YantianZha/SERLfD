# eatGhostPacmanGame.py
# ------------------------
# helper functions for pacman game
from pacman_src import layout
from pacman_src import pacman
from pacman_src import game as game_source
from pacman_src.game import GameStateData
from pacman_src.util import nearestPoint
from pacman_src.util import manhattanDistance
from pacman_src import graphicsDisplay
from utils.utils import manhattanDistance

import configparser
import sys
import enum
import os


############################################################################
###################### Game Parameters #####################################
############################################################################
SCARED_TIME = 30    # Moves ghosts are scared
COLLISION_TOLERANCE = 0.7  # How close ghosts must be to Pacman to kill
TIME_PENALTY = 0  # Number of points lost each round
EAT_FOOD_REWARD = 0     # The reward of eating the food
WIN_GAME_REWARD = 1     # The reward of winning the game
LOSE_GAME_REWARD = 0   # The reward of losing the game
EAT_GHOST_REWARD = 0    # The reward of eating ghost
############################################################################
############################ Predicate Related #############################
############################################################################
GHOST_NEARBY_DIST = 6
############################################################################
############################################################################


class PREDICATES(enum.Enum):
    CAPSULE_PREDICATE = 'is_eat_capsule'
    GHOST_PREDICATE = 'is_ghost_nearby'


class EatGhostGame:
    """
    The Game manages the control flow, soliciting actions from agents.
    """
    def __init__(self, agents, display, rules, init_state, starting_index=0):
        self.agent_crashed = False
        self.agents = agents
        self.display = display
        self.rules = rules
        self.starting_index = starting_index
        self.game_over = False
        self.state = init_state
        self.last_move = pacman.Directions.STOP
        self.last_score = 0

    def initialize(self):
        self.display.initialize(self.state.data)
        self.last_score = 0
        # initialize agents
        for i in range(len(self.agents)):
            agent = self.agents[i]
            if agent is None:
                print("Agent %d failed to load" % i, file=sys.stderr)
                return
            if "registerInitialState" in dir(agent):
                agent.registerInitialState(self.state.deepCopy())

    def set_action(self, action):
        """ Set the action of the proxy agent """
        self.agents[self.starting_index].setAction(action)

    def render(self):
        if 'render' in dir(self.display):
            self.display.render()

    def update(self):
        """
        Render the environment at current step,
        including executing the specified action, updating the ghosts
        Note: it will also update the utility map if utilities is not None
        """
        if self.game_over:
            return

        agents = self.agents
        for agent_idx in range(len(agents)):
            agent = agents[agent_idx]
            # never update a dead agent
            if self.state.data.is_agent_dead[agent_idx]:
                continue

            old_agent_dead = self.state.data.is_agent_dead.copy()
            observation = self.state.deepCopy()
            action = agent.getAction(observation)
            # record the keyboard agent's last move
            if agent_idx == 0 and 'getLastMove' in dir(agent):
                self.last_move = action
            # Execute the action
            self.state = self.state.generateSuccessor(agent_idx, action)

            # update the display
            self.display.update(self.state.data)
            # remove dead ghosts from the screen
            for agent_i in range(1, len(agents)):
                if not old_agent_dead[agent_i] and self.state.data.is_agent_dead[agent_i]:
                    if 'removeGhost' in dir(self.display):
                        self.display.removeGhost(self.state, agent_i)

            # Allow for game specific conditions (winning, losing, etc.)
            self.rules.process(self.state, self)
            if self.game_over:
                break

        rgb_numpy = self.get_current_observation()
        info = {}

        # if the game is over, info agents and display
        if self.game_over:
            self.final()
        # compute the reward
        reward = self.state.data.score - self.last_score
        self.last_score = self.state.data.score
        return rgb_numpy, reward, self.game_over, info

    def final(self):
        for agentIndex, agent in enumerate(self.agents):
            if "final" in dir(agent):
                agent.final(self.state)
        self.display.finish()

    def update_utility_map(self, utilities, get_mask=False):
        if self.game_over:
            return None
        self.display.getRgbUtilityMap(utilities, get_mask=get_mask)

    def set_game_over(self, is_over):
        self.game_over = is_over

    def get_current_observation(self):
        if 'getRgbObservation' in dir(self.display):
            return self.display.getRgbObservation()
        return self.state

    def get_current_utility_map(self, utilities_values, get_mask=False):
        """ get current utility map (numpy, img) """
        if self.game_over:
            return None
        else:
            return self.display.getRgbUtilityMap(utilities_values, get_mask=get_mask)

    def get_current_predicate_values(self):
        state = self.state
        is_eat_capsule = -1
        is_ghost_nearby = -1

        # get pacman position
        pacman_pos = state.getPacmanPosition()

        for agent_idx in range(1, len(self.agents)):
            if state.data.is_agent_dead[agent_idx]:
                continue

            agent_state = state.data.agentStates[agent_idx]
            # compute the capsule predicate value (is_eat_capsule)
            if agent_state.scaredTimer > 0:
                is_eat_capsule = 1
            # compute the ghost predicate (is_ghost_nearby)
            ghost_pos = agent_state.configuration.getPosition()
            if manhattanDistance(pacman_pos, ghost_pos) < GHOST_NEARBY_DIST:
                is_ghost_nearby = 1

        return {PREDICATES.GHOST_PREDICATE.value: is_ghost_nearby, PREDICATES.CAPSULE_PREDICATE.value: is_eat_capsule}

    def get_keyboard_agent_last_move(self):
        return self.last_move

    def is_win(self):
        return self.state.isWin()

    def is_lose(self):
        return self.state.isLose()


class EatGhostsGameRules:
    """
    These game rules manage the control flow of a game, deciding when
    and how the game starts and ends.
    """

    def __init__(self, timeout=30):
        self.timeout = timeout
        self.initial_state = None

    def new_game(self, layout, pacmanAgent, ghostAgents, display):
        agents = [pacmanAgent] + ghostAgents[:layout.getNumGhosts()]
        # init initial state
        init_state = EatGhostGameState()
        init_state.initialize(layout, len(ghostAgents))

        game = EatGhostGame(agents, display, self, init_state)
        self.initial_state = init_state.deepCopy()
        return game

    def process(self, state, game):
        """
        Checks to see whether it is time to end the game.
        """
        if state.isWin():
            self.win(state, game)
        if state.isLose():
            self.lose(state, game)
        # check if there is no capsule left but there is still some ghost alive
        remaining_capsules = len(state.data.capsules)
        if remaining_capsules == 0:
            for agent_idx in range(1, len(state.data.agentStates)):
                # if ghost is alive and not scared
                if not state.data.is_agent_dead[agent_idx] and state.data.agentStates[agent_idx].scaredTimer < 1:
                    state.data.score += LOSE_GAME_REWARD
                    self.lose(state, game)

    def win(self, state, game):
        # print("Pacman wins! Score: %d" % state.data.score)
        game.set_game_over(True)

    def lose(self, state, game): 
        # print("Pacman died! Score: %d" % state.data.score)
        game.set_game_over(True)


# noinspection DuplicatedCode,PyMissingConstructor
class EatGhostGameState(pacman.GameState):

    def __init__(self, prev_state=None):
        if prev_state is not None:  # Initial state
            self.data = EatGhostGameStateData(prev_state.data)
        else:
            self.data = EatGhostGameStateData()

    def generateSuccessor(self, agent_index, action):
        """
        Returns the successor state after the specified agent takes the action.
        """
        # Check that successors exist
        if self.isWin() or self.isLose():
            raise Exception('Can\'t generate a successor of a terminal state.')

        # Copy current state
        state = EatGhostGameState(self)

        # Let agent's logic deal with its action's effects on the board
        if agent_index == 0:  # Pacman is moving
            state.data._eaten = [False for i in range(state.getNumAgents())]
            EatGhostPacmanRules.applyAction(state, action)
        else:   # A ghost is moving
            EatGhostGhostRules.applyAction(state, action, agent_index)

        # Time passes
        if agent_index == 0:
            state.data.scoreChange += -TIME_PENALTY  # Penalty for waiting around
        else:
            EatGhostGhostRules.decrementTimer(state.data.agentStates[agent_index])

        # Resolve multi-agent effects
        EatGhostGhostRules.checkDeath(state, agent_index)

        # check if all the ghosts are eaten
        for agent_i, is_eaten in enumerate(state.data._eaten):
            if is_eaten:
                state.data.is_agent_dead[agent_i] = True
        # count eaten ghosts
        n_eaten_ghosts = 0
        for agent_i in range(1, state.getNumAgents()):
            if state.data.is_agent_dead[agent_i]:
                n_eaten_ghosts += 1
        # check terminal state
        if not state.data.is_agent_dead[0] and n_eaten_ghosts == state.getNumAgents()-1:
            state.data._win = True
            state.data.scoreChange += WIN_GAME_REWARD

        # Book keeping
        state.data._agentMoved = agent_index
        state.data.score += state.data.scoreChange
        pacman.GameState.explored.add(self)     # use the explored from GameState rather than EatGhostGameState
        pacman.GameState.explored.add(state)
        return state


# noinspection PyPep8Naming,DuplicatedCode
class EatGhostGameStateData(GameStateData):
    def __init__(self, prev_state=None):
        super().__init__(prev_state)
        # is agent dead
        if prev_state is not None:
            self.is_agent_dead = prev_state.is_agent_dead

    def initialize(self, layout, numGhostAgents):
        super().initialize(layout, numGhostAgents)
        self.is_agent_dead = [False for _ in range(len(self.agentStates))]

    def deepCopy(self):
        state = EatGhostGameStateData(self)
        state.food = self.food.deepCopy()
        state.layout = self.layout.deepCopy()
        state._agentMoved = self._agentMoved
        state._foodEaten = self._foodEaten
        state._foodAdded = self._foodAdded
        state._capsuleEaten = self._capsuleEaten
        state.is_agent_dead = self.is_agent_dead.copy()
        return state

    def __eq__(self, other):
        if other is None:
            return False
        if other.is_agent_dead == self.is_agent_dead:
            return super().__eq__(other)
        else:
            return False

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.
        """
        for i, state in enumerate(self.agentStates):
            try:
                int(hash(state))
            except TypeError as e:
                print(e)
                # hash(state)
        return int((hash(tuple(self.agentStates)) + hash(tuple(self.is_agent_dead)) + 13*hash(self.food) + 113 * hash(tuple(self.capsules)) + 7 * hash(self.score)) % 1048575)

    def __str__(self):
        width, height = self.layout.width, self.layout.height
        map = game_source.Grid(width, height)
        if type(self.food) == type((1, 2)):
            self.food = game_source.reconstituteGrid(self.food)
        for x in range(width):
            for y in range(height):
                food, walls = self.food, self.layout.walls
                map[x][y] = self._foodWallStr(food[x][y], walls[x][y])

        for agent_idx, agentState in enumerate(self.agentStates):
            if agentState is None or agentState.configuration is None:
                continue
            if self.is_agent_dead[agent_idx]:
                continue

            x, y = [int(i) for i in nearestPoint(agentState.configuration.pos)]
            agent_dir = agentState.configuration.direction
            if agentState.isPacman:
                map[x][y] = self._pacStr(agent_dir)
            else:
                map[x][y] = self._ghostStr(agent_dir)

        for x, y in self.capsules:
            map[x][y] = 'o'

        return str(map) + ("\nScore: %d\n" % self.score)


# noinspection PyPep8Naming
class EatGhostPacmanRules(pacman.PacmanRules):
    """
    These functions govern how pacman interacts with his environment under
    the classic game rules.
    """
    def __init__(self):
        super().__init__()

    PACMAN_SPEED = 1

    def consume(position, state):
        x, y = position
        # Eat food
        if state.data.food[x][y]:
            state.data.scoreChange += EAT_FOOD_REWARD
            state.data.food = state.data.food.copy()
            state.data.food[x][y] = False
            state.data._foodEaten = position

        # Eat capsule
        if position in state.getCapsules():
            state.data.capsules.remove(position)
            state.data._capsuleEaten = position
            # Reset all ghosts' scared timers
            for index in range(1, len(state.data.agentStates)):
                state.data.agentStates[index].scaredTimer = SCARED_TIME
    consume = staticmethod(consume)

    def applyAction(state, action):
        """
        Edits the state to reflect the results of the action.
        """
        legal = EatGhostPacmanRules.getLegalActions(state)
        if action not in legal:
            action = game_source.Directions.STOP

        pacmanState = state.data.agentStates[0]

        # Update Configuration
        vector = pacman.Actions.directionToVector(action, EatGhostPacmanRules.PACMAN_SPEED)
        pacmanState.configuration = pacmanState.configuration.generateSuccessor(vector)

        # Eat
        next = pacmanState.configuration.getPosition()
        nearest = nearestPoint(next)
        if manhattanDistance(nearest, next) <= 0.5:
            # Remove food
            EatGhostPacmanRules.consume(nearest, state)
    applyAction = staticmethod(applyAction)


# noinspection PyPep8Naming,DuplicatedCode
class EatGhostGhostRules(pacman.GhostRules):
    """
    These functions dictate how ghosts interact with their environment.
    """
    def __init__(self):
        super().__init__()

    GHOST_SPEED = 1.0

    def applyAction(state, action, ghostIndex):
        legal = EatGhostGhostRules.getLegalActions(state, ghostIndex)
        if action not in legal:
            raise Exception("Illegal ghost action " + str(action))

        ghostState = state.data.agentStates[ghostIndex]
        speed = EatGhostGhostRules.GHOST_SPEED
        if ghostState.scaredTimer > 0:
            speed /= 2.0
        vector = pacman.Actions.directionToVector(action, speed)
        ghostState.configuration = ghostState.configuration.generateSuccessor(vector)
    applyAction = staticmethod(applyAction)

    def checkDeath(state, agentIndex):
        pacmanPosition = state.getPacmanPosition()
        if agentIndex == 0:  # Pacman just moved; Anyone can kill him
            for index in range(1, len(state.data.agentStates)):
                # if the encountered ghost is dead, no need to check
                if state.data.is_agent_dead[index]:
                    continue
                ghostState = state.data.agentStates[index]
                ghostPosition = ghostState.configuration.getPosition()
                if EatGhostGhostRules.canKill(pacmanPosition, ghostPosition):
                    EatGhostGhostRules.collide(state, ghostState, index)
        else:
            # no need to check a dead ghost
            if state.data.is_agent_dead[agentIndex]:
                return
            ghostState = state.data.agentStates[agentIndex]
            ghostPosition = ghostState.configuration.getPosition()
            if EatGhostGhostRules.canKill(pacmanPosition, ghostPosition):
                EatGhostGhostRules.collide(state, ghostState, agentIndex)
    checkDeath = staticmethod(checkDeath)

    def collide(state, ghostState, agentIndex):
        # ghost is eaten by pacman
        if ghostState.scaredTimer > 0:
            state.data.scoreChange += EAT_GHOST_REWARD
            EatGhostGhostRules.removeGhost(state, ghostState, agentIndex)
            ghostState.scaredTimer = 0
            # Added for first-person
            state.data._eaten[agentIndex] = True
        # pacman is eaten by a ghost
        else:
            if not state.data._win:
                state.data.scoreChange += LOSE_GAME_REWARD
                state.data._eaten[0] = True
                state.data._lose = True
    collide = staticmethod(collide)

    def removeGhost(state, ghostState, agentIndex):
        ghostState.configuration = ghostState.start
    removeGhost = staticmethod(removeGhost)


def run_game(layout, pacman, ghosts, display, timeout=30, **kwargs):
    import __main__
    __main__.__dict__['_display'] = display

    rules = EatGhostsGameRules(timeout)
    game = rules.new_game(layout, pacman, ghosts, display)
    game.initialize()
    return game


def read_config(config_file=None):
    args = {}
    # if config file is None, use the default env setting
    if config_file is None:
        config_file = os.path.dirname(os.path.abspath(__file__)) + '/default_pacman.cnf'
    config = configparser.ConfigParser()
    config.read(config_file)

    # Choose a layout
    layout_name = config.get('CUSTOMIZE', 'layout')
    args['layout'] = layout.getLayout(layout_name)
    # Whether use graphics output
    args['is_graphics_output'] = config.getboolean('CUSTOMIZE', 'graphicsOutput')
    # Whether to use efficient graphics
    args['is_efficient_graphics'] = config.getboolean('CUSTOMIZE', 'efficientGraphics')

    # Choose a Pacman agent
    pacman_type = config.get('CUSTOMIZE', 'pacman')
    if pacman_type == 'KeyboardAgent':
        pacman_class = pacman.loadAgent(pacman_type, nographics=False)
        args['is_graphics_output'] = True
    else:
        # for all other types of pacman, we use a proxy agent which
        # takes the action given in the step function
        pacman_class = pacman.loadAgent(pacman_type, nographics=args['is_graphics_output'])
    args['pacman'] = pacman_class()

    # Choose the number of ghosts
    num_ghosts = config.getint('CUSTOMIZE', 'numGhosts')

    # Choose a ghost agent
    ghost_type = config.get('CUSTOMIZE', 'ghost')
    ghost_class = pacman.loadAgent(ghost_type, nographics=args['is_graphics_output'])
    args['ghosts'] = [ghost_class(i+1) for i in range(num_ghosts)]

    # Choose a frame time
    args['frame_time'] = config.getfloat('CUSTOMIZE', 'frameTime')

    # Choose a zoom scale
    args['zoom_scale'] = config.getfloat('CUSTOMIZE', 'zoom')

    # Choose a display format
    if not args['is_graphics_output']:
        from pacman_src import textDisplay
        args['display'] = textDisplay.NullGraphics()
    elif not args['is_efficient_graphics']:
        from eat_ghost_env.utilityMapGraphicsDisplay import UtilityPacmanGraphics
        args['display'] = UtilityPacmanGraphics(args['zoom_scale'], frameTime=args['frame_time'])
    else:
        from eat_ghost_env import efficient_graphics
        args['display'] = efficient_graphics.PacmanEfficientGraphics(args['zoom_scale'], frameTime=args['frame_time'], show_utility=True)

    # Choose whether to record game
    args['is_record'] = config.getboolean('CUSTOMIZE', 'record')

    return args


if __name__ == '__main__':
    pass

