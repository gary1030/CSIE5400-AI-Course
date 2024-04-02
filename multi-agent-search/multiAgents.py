# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import random

import util
from game import Agent, Directions
from pacman import GameState
from util import manhattanDistance


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        # print("Scores: ", scores)
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [
            ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # print("newPos: ", newPos)
        # print("newFood: ", newFood)
        # print("newGhostStates: ", newGhostStates)
        # print("newScaredTimes: ", newScaredTimes)
        # print("successorGameState Score: ", successorGameState.getScore())
        FOOD_SCORE_WEIGHT = 30
        GHOST_SCORE_WEIGHT = 30
        STOP_PENALTY = 2
        NEW_POS_IS_FOOD = 30

        foodDist = 0
        # Get the closest food
        foodList = newFood.asList()
        if len(foodList) > 0:
            closestFood = min(
                foodList, key=lambda x: manhattanDistance(x, newPos))
            foodDist = manhattanDistance(closestFood, newPos)
        # print("Food Distance: ", foodDist)

        # If the new position is food, return a high score
        newPosFoodScore = 0
        oldFoodList = currentGameState.getFood().asList()
        if newPos in oldFoodList:
            newPosFoodScore = 1

        # split the ghost score into 3 parts
        # 1. very close to the ghost -> avoid it
        # 2. won't be caught by the ghost -> ignore it
        # 3. newScaredTimes - distance to the ghost -> eat it
        ghostDist = float("inf")
        for i, ghost in enumerate(newGhostStates):
            ghostPos = ghost.getPosition()
            ghostDistance = manhattanDistance(ghostPos, newPos)
            if ghostDistance < 2:
                ghostDist = 0
            elif ghostDistance < newScaredTimes[i]:
                ghostDist = min(ghostDist, float("inf"))
            else:
                ghostDist = min(ghostDist, ghostDistance)

        foodScore = 1.0 / ((foodDist) + 1)
        ghostScore = 1.0 / ((ghostDist) + 1)
        return FOOD_SCORE_WEIGHT * foodScore + newPosFoodScore * NEW_POS_IS_FOOD - GHOST_SCORE_WEIGHT * ghostScore - STOP_PENALTY * (action == Directions.STOP) + successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        agentCnt = gameState.getNumAgents()
        self.bestAction = None
        self.generateSuccessorCalls = 0

        def maxValue(gameState: GameState, depth: int):
            global bestAction
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = -float("inf")
            for action in gameState.getLegalActions(0):
                tempMin = minValue(
                    gameState.generateSuccessor(0, action), depth, 1)
                self.generateSuccessorCalls += 1
                if tempMin > v:
                    v = tempMin
                    if depth == self.depth:
                        self.bestAction = action
            return v

        def minValue(gameState: GameState, depth: int, agentIndex: int):
            # print("Agent Index: ", agentIndex)
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == agentCnt - 1:
                    v = min(v, maxValue(
                        gameState.generateSuccessor(agentIndex, action), depth - 1))
                    self.generateSuccessorCalls += 1
                else:
                    v = min(v, minValue(
                        gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1))
                    self.generateSuccessorCalls += 1
            return v

        score = maxValue(gameState, self.depth)
        # print("Best Action: ", self.bestAction)
        # print("Generate Successor Calls: ", self.generateSuccessorCalls)
        return self.bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        agentCnt = gameState.getNumAgents()
        self.bestAction = None

        # count generateSuccessor calls
        self.generateSuccessorCalls = 0

        def maxValue(gameState: GameState, depth: int, alpha: float, beta: float):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState)
            v = -float("inf")
            for action in gameState.getLegalActions(0):
                tempMin = minValue(
                    gameState.generateSuccessor(0, action), depth, 1, alpha, beta)
                self.generateSuccessorCalls += 1
                if tempMin > v:
                    v = tempMin
                    if depth == self.depth:
                        self.bestAction = action
                if v > beta:
                    return v
                alpha = max(alpha, v)
            return v

        def minValue(gameState: GameState, depth: int, agentIndex: int, alpha: float, beta: float):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            v = float("inf")
            for action in gameState.getLegalActions(agentIndex):
                if agentIndex == agentCnt - 1:
                    v = min(v, maxValue(
                        gameState.generateSuccessor(agentIndex, action), depth - 1, alpha, beta))
                    self.generateSuccessorCalls += 1
                else:
                    v = min(v, minValue(
                        gameState.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta))
                    self.generateSuccessorCalls += 1
                if v < alpha:
                    return v
                beta = min(beta, v)
            return v

        score = maxValue(gameState, self.depth, -float("inf"), float("inf"))
        # print("Best Action: ", self.bestAction)
        # print("Generate Successor Calls: ", self.generateSuccessorCalls)
        return self.bestAction


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
