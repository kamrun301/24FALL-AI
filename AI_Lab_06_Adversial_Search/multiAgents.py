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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]
    def evaluationFunction(self, currentGameState, action):
        """
        Enhanced evaluation function considering food, ghosts, power pellets,
        and efficiency of Pacman's movements.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        powerPellets = successorGameState.getCapsules()

        foodList = newFood.asList()
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        # Food distance score: closer food is better
        if foodList:
           closestFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
        else:
           closestFoodDist = 1

    # Power pellet proximity: getting closer is better
        powerPelletDist = min([manhattanDistance(newPos, pellet) for pellet in powerPellets], default=float('inf'))

    # Ghost proximity: Avoid ghosts unless they're scared
        ghostScore = 0
        for i, dist in enumerate(ghostDistances):
         if newScaredTimes[i] > 0:  # Ghost is scared
            ghostScore += 200 / (dist + 1)  # Reward for eating scared ghosts
         else:  # Active ghost
            if dist < 2:  # Punish if ghosts are too close
                ghostScore -= 1000 / (dist + 1)

    # Total score
         totalScore = successorGameState.getScore()

    # Weights to fine-tune the importance of food, ghosts, and power pellets
        foodWeight = 10
        ghostWeight = 1
        powerPelletWeight = 50

    # Consider the closest food, power pellet, and ghosts
        return (totalScore + foodWeight / (closestFoodDist + 1) + ghostWeight * ghostScore + (powerPelletWeight / (powerPelletDist + 1) if powerPelletDist != float('inf') else 0))


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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
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
        """
        action, score = self.minimax(0, 0, gameState)  # Get the action and score for pacman (agent_index=0)
        return action  # Return the action to be done as per minimax algorithm

    def minimax(self, curr_depth, agent_index, gameState):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        best_score, best_action = None, None
        if agent_index == 0:  # Max player's turn (Pacman)
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
        else:  # Min player's turn (Ghosts)
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.minimax(curr_depth, agent_index + 1, next_game_state)
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action

        return best_action, best_score

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        action, _ = self.alphabeta(0, 0, gameState, float('-inf'), float('inf'))
        return action

    def alphabeta(self, curr_depth, agent_index, gameState, alpha, beta):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        best_score, best_action = None, None
        if agent_index == 0:  # Max player's turn (Pacman)
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.alphabeta(curr_depth, agent_index + 1, next_game_state, alpha, beta)
                if best_score is None or score > best_score:
                    best_score = score
                    best_action = action
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break
        else:  # Min player's turn (Ghosts)
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.alphabeta(curr_depth, agent_index + 1, next_game_state, alpha, beta)
                if best_score is None or score < best_score:
                    best_score = score
                    best_action = action
                beta = min(beta, best_score)
                if beta <= alpha:
                    break

        return best_action, best_score

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        action, _ = self.expectimax(0, 0, gameState)
        return action

    def expectimax(self, curr_depth, agent_index, gameState):
        if agent_index >= gameState.getNumAgents():
            agent_index = 0
            curr_depth += 1

        if curr_depth == self.depth or gameState.isWin() or gameState.isLose():
            return None, self.evaluationFunction(gameState)

        if agent_index == 0:  # Max player's turn (Pacman)
            best_score, best_action = float('-inf'), None
            for action in gameState.getLegalActions(agent_index):
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.expectimax(curr_depth, agent_index + 1, next_game_state)
                if score > best_score:
                    best_score = score
                    best_action = action
            return best_action, best_score
        else:  # Ghost's turn (Chance node)
            actions = gameState.getLegalActions(agent_index)
            total_score = 0
            for action in actions:
                next_game_state = gameState.generateSuccessor(agent_index, action)
                _, score = self.expectimax(curr_depth, agent_index + 1, next_game_state)
                total_score += score
            return None, total_score / len(actions)

    def evaluationFunction(self, currentGameState, action):
        """
        Enhanced evaluation function considering food, ghosts, power pellets,
        and efficiency of Pacman's movements.
        """
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        powerPellets = successorGameState.getCapsules()

        foodList = newFood.asList()
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        # Food distance score: closer food is better
        if foodList:
           closestFoodDist = min(manhattanDistance(newPos, food) for food in foodList)
        else:
           closestFoodDist = 1

    # Power pellet proximity: getting closer is better
        powerPelletDist = min([manhattanDistance(newPos, pellet) for pellet in powerPellets], default=float('inf'))

    # Ghost proximity: Avoid ghosts unless they're scared
        ghostScore = 0
        for i, dist in enumerate(ghostDistances):
         if newScaredTimes[i] > 0:  # Ghost is scared
            ghostScore += 200 / (dist + 1)  # Reward for eating scared ghosts
         else:  # Active ghost
            if dist < 2:  # Punish if ghosts are too close
                ghostScore -= 1000 / (dist + 1)

    # Total score
        totalScore = successorGameState.getScore()

    # Weights to fine-tune the importance of food, ghosts, and power pellets
        foodWeight = 10
        ghostWeight = 1
        powerPelletWeight = 50

    # Consider the closest food, power pellet, and ghosts
        return (totalScore + foodWeight / (closestFoodDist + 1) + ghostWeight * ghostScore + (powerPelletWeight / (powerPelletDist + 1) if powerPelletDist != float('inf') else 0))



# Abbreviation
    better = evaluationFunction
  
