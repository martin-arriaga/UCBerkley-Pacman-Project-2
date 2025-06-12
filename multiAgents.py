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
#from typing import final

from textDisplay import PacmanGraphics
from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

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
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        ghostpositions = [ghost.getPosition() for ghost in newGhostStates]
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        powerpellets = successorGameState.getCapsules()
        "*** YOUR CODE HERE ***"

        final_score = 0
        food_reward = 0
        ghost_penalty = 0
        powerpellet_reward = 0

        currentfood = currentGameState.getFood()
        currentfoodaslist = currentfood.asList()
        newfoodaslist = newFood.asList()

        if len(currentfoodaslist) > len(newfoodaslist):  # automatic reward for food, less food after action --> better action
            food_reward += 1000
        else:
            food_reward += 0

        if  newfoodaslist: # reward for food
            min_newfood = min(manhattanDistance(newPos, food) for food in newfoodaslist)
            food_reward += max(1000 - min_newfood * 10, 0)# dynamically adjusts reward based on the distance and then scaled by 10, instead of a flat reward
        else:
            if not currentfoodaslist: # no food left at current position
                food_reward += 0
            if not newfoodaslist: #no food left after move
                food_reward += 0
        # penalty for being close to ghosts
        scared_time = sum(newScaredTimes)
        for ghost_position in ghostpositions:
            ghost_distance = manhattanDistance(newPos, ghost_position)

            if scared_time > 0: # if ghost is scared
                if ghost_distance < 2: # and ghost is close by, eat ghost
                    ghost_penalty += 900
                elif ghost_distance < 4:# ghost is scared, and close, small penalty
                    ghost_penalty += 200
            else:# ghost is not scare
                if ghost_distance < 2: # ghost is really close, avoid ghost ---> penalize, negative because im just adding everything at the end
                    ghost_penalty -= 2200
                elif ghost_distance < 5: # ghost is moderately close, penalize less
                    ghost_penalty -= 900
        # reward for eating a power pellet or moving close to a pellet
        if powerpellets:
            min_distancepowerpellet = min([manhattanDistance(currentGameState.getPacmanPosition(), powerpellet) for powerpellet in powerpellets])
            if min_distancepowerpellet < 2:
                powerpellet_reward += 2000
            elif min_distancepowerpellet < 5:
                powerpellet_reward += 400

        final_score =( food_reward + ((ghost_penalty)) + powerpellet_reward) - currentGameState.getScore()

        return final_score

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
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
        "*** YOUR CODE HERE ***"

        """
        Algorithm Design:
        minimaxdecision: Starts with Pac-Man’s turn (maximizing), and iterates through possible actions.
        maxvalue: Handles Pac-Man's turn. It will call minvalue for all ghosts (alternating their turns).
        minvalue: Handles each ghost’s turn. It will call maxvalue for Pac-Man's turn, and after each ghost’s turn, it will alternate to the next ghost.
        
        next_ghost_idx = ghostidx + 1 if ghostidx + 1 < num_agents else 0

        """
        action = self.minimaxdecision(gameState)  # just calls the decision function and rest of the algorithm, returns the action
        return action


    def minimaxdecision(self,gameState):

        actions = gameState.getLegalActions(0)
        decision = None
        max_utilityvalue = -float("inf")

        for action in actions:
            possibleaction = gameState.generateSuccessor(0, action)
            #recursively go down the tree
            current_max = self.minvalue(possibleaction, 0, 1 )
            if current_max > max_utilityvalue:
                max_utilityvalue = current_max
                decision = action
        return decision

    def minvalue(self, gameState, depth, ghostidx ):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        ghost = ghostidx
        utility_value = float('inf')
        actions = gameState.getLegalActions(ghost)

        for action in actions:
            nextstate = gameState.generateSuccessor(ghost, action)
            if ghost + 1 < gameState.getNumAgents():# checks if we are done with all ghosts,ie total number of agents is 0 + all ghosts
                agent = ghost + 1                   # after last ghost, the ghostidx + 1  will be equal to total agents and agent will be set to 0
            else:
                agent = 0

            if agent != 0: # all ghost have not made their move
                utility_value = min(utility_value, self.minvalue(nextstate, depth , ghost + 1))
            else:
                utility_value = min(utility_value, self.maxvalue(nextstate, depth + 1 , None))

        return utility_value

    def maxvalue(self, gameState, depth, ghostidx):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        utility_value = -float('inf')
        for action in actions:
            nextstate = gameState.generateSuccessor(0, action)
            utility_value = max(utility_value, self.minvalue(nextstate, depth   ,1))
        return utility_value

        #util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        '''
        pretty much only thing modified is the condition after the recursion return and updating alpha and beta 
        '''
        action = self.alphabetaminimax(gameState)  # just calls the decision function and rest of the algorithm, returns the action
        return action

    def alphabetaminimax(self, gameState):

        actions = gameState.getLegalActions(0)
        decision = None
        max_utilityvalue = float('-inf')
        alpha = float("-inf")
        beta = float("inf")

        for action in actions:
            possibleaction = gameState.generateSuccessor(0, action)
            # recursively go down the tree
            current_max = self.alphabetaminvalue(possibleaction, 0, 1, alpha, beta)
            if current_max > max_utilityvalue:
                max_utilityvalue = current_max
                decision = action
            if current_max > beta:
                break
            alpha = max(alpha, current_max)
        return decision

    def alphabetaminvalue(self, gameState, depth, ghostidx, alpha, beta ):

        if gameState.isWin() or gameState.isLose() :
            return self.evaluationFunction(gameState)

        ghost = ghostidx
        utility_value = float('inf')
        actions = gameState.getLegalActions(ghost)

        for action in actions:
            nextstate = gameState.generateSuccessor(ghost, action)

            if ghost + 1 < gameState.getNumAgents():# checks if we are done with all ghosts,ie total number of agents is 0 + all ghosts
                agent = ghost + 1                   # after last ghost, the ghostidx + 1 will be equal to total agents and agent will be set to 0
            else:
                agent = 0

            if agent != 0: # all ghost have not made their move
                 utility_value = min(utility_value, self.alphabetaminvalue(nextstate, depth, ghost + 1, alpha, beta))
            else:
                utility_value = min(utility_value, self.alphabetamaxvalue(nextstate, depth + 1, None, alpha, beta))

            if utility_value < alpha:
                return utility_value
            beta = min(beta, utility_value)

        return utility_value

    def alphabetamaxvalue(self, gameState, depth, ghostidx, alpha, beta):

        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState)

        actions = gameState.getLegalActions(0)
        utility_value = float('-inf')

        for action in actions:
            nextstate = gameState.generateSuccessor(0, action)
            utility_value = max(utility_value, self.alphabetaminvalue(nextstate, depth, 1, alpha, beta))

            if utility_value > beta:
                return utility_value
            alpha = max(alpha, utility_value)

        return utility_value
        #util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState:GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        """
        basically the only thing that really changes is the min layer, instead of choosing the min we just average them, or expected value 
        we need to keep a running sum then divide by total number of actions, so, the expected value 
        """
        action = self.expectimaxdecision(gameState) # just calls the decision function and rest of the algorithm, returns the action
        return action
        #util.raiseNotDefined()
    def expectimaxdecision(self,gameState):

        actions = gameState.getLegalActions(0)
        decision = None
        max_utilityvalue = -float('inf')

        for action in actions:
            possible_next_state = gameState.generateSuccessor(0, action)
            current_utility_value = self.expectichance(possible_next_state, 0, 1)
            if current_utility_value > max_utilityvalue:
                max_utilityvalue = current_utility_value
                decision = action

        return decision


    def expectichance(self, gameState, depth, ghostidx):

        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        ghost = ghostidx
        actions = gameState.getLegalActions(ghostidx)
        utility_value_sum = 0

        for action in actions:
            possible_next_state = gameState.generateSuccessor(ghost, action)
            if ghost + 1 < gameState.getNumAgents():# checks if we are done with all ghosts,ie total number of agents is 0 + all ghosts
                agent = ghost + 1                   # after last ghost, the ghostidx + 1  will be equal to total agents and agent will be set to 0
            else:
                agent = 0

            if agent != 0:# all ghost have not made their move
                utility_value = self.expectichance(possible_next_state, depth, ghost + 1)
            else:
                utility_value= self.expectimax(possible_next_state, depth + 1, None)
            utility_value_sum += utility_value

        return utility_value_sum / len(actions)

    def expectimax(self, gameState, depth, ghostidx):
        if gameState.isWin() or gameState.isLose() or depth == self.depth :
            return self.evaluationFunction(gameState)
        actions = gameState.getLegalActions(0)
        utility_value = float('-inf')
        for action in actions:
            possible_next_state = gameState.generateSuccessor(0, action)
            utility_value = max(utility_value, self.expectichance(possible_next_state, depth  , 1))

        return utility_value

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
     5 paramaters:
        food proximity
            give higher evaluation with states that have closer food
        ghost proximity
            penalize states where ghost are close to pacman
        power pellets
            pacman should eat the pellets, for states with closer pellets -> higher reward
        current score
            factor in current score of the game
        number of remaining food pellets
            the less remaining pellets the better, reward states with less pellets

        final design:
                    variables used
                    Hyperparameters, easily tune my eval function
                    find min distance for ghosts, food and powerpellets
                    stuck penalty
                        pacman seemed to get stuck inside the cubicle were the power pellets are
                        penalize heavily if that happens

                    ghost penalty
                        if ghosts are scared and depending on food distance
                            penalty is less, we want pacman to eat the food
                        else: ghost are not scared
                            depending on the distance of ghost to pacman
                            penalty is more severe for smaller distances
                            i.e ghost is close so stay away from that state
                    food reward
                        states with less food on the board are rewarded better
                    powerpellet reward
                        states that bring pacman close to eating power pellets are rewarded better
                        else reward is less
                    final eval
                        add reciprocal of penalties and rewards and scale by scalars

    """
    "*** YOUR CODE HERE ***"
    # variables to evaluate the state
    foodaslist = currentGameState.getFood().asList()
    pacman_position = currentGameState.getPacmanPosition()
    ghoststate = currentGameState.getGhostStates()
    ghostpositions = [ghost.getPosition() for ghost in ghoststate]
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghoststate]
    powerpellets = currentGameState.getCapsules()
    final_evaluation = 0

    #constants to easily change the formula to how we scale our rewards or penalties
    CLOSE_GHOST_DISTANCE = 3 #treshold to determine if ghost distance is "close" 2,4,7
    MED_GHOST_DISTANCE = 4 #treshold to determine if ghost distance is "medium"
    FAR_GHOST_DISTANCE = 10 #ghost distance is far

    CLOSE_POWERPELLET_DISTANCE = 2 # threshold for distance of closest powerpellet
    POWERPELLET_REWARD = 100 # constant to determine how much reward to give if powerpellet is close

    CLOSE_FOOD_DISTANCE = 1 #thresholds to determine if food distance is close, med, or far
    MED_FOOD_DISTANCE = 3
    FAR_FOOD_DISTANCE = 7
    # Penalties for -> if ghost is scared and food is close
    PENALTY_FORCLOSE_FOOD_DISTANCE = 20
    PENALTY_FORMEDFOOD_DISTANCE = 40
    PENALTY_FORGHOSTFAR_DISTANCE = 600
    #Penalties for -> if ghost is NOT scared and it is close to pacman
    PENALTY_IFGHOSTDISTANCE_CLOSE = 500
    PENALTY_IFGHOSTDISTANCE_MED = 300
    PENALTY_IFGHOSTDISTANCE_FAR = 100
    #reward for total food left on the board, i.e states with less food are rewarded more
    REWARD_FORTOTAL_FOOD= 200
    #scalars for final evaluation
    SCALEFOOD_RWARD_BY = 10
    SCALEPOWERPELLET_RWARD_BY = 10
    SCALEGHOSTPENALTY_BY = 100
    SCALESTUCK_PENALTY_BY = 300


    # list comprehension, finds min distance for food, ghosts and power pellets
    # food
    minfooddistance = min([manhattanDistance(pacman_position, food) for food in foodaslist], default= 0)
    #ghosts
    minghostdistance = min([manhattanDistance(pacman_position, ghost) for ghost in ghostpositions], default= 0)
    # powerpellets
    minpowerpelletdistance = min([manhattanDistance(pacman_position, powerpellet) for powerpellet in powerpellets], default= 0)

    # stuck in wall penalty
    actions = currentGameState.getLegalActions(0)
    stuck_penalty = 0
    if len(actions) < 2:  # checks if pacmans legal moves are 1, i.e he is stuck inside the cubicle where the power pellets are
        stuck_penalty = 1000  # DO NOT GET STUCK IN CUBICLE PACMAN!!!!!

    # start evaluation, Penalize if ghosts are too close, if ghosts are scared and distance is less than 2 eat ghosts
    current_score = currentGameState.getScore()
    penalty_forghosts = 0
    scaredghosts_time = sum(newScaredTimes)
    if scaredghosts_time > 0: #if food is close and ghost are scared, penalize less, i.e eat food
        if minfooddistance < CLOSE_FOOD_DISTANCE:
            penalty_forghosts += PENALTY_FORCLOSE_FOOD_DISTANCE
        elif minfooddistance >  MED_FOOD_DISTANCE:
            penalty_forghosts += PENALTY_FORMEDFOOD_DISTANCE
        elif minfooddistance > FAR_FOOD_DISTANCE:
            penalty_forghosts += PENALTY_FORGHOSTFAR_DISTANCE
    else:
        if minghostdistance < CLOSE_GHOST_DISTANCE: # if ghost are close and ghost are not scared, penalize i.e do not eat food
            penalty_forghosts += PENALTY_IFGHOSTDISTANCE_CLOSE
        elif minghostdistance< MED_GHOST_DISTANCE:
            penalty_forghosts += PENALTY_IFGHOSTDISTANCE_MED
        elif minghostdistance < FAR_GHOST_DISTANCE:
            penalty_forghosts += PENALTY_IFGHOSTDISTANCE_FAR
    #reward for state with less food
    reward_forfood = 0
    total_food = len(foodaslist)
    reward_forfood= REWARD_FORTOTAL_FOOD * (1 / (total_food+ 1))# total_food is large ---> fraction is smaller, total_food is small--> fraction is bigger , then scale up

    #reward for eating power pellets
    powerpellet_reward = 0
    totalpowerpellets = len(powerpellets)
    if minpowerpelletdistance < CLOSE_POWERPELLET_DISTANCE:
        powerpellet_reward = POWERPELLET_REWARD / (totalpowerpellets + 1) #if pacman is close---> reward that state i.e eat the pellet
    else:
        powerpellet_reward -= POWERPELLET_REWARD / (total_food + 1) # if im not close, reward is less

    final_evaluation = (
                        (
                         ((1/ (reward_forfood + 1)) * SCALEFOOD_RWARD_BY) +
                         ((1/ (powerpellet_reward + 1)) * SCALEPOWERPELLET_RWARD_BY) -
                         ((1/ (penalty_forghosts + 1)) * SCALEGHOSTPENALTY_BY) -
                         ((1/ (stuck_penalty + 1)) * SCALESTUCK_PENALTY_BY)) +
                         current_score)

    return final_evaluation

    #util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
