from Agents import Agent
import util
import random
import math
import numpy as np


class ReflexAgent(Agent):
    
    def __init__(self, *args, **kwargs) -> None:
        self.index = 0 # your agent always has index 0

    def getAction(self, gameState):
        
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions(self.index)

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
       
        nextGameState = currentGameState.generateSuccessor(self.index, action)
        return nextGameState.getScore(self.index) - currentGameState.getScore(self.index)


def scoreEvaluationFunction(currentGameState):
    
    return currentGameState.getScore(0)


class MultiAgentSearchAgent(Agent):
    

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2', **kwargs):
        self.index = 0 # your agent always has index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, state):
        
        return self.minimax(self.index, state, 0)[0]
        
    def minimax(self, index, state, depth):
        # Collect legal moves and successor states
        legalMoves = state.getLegalActions(index)
        if self.depth == depth or len(legalMoves) == 0 :
            return (None, self.evaluationFunction(state))

        # Choose one of the best actions
        scores = [self.minimax((index + 1) % state.getNumAgents(),
         state.generateSuccessor(index, action), depth + 1)[1] for action in legalMoves]
        if index == 0:
            bestScore = max(scores)
            bestIndices = [idx for idx in range(len(scores)) if scores[idx] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return (legalMoves[chosenIndex], bestScore)
        else:
            bestScore = min(scores)
            bestIndices = [idx for idx in range(len(scores)) if scores[idx] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return (legalMoves[chosenIndex], bestScore)




        


class AlphaBetaAgent(MultiAgentSearchAgent):
    

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        
        
        return self.value(self.index, gameState, 0, -math.inf, math.inf)[0]

    def max_value(self, index, state, depth, alpha, beta):
        # Collect legal moves and successor states
        legalMoves = state.getLegalActions(index)
        if self.depth == depth or len(legalMoves) == 0 :
            return (None, self.evaluationFunction(state))
        bestActin = legalMoves[0]
        v = -math.inf
        for action in legalMoves:
            s = self.value((index + 1) % state.getNumAgents(), state.generateSuccessor(index, action), 
            depth + 1, alpha, beta)[1]
            if v < s:
                bestActin = action
            v = max(v, s)
            if v >= beta:
                return (action, v)
            alpha = max(alpha, v)

        return (bestActin, v)
        
        
        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # return (legalMoves[chosenIndex], bestScore)
        
    def min_value(self, index, state, depth, alpha, beta):
        # Collect legal moves and successor states
        legalMoves = state.getLegalActions(index)
        if self.depth == depth or len(legalMoves) == 0 :
            return (None, self.evaluationFunction(state))

        bestActin = legalMoves[0]
        v = math.inf
        for action in legalMoves:
            s = self.value((index + 1) % state.getNumAgents(), state.generateSuccessor(index, action), 
            depth + 1, alpha, beta)[1]
            if v > s:
                bestActin = action
            v = min(v, s)
            if v <= alpha:
                return (action, v)
            beta = min(beta, v)
        return (bestActin, v)

        # bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        # chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        # return (legalMoves[chosenIndex], bestScore)
    def value(self, index, state, depth, alpha, beta):
        if index == 0:
            return self.max_value(index, state, depth, alpha, beta)
        else:
            return self.min_value(index, state, depth, alpha, beta)
    
    

class ExpectimaxAgent(MultiAgentSearchAgent):
  

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(**kwargs)

    def getAction(self, gameState):
        
        return self.expectimax(self.index, gameState, 0)[0]
    
    def expectimax(self, index, state, depth):
        legalMoves = state.getLegalActions(index)
        
        # Collect legal moves and successor states
        if self.depth == depth or len(legalMoves) == 0 :
            return (None, self.evaluationFunction(state))

        # Choose one of the best actions
        scores = [self.expectimax((index + 1) % state.getNumAgents(), 
        state.generateSuccessor(index, action), depth + 1)[1] for action in legalMoves]
        if index == 0:
            bestScore = max(scores)
            bestIndices = [idx for idx in range(len(scores)) if scores[idx] == bestScore]
            chosenIndex = random.choice(bestIndices) # Pick randomly among the best
            return (legalMoves[chosenIndex], bestScore)
        else:
            return (None, sum(scores)/len(scores))
    


def betterEvaluationFunction(currentGameState):
    

    # parity
    def parity():
        x = currentGameState.getScore(0)
        y = sum(currentGameState.getScore())
        return 100 * (2 * x - y)/y

    # corners
    def corners():
        n_corners = 0
        sum = 0
        for i in currentGameState.getCorners():
            if i == 0:
                n_corners += 1
                sum += 1
            elif i != -1:
                n_corners -= 1
                sum += 1
        if sum == 0:
            return 0
        else:
            return 100 * n_corners/sum

    # mobility
    def mobility():
        x = len(currentGameState.getLegalActions(0))
        y = 0
        for i in range(currentGameState.getNumAgents()):
            y += len(currentGameState.getLegalActions(i))
        if (y) == 0:
            return 0
        else:
            return 100 * (2 * x - y)/(y)

    # stability
    def stability():
        l = [0, 0]
        weights = [[4,  -3,  2,  2,  2,  2, -3,  4],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [2,  -1,  1,  0,  0,  1, -1,  2],
        [2,  -1,  0,  1,  1,  0, -1,  2],
        [2,  -1,  0,  1,  1,  0, -1,  2],
        [2,  -1,  1,  0,  0,  1, -1,  2],
        [-3, -4, -1, -1, -1, -1, -4, -3],
        [4,  -3,  2,  2,  2,  2, -3,  4]]
        for i in range(8):
            for j in range(8):
                index = currentGameState.data.board[i][j]
                if index == 0:
                    l[0] += weights[i][j]
                elif index != -1:
                    l[1] += weights[i][j]
        x = l[0]
        y = sum(l)
        if y == 0:
            return 0
        else:
            return 100 * (2 * x - y)/y

    
    


    return (2) * parity() + (20) * corners() + (10) * mobility() + (10) * stability()

# Abbreviation
better = betterEvaluationFunction