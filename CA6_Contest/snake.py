from cube import Cube
from constants import *
from utility import *

import random
import random
import numpy as np
SNAKE_1_Q_TABLE = "s1_qtble.npy"
SNAKE_2_Q_TABLE = "s2_qtble.npy"

WIDTH = 500
HEIGHT = 500

ROWS = 20

LEARNING_RATE = 0.4
DISCOUNT_FACTOR = 0.9
EPSILON = 0.5
class Snake:
    body = []
    turns = {}

    def __init__(self, color, pos, file_name=None):
        self.SELF_COL = 0 
        self.OTHER_COL = 0
        self.BOARD_COL = 0 
        self.EATEN_SNACKS = 0
        self.BEING_EATEN = 0
        self.KILL = 0 
        self.color = color
        self.head = Cube(pos, color=color)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.current_state = [0,0,0,0,0,0]
        print("file name:",file_name)
        try:
            self.q_table = np.load(file_name)
        except:
            self.q_table = np.zeros((3,3,4,4,4,4,4))

        self.lr = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        self.epsilon = EPSILON      
        self.epsilon_decay = 0.995  
        self.epsilon_min = 0.01

    def get_optimal_policy(self, state):
        return np.argmax(self.q_table[tuple(state)])
    def check_hit(self,x,y,other_snake):
        if (x, y) in list(map(lambda z: z.pos, self.body)):
            return 0
        elif (x, y) in list(map(lambda z: z.pos, other_snake.body)):
            return 1 
        elif x==0 or x== ROWS - 1 or y==0 or y == ROWS-1: 
            return 2 
        return 3 
        
    def check_move(self,direction,other_snake):
        if direction == 0 :
            x = self.head.pos[0] - 1
            y = self.head.pos[1]
            
        elif direction == 1:
            x = self.head.pos[0] + 1
            y = self.head.pos[1]
            
        elif direction == 2:
            x = self.head.pos[0]
            y = self.head.pos[1] - 1
            
        elif direction == 3:
            x = self.head.pos[0]
            y = self.head.pos[1] + 1
        return self.check_hit(x,y,other_snake) 
            
    def make_action(self, state):
        
        chance = random.random()
        if chance < self.epsilon:
            action = random.randint(0, 3)
        else:
            action = self.get_optimal_policy(state)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)    
        
        return action

    def update_q_table(self, state, action, next_state, reward):
        sample = reward + self.discount_factor*np.max(self.q_table[tuple(next_state)])
        self.q_table[tuple(state)][action] = (1-self.lr)*self.q_table[tuple(state)][action] + self.lr*sample
        pass

    def move(self, snack, other_snake):

        state = self.current_state
        action = self.make_action(state)

        if action == 0: # Left
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 1: # Right
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 2: # Up
            self.dirny = -1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
        elif action == 3: # Down
            self.dirny = 1
            self.dirnx = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]

        for i, c in enumerate(self.body):
            p = c.pos[:]
            if p in self.turns:
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)
                          
        new_state = [
        (snack.pos[0] - self.head.pos[0]) // max(1, abs(snack.pos[0] - self.head.pos[0])) + 1 ,
        (snack.pos[1] - self.head.pos[1]) // max(1, abs(snack.pos[1] - self.head.pos[1])) + 1, 
        self.check_move(0,other_snake),
        self.check_move(1,other_snake),
        self.check_move(2,other_snake),
        self.check_move(3,other_snake)
        ] 
        self.current_state = new_state
        return state , new_state,action
    
    def check_out_of_board(self):
        headPos = self.head.pos
        if headPos[0] >= ROWS - 1 or headPos[0] < 1 or headPos[1] >= ROWS - 1 or headPos[1] < 1:
            self.reset((random.randint(3, 18), random.randint(3, 18)))
            return True
        return False 
    def print_info(self):
        print(self.color)
        print("self.SELF_COL:",self.SELF_COL)
        print("self.OTHER_COL",self.OTHER_COL)
        print("self.BOARD_COL",self.BOARD_COL)
        print("self.EATEN_SNACKS",self.EATEN_SNACKS)
        print("self.BEING_EATEN",self.BEING_EATEN)
        print("self.KILL",self.KILL)
        print("--------------------------------------------------")
    def calc_reward(self, snack, other_snake):
        reward = 0
        win_self, win_other = False, False

        if self.check_out_of_board():
            self.BOARD_COL += 1 
            reward -= 750  
            win_other = True
            self.reset((random.randint(3, 18), random.randint(3, 18))) 

        if self.head.pos == snack.pos:
            self.EATEN_SNACKS += 1
            self.addCube()
            snack = Cube(randomSnack(ROWS, self), color=(0, 255, 0))
            reward += 400  

        if self.head.pos in list(map(lambda z: z.pos, self.body[1:])):
            self.SELF_COL += 1
            reward -= 450  
            win_other = True
            self.reset((random.randint(3, 18), random.randint(3, 18))) 

        if self.head.pos in list(map(lambda z: z.pos, other_snake.body)):
            if self.head.pos != other_snake.head.pos:
                self.OTHER_COL += 1
                reward -= 450  
                win_other = True
            else:
                if len(self.body) > len(other_snake.body):
                    self.KILL += 1 
                    reward += 450  
                    win_self = True
                elif len(self.body) == len(other_snake.body):
                    pass  # No winner, no reward
                else:
                    self.BEING_EATEN += 1 
                    reward -= 300  
                    win_other = True
            if win_other:
                self.reset((random.randint(3, 18), random.randint(3, 18)))  
            elif win_self:
                other_snake.reset((random.randint(3, 18), random.randint(3, 18))) 

         
        dist_to_snack_before = abs(self.head.pos[0] - snack.pos[0]) + abs(self.head.pos[1] - snack.pos[1])
        
        simulated_head_pos = (self.head.pos[0] + self.dirnx, self.head.pos[1] + self.dirny)
        dist_to_snack_after = abs(simulated_head_pos[0] - snack.pos[0]) + abs(simulated_head_pos[1] - snack.pos[1])

        if dist_to_snack_after < dist_to_snack_before:
            reward += 200 
        else :
            reward -= 200 
        return snack, reward, win_self, win_other
    
    def reset(self, pos):
        self.head = Cube(pos, color=self.color)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(Cube((tail.pos[0] - 1, tail.pos[1]), color=self.color))
        elif dx == -1 and dy == 0:
            self.body.append(Cube((tail.pos[0] + 1, tail.pos[1]), color=self.color))
        elif dx == 0 and dy == 1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] - 1), color=self.color))
        elif dx == 0 and dy == -1:
            self.body.append(Cube((tail.pos[0], tail.pos[1] + 1), color=self.color))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)

    def save_q_table(self, file_name):
        np.save(file_name, self.q_table)
    
        