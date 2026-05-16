import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
#font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0,0,0)
GREEN = (0,255,0)

BLOCK_SIZE = 20
SPEED = 50

class Environment:

    def __init__(self, w=26*20, h=37*20):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Simulateur')
        self.clock = pygame.time.Clock()
        self.cpt = 0
        self.reset()


    def reset(self):
        # init game state
        self.direction = Direction.UP

        self.head = Point(4*BLOCK_SIZE, ((self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE//2)
        self.snake = [self.head,
                      Point(self.head.x-BLOCK_SIZE, self.head.y)]

        self.score = 0
        self.circuit = None
        self.foods = 0
        self._create_circuit()
        self.frame_iteration = 0

    def _create_circuit(self):
        self.circuit = []
        self.foods = 0
        with open("circuit.csv", "r") as f:
            file = f.readlines()
        for i in range(len(file)):
            line = file[i].strip().split(",")
            self.circuit.append([])
            for j in range(len(line)):
                self.circuit[i].append(int(line[j]))
                if int(line[j]) == 2:
                    self.foods += 1


    def play_step(self, action):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. move
        self._move(action) # update the head
        self.snake.insert(0, self.head)
        
        # 3. check if game over
        reward = -10
        game_over = False
        if self.is_collision() or self.frame_iteration > 50:
            game_over = True
            reward = -100
            return reward, game_over, self.score

        # 4. place new food or just move
        x = int(self.head.x/BLOCK_SIZE)
        y = int(self.head.y/BLOCK_SIZE)
        if self.circuit[y][x] == 2:
            self.score += 1
            reward = 100
            self.circuit[y][x] = 0
            self.foods -= 1
            self.cpt = 0
        elif self.cpt >= 2:
            self.frame_iteration += 1
            self.cpt = 0
        else:
            self.cpt += 1
        self.snake.pop()
        
        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        if self.foods == 0:
            self._create_circuit()

        # 6. return game over and score
        return reward, game_over, self.score


    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself
        if pt in self.snake[1:]:
            return True
        #hits circuit
        x = int(pt.x/BLOCK_SIZE)
        y = int(pt.y/BLOCK_SIZE)
        if self.circuit[y][x] == 1:
            return True

        return False


    def _update_ui(self):
        self.display.fill(BLACK)

        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            # pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        # Draw circuit
        for y in range(len(self.circuit)):
            for x in range(len(self.circuit[0])):
                if self.circuit[y][x] == 1:
                    pygame.draw.rect(self.display, WHITE, pygame.Rect(x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
                if self.circuit[y][x] == 2:
                    pygame.draw.rect(self.display, RED, pygame.Rect(x*BLOCK_SIZE, y*BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))

        # Champ de vision
        pygame.draw.rect(self.display, GREEN, pygame.Rect(self.head.x-BLOCK_SIZE*3, self.head.y-BLOCK_SIZE*3, 7*BLOCK_SIZE, 7*BLOCK_SIZE), width = 1)

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()


    def _move(self, action):
        # [straight, right, left]

        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx] # right turn r -> d -> l -> u
        else: # [0, 0, 1]
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] # left turn r -> u -> l -> d

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)