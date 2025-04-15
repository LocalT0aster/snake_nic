from collections import deque
from copy import deepcopy
import random
import numpy as np
import math
import operator
from enum import Enum

# Disable the welcome message during import.
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame

class Direction(Enum):
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

class SnakeGameAI:
    """
    Snake game environment modified for AI control.
    
    The game uses a grid of cells (field_size) with each cell of size block_size.
    The AI agent receives a 21x21x3 vision matrix (walls, snake body, food) centered 
    on the snake's head along with normalized food distance (dx, dy) and normalized snake length.
    """
    def __init__(self, field_size=(30, 30), block_size=20, snake_speed=15, render=False):
        # Game configuration
        self.field_size = field_size
        self.block_size = block_size
        self.snake_speed = snake_speed
        self.max_steps_without_food = field_size[0] * field_size[1] // 2
        self.steps_without_food = 0
        self.window_width = self.field_size[0] * self.block_size
        self.window_height = self.field_size[1] * self.block_size

        # Colors for drawing (if rendering is enabled)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.green = (0, 255, 0)
        self.blue = (50, 153, 213)

        self.render = render
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Snake Game AI")
        self.clock = pygame.time.Clock()

        # Initialize the game state
        self.reset()

    def reset(self):
        """Reset the game to the starting state."""
        self.direction = Direction.RIGHT
        self.head = (self.field_size[0] // 2, self.field_size[1] // 2)
        self.snake_deque: deque[tuple[int, int]] = deque(maxlen=self.field_size[0] * self.field_size[1])
        self.snake_deque.append(self.head)
        self.snake_deque.append((self.head[0] - 1, self.head[1]))
        self.snake_length = 2
        self.score = 0
        self.frame_iteration = 0
        self._spawn_food()

    def _spawn_food(self):
        """Randomly place food on the grid."""
        if self.snake_length >= self.field_size[0] * self.field_size[1]:
            raise ValueError("Food does not fit inside the field")
            return
        
        while True:
            self.food = (random.randrange(0, self.field_size[0]), random.randrange(0, self.field_size[1]))
            if self.food not in self.snake_deque:
                return



    def _is_collision(self, point=None):
        """Check if the given point (or the snake's head) collides with boundaries or its own body."""
        if point is None:
            point = self.head
        # Check boundaries
        if not in_rect(point, (0, 0), self.field_size):
            return True
        # Check self-collision
        if self.snake_length > 4 and point in self.snake_deque[4:]:
            return True
        return False

    def _move(self, action):
        """
        Update the snake's direction and head position based on the action.
        The action is an integer (0: LEFT, 1: RIGHT, 2: UP, 3: DOWN).
        Prevents reversing direction.
        """
        desired_direction = None
        if action == 0:
            desired_direction = Direction.LEFT
        elif action == 1:
            desired_direction = Direction.RIGHT
        elif action == 2:
            desired_direction = Direction.UP
        elif action == 3:
            desired_direction = Direction.DOWN

        # Prevent reversing: if current direction is opposite, ignore new action.
        if self.direction == Direction.LEFT and desired_direction == Direction.RIGHT:
            desired_direction = self.direction
        elif self.direction == Direction.RIGHT and desired_direction == Direction.LEFT:
            desired_direction = self.direction
        elif self.direction == Direction.UP and desired_direction == Direction.DOWN:
            desired_direction = self.direction
        elif self.direction == Direction.DOWN and desired_direction == Direction.UP:
            desired_direction = self.direction
        
        self.direction = desired_direction

        # Update head position based on current direction.
        if self.direction == Direction.LEFT:
            self.head = sum(self.head, (-1, 0))
        elif self.direction == Direction.RIGHT:
            self.head = sum(self.head, (1, 0))
        elif self.direction == Direction.UP:
            self.head = sum(self.head, (0, -1))
        elif self.direction == Direction.DOWN:
            self.head = sum(self.head, (0, 1))

    def _update_snake(self):
        """Add new head to the snake and remove the tail if not growing."""
        self.snake_deque.appendleft(deepcopy(self.head))
        if len(self.snake_deque) > self.snake_length:
            self.snake_deque.pop()

    def _draw_elements(self):
        """Render the game graphics."""
        self.display.fill(self.blue)
        pygame.draw.rect(self.display, self.green, [self.food[0] * self.block_size, self.food[1] * self.block_size, self.block_size, self.block_size])
        for segment in self.snake_deque:
            pygame.draw.rect(self.display, self.black, [segment[0] * self.block_size, segment[1] * self.block_size, self.block_size, self.block_size])
        pygame.display.flip()

    def get_state(self):
        """
        Construct the input representation for the neural network.
        """
        vision_size = 21
        half = vision_size // 2
        vision = np.zeros((vision_size, vision_size, 3), dtype=np.float16)
        
        # Top-left position of the vision grid in pixel coordinates
        start = diff(self.head, (vision_size, vision_size))

        # Populate the vision matrix cell by cell.
        for cell_x in range(0, vision_size):
            for cell_y in range(0, vision_size):
                cell = (cell_x, cell_y)
                # Mark walls if out of bounds.
                if not in_rect(sum(cell, start), (0, 0), self.field_size):
                    vision[*cell, 0] = 1.0  # Wall
                else:
                    # Mark food presence.
                    if self.food == cell:
                        vision[*cell, 2] = 1.0
        
        # Mark snake body presence.
        for segment in self.snake_deque:
            if in_rect(segment, start, sum(start, (vision_size, vision_size))):
                vision[*diff(segment, start), 1] = 1.0
        
        # Flatten the vision matrix.
        # vision_flat = vision.flatten()  # 21*21*3 = 1323

        # Compute normalized food distance vector.
        # max_distance = math.sqrt(self.width ** 2 + self.height ** 2)
        dx = (self.food[0] - self.head[0]) / self.field_size[0] #max_distance
        dy = (self.food[1] - self.head[1]) / self.field_size[1] #max_distance

        # Normalized snake length.
        normalized_length = self.snake_length / (self.field_size[0] * self.field_size[1])

        return vision, np.array([dx, dy, normalized_length], dtype=np.float16)

    def play_step(self, action):
        """
        Execute one time step of the game:
          - Move the snake given an action.
          - Update the snake's body and check for food consumption.
          - Calculate reward.
        
        Parameters:
            action (int): An integer in {0, 1, 2, 3} representing the movement direction.
            
        Returns:
            reward (float): Reward for the current step.
            game_over (bool): Flag indicating if the game ended (collision).
            score (int): The current score (food eaten).
        """
        self.frame_iteration += 1

        self._move(action)
        self._update_snake()
        self.steps_without_food += 1

        reward = 0.0
        
        game_over = False
        if self._is_collision():
            game_over = True
            reward -= 100.0  # Death penalty
            return reward, game_over, self.score
        
        # Check if food is eaten.
        if self.head == self.food:
            self.snake_length += 1
            self.score += 1
            reward += 150.0  # Reward for eating food
            self.steps_without_food = 0
            self._spawn_food()

        if self.steps_without_food > self.max_steps_without_food:
            game_over = True
            reward -= 100.0  # Death penalty
            return reward, game_over, self.score
        
        # Reward for survival (small incentive for each step taken)
        reward += 0.01

        if self.render:
            self._draw_elements()
            self.clock.tick(self.snake_speed)
        
        return reward, game_over, self.score
    

def in_rect(point: tuple, start: tuple, end: tuple) -> bool:
    return point[0] >= start[0] and point[0] < end[0] and point[1] >= start[1] and point[1] < end[1]

def sum(a: tuple, b:tuple):
    return tuple(map(operator.add, a, b))

def diff(a: tuple, b:tuple):
    return tuple(map(operator.sub, a, b))

if __name__ == '__main__':
    # Quick test: run a manual game with random actions.
    game = SnakeGameAI(render=True)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        action = random.randint(0, 3)
        reward, game_over, score = game.play_step(action)
        if game_over:
            print("Score:", score)
            game.reset()
