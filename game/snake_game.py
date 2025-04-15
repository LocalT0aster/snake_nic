import pygame
import random
import numpy as np
import math
from enum import Enum

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
        self.field_width, self.field_height = field_size  # grid dimensions
        self.block_size = block_size
        self.snake_speed = snake_speed
        self.width = self.field_width * self.block_size
        self.height = self.field_height * self.block_size

        # Colors for drawing (if rendering is enabled)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.green = (0, 255, 0)
        self.blue = (50, 153, 213)

        self.render = render
        if self.render:
            pygame.init()
            self.display = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Snake Game AI")
        self.clock = pygame.time.Clock()

        # Initialize the game state
        self.reset()

    def reset(self):
        """Reset the game to the starting state."""
        self.direction = Direction.RIGHT
        self.head_x = self.width // 2
        self.head_y = self.height // 2
        self.snake_list = [[self.head_x, self.head_y]]  # snake starts with only the head
        self.snake_length = 1
        self.score = 0
        self.frame_iteration = 0
        self._spawn_food()

    def _spawn_food(self):
        """Randomly place food on the grid."""
        self.food_x = random.randrange(0, self.field_width) * self.block_size
        self.food_y = random.randrange(0, self.field_height) * self.block_size

    def _is_collision(self, point=None):
        """Check if the given point (or the snake's head) collides with boundaries or its own body."""
        if point is None:
            point = [self.head_x, self.head_y]
        # Check boundaries
        if point[0] < 0 or point[0] >= self.width or point[1] < 0 or point[1] >= self.height:
            return True
        # Check self-collision
        if point in self.snake_list[1:]:
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
            self.head_x -= self.block_size
        elif self.direction == Direction.RIGHT:
            self.head_x += self.block_size
        elif self.direction == Direction.UP:
            self.head_y -= self.block_size
        elif self.direction == Direction.DOWN:
            self.head_y += self.block_size

    def _update_snake(self):
        """Add new head to the snake and remove the tail if not growing."""
        self.snake_list.append([self.head_x, self.head_y])
        if len(self.snake_list) > self.snake_length:
            self.snake_list.pop(0)

    def _draw_elements(self):
        """Render the game graphics."""
        self.display.fill(self.blue)
        pygame.draw.rect(self.display, self.green, [self.food_x, self.food_y, self.block_size, self.block_size])
        for segment in self.snake_list:
            pygame.draw.rect(self.display, self.black, [segment[0], segment[1], self.block_size, self.block_size])
        pygame.display.flip()

    def get_state(self):
        """
        Construct the input representation for the neural network.
        
        Returns:
            state (np.ndarray): A flattened vector of size 1326, comprising:
                - A 21x21x3 vision matrix centered on the snake's head.
                - A normalized (dx, dy) vector to the food.
                - The normalized snake length.
        """
        vision_size = 21
        half = vision_size // 2
        vision = np.zeros((vision_size, vision_size, 3), dtype=np.float32)
        
        # Top-left position of the vision grid in pixel coordinates
        start_x = self.head_x - half * self.block_size
        start_y = self.head_y - half * self.block_size

        # Populate the vision matrix cell by cell.
        for i in range(vision_size):
            for j in range(vision_size):
                cell_x = start_x + i * self.block_size
                cell_y = start_y + j * self.block_size
                
                # Mark walls if out of bounds.
                if cell_x < 0 or cell_x >= self.width or cell_y < 0 or cell_y >= self.height:
                    vision[i, j, 0] = 1.0  # Wall
                else:
                    # Mark snake body presence.
                    for segment in self.snake_list:
                        if segment[0] == cell_x and segment[1] == cell_y:
                            vision[i, j, 1] = 1.0
                            break
                    # Mark food presence.
                    if cell_x == self.food_x and cell_y == self.food_y:
                        vision[i, j, 2] = 1.0
        
        # Flatten the vision matrix.
        vision_flat = vision.flatten()  # 21*21*3 = 1323

        # Compute normalized food distance vector.
        # max_distance = math.sqrt(self.width ** 2 + self.height ** 2)
        dx = (self.food_x - self.head_x) / self.width #max_distance
        dy = (self.food_y - self.head_y) / self.height #max_distance
        
        # Normalized snake length.
        normalized_length = self.snake_length / (self.field_width * self.field_height)
        
        state = np.concatenate((vision_flat, np.array([dx, dy, normalized_length], dtype=np.float32)))
        return state  # shape: (1326,)

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

        reward = 0.0
        
        game_over = False
        if self._is_collision():
            game_over = True
            reward -= 30.0  # Death penalty
            return reward, game_over, self.score
        
        # Check if food is eaten.
        if self.head_x == self.food_x and self.head_y == self.food_y:
            self.snake_length += 1
            self.score += 1
            reward += 100.0  # Reward for eating food
            self._spawn_food()
        
        # Reward for survival (small incentive for each step taken)
        reward += 0.1

        if self.render:
            self._draw_elements()
            self.clock.tick(self.snake_speed)
        
        return reward, game_over, self.score

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
