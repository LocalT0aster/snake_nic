import pygame
import random
from enum import Enum

class Direction(Enum):
    """
    Enum for storing snake movement direction.
    """
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4

class SnakeGame:
    """
    A simple Snake game using Pygame where the window dimensions are
    computed based on a field size (grid cells) and block size.

    Parameters
    ----------
    field_size : tuple of int
        The size of the field in grid cells, given as (columns, rows).
    block_size : int, optional
        The size of each grid cell (snake segment and food) in pixels (default is 20).
    snake_speed : int, optional
        The speed of the snake which controls the game clock (default is 15).

    Attributes
    ----------
    display : pygame.Surface
        The main game display surface.
    clock : pygame.time.Clock
        Pygame clock object to control the game speed.
    snake_list : list
        A list of [x, y] positions (in pixels) representing the snake's segments.
    snake_length : int
        The current length of the snake.
    direction : Direction
        The current movement direction of the snake as an instance of the Direction enum.
    head_x : int
        The x-coordinate of the snake's head in pixels.
    head_y : int
        The y-coordinate of the snake's head in pixels.
    food_x : int
        The x-coordinate of the food in pixels.
    food_y : int
        The y-coordinate of the food in pixels.
    width : int
        The width of the game window in pixels.
    height : int
        The height of the game window in pixels.
    """
    def __init__(self, field_size=(30, 20), block_size=20, snake_speed=15):
        # Game configuration parameters
        self.field_width, self.field_height = field_size  # Number of grid cells (columns, rows)
        self.block_size = block_size
        self.snake_speed = snake_speed

        # Compute the window dimensions in pixels from field size and block size.
        self.width = self.field_width * self.block_size
        self.height = self.field_height * self.block_size

        # Define basic colors (RGB)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        self.red = (213, 50, 80)
        self.green = (0, 255, 0)
        self.blue = (50, 153, 213)

        # Initialize snake properties in pixel coordinates.
        self.snake_list = []              # List storing snake segments ([x, y]).
        self.snake_length = 1             # Start with a single segment.
        self.direction = Direction.RIGHT  # Start with RIGHT as the initial direction.
        self.head_x = self.width // 2     # Center the snake horizontally.
        self.head_y = self.height // 2    # Center the snake vertically.

        # Initialize food position (will be set in _spawn_food).
        self.food_x = None
        self.food_y = None

        # Initialize Pygame and set up the display.
        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        # Place the first food item.
        self._spawn_food()

    def _spawn_food(self):
        """
        Randomly generate a new food position on the grid.

        The food's coordinates are aligned to the grid by using multiples of block_size.
        """
        self.food_x = round(random.randrange(0, self.width - self.block_size) / self.block_size) * self.block_size
        self.food_y = round(random.randrange(0, self.height - self.block_size) / self.block_size) * self.block_size

    def _handle_events(self):
        """
        Process Pygame events such as key presses and window close events.

        To prevent multiple direction changes in the same game tick, a flag is used.
        
        Returns
        -------
        bool
            Returns False if a quit event is detected; otherwise, True.
        """
        # Allow only one change in direction per tick.
        direction_updated = False
        # Save the direction at the beginning of this tick.
        initial_direction = self.direction
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to exit the game.
            if event.type == pygame.KEYDOWN and not direction_updated:
                # Check key events but always use the initial direction for validation.
                if event.key == pygame.K_LEFT and initial_direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                    direction_updated = True
                elif event.key == pygame.K_RIGHT and initial_direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                    direction_updated = True
                elif event.key == pygame.K_UP and initial_direction != Direction.DOWN:
                    self.direction = Direction.UP
                    direction_updated = True
                elif event.key == pygame.K_DOWN and initial_direction != Direction.UP:
                    self.direction = Direction.DOWN
                    direction_updated = True
        return True

    def _update_snake_position(self):
        """
        Update the snake's head coordinates based on the current direction and manage its body segments.
        """
        if self.direction == Direction.LEFT:
            self.head_x -= self.block_size
        elif self.direction == Direction.RIGHT:
            self.head_x += self.block_size
        elif self.direction == Direction.UP:
            self.head_y -= self.block_size
        elif self.direction == Direction.DOWN:
            self.head_y += self.block_size

        # Append new head position.
        self.snake_list.append([self.head_x, self.head_y])
        # Remove tail segment if snake hasn't grown.
        if len(self.snake_list) > self.snake_length:
            del self.snake_list[0]

    def _check_collisions(self):
        """
        Check for collisions with boundaries, the snake itself, or food.

        Returns
        -------
        bool
            Returns True if a collision occurs that should end the game.
        """
        # Collision with game boundaries.
        if self.head_x < 0 or self.head_x >= self.width or self.head_y < 0 or self.head_y >= self.height:
            return True
        
        # Collision with itself (excluding the head).
        for segment in self.snake_list[:-1]:
            if segment == [self.head_x, self.head_y]:
                return True
        
        # Collision with food: increase snake length and reposition food.
        if self.head_x == self.food_x and self.head_y == self.food_y:
            self.snake_length += 1
            self._spawn_food()
        
        return False

    def _draw_elements(self):
        """
        Draw the background, snake, and food on the display.
        """
        self.display.fill(self.blue)
        pygame.draw.rect(self.display, self.green,
                         [self.food_x, self.food_y, self.block_size, self.block_size])
        for segment in self.snake_list:
            pygame.draw.rect(self.display, self.black,
                             [segment[0], segment[1], self.block_size, self.block_size])
        pygame.display.update()

    def run(self):
        """
        Run the main game loop until a collision ends the game.

        The loop processes events, updates the snake's position, checks for collisions,
        and controls the game speed.
        """
        running = True
        while running:
            running = self._handle_events()
            self._update_snake_position()
            if self._check_collisions():
                running = False
            self._draw_elements()
            self.clock.tick(self.snake_speed)
        self._game_over()

    def _game_over(self):
        """
        Display the game over screen and allow the player to quit or restart.

        The screen shows a message and waits for user input: Q to quit, C to restart.
        """
        font_style = pygame.font.SysFont("bahnschrift", 35)
        message = font_style.render("Game Over! Press Q-Quit or C-Play Again", True, self.red)
        self.display.blit(message, [self.width / 6, self.height / 3])
        pygame.display.update()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        waiting = False
                    elif event.key == pygame.K_c:
                        self.__init__((self.field_width, self.field_height), self.block_size, self.snake_speed)
                        self.run()
                        waiting = False
                if event.type == pygame.QUIT:
                    waiting = False
        pygame.quit()

if __name__ == "__main__":
    """
    Entry point for the Snake game.

    Customize the game by setting field size (cells), block size (pixels), and snake speed.
    """
    game = SnakeGame(field_size=(30, 20), block_size=20, snake_speed=15)
    game.run()
