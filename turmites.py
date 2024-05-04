import colorsys
import random
import sys
from typing import List, Tuple, Dict

import jsonpickle
import pygame

# Define constants for the display
CELL_SIZE = 10
NUM_CELLS = 100
SCREEN_SIZE = CELL_SIZE * NUM_CELLS
FPS = 30

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

NUM_COLORS = 4


def get_color_palette(num_colors: int, saturation: float = 1.0, lightness: float = 0.5, hue_offset: float = 0) -> List[
    Tuple[int, int, int]]:
    colors = []
    for i in range(num_colors):
        hue = i / num_colors  # Evenly spaced hue values
        hue = (hue + hue_offset) % 1.0  # Apply offset
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        colors.append(rgb)
    return colors


# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
TEAL = (0, 255, 255)
PURPLE = (255, 0, 255)
ORANGE = (255, 165, 0)
PINK = (255, 192, 203)

# COLORS = [BLACK, RED, BLUE, GREEN, YELLOW, TEAL, PURPLE, ORANGE, PINK]

COLORS = [BLACK] + get_color_palette(NUM_COLORS, saturation=random.uniform(0.5, 1.0),
                                     lightness=random.uniform(0.4, 0.6), hue_offset=random.random())


class Turmite:
    def __init__(self, x: int, y: int, direction: int, state: int,
                 transition_table: Dict[Tuple[int, int], Tuple[int, str, int]]):
        self.x = x
        self.y = y
        self.direction = direction
        self.state = state
        self.transition_table = transition_table
        self.recent_positions = []
        self.turns_stuck = 0


# Example state transition table for the turmites
# (current_color, current_state): (write_color, turn, next_state)
# transition_table = {
#     (0, 0): (1, 'R', 0),
#     (1, 0): (0, 'R', 1),
#     (0, 1): (0, 'N', 0),
#     (1, 1): (0, 'N', 1),
# }


class TurmiteHandler(jsonpickle.handlers.BaseHandler):
    def flatten(self, obj, data):
        # Convert object to a serializable form
        data['x'] = obj.x
        data['y'] = obj.y
        data['direction'] = obj.direction
        data['state'] = obj.state
        data['transition_table'] = {f"{k[0]},{k[1]}": v for k, v in obj.transition_table.items()}
        return data

    def restore(self, obj):
        # Restore object from serialized form
        x = obj['x']
        y = obj['y']
        direction = obj['direction']
        state = obj['state']
        transition_table = {tuple(map(int, k.split(','))): v for k, v in obj['transition_table'].items()}
        return Turmite(x, y, direction, state, transition_table)


# Register the handler for Turmite class
jsonpickle.handlers.registry.register(Turmite, TurmiteHandler)


def generate_random_transition_table(num_colors: int = -1, num_states: int = -1) -> Dict[
    Tuple[int, int], Tuple[int, str, int]]:
    if num_colors == -1:
        num_colors = NUM_COLORS
    if num_states == -1:
        num_states = random.randint(2, 5)

    directions = ['L', 'R', 'N', 'U']  # Left, Right, No turn, U-turn
    transition_table = {}

    for state in range(num_states):
        for color in range(num_colors):
            new_color = random.randint(0, num_colors - 1)
            turn_direction = random.choice(directions)
            next_state = random.randint(0, num_states - 1)
            transition_table[(color, state)] = (new_color, turn_direction, next_state)

    return transition_table


def turn(direction: int, turn: str) -> int:
    if turn == 'L':
        return (direction - 1) % 4
    elif turn == 'R':
        return (direction + 1) % 4
    elif turn == 'U':
        return (direction + 2) % 4
    return direction


def move_forward(x: int, y: int, direction: int) -> Tuple[int, int]:
    if direction == UP:
        return x, (y - 1) % NUM_CELLS
    elif direction == RIGHT:
        return (x + 1) % NUM_CELLS, y
    elif direction == DOWN:
        return x, (y + 1) % NUM_CELLS
    elif direction == LEFT:
        return (x - 1) % NUM_CELLS, y
    return x, y


def create_turmites(n: int) -> List[Turmite]:
    turmites = []
    for _ in range(n):
        x = random.randint(0, NUM_CELLS - 1)
        y = random.randint(0, NUM_CELLS - 1)
        direction = random.choice([UP, RIGHT, DOWN, LEFT])
        state = random.randint(0, 1)  # assuming there are two states (0 and 1)
        turmites.append(Turmite(x, y, direction, state, generate_random_transition_table()))
    return turmites


def get_triangle_vertices(x: int, y: int, direction: int) -> List[Tuple[int, int]]:
    # Middle point of the square
    center_x = x * CELL_SIZE + CELL_SIZE // 2
    center_y = y * CELL_SIZE + CELL_SIZE // 2

    # Size of the triangle
    size = CELL_SIZE // 4

    if direction == UP:
        vertices = [(center_x, center_y - size), (center_x - size, center_y + size), (center_x + size, center_y + size)]
    elif direction == RIGHT:
        vertices = [(center_x + size, center_y), (center_x - size, center_y - size), (center_x - size, center_y + size)]
    elif direction == DOWN:
        vertices = [(center_x, center_y + size), (center_x - size, center_y - size), (center_x + size, center_y - size)]
    elif direction == LEFT:
        vertices = [(center_x - size, center_y), (center_x + size, center_y - size), (center_x + size, center_y + size)]
    else:
        raise ValueError('Invalid direction')

    return vertices


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock = pygame.time.Clock()
    grid = [[0 for _ in range(NUM_CELLS)] for _ in range(NUM_CELLS)]
    turmites: List[Turmite] = create_turmites(10)

    serialized_turmites = jsonpickle.encode(turmites, indent=4)
    # print(serialized_turmites)

    with open('turmites.json', 'w') as f:
        f.write(serialized_turmites)

    # deserialized_turmites = jsonpickle.decode(serialized_turmites)
    # reserialized_turmites = jsonpickle.encode(deserialized_turmites, indent=4)
    #
    # # Compare the two strings
    # print(serialized_turmites == reserialized_turmites)
    #
    # turmites = deserialized_turmites

    running = True
    while running:
        if pygame.time.get_ticks() % 100 == 0 and False:
            with open('turmites.json', 'r') as f:
                print("Decoding turmites...")
                edited_turmites = jsonpickle.decode(f.read())

                # Copy over positions and states from turmites to edited_turmites
                for turmite, edited_turmite in zip(turmites, edited_turmites):
                    edited_turmite.x = turmite.x
                    edited_turmite.y = turmite.y
                    edited_turmite.state = turmite.state

                turmites = edited_turmites

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(BLACK)
        for turmite in turmites:
            current_color = grid[turmite.y][turmite.x]
            write_color, turn_direction, next_state = turmite.transition_table[(current_color, turmite.state)]
            grid[turmite.y][turmite.x] = write_color
            turmite.direction = turn(turmite.direction, turn_direction)
            turmite.x, turmite.y = move_forward(turmite.x, turmite.y, turmite.direction)
            turmite.state = next_state

        for row in range(NUM_CELLS):
            for col in range(NUM_CELLS):
                color = COLORS[grid[row][col]]
                pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for turmite in turmites:
            pygame.draw.rect(screen, WHITE, (turmite.x * CELL_SIZE, turmite.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            # Draw direction triangle
            triangle_vertices = get_triangle_vertices(turmite.x, turmite.y, turmite.direction)
            pygame.draw.polygon(screen, BLACK, triangle_vertices)  # Draw a triangle for visibility

        # Add to the recent positions of all turmites
        for turmite in turmites:
            turmite.recent_positions.append((turmite.x, turmite.y))
            if len(turmite.recent_positions) > 10:
                turmite.recent_positions.pop(0)

            # Mark them as stuck if they are stuck
            if len(set(turmite.recent_positions)) <= 9:
                turmite.turns_stuck += 1
            else:
                turmite.turns_stuck = 0

            # Draw a trail
            for i, (x, y) in enumerate(turmite.recent_positions):
                pygame.draw.circle(screen, color=WHITE,
                                   center=(x * CELL_SIZE + CELL_SIZE * 0.5, y * CELL_SIZE + CELL_SIZE * 0.5),
                                   radius=CELL_SIZE / 4, )

        # Respawn the turmites that are stuck
        for i, turmite in enumerate(turmites):
            if turmite.turns_stuck > 40:
                turmites[i] = Turmite(random.randint(0, NUM_CELLS - 1),
                                      random.randint(0, NUM_CELLS - 1),
                                      random.choice([UP, RIGHT, DOWN, LEFT]),
                                      random.randint(0, 1),
                                      generate_random_transition_table())

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
