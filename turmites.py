import colorsys
import random
import sys
from typing import List, Tuple, Dict, Optional

import jsonpickle
import pygame

# Define constants for the display
CELL_SIZE = 10
NUM_CELLS_WIDE = 150
NUM_CELLS_HIGH = 100
SCREEN_SIZE_WIDE = CELL_SIZE * NUM_CELLS_WIDE
SCREEN_SIZE_HIGH = CELL_SIZE * NUM_CELLS_HIGH
FPS = 60

# Set seed for random library
random_seed = random.randint(0, 10_000)
random.seed(random_seed)
print(f"Random seed: {random_seed}")

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

NUM_TURMITES = 10

NUM_COLORS = 4
NUM_STATES_PER_TURMITE = 3
PERCENT_DARKNESS = 0.4

STUCK_THRESHOLD = 9  # If 0, they will never be considered stuck
STUCK_MEMORY_LEN = 30
VISUAL_TRAIL_LENGTH = 7
STUCK_DURATION = 60
NUM_TOURNAMENT_LOSERS = 0
TOURNAMENT_THRESHOLD_MULTIPLIER = 3

ALL_SAME_AT_START = True
ONLY_REGEN_ALL_IF_ALL_STUCK = True
RENDER_CHAMPION = False


def get_color_palette(num_colors: int, saturation: float = 1.0, lightness: float = 0.5, hue_offset: float = 0,
                      random_jitter: float = 0.0) -> List[
    Tuple[int, int, int]]:
    colors = []
    for i in range(num_colors):
        hue = i / num_colors  # Evenly spaced hue values
        hue = (hue + hue_offset) % 1.0  # Apply offset
        r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
        rgb = (int(r * 255), int(g * 255), int(b * 255))
        # Apply some random jitter
        rgb = tuple(int(min(255, max(0, c + int(random.uniform(-random_jitter, random_jitter) * 255)))) for c in rgb)
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
    def __init__(self, x: int, y: int, num_states: int, direction: int, state: int,
                 transition_table: Dict[Tuple[int, int], Tuple[int, str, int]]):
        self.x = x
        self.y = y
        self.num_states = num_states
        self.direction = direction
        self.state = state
        self.transition_table = transition_table
        self.recent_positions = []
        self.turns_stuck = 0
        self.id = random.randint(0, sys.maxsize)
        self.credit_count = 0


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
        raise NotImplementedError("Restore not implemented")
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
            new_color = 0 if random.uniform(0.0, 1.0) < PERCENT_DARKNESS else random.randint(1, num_colors - 1)
            turn_direction = random.choice(directions)
            next_state = random.randint(0, num_states - 1)
            transition_table[(color, state)] = (new_color, turn_direction, next_state)

    return transition_table


def get_random_turmite(turmites: Optional[list[Turmite]] = None) -> Turmite:
    print("Getting random turmite...")
    x = random.randint(0, NUM_CELLS_WIDE - 1)
    y = random.randint(0, NUM_CELLS_HIGH - 1)
    if random.randint(0, 2) == 0 and turmites is not None:
        t = random.choice(turmites)
        t.x = x
        t.y = y
        return t
    elif random.randint(0, 2) == 0 and turmites is not None:
        t = turmites[0]  # The turmites are periodically sorted based on their credit count
        t.x = x
        t.y = y
        return t
    else:
        direction = random.randint(0, 3)
        num_states = random.randint(2, NUM_STATES_PER_TURMITE)
        state = random.randint(0, num_states - 1)
        transition_table = generate_random_transition_table(num_states=num_states)
        return Turmite(x, y, num_states, direction, state, transition_table)


def create_turmites(n: int = NUM_TURMITES) -> List[Turmite]:
    turmites = []
    if ALL_SAME_AT_START:
        for i in range(n):
            if i == 0:
                turmites.append(get_random_turmite())
            else:
                t0 = turmites[0]
                randx = random.randint(0, NUM_CELLS_WIDE - 1)
                randy = random.randint(0, NUM_CELLS_HIGH - 1)
                randdir = random.randint(0, 3)
                turmites.append(Turmite(randx, randy, t0.num_states, randdir, t0.state, t0.transition_table))
        print("Initial turmite:")
        print(turmites[0].transition_table)
    else:
        for i in range(n):
            turmites.append(get_random_turmite())
    return turmites


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
        return x, (y - 1) % NUM_CELLS_HIGH
    elif direction == RIGHT:
        return (x + 1) % NUM_CELLS_WIDE, y
    elif direction == DOWN:
        return x, (y + 1) % NUM_CELLS_HIGH
    elif direction == LEFT:
        return (x - 1) % NUM_CELLS_WIDE, y
    return x, y


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
    screen = pygame.display.set_mode((SCREEN_SIZE_WIDE, SCREEN_SIZE_HIGH))
    clock = pygame.time.Clock()
    grid = [[0 for _ in range(NUM_CELLS_WIDE)] for _ in range(NUM_CELLS_HIGH)]
    credit_grid = [[0 for _ in range(NUM_CELLS_WIDE)] for _ in range(NUM_CELLS_HIGH)]
    turmites: List[Turmite] = create_turmites()

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
            credit_grid[turmite.y][turmite.x] = turmite.id
            turmite.direction = turn(turmite.direction, turn_direction)
            turmite.x, turmite.y = move_forward(turmite.x, turmite.y, turmite.direction)
            turmite.state = next_state

        for row in range(NUM_CELLS_HIGH):
            for col in range(NUM_CELLS_WIDE):
                color = COLORS[grid[row][col]]
                pygame.draw.rect(screen, color, (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE))

        for i, turmite in enumerate(turmites):
            pygame.draw.rect(screen, YELLOW if (i == 0 and RENDER_CHAMPION) else WHITE,
                             (turmite.x * CELL_SIZE, turmite.y * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            # Draw direction triangle
            triangle_vertices = get_triangle_vertices(turmite.x, turmite.y, turmite.direction)
            pygame.draw.polygon(screen, BLACK, triangle_vertices)  # Draw a triangle for visibility

        # Add to the recent positions of all turmites
        for turmite in turmites:
            turmite.recent_positions.append((turmite.x, turmite.y))
            if len(turmite.recent_positions) > STUCK_MEMORY_LEN:
                turmite.recent_positions.pop(0)

            # Mark them as stuck if they are stuck
            if len(set(turmite.recent_positions)) <= STUCK_THRESHOLD:
                turmite.turns_stuck += 1
            else:
                turmite.turns_stuck = 0

            # Draw a trail
            for i, (x, y) in enumerate(turmite.recent_positions[-VISUAL_TRAIL_LENGTH:]):
                pygame.draw.circle(screen, WHITE,
                                   center=(x * CELL_SIZE + CELL_SIZE * 0.5, y * CELL_SIZE + CELL_SIZE * 0.5),
                                   radius=CELL_SIZE / 4, )

        # Respawn the turmites that are stuck
        if ONLY_REGEN_ALL_IF_ALL_STUCK:
            if all(turmite.turns_stuck > STUCK_DURATION for turmite in turmites):
                turmites = create_turmites(len(turmites))
        else:
            for i, turmite in enumerate(turmites):
                if turmite.turns_stuck > STUCK_DURATION + random.randint(0, STUCK_DURATION):
                    turmites[i] = get_random_turmite(turmites)

        # Periodic tournament
        if pygame.time.get_ticks() % 200 == 0:
            # Iterate over the whole grid
            for t in turmites:
                t.credit_count = 0
            for row in range(NUM_CELLS_HIGH):
                for col in range(NUM_CELLS_WIDE):
                    # Get the turmite id
                    turmite_id = credit_grid[row][col]

                    # If there is a turmite id
                    if turmite_id != 0:
                        # Get the turmite
                        turmite = None
                        for t in turmites:
                            if t.id == turmite_id:
                                turmite = t
                                break
                        if turmite is None:
                            continue

                        # Increment the turmite's credit
                        # turmite.credit_count += grid[row][col]  # BLACK is worth nothing I guess
                        turmite.credit_count += 2 if grid[row][col] == 0 else 1  # BLACK IS THE BEST COLOR
                        turmite.credit_count += 1  # All colors are valuable

                        # Reset the credit grid probabilistically, to get exponential decay
                        if random.randint(0, 2) == 0:
                            credit_grid[row][col] = 0

            # Sort the turmites by credit count
            turmites = sorted(turmites, key=lambda t: t.credit_count, reverse=True)

            # Print credit counts
            # print("\nTurmite Credits:")
            # for t in turmites:
            #     print(f"Turmite {t.id}: {t.credit_count}")

            # Respawn the worst N turmites
            worst_score = turmites[-1].credit_count
            for i in range(NUM_TOURNAMENT_LOSERS):
                if turmites[-i].credit_count > worst_score * TOURNAMENT_THRESHOLD_MULTIPLIER:
                    break  # It's fine
                turmites[-i] = get_random_turmite(turmites)

        # Keyboard inputs
        keys = pygame.key.get_pressed()
        if keys[pygame.K_r]:
            turmites = create_turmites(len(turmites))
        if keys[pygame.K_c]:
            grid = [[0 for _ in range(NUM_CELLS_WIDE)] for _ in range(NUM_CELLS_HIGH)]
            credit_grid = [[0 for _ in range(NUM_CELLS_WIDE)] for _ in range(NUM_CELLS_HIGH)]
        if keys[pygame.K_ESCAPE]:
            running = False

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
