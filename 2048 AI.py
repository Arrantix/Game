import pygame
import random
import sys
import math
from copy import deepcopy

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (64, 64, 64)

# Tile colors
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

# Screen dimensions
SCREEN_WIDTH = 500
SCREEN_HEIGHT = 600

# Grid size
GRID_SIZE = 4
CELL_SIZE = 100
CELL_MARGIN = 10
GRID_OFFSET_X = (SCREEN_WIDTH - (GRID_SIZE * CELL_SIZE + (GRID_SIZE - 1) * CELL_MARGIN)) // 2
GRID_OFFSET_Y = 150

# FPS
FPS = 60

# Font
FONT = pygame.font.SysFont('arial', 24)
LARGE_FONT = pygame.font.SysFont('arial', 36)
TITLE_FONT = pygame.font.SysFont('arial', 48)

class Game2048:
    def __init__(self):
        self.grid = [[0] * GRID_SIZE for _ in range(GRID_SIZE)]
        self.score = 0
        self.game_over = False
        self.won = False
        self.spawn_tile()
        self.spawn_tile()

    def spawn_tile(self):
        empty_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if self.grid[i][j] == 0]
        if empty_cells:
            i, j = random.choice(empty_cells)
            self.grid[i][j] = 2 if random.random() < 0.9 else 4

    def can_move(self):
        # Check for empty cells
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if self.grid[i][j] == 0:
                    return True

        # Check for possible merges
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i][j + 1]:
                    return True

        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE - 1):
                if self.grid[i][j] == self.grid[i + 1][j]:
                    return True

        return False

    def move_left(self):
        moved = False
        new_score = 0

        for i in range(GRID_SIZE):
            # Compress
            row = [x for x in self.grid[i] if x != 0]
            row += [0] * (GRID_SIZE - len(row))

            # Merge
            for j in range(GRID_SIZE - 1):
                if row[j] == row[j + 1] and row[j] != 0:
                    row[j] *= 2
                    new_score += row[j]
                    row[j + 1] = 0
                    moved = True

            # Compress again
            row = [x for x in row if x != 0]
            row += [0] * (GRID_SIZE - len(row))

            if row != self.grid[i]:
                moved = True
            self.grid[i] = row

        self.score += new_score
        return moved

    def move_right(self):
        moved = False
        new_score = 0

        for i in range(GRID_SIZE):
            # Compress
            row = [x for x in self.grid[i] if x != 0]
            row = [0] * (GRID_SIZE - len(row)) + row

            # Merge
            for j in range(GRID_SIZE - 1, 0, -1):
                if row[j] == row[j - 1] and row[j] != 0:
                    row[j] *= 2
                    new_score += row[j]
                    row[j - 1] = 0
                    moved = True

            # Compress again
            row = [x for x in row if x != 0]
            row = [0] * (GRID_SIZE - len(row)) + row

            if row != self.grid[i]:
                moved = True
            self.grid[i] = row

        self.score += new_score
        return moved

    def move_up(self):
        moved = False
        new_score = 0

        for j in range(GRID_SIZE):
            # Compress
            col = [self.grid[i][j] for i in range(GRID_SIZE) if self.grid[i][j] != 0]
            col += [0] * (GRID_SIZE - len(col))

            # Merge
            for i in range(GRID_SIZE - 1):
                if col[i] == col[i + 1] and col[i] != 0:
                    col[i] *= 2
                    new_score += col[i]
                    col[i + 1] = 0
                    moved = True

            # Compress again
            col = [x for x in col if x != 0]
            col += [0] * (GRID_SIZE - len(col))

            for i in range(GRID_SIZE):
                if self.grid[i][j] != col[i]:
                    moved = True
                self.grid[i][j] = col[i]

        self.score += new_score
        return moved

    def move_down(self):
        moved = False
        new_score = 0

        for j in range(GRID_SIZE):
            # Compress
            col = [self.grid[i][j] for i in range(GRID_SIZE) if self.grid[i][j] != 0]
            col = [0] * (GRID_SIZE - len(col)) + col

            # Merge
            for i in range(GRID_SIZE - 1, 0, -1):
                if col[i] == col[i - 1] and col[i] != 0:
                    col[i] *= 2
                    new_score += col[i]
                    col[i - 1] = 0
                    moved = True

            # Compress again
            col = [x for x in col if x != 0]
            col = [0] * (GRID_SIZE - len(col)) + col

            for i in range(GRID_SIZE):
                if self.grid[i][j] != col[i]:
                    moved = True
                self.grid[i][j] = col[i]

        self.score += new_score
        return moved

    def make_move(self, direction):
        if direction == 'left':
            moved = self.move_left()
        elif direction == 'right':
            moved = self.move_right()
        elif direction == 'up':
            moved = self.move_up()
        elif direction == 'down':
            moved = self.move_down()
        else:
            return False

        if moved:
            self.spawn_tile()
            if not self.can_move():
                self.game_over = True
            if any(2048 in row for row in self.grid):
                self.won = True

        return moved

    def get_possible_moves(self):
        moves = []
        for direction in ['left', 'right', 'up', 'down']:
            temp_game = deepcopy(self)
            if temp_game.make_move(direction):
                moves.append(direction)
        return moves

class AI2048:
    def __init__(self, game):
        self.game = game

    def evaluate_grid(self, grid):
        # Heuristic evaluation function
        score = 0

        # Prefer higher values in corners
        corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for i, j in corners:
            if grid[i][j] >= 128:
                score += grid[i][j] * 10

        # Prefer monotonicity (values decreasing in rows and columns)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] > grid[i][j + 1]:
                    score += grid[i][j] - grid[i][j + 1]

        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE - 1):
                if grid[i][j] > grid[i + 1][j]:
                    score += grid[i][j] - grid[i + 1][j]

        # Prefer empty cells
        empty_count = sum(1 for row in grid for cell in row if cell == 0)
        score += empty_count * 100

        # Prefer merges possible
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE - 1):
                if grid[i][j] == grid[i][j + 1] and grid[i][j] != 0:
                    score += grid[i][j] * 2

        for j in range(GRID_SIZE):
            for i in range(GRID_SIZE - 1):
                if grid[i][j] == grid[i + 1][j] and grid[i][j] != 0:
                    score += grid[i][j] * 2

        return score

    def expectimax(self, game, depth, is_max_player):
        if depth == 0 or game.game_over:
            return self.evaluate_grid(game.grid), None

        if is_max_player:
            max_eval = -math.inf
            best_move = None
            for move in game.get_possible_moves():
                temp_game = deepcopy(game)
                temp_game.make_move(move)
                eval_score, _ = self.expectimax(temp_game, depth - 1, False)
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
            return max_eval, best_move
        else:
            # Chance node - average over possible tile spawns
            total_eval = 0
            count = 0
            empty_cells = [(i, j) for i in range(GRID_SIZE) for j in range(GRID_SIZE) if game.grid[i][j] == 0]
            if not empty_cells:
                return self.evaluate_grid(game.grid), None

            # Sample a few possible spawns for efficiency
            sample_size = min(5, len(empty_cells))
            sampled_cells = random.sample(empty_cells, sample_size)

            for i, j in sampled_cells:
                for tile_value in [2, 4]:
                    temp_game = deepcopy(game)
                    temp_game.grid[i][j] = tile_value
                    eval_score, _ = self.expectimax(temp_game, depth - 1, True)
                    total_eval += eval_score
                    count += 1

            return total_eval / count if count > 0 else self.evaluate_grid(game.grid), None

    def get_best_move(self):
        _, best_move = self.expectimax(self.game, 3, True)  # Depth 3 for reasonable performance
        return best_move

def draw_grid(screen, game):
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            x = GRID_OFFSET_X + j * (CELL_SIZE + CELL_MARGIN)
            y = GRID_OFFSET_Y + i * (CELL_SIZE + CELL_MARGIN)

            # Draw cell background
            pygame.draw.rect(screen, TILE_COLORS.get(game.grid[i][j], DARK_GRAY),
                           (x, y, CELL_SIZE, CELL_SIZE))

            # Draw cell border
            pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 2)

            # Draw number
            if game.grid[i][j] != 0:
                text = LARGE_FONT.render(str(game.grid[i][j]), True, BLACK)
                text_rect = text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                screen.blit(text, text_rect)

def draw_text(screen, text, font, color, x, y):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    screen.blit(textobj, textrect)

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('2048 with AI')
    clock = pygame.time.Clock()

    game = Game2048()
    ai = AI2048(game)

    ai_mode = True
    move_delay = 0
    move_timer = 0

    while True:
        screen.fill(LIGHT_GRAY)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game = Game2048()
                    ai = AI2048(game)
                elif event.key == pygame.K_a:
                    ai_mode = not ai_mode
                elif not ai_mode and not game.game_over and not game.won:
                    # Manual controls when AI is off
                    if event.key == pygame.K_LEFT:
                        game.make_move('left')
                    elif event.key == pygame.K_RIGHT:
                        game.make_move('right')
                    elif event.key == pygame.K_UP:
                        game.make_move('up')
                    elif event.key == pygame.K_DOWN:
                        game.make_move('down')

        if ai_mode and not game.game_over and not game.won:
            move_timer += 1
            if move_timer >= move_delay:
                best_move = ai.get_best_move()
                if best_move:
                    game.make_move(best_move)
                move_timer = 0
                move_delay = random.randint(10, 30)  # Random delay for visual effect

        draw_grid(screen, game)

        # Draw UI
        draw_text(screen, "2048", TITLE_FONT, BLACK, 200, 20)
        draw_text(screen, f"Score: {game.score}", FONT, BLACK, 20, 100)
        draw_text(screen, f"AI: {'ON' if ai_mode else 'OFF'}", FONT, BLACK, 350, 100)
        draw_text(screen, "A: Toggle AI | R: Restart", FONT, BLACK, 20, 570)

        if game.game_over:
            draw_text(screen, "Game Over!", LARGE_FONT, (255, 0, 0), 150, 500)
            draw_text(screen, "Press R to restart", FONT, BLACK, 150, 540)
        elif game.won:
            draw_text(screen, "You Won!", LARGE_FONT, (0, 255, 0), 150, 500)
            draw_text(screen, "Press R to restart", FONT, BLACK, 150, 540)

        pygame.display.update()
        clock.tick(FPS)

if __name__ == '__main__':
    main()
