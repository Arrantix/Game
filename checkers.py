import pygame
import sys
import copy
import random
import math
import numpy as np
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
CREAM = (255, 248, 220)
BROWN = (139, 69, 19)
DARK_BROWN = (101, 67, 33)
RED = (220, 20, 60)
DARK_RED = (139, 0, 0)
LIGHT_RED = (255, 99, 71)
BLUE = (30, 144, 255)
DARK_BLUE = (0, 0, 139)
LIGHT_BLUE = (135, 206, 250)
GOLD = (255, 215, 0)
SILVER = (192, 192, 192)
GREEN = (50, 205, 50)
LIGHT_GREEN = (144, 238, 144)
PURPLE = (147, 112, 219)
GRAY = (128, 128, 128)
LIGHT_GRAY = (211, 211, 211)
DARK_GRAY = (64, 64, 64)

# Screen dimensions
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 1000

# Board dimensions
BOARD_SIZE = 8
SQUARE_SIZE = 70
BOARD_OFFSET_X = 200  # Moved further right to give space for statistics
BOARD_OFFSET_Y = 150

# AI Visualization
AI_PANEL_WIDTH = 450
AI_PANEL_HEIGHT = 750
AI_PANEL_X = SCREEN_WIDTH - AI_PANEL_WIDTH - 20
AI_PANEL_Y = 120

# FPS
FPS = 60

# Font
FONT = pygame.font.SysFont('arial', 20)
LARGE_FONT = pygame.font.SysFont('arial', 32)
TITLE_FONT = pygame.font.SysFont('arial', 48)
SMALL_FONT = pygame.font.SysFont('arial', 16)

# Piece types
EMPTY = 0
RED_PIECE = 1
RED_KING = 2
BLUE_PIECE = 3
BLUE_KING = 4

class CheckersGame:
    def __init__(self):
        self.board = self.initialize_board()
        self.current_player = RED_PIECE
        self.selected_piece = None
        self.possible_moves = []
        self.game_over = False
        self.winner = None
        self.must_jump = False

    def initialize_board(self):
        board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]

        # Place red pieces (top 3 rows)
        for row in range(3):
            for col in range(BOARD_SIZE):
                if (row + col) % 2 == 1:
                    board[row][col] = RED_PIECE

        # Place blue pieces (bottom 3 rows)
        for row in range(5, BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if (row + col) % 2 == 1:
                    board[row][col] = BLUE_PIECE

        return board

    def is_valid_square(self, row, col):
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE and (row + col) % 2 == 1

    def get_piece_moves(self, row, col):
        if not self.is_valid_square(row, col):
            return []

        piece = self.board[row][col]
        if piece == EMPTY:
            return []

        moves = []
        jumps = []

        # Determine enemy pieces
        if piece in [RED_PIECE, RED_KING]:
            enemy_pieces = [BLUE_PIECE, BLUE_KING]
        else:
            enemy_pieces = [RED_PIECE, RED_KING]

        # Determine movement directions based on piece type
        if piece == RED_PIECE:
            directions = [(1, -1), (1, 1)]    # Red moves down
        elif piece == BLUE_PIECE:
            directions = [(-1, -1), (-1, 1)]  # Blue moves up
        else:  # Kings can move in all directions
            directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dr, dc in directions:
            if piece in [RED_PIECE, BLUE_PIECE]:
                # Regular move: 1 step
                new_row, new_col = row + dr, col + dc
                if self.is_valid_square(new_row, new_col) and self.board[new_row][new_col] == EMPTY:
                    moves.append((new_row, new_col, False))  # False = not a jump

                # Jump move for regular pieces
                jump_row, jump_col = row + 2 * dr, col + 2 * dc
                if (self.is_valid_square(new_row, new_col) and
                    self.is_valid_square(jump_row, jump_col) and
                    self.board[new_row][new_col] in enemy_pieces and  # Enemy piece
                    self.board[jump_row][jump_col] == EMPTY):
                    jumps.append((jump_row, jump_col, True, (new_row, new_col)))  # True = jump, with jumped piece position
                    logging.debug(f"Jump move found for piece at ({row},{col}) to ({jump_row},{jump_col}) over enemy at ({new_row},{new_col})")
            else:  # Kings: sliding moves and jumps
                # First, check for jumps in this direction
                found_enemy = False
                enemy_pos = None
                landing_pos = None

                for step in range(1, BOARD_SIZE):
                    new_row = row + step * dr
                    new_col = col + step * dc

                    if not self.is_valid_square(new_row, new_col):
                        break

                    current_piece = self.board[new_row][new_col]

                    if current_piece == EMPTY:
                        if found_enemy:
                            # This is a potential landing square after jumping over enemy
                            if landing_pos is None:
                                landing_pos = (new_row, new_col)
                        else:
                            # Normal sliding move
                            moves.append((new_row, new_col, False))
                    elif current_piece in enemy_pieces:
                        if found_enemy:
                            # Multiple enemies in a row - invalid jump
                            break
                        else:
                            # Found first enemy
                            found_enemy = True
                            enemy_pos = (new_row, new_col)
                    else:
                        # Blocked by own piece
                        break

                # If we found exactly one enemy followed by an empty square, it's a valid jump
                if found_enemy and landing_pos:
                    jumps.append((landing_pos[0], landing_pos[1], True, enemy_pos))
                    logging.debug(f"King jump found for piece at ({row},{col}) to ({landing_pos[0]},{landing_pos[1]}) over enemy at ({enemy_pos[0]},{enemy_pos[1]})")

        # If there are jumps available, only jumps are allowed
        if jumps:
            return jumps
        else:
            return moves

    def make_move(self, start_row, start_col, end_row, end_col):
        if not self.is_valid_square(start_row, start_col) or not self.is_valid_square(end_row, end_col):
            return False

        piece = self.board[start_row][start_col]
        if piece == EMPTY or not self.is_player_piece(piece):
            return False

        # Check if this is a valid move
        valid_moves = self.get_piece_moves(start_row, start_col)
        target_move = None

        for move in valid_moves:
            if move[0] == end_row and move[1] == end_col:
                target_move = move
                break

        if not target_move:
            return False

        logging.debug(f"Making move for piece {piece} from ({start_row},{start_col}) to ({end_row},{end_col})")

        # Make the move
        self.board[end_row][end_col] = piece
        self.board[start_row][start_col] = EMPTY

        # Handle jump
        if len(target_move) > 2 and target_move[2]:  # It's a jump
            jumped_row, jumped_col = target_move[3]
            logging.debug(f"Jumped over enemy piece at ({jumped_row},{jumped_col})")
            self.board[jumped_row][jumped_col] = EMPTY

            # Check for additional jumps with the same piece
            additional_jumps = self.get_piece_moves(end_row, end_col)
            jumps_only = [move for move in additional_jumps if len(move) > 2 and move[2]]

            if jumps_only:
                # Don't switch turns if more jumps are possible
                return True

        # Check for king promotion
        if piece == RED_PIECE and end_row == BOARD_SIZE - 1:
            logging.debug(f"Promoting piece at ({end_row},{end_col}) to RED_KING")
            self.board[end_row][end_col] = RED_KING
        elif piece == BLUE_PIECE and end_row == 0:
            logging.debug(f"Promoting piece at ({end_row},{end_col}) to BLUE_KING")
            self.board[end_row][end_col] = BLUE_KING

        # Switch turns
        self.current_player = BLUE_PIECE if self.current_player == RED_PIECE else RED_PIECE

        # Check for game over
        self.check_game_over()

        return True

    def is_player_piece(self, piece):
        if self.current_player in [RED_PIECE, RED_KING]:
            return piece in [RED_PIECE, RED_KING]
        else:
            return piece in [BLUE_PIECE, BLUE_KING]

    def get_all_possible_moves(self):
        moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.is_player_piece(self.board[row][col]):
                    piece_moves = self.get_piece_moves(row, col)
                    for move in piece_moves:
                        moves.append((row, col, move[0], move[1]))
        return moves

    def check_game_over(self):
        opponent = BLUE_PIECE if self.current_player == RED_PIECE else RED_PIECE
        opponent_pieces = [BLUE_PIECE, BLUE_KING] if opponent == BLUE_PIECE else [RED_PIECE, RED_KING]

        # Check if opponent has any pieces left
        has_pieces = any(self.board[row][col] in opponent_pieces
                        for row in range(BOARD_SIZE) for col in range(BOARD_SIZE))

        # Check if current player has any pieces left
        current_player_pieces = [RED_PIECE, RED_KING] if self.current_player == RED_PIECE else [BLUE_PIECE, BLUE_KING]
        has_current_pieces = any(self.board[row][col] in current_player_pieces
                                for row in range(BOARD_SIZE) for col in range(BOARD_SIZE))

        if not has_pieces or not has_current_pieces:
            self.game_over = True
            # Winner is the player who still has pieces
            if has_pieces and not has_current_pieces:
                self.winner = opponent
            elif has_current_pieces and not has_pieces:
                self.winner = self.current_player
            else:
                self.winner = None  # Draw or no pieces left
            logging.debug(f"Game over: winner determined by pieces. Winner: {self.winner}")
            return

        # Check if opponent has any valid moves
        has_moves = len(self.get_all_possible_moves()) > 0

        if not has_moves:
            self.game_over = True
            self.winner = self.current_player
            logging.debug(f"Game over: no moves available for opponent. Winner: {self.winner}")

class DeepQNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2. / self.input_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2. / self.hidden_size)
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def forward(self, x):
        self.z1 = np.dot(x, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        return self.z2  # Output layer (Q-values)

    def backward(self, x, y_true, y_pred):
        error = y_pred - y_true
        dW2 = np.dot(self.a1.T, error)
        db2 = np.sum(error, axis=0, keepdims=True)

        da1 = np.dot(error, self.W2.T)
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = np.dot(x.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # Update weights
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def train(self, x, y_true):
        y_pred = self.forward(x)
        self.backward(x, y_true, y_pred)

    def predict(self, x):
        return self.forward(x)

class CheckersAI:
    def __init__(self, game, player):
        self.game = game
        self.player = player
        self.model = DeepQNetwork(input_size=BOARD_SIZE*BOARD_SIZE*5, hidden_size=128, output_size=BOARD_SIZE*BOARD_SIZE)
        self.memory = []
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.games_played = 0
        self.wins = 0

    def board_to_input(self, board):
        # One-hot encode board pieces into 5 channels: empty, red, red king, blue, blue king
        input_array = np.zeros((BOARD_SIZE, BOARD_SIZE, 5))
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = board[r][c]
                input_array[r, c, piece] = 1
        return input_array.flatten().reshape(1, -1)

    def get_possible_moves(self):
        return self.game.get_all_possible_moves()

    def get_best_move(self):
        moves = self.get_possible_moves()
        if not moves:
            return None

        if np.random.rand() < self.epsilon:
            return random.choice(moves)

        best_move = None
        best_q = -float('inf')
        state_input = self.board_to_input(self.game.board)

        for move in moves:
            # Simulate move
            temp_game = CheckersGame()
            temp_game.board = copy.deepcopy(self.game.board)
            temp_game.current_player = self.game.current_player
            if temp_game.make_move(*move):
                next_state_input = self.board_to_input(temp_game.board)
                q_values = self.model.predict(next_state_input)
                q_value = np.max(q_values)
                if q_value > best_q:
                    best_q = q_value
                    best_move = move

        return best_move or random.choice(moves)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > 10000:
            self.memory.pop(0)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = self.model.predict(state)
            if done:
                target[0, action] = reward
            else:
                t = self.model.predict(next_state)
                target[0, action] = reward + self.gamma * np.max(t)
            self.model.train(state, target)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_stats(self, won):
        self.games_played += 1
        if won:
            self.wins += 1
        # Decay epsilon faster after each game
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def evaluate_board(self, board):
        score = 0

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                piece = board[row][col]
                if piece == EMPTY:
                    continue

                piece_value = 1
                if piece in [RED_KING, BLUE_KING]:
                    piece_value = 3  # Kings are more valuable

                # Position bonus for center control
                center_distance = abs(3.5 - row) + abs(3.5 - col)
                position_bonus = max(0, 4 - center_distance)

                # Edge control bonus
                edge_bonus = 0
                if row in [0, 7] or col in [0, 7]:
                    edge_bonus = 1

                if piece in [RED_PIECE, RED_KING]:
                    score += piece_value + position_bonus + edge_bonus
                else:
                    score -= piece_value + position_bonus + edge_bonus

        return score if self.player == RED_PIECE else -score

    def get_board_state(self, board):
        # Convert board to a hashable state representation
        return tuple(tuple(row) for row in board)

    def learn(self, old_state, action, reward, new_state):
        # For compatibility, but now using replay
        pass

    def get_move_index(self, move):
        # Map move to index for neural network output
        moves = self.get_possible_moves()
        if move in moves:
            return moves.index(move)
        return 0

def draw_gradient_rect(screen, rect, start_color, end_color, vertical=True):
    """Draw a gradient rectangle"""
    x, y, width, height = rect

    if vertical:
        for i in range(height):
            ratio = i / height
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            pygame.draw.line(screen, (r, g, b), (x, y + i), (x + width, y + i))
    else:
        for i in range(width):
            ratio = i / width
            r = int(start_color[0] * (1 - ratio) + end_color[0] * ratio)
            g = int(start_color[1] * (1 - ratio) + end_color[1] * ratio)
            b = int(start_color[2] * (1 - ratio) + end_color[2] * ratio)
            pygame.draw.line(screen, (r, g, b), (x + i, y), (x + i, y + height))

def draw_board(screen, game):
    # Draw board background with gradient
    board_width = BOARD_SIZE * SQUARE_SIZE
    board_height = BOARD_SIZE * SQUARE_SIZE
    board_rect = (BOARD_OFFSET_X, BOARD_OFFSET_Y, board_width, board_height)

    # Draw wooden board effect
    draw_gradient_rect(screen, board_rect, BROWN, DARK_BROWN)

    # Draw board squares with 3D effect
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            x = BOARD_OFFSET_X + col * SQUARE_SIZE
            y = BOARD_OFFSET_Y + row * SQUARE_SIZE

            if (row + col) % 2 == 0:
                # Light squares
                base_color = CREAM
                highlight_color = WHITE
            else:
                # Dark squares
                base_color = DARK_BROWN
                highlight_color = BROWN

            # Highlight selected piece
            if game.selected_piece and game.selected_piece == (row, col):
                base_color = LIGHT_GREEN
                highlight_color = GREEN

            # Highlight possible moves
            if (row, col) in [(move[0], move[1]) for move in game.possible_moves]:
                base_color = LIGHT_BLUE
                highlight_color = BLUE

            # Draw square with 3D effect
            square_rect = (x, y, SQUARE_SIZE, SQUARE_SIZE)
            draw_gradient_rect(screen, square_rect, base_color, highlight_color)

            # Draw square border
            pygame.draw.rect(screen, BLACK, square_rect, 1)

            # Draw pieces with enhanced graphics
            piece = game.board[row][col]
            if piece != EMPTY:
                center_x = x + SQUARE_SIZE // 2
                center_y = y + SQUARE_SIZE // 2
                radius = SQUARE_SIZE // 2 - 8

                # Draw piece shadow
                shadow_offset = 3
                pygame.draw.circle(screen, DARK_GRAY, (center_x + shadow_offset, center_y + shadow_offset), radius)

                if piece in [RED_PIECE, RED_KING]:
                    # Red pieces with gradient
                    piece_color = RED
                    highlight_color = LIGHT_RED
                    if piece == RED_KING:
                        # King crown effect
                        pygame.draw.circle(screen, DARK_RED, (center_x, center_y), radius)
                        pygame.draw.circle(screen, piece_color, (center_x, center_y), radius - 3)
                        # Draw crown
                        crown_points = [
                            (center_x - 8, center_y - 8), (center_x - 4, center_y - 12),
                            (center_x, center_y - 8), (center_x + 4, center_y - 12),
                            (center_x + 8, center_y - 8), (center_x + 6, center_y - 4),
                            (center_x - 6, center_y - 4)
                        ]
                        pygame.draw.polygon(screen, GOLD, crown_points)
                else:
                    # Blue pieces with gradient
                    piece_color = BLUE
                    highlight_color = LIGHT_BLUE
                    if piece == BLUE_KING:
                        # King crown effect
                        pygame.draw.circle(screen, DARK_BLUE, (center_x, center_y), radius)
                        pygame.draw.circle(screen, piece_color, (center_x, center_y), radius - 3)
                        # Draw crown
                        crown_points = [
                            (center_x - 8, center_y - 8), (center_x - 4, center_y - 12),
                            (center_x, center_y - 8), (center_x + 4, center_y - 12),
                            (center_x + 8, center_y - 8), (center_x + 6, center_y - 4),
                            (center_x - 6, center_y - 4)
                        ]
                        pygame.draw.polygon(screen, GOLD, crown_points)

                # Draw main piece with gradient
                for r in range(radius, 0, -1):
                    ratio = r / radius
                    if piece in [RED_PIECE, RED_KING]:
                        color = (
                            int(DARK_RED[0] * (1 - ratio) + LIGHT_RED[0] * ratio),
                            int(DARK_RED[1] * (1 - ratio) + LIGHT_RED[1] * ratio),
                            int(DARK_RED[2] * (1 - ratio) + LIGHT_RED[2] * ratio)
                        )
                    else:
                        color = (
                            int(DARK_BLUE[0] * (1 - ratio) + LIGHT_BLUE[0] * ratio),
                            int(DARK_BLUE[1] * (1 - ratio) + LIGHT_BLUE[1] * ratio),
                            int(DARK_BLUE[2] * (1 - ratio) + LIGHT_BLUE[2] * ratio)
                        )
                    pygame.draw.circle(screen, color, (center_x, center_y), r)

                # Draw piece border
                pygame.draw.circle(screen, BLACK, (center_x, center_y), radius, 2)

    # Draw board border
    pygame.draw.rect(screen, BLACK, board_rect, 3)

def draw_text(screen, text, font, color, x, y):
    textobj = font.render(text, True, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    screen.blit(textobj, textrect)

def draw_ai_brain_visualization(screen, ai, x, y, width, height, color):
    """Draw a visual representation of the AI's neural network"""
    # Draw panel background
    pygame.draw.rect(screen, DARK_GRAY, (x, y, width, height))
    pygame.draw.rect(screen, color, (x+2, y+2, width-4, height-4))

    # Draw title
    ai_name = "RED AI" if ai.player == RED_PIECE else "BLUE AI"
    draw_text(screen, ai_name, FONT, WHITE, x + 10, y + 10)

    # Draw layer labels
    layer_names = ["Board State", "Hidden", "Actions"]
    layer_sizes = [BOARD_SIZE*BOARD_SIZE*5, 128, BOARD_SIZE*BOARD_SIZE]
    layer_positions = []

    for i, (name, size) in enumerate(zip(layer_names, layer_sizes)):
        layer_x = x + 30 + (i * (width - 60) // (len(layer_names) - 1))
        layer_y = y + 50
        layer_height = height - 140
        layer_positions.append((layer_x, layer_y, size))

        # Draw layer label
        draw_text(screen, name, SMALL_FONT, WHITE, layer_x - 15, layer_y - 25)

        # Draw neurons
        if i == 0:  # Input layer - show board representation
            # Show simplified board state
            board_squares = []
            for r in range(4):  # Show 4x4 grid
                for c in range(4):
                    if (r + c) % 2 == 1:
                        square_x = layer_x - 12 + c * 6
                        square_y = layer_y + r * 6
                        pygame.draw.rect(screen, CREAM, (square_x, square_y, 4, 4))
                        board_squares.append((square_x, square_y))
        elif i == 2:  # Output layer - show action grid
            # Show move grid
            move_squares = []
            for r in range(4):
                for c in range(4):
                    if (r + c) % 2 == 1:
                        square_x = layer_x - 12 + c * 6
                        square_y = layer_y + r * 6
                        pygame.draw.rect(screen, LIGHT_BLUE, (square_x, square_y, 4, 4))
                        move_squares.append((square_x, square_y))
        else:  # Hidden layer
            neuron_spacing = layer_height / max(size, 16)
            for j in range(min(size, 16)):
                neuron_y = layer_y + j * neuron_spacing
                neuron_radius = 2
                pygame.draw.circle(screen, WHITE, (layer_x, int(neuron_y)), neuron_radius)

    # Draw connections between layers with better organization
    for i in range(len(layer_positions) - 1):
        current_layer = layer_positions[i]
        next_layer = layer_positions[i + 1]

        if i == 0:  # From input to hidden
            current_neurons = 16  # Simplified
            next_neurons = min(next_layer[2], 16)
        elif i == 1:  # From hidden to output
            current_neurons = min(current_layer[2], 16)
            next_neurons = 16  # Simplified output
        else:
            current_neurons = min(current_layer[2], 16)
            next_neurons = min(next_layer[2], 16)

        for j in range(current_neurons):
            for k in range(next_neurons):
                if random.random() < 0.2:  # Fewer connections for clarity
                    if i == 0:
                        start_x = current_layer[0] - 10 + (j % 4) * 6
                        start_y = current_layer[1] + (j // 4) * 6
                    else:
                        start_x = current_layer[0]
                        start_y = current_layer[1] + j * (current_layer[2] / current_neurons)

                    if i == 1:
                        end_x = next_layer[0] - 10 + (k % 4) * 6
                        end_y = next_layer[1] + (k // 4) * 6
                    else:
                        end_x = next_layer[0]
                        end_y = next_layer[1] + k * (next_layer[2] / next_neurons)

                    pygame.draw.line(screen, LIGHT_GRAY, (start_x, start_y), (end_x, end_y), 1)

    # Draw AI statistics in a more compact layout
    stats_y = y + height - 80
    draw_text(screen, f"Games: {ai.games_played}", SMALL_FONT, WHITE, x + 10, stats_y)
    draw_text(screen, f"Wins: {ai.wins}", SMALL_FONT, WHITE, x + 10, stats_y + 15)
    draw_text(screen, f"Epsilon: {ai.epsilon:.3f}", SMALL_FONT, WHITE, x + 10, stats_y + 30)
    draw_text(screen, f"Memory: {len(ai.memory)}", SMALL_FONT, WHITE, x + 10, stats_y + 45)
    draw_text(screen, f"LR: {ai.model.learning_rate}", SMALL_FONT, WHITE, x + 10, stats_y + 60)

def draw_q_values_visualization(screen, ai, x, y, width, height):
    """Draw a visualization of Q-values for current state"""
    pygame.draw.rect(screen, DARK_GRAY, (x, y, width, height))
    pygame.draw.rect(screen, BLACK, (x+2, y+2, width-4, height-4))

    draw_text(screen, "Action Values", SMALL_FONT, WHITE, x + 10, y + 10)

    # Get current state Q-values
    state_input = ai.board_to_input(ai.game.board)
    q_values = ai.model.predict(state_input)[0]

    # Replace NaN values with 0
    q_values = np.nan_to_num(q_values, nan=0.0)

    # Get possible moves to map Q-values to actual actions
    possible_moves = ai.get_possible_moves()
    if not possible_moves:
        return

    # Find max and min Q-values for scaling
    max_q = np.max(q_values)
    min_q = np.min(q_values)

    # Draw Q-value bars with action labels (show first 12 for better fit)
    num_bars = min(12, len(possible_moves))
    bar_width = (width - 20) // num_bars

    for i in range(num_bars):
        move = possible_moves[i]
        q_value = q_values[i] if i < len(q_values) else 0

        bar_height = int((q_value - min_q) / (max_q - min_q + 1e-6) * (height - 80))
        bar_x = x + 10 + i * bar_width
        bar_y = y + height - 30 - bar_height

        # Color based on Q-value (green for positive, red for negative)
        if q_value > 0:
            bar_color = GREEN
        else:
            bar_color = RED

        pygame.draw.rect(screen, bar_color, (bar_x, bar_y, bar_width - 2, bar_height))

        # Draw action label (from_row,from_col -> to_row,to_col)

def get_square_from_mouse(pos):
    x, y = pos
    col = (x - BOARD_OFFSET_X) // SQUARE_SIZE
    row = (y - BOARD_OFFSET_Y) // SQUARE_SIZE

    if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
        return row, col
    return None

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption('Checkers with Learning AI')
    clock = pygame.time.Clock()

    game = CheckersGame()
    red_ai = CheckersAI(game, RED_PIECE)   # AI plays as red
    blue_ai = CheckersAI(game, BLUE_PIECE) # AI plays as blue

    ai_vs_ai_mode = True
    human_mode = False
    ai_turn = True
    move_delay = 0
    move_timer = 0

    # Game statistics
    total_games = 0
    red_wins = 0
    blue_wins = 0

    # Learning variables
    last_state = None
    last_action = None
    last_player = None

    while True:
        # Background gradient
        draw_gradient_rect(screen, (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), LIGHT_GRAY, GRAY)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    # Reset game
                    game = CheckersGame()
                    red_ai.game = game
                    blue_ai.game = game
                    ai_turn = True
                    last_state = None
                    last_action = None
                    last_player = None
                elif event.key == pygame.K_a:
                    ai_vs_ai_mode = not ai_vs_ai_mode
                    human_mode = not human_mode
                    if human_mode:
                        ai_turn = False
                elif event.key == pygame.K_v:
                    # Toggle AI vs AI mode
                    ai_vs_ai_mode = not ai_vs_ai_mode
            elif event.type == pygame.MOUSEBUTTONDOWN and human_mode:
                if event.button == 1:  # Left click
                    square = get_square_from_mouse(event.pos)
                    if square:
                        row, col = square

                        if game.selected_piece:
                            # Try to make a move
                            if game.make_move(game.selected_piece[0], game.selected_piece[1], row, col):
                                ai_turn = True
                            game.selected_piece = None
                            game.possible_moves = []
                        else:
                            # Select a piece
                            if game.is_player_piece(game.board[row][col]):
                                game.selected_piece = (row, col)
                                game.possible_moves = game.get_piece_moves(row, col)

        # AI turns
        if not game.game_over:
            if ai_vs_ai_mode:
                # AI vs AI mode
                move_timer += 1
                if move_timer >= move_delay:
                    if game.current_player == RED_PIECE:
                        current_ai = red_ai
                    else:
                        current_ai = blue_ai

                    # Store state before move for learning
                    if last_state is not None and last_action is not None:
                        reward = current_ai.evaluate_board(game.board)
                        next_state = current_ai.board_to_input(game.board)
                        done = game.game_over
                        current_ai.remember(last_state, last_action, reward, next_state, done)

                    best_move = current_ai.get_best_move()
                    if best_move:
                        # Store state and action for learning
                        last_state = current_ai.board_to_input(game.board)
                        last_action = current_ai.get_move_index(best_move)
                        last_player = game.current_player

                        game.make_move(*best_move)

                        # Replay experience
                        current_ai.replay()

                    move_timer = 0
                    move_delay = random.randint(5, 15)  # Random delay for visual effect

            elif human_mode and ai_turn and game.current_player == BLUE_PIECE:
                # Human vs AI mode
                move_timer += 1
                if move_timer >= move_delay:
                    # Store state before move for learning
                    if last_state is not None and last_action is not None:
                        reward = blue_ai.evaluate_board(game.board)
                        next_state = blue_ai.board_to_input(game.board)
                        done = game.game_over
                        blue_ai.remember(last_state, last_action, reward, next_state, done)

                    best_move = blue_ai.get_best_move()
                    if best_move:
                        # Store state and action for learning
                        last_state = blue_ai.board_to_input(game.board)
                        last_action = blue_ai.get_move_index(best_move)
                        last_player = game.current_player

                        game.make_move(*best_move)
                        ai_turn = False

                        # Replay experience
                        blue_ai.replay()

                    move_timer = 0
                    move_delay = random.randint(10, 20)

        # Learning after game ends
        if game.game_over and last_state is not None and last_action is not None:
            if game.winner == RED_PIECE:
                reward = 100  # Win reward
                red_ai.update_stats(True)
                blue_ai.update_stats(False)
            elif game.winner == BLUE_PIECE:
                reward = 100  # Win reward
                blue_ai.update_stats(True)
                red_ai.update_stats(False)
            else:
                reward = 0  # Draw

            if last_player == RED_PIECE:
                red_ai.learn(last_state, last_action, reward, red_ai.board_to_input(game.board))
            else:
                blue_ai.learn(last_state, last_action, reward, blue_ai.board_to_input(game.board))

            # Update statistics
            total_games += 1
            if game.winner == RED_PIECE:
                red_wins += 1
            elif game.winner == BLUE_PIECE:
                blue_wins += 1

            last_state = None
            last_action = None
            last_player = None

            # Auto-restart game in AI vs AI mode
            if ai_vs_ai_mode:
                game = CheckersGame()
                red_ai.game = game
                blue_ai.game = game

        draw_board(screen, game)

        # Draw AI Brain Visualizations
        # Red AI Brain (top panel)
        draw_ai_brain_visualization(screen, red_ai, AI_PANEL_X, AI_PANEL_Y, AI_PANEL_WIDTH, AI_PANEL_HEIGHT // 2 - 10, DARK_RED)

        # Blue AI Brain (bottom panel)
        draw_ai_brain_visualization(screen, blue_ai, AI_PANEL_X, AI_PANEL_Y + AI_PANEL_HEIGHT // 2 + 10, AI_PANEL_WIDTH, AI_PANEL_HEIGHT // 2 - 10, DARK_BLUE)

        # Draw Q-Values Visualization (small panel at bottom)
        q_panel_height = 150
        draw_q_values_visualization(screen, red_ai if game.current_player == RED_PIECE else blue_ai,
                                  AI_PANEL_X, AI_PANEL_Y + AI_PANEL_HEIGHT - q_panel_height, AI_PANEL_WIDTH, q_panel_height)

        # Draw UI with enhanced styling
        # Title with shadow
        draw_text(screen, "Checkers with AI Learning", TITLE_FONT, BLACK, 302, 22)
        draw_text(screen, "Checkers with AI Learning", TITLE_FONT, GOLD, 300, 20)

        # Game mode indicator
        mode_text = "AI vs AI" if ai_vs_ai_mode else "Human vs AI"
        draw_text(screen, f"Mode: {mode_text}", FONT, BLACK, 10, 70)

        # Current player
        current_player_text = "Red" if game.current_player == RED_PIECE else "Blue"
        player_color = RED if game.current_player == RED_PIECE else BLUE
        draw_text(screen, f"Current: {current_player_text}", FONT, player_color, 10, 100)

        # Statistics - organized in a more compact layout
        draw_text(screen, f"Games: {total_games}", SMALL_FONT, BLACK, 10, 130)
        draw_text(screen, f"Red: {red_wins} wins", SMALL_FONT, RED, 10, 150)
        draw_text(screen, f"Blue: {blue_wins} wins", SMALL_FONT, BLUE, 10, 170)

        # Win rates - more compact
        red_win_rate = (red_wins / total_games * 100) if total_games > 0 else 0
        blue_win_rate = (blue_wins / total_games * 100) if total_games > 0 else 0
        draw_text(screen, f"Red Rate: {red_win_rate:.1f}%", SMALL_FONT, DARK_RED, 10, 190)
        draw_text(screen, f"Blue Rate: {blue_win_rate:.1f}%", SMALL_FONT, DARK_BLUE, 10, 210)

        # Controls - positioned to avoid overlap
        draw_text(screen, "R: Restart | A: Toggle Mode | V: AI vs AI", SMALL_FONT, BLACK, 10, 250)

        if game.game_over:
            winner_text = "Red" if game.winner == RED_PIECE else "Blue"
            winner_color = RED if game.winner == RED_PIECE else BLUE

            # Winner announcement with shadow
            draw_text(screen, f"{winner_text} Wins!", LARGE_FONT, BLACK, 202, 302)
            draw_text(screen, f"{winner_text} Wins!", LARGE_FONT, winner_color, 200, 300)
            draw_text(screen, "Press R to restart", FONT, BLACK, 200, 350)

        pygame.display.update()
        clock.tick(FPS)

def test_win_conditions():
    # Test 1: Normal win by capturing all pieces
    game = CheckersGame()
    game.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    game.board[0][0] = RED_PIECE
    game.board[7][7] = BLUE_PIECE
    game.current_player = BLUE_PIECE
    game.check_game_over()
    assert game.game_over == True
    assert game.winner == BLUE_PIECE
    print("Test 1 passed: Win by capturing all pieces")

    # Test 2: Win by no moves
    game = CheckersGame()
    game.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    game.board[0][0] = RED_PIECE
    game.board[1][1] = BLUE_PIECE  # Enemy to jump over
    game.board[2][2] = RED_PIECE  # Block landing square
    game.current_player = BLUE_PIECE  # So opponent is red
    game.check_game_over()
    assert game.game_over == True
    assert game.winner == BLUE_PIECE
    print("Test 2 passed: Win by no moves")

    # Test 3: King movement
    game = CheckersGame()
    game.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
    game.board[0][1] = RED_KING
    moves = game.get_piece_moves(0, 1)
    assert len(moves) > 0  # Should have moves in all directions
    print("Test 3 passed: King can move")

    print("All tests passed!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_win_conditions()
    else:
        main()
