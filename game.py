import pygame
import random
import sys
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
import tkinter as tk
from tkinter import messagebox
import threading
import time

# Initialize Pygame
pygame.init()

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Screen dimensions
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 900

# Block size
BLOCK_SIZE = 20

# FPS
FPS = 15  # Slightly faster for AI

# Font
FONT = pygame.font.SysFont(None, 35)

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class Snake:
    def __init__(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
        self.color = GREEN

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if self.length > 1 and (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = (cur[0] + (x * BLOCK_SIZE), cur[1] + (y * BLOCK_SIZE))

        # Check for wall collision first
        if (new[0] < 0 or new[0] >= SCREEN_WIDTH or
            new[1] < 0 or new[1] >= SCREEN_HEIGHT):
            return False  # Wall collision

        # Check for self collision
        if len(self.positions) > 1 and new in self.positions[1:]:
            return False  # Self collision

        self.positions.insert(0, new)
        if len(self.positions) > self.length:
            self.positions.pop()
        return True  # Move successful

    def reset(self):
        self.length = 1
        self.positions = [((SCREEN_WIDTH // 2), (SCREEN_HEIGHT // 2))]
        self.direction = random.choice([UP, DOWN, LEFT, RIGHT])

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BLACK, r, 1)

    def handle_keys(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.turn(LEFT)
                elif event.key == pygame.K_RIGHT:
                    self.turn(RIGHT)
                elif event.key == pygame.K_UP:
                    self.turn(UP)
                elif event.key == pygame.K_DOWN:
                    self.turn(DOWN)

class Food:
    def __init__(self):
        self.position = (0, 0)
        self.color = RED
        self.randomize_position()

    def randomize_position(self, snake_positions=None):
        """Randomize food position, ensuring it doesn't spawn on snake body"""
        if snake_positions is None:
            snake_positions = set()

        while True:
            x = random.randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
            y = random.randint(0, (SCREEN_HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
            new_position = (x, y)

            # Ensure food doesn't spawn on snake body
            if new_position not in snake_positions:
                self.position = new_position
                break

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, BLACK, r, 1)

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SnakeAI:
    def __init__(self, snake, food):
        self.snake = snake
        self.food = food
        self.vision_range = 8  # Increased vision to 8 blocks

    def get_vision_area(self, head_pos):
        """Get all positions within vision range of the head"""
        vision_positions = []
        hx, hy = head_pos
        for dx in range(-self.vision_range, self.vision_range + 1):
            for dy in range(-self.vision_range, self.vision_range + 1):
                vx = (hx + dx * BLOCK_SIZE) % SCREEN_WIDTH
                vy = (hy + dy * BLOCK_SIZE) % SCREEN_HEIGHT
                vision_positions.append((vx, vy))
        return vision_positions

    def is_position_safe(self, pos, snake_positions):
        """Check if a position is safe (no collision with snake body or walls)"""
        x, y = pos
        # Check wall boundaries
        if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
            return False
        # Check snake body collision
        return pos not in snake_positions

    def get_neighbors(self, pos, snake_positions):
        """Get valid neighboring positions"""
        x, y = pos
        neighbors = []
        for dx, dy in [UP, DOWN, LEFT, RIGHT]:
            nx = x + dx * BLOCK_SIZE
            ny = y + dy * BLOCK_SIZE
            new_pos = (nx, ny)
            if self.is_position_safe(new_pos, snake_positions):
                neighbors.append(new_pos)
        return neighbors

    def bfs_path(self, start, goal, snake_positions):
        """Find shortest path using BFS"""
        queue = deque([(start, [])])
        visited = set([start])

        while queue:
            current, path = queue.popleft()
            if current == goal:
                return path

            for neighbor in self.get_neighbors(current, snake_positions):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def can_reach_percentage(self, head_pos, snake_positions):
        """Check if snake can reach 80% of free blocks"""
        total_blocks = (SCREEN_WIDTH // BLOCK_SIZE) * (SCREEN_HEIGHT // BLOCK_SIZE)
        free_blocks = total_blocks - len(snake_positions)

        # Sample some free blocks to check reachability
        reachable = 0
        sample_size = min(100, free_blocks)  # Sample up to 100 blocks

        for _ in range(sample_size):
            # Pick a random free block
            while True:
                x = random.randint(0, (SCREEN_WIDTH // BLOCK_SIZE) - 1) * BLOCK_SIZE
                y = random.randint(0, (SCREEN_HEIGHT // BLOCK_SIZE) - 1) * BLOCK_SIZE
                test_pos = (x, y)
                if test_pos not in snake_positions:
                    break

            # Check if we can reach it
            if self.bfs_path(head_pos, test_pos, snake_positions) is not None:
                reachable += 1

        return (reachable / sample_size) >= 0.8 if sample_size > 0 else True

    def get_best_direction(self):
        """Determine the best direction for the snake"""
        head_pos = self.snake.get_head_position()
        snake_positions = set(self.snake.positions)

        # First priority: path to food
        food_path = self.bfs_path(head_pos, self.food.position, snake_positions)
        if food_path:
            next_pos = food_path[0]
            # Check if this move keeps us in a good position
            temp_positions = snake_positions.copy()
            temp_positions.add(next_pos)
            temp_positions.remove(self.snake.positions[-1])  # Remove tail

            if self.can_reach_percentage(next_pos, temp_positions):
                return self.get_direction_from_positions(head_pos, next_pos)

        # Second priority: find a safe direction that maintains accessibility
        possible_moves = []
        for direction in [UP, DOWN, LEFT, RIGHT]:
            dx, dy = direction
            new_x = head_pos[0] + dx * BLOCK_SIZE
            new_y = head_pos[1] + dy * BLOCK_SIZE
            new_pos = (new_x, new_y)

            if self.is_position_safe(new_pos, snake_positions):
                # Simulate the move
                temp_positions = snake_positions.copy()
                temp_positions.add(new_pos)
                if len(self.snake.positions) > 1:
                    temp_positions.remove(self.snake.positions[-1])

                if self.can_reach_percentage(new_pos, temp_positions):
                    possible_moves.append((direction, new_pos))

        if possible_moves:
            # Choose the move that gives the longest path to food if possible
            best_move = None
            best_distance = -1

            for direction, new_pos in possible_moves:
                path_to_food = self.bfs_path(new_pos, self.food.position, snake_positions)
                if path_to_food and len(path_to_food) > best_distance:
                    best_distance = len(path_to_food)
                    best_move = direction

            if best_move:
                return best_move

            # If no good food path, just pick the first safe move
            return possible_moves[0][0]

        # Emergency: if no safe moves, pick any valid direction
        for direction in [UP, DOWN, LEFT, RIGHT]:
            dx, dy = direction
            new_x = head_pos[0] + dx * BLOCK_SIZE
            new_y = head_pos[1] + dy * BLOCK_SIZE
            new_pos = (new_x, new_y)
            if self.is_position_safe(new_pos, snake_positions):
                return direction

        return self.snake.direction  # Keep current direction as last resort

    def get_direction_from_positions(self, current, target):
        """Convert position difference to direction"""
        dx = (target[0] - current[0]) // BLOCK_SIZE
        dy = (target[1] - current[1]) // BLOCK_SIZE

        if dx == 1: return RIGHT
        if dx == -1: return LEFT
        if dy == 1: return DOWN
        if dy == -1: return UP

        return self.snake.direction

class DQLAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = 18  # 4 dir + 2 food + 4 dist + 4 safe + 4 food_dist
        self.output_size = 4
        self.policy_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net = DQN(self.input_size, self.output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = deque(maxlen=10000)
        self.batch_size = 128
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10
        self.steps_done = 0

    def get_state(self, snake, food):
        head = snake.get_head_position()
        dir = snake.direction
        food_pos = food.position
        snake_positions = set(snake.positions)

        # Relative food
        food_dx = (food_pos[0] - head[0]) / SCREEN_WIDTH
        food_dy = (food_pos[1] - head[1]) / SCREEN_HEIGHT

        # Direction one-hot
        dir_up = 1 if dir == UP else 0
        dir_down = 1 if dir == DOWN else 0
        dir_left = 1 if dir == LEFT else 0
        dir_right = 1 if dir == RIGHT else 0

        # Distances to obstacles (walls and snake body)
        dist_up = self.get_distance(head, UP, snake_positions)
        dist_down = self.get_distance(head, DOWN, snake_positions)
        dist_left = self.get_distance(head, LEFT, snake_positions)
        dist_right = self.get_distance(head, RIGHT, snake_positions)
        max_dist = (SCREEN_WIDTH // BLOCK_SIZE) + (SCREEN_HEIGHT // BLOCK_SIZE)

        # Safety of immediate next step in each direction
        safe_up = 1 if self.is_position_safe((head[0], head[1] - BLOCK_SIZE), snake_positions) else 0
        safe_down = 1 if self.is_position_safe((head[0], head[1] + BLOCK_SIZE), snake_positions) else 0
        safe_left = 1 if self.is_position_safe((head[0] - BLOCK_SIZE, head[1]), snake_positions) else 0
        safe_right = 1 if self.is_position_safe((head[0] + BLOCK_SIZE, head[1]), snake_positions) else 0

        # Distance to food in each direction (Manhattan distance)
        food_dist_up = abs(food_pos[1] - head[1]) + abs(food_pos[0] - head[0]) if food_pos[1] < head[1] else 0
        food_dist_down = abs(food_pos[1] - head[1]) + abs(food_pos[0] - head[0]) if food_pos[1] > head[1] else 0
        food_dist_left = abs(food_pos[0] - head[0]) + abs(food_pos[1] - head[1]) if food_pos[0] < head[0] else 0
        food_dist_right = abs(food_pos[0] - head[0]) + abs(food_pos[1] - head[1]) if food_pos[0] > head[0] else 0

        # Normalize food distances
        max_food_dist = SCREEN_WIDTH + SCREEN_HEIGHT
        food_dist_up /= max_food_dist
        food_dist_down /= max_food_dist
        food_dist_left /= max_food_dist
        food_dist_right /= max_food_dist

        state = [
            dir_up, dir_down, dir_left, dir_right,
            food_dx, food_dy,
            dist_up / max_dist, dist_down / max_dist, dist_left / max_dist, dist_right / max_dist,
            safe_up, safe_down, safe_left, safe_right,
            food_dist_up, food_dist_down, food_dist_left, food_dist_right
        ]
        return np.array(state, dtype=np.float32)

    def is_position_safe(self, pos, snake_positions):
        """Check if a position is safe (no collision with snake body or walls)"""
        x, y = pos
        # Check wall boundaries
        if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
            return False
        # Check snake body collision
        return pos not in snake_positions

    def get_distance(self, head, direction, snake_positions):
        x, y = head
        dx, dy = direction
        dist = 0
        while True:
            x += dx * BLOCK_SIZE
            y += dy * BLOCK_SIZE
            dist += 1
            if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT or (x, y) in snake_positions:
                return dist
            if dist > 100:  # max
                return 100

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()

    def store_transition(self, state, action, next_state, reward):
        self.memory.append(Transition(state, action, next_state, reward))

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.tensor(batch.state, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long).to(self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32).to(self.device)
        next_state_batch = torch.tensor(batch.next_state, dtype=torch.float32).to(self.device)
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0]
        # Compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # Compute loss
        loss = F.mse_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()

def draw_text(surface, text, color, x, y):
    textobj = FONT.render(text, 1, color)
    textrect = textobj.get_rect()
    textrect.topleft = (x, y)
    surface.blit(textobj, textrect)

class ControlMenu:
    def __init__(self, agent, game_state):
        self.agent = agent
        self.game_state = game_state
        self.root = tk.Tk()
        self.root.title("AI Snake Control Panel")
        self.root.geometry("300x200")

        # Pause/Resume button
        self.pause_button = tk.Button(self.root, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(pady=10)

        # Save Model button
        self.save_button = tk.Button(self.root, text="Save Model", command=self.save_model)
        self.save_button.pack(pady=10)

        # Load Model button
        self.load_button = tk.Button(self.root, text="Load Model", command=self.load_model)
        self.load_button.pack(pady=10)

        # Status label
        self.status_label = tk.Label(self.root, text="Game Running")
        self.status_label.pack(pady=10)

    def toggle_pause(self):
        self.game_state['paused'] = not self.game_state['paused']
        if self.game_state['paused']:
            self.pause_button.config(text="Resume")
            self.status_label.config(text="Game Paused")
        else:
            self.pause_button.config(text="Pause")
            self.status_label.config(text="Game Running")

    def save_model(self):
        try:
            self.agent.save_model('snake_dql.pth')
            messagebox.showinfo("Success", "Model saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        try:
            self.agent.load_model('snake_dql.pth')
            messagebox.showinfo("Success", "Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")

    def run(self):
        self.root.mainloop()

class NeuralNetworkVisualizer:
    def __init__(self, agent):
        self.agent = agent
        self.screen = pygame.display.set_mode((600, 400))
        pygame.display.set_caption('Neural Network Visualization')
        self.clock = pygame.time.Clock()
        self.running = True

    def draw_neuron(self, surface, x, y, activation, max_activation=1.0):
        # Color based on activation (green for positive, red for negative)
        if activation > 0:
            color = (0, int(255 * min(activation / max_activation, 1.0)), 0)
        else:
            color = (int(255 * min(-activation / max_activation, 1.0)), 0, 0)

        pygame.draw.circle(surface, color, (x, y), 20)
        pygame.draw.circle(surface, WHITE, (x, y), 20, 2)

        # Draw activation value
        font = pygame.font.SysFont(None, 20)
        text = font.render(f"{activation:.2f}", True, WHITE)
        text_rect = text.get_rect(center=(x, y))
        surface.blit(text, text_rect)

    def draw_layer(self, surface, layer_idx, activations, x_offset, y_offset, neuron_spacing):
        for i, activation in enumerate(activations):
            x = x_offset + layer_idx * 150
            y = y_offset + i * neuron_spacing
            self.draw_neuron(surface, x, y, activation)

    def visualize(self, state):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            self.screen.fill(BLACK)

            # Get activations from the neural network
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.agent.device)

            # Forward pass to get activations
            with torch.no_grad():
                x = F.relu(self.agent.policy_net.fc1(state_tensor))
                hidden1 = x.squeeze().cpu().numpy()
                x = F.relu(self.agent.policy_net.fc2(x))
                hidden2 = x.squeeze().cpu().numpy()
                output = self.agent.policy_net.fc3(x).squeeze().cpu().numpy()

            # Draw layers
            input_layer = state
            self.draw_layer(self.screen, 0, input_layer, 50, 50, 30)
            self.draw_layer(self.screen, 1, hidden1, 50, 50, 30)
            self.draw_layer(self.screen, 2, hidden2, 50, 50, 30)
            self.draw_layer(self.screen, 3, output, 50, 150, 60)

            # Draw connections (simplified)
            for i in range(len(hidden1)):
                for j in range(len(hidden2)):
                    pygame.draw.line(self.screen, WHITE, (200, 50 + i * 30), (350, 50 + j * 30), 1)

            for i in range(len(hidden2)):
                for j in range(len(output)):
                    pygame.draw.line(self.screen, WHITE, (350, 50 + i * 30), (500, 150 + j * 60), 1)

            # Draw layer labels
            font = pygame.font.SysFont(None, 24)
            labels = ["Input", "Hidden 1", "Hidden 2", "Output"]
            for i, label in enumerate(labels):
                text = font.render(label, True, WHITE)
                self.screen.blit(text, (50 + i * 150, 10))

            pygame.display.flip()
            self.clock.tick(10)  # Update at 10 FPS

    def visualize_once(self, state):
        """Update the visualization with a single frame"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        self.screen.fill(BLACK)

        # Get activations from the neural network
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.agent.device)

        # Forward pass to get activations
        with torch.no_grad():
            x = F.relu(self.agent.policy_net.fc1(state_tensor))
            hidden1 = x.squeeze().cpu().numpy()
            x = F.relu(self.agent.policy_net.fc2(x))
            hidden2 = x.squeeze().cpu().numpy()
            output = self.agent.policy_net.fc3(x).squeeze().cpu().numpy()

        # Draw layers
        input_layer = state
        self.draw_layer(self.screen, 0, input_layer, 50, 50, 30)
        self.draw_layer(self.screen, 1, hidden1, 50, 50, 30)
        self.draw_layer(self.screen, 2, hidden2, 50, 50, 30)
        self.draw_layer(self.screen, 3, output, 50, 150, 60)

        # Draw connections (simplified)
        for i in range(len(hidden1)):
            for j in range(len(hidden2)):
                pygame.draw.line(self.screen, WHITE, (200, 50 + i * 30), (350, 50 + j * 30), 1)

        for i in range(len(hidden2)):
            for j in range(len(output)):
                pygame.draw.line(self.screen, WHITE, (350, 50 + i * 30), (500, 150 + j * 60), 1)

        # Draw layer labels
        font = pygame.font.SysFont(None, 24)
        labels = ["Input", "Hidden 1", "Hidden 2", "Output"]
        for i, label in enumerate(labels):
            text = font.render(label, True, WHITE)
            self.screen.blit(text, (50 + i * 150, 10))

        pygame.display.flip()

    def stop(self):
        self.running = False

def main(train=False):
    if train:
        # Training mode with visualization
        pygame.init()
        clock = pygame.time.Clock()

        # Create a larger window that shows both game and neural network
        TOTAL_WIDTH = SCREEN_WIDTH + 600  # Reduced extra width to cut empty space on right
        TOTAL_HEIGHT = max(SCREEN_HEIGHT, 600)  # Max of game height and NN height
        screen = pygame.display.set_mode((TOTAL_WIDTH, TOTAL_HEIGHT))
        pygame.display.set_caption('AI Snake Training - Live Neural Network')

        agent = DQLAgent()
        num_episodes = 1000

        for episode in range(num_episodes):
            snake = Snake()
            food = Food()
            score = 0
            done = False
            steps = 0

            while not done:
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return

                # AI decision
                state = agent.get_state(snake, food)
                action = agent.select_action(state)
                directions = [UP, DOWN, LEFT, RIGHT]
                snake.turn(directions[action])

                # Move snake
                move_successful = snake.move()
                reward = -1
                if not move_successful:
                    reward = -10
                    done = True
                else:
                    if snake.get_head_position() == food.position:
                        snake.length += 1
                        score += 1
                        reward += 10
                        food.randomize_position(set(snake.positions))

                next_state = agent.get_state(snake, food)
                agent.store_transition(state, action, next_state, reward)
                agent.optimize_model()

                steps += 1
                if done or steps > 1000:  # Prevent infinite episodes
                    break

                # Clear screen
                screen.fill(BLACK)

                # Draw game on left side
                game_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
                game_surface.fill(BLACK)
                snake.draw(game_surface)
                food.draw(game_surface)

                # Draw training info on game surface
                draw_text(game_surface, f"Episode: {episode}", WHITE, 10, 10)
                draw_text(game_surface, f"Score: {score}", WHITE, 10, 40)
                draw_text(game_surface, f"Epsilon: {agent.epsilon:.3f}", WHITE, 10, 70)
                draw_text(game_surface, f"Steps: {steps}", WHITE, 10, 100)
                draw_text(game_surface, "TRAINING MODE", BLUE, SCREEN_WIDTH - 150, 10)
                draw_text(game_surface, "ESC: Stop | Live NN →", WHITE, 10, SCREEN_HEIGHT - 30)

                # Blit game surface to main screen
                screen.blit(game_surface, (0, 0))

                # Draw neural network visualization on right side
                nn_x_offset = SCREEN_WIDTH + 50

                # Get activations from the neural network
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(agent.device)
                with torch.no_grad():
                    x = F.relu(agent.policy_net.fc1(state_tensor))
                    hidden1 = x.squeeze().cpu().numpy()
                    x = F.relu(agent.policy_net.fc2(x))
                    hidden2 = x.squeeze().cpu().numpy()
                    output = agent.policy_net.fc3(x).squeeze().cpu().numpy()

                # Draw neural network layers
                def draw_neuron(x, y, activation, max_activation=1.0):
                    if activation > 0:
                        color = (0, int(255 * min(activation / max_activation, 1.0)), 0)
                    else:
                        color = (int(255 * min(-activation / max_activation, 1.0)), 0, 0)
                    pygame.draw.circle(screen, color, (x, y), 15)
                    pygame.draw.circle(screen, WHITE, (x, y), 15, 2)
                    font = pygame.font.SysFont(None, 16)
                    text = font.render(f"{activation:.1f}", True, WHITE)
                    text_rect = text.get_rect(center=(x, y))
                    screen.blit(text, text_rect)

                # Input layer (18 neurons)
                input_layer = state
                for i, activation in enumerate(input_layer):
                    x = nn_x_offset
                    y = 50 + i * 25
                    draw_neuron(x, y, activation)

                # Hidden layer 1 (128 neurons, show first 20)
                for i in range(min(20, len(hidden1))):
                    x = nn_x_offset + 120
                    y = 50 + i * 25
                    draw_neuron(x, y, hidden1[i])

                # Hidden layer 2 (128 neurons, show first 20)
                for i in range(min(20, len(hidden2))):
                    x = nn_x_offset + 240
                    y = 50 + i * 25
                    draw_neuron(x, y, hidden2[i])

                # Output layer (4 neurons)
                for i, activation in enumerate(output):
                    x = nn_x_offset + 360
                    y = 200 + i * 50
                    draw_neuron(x, y, activation)

                # Draw layer labels
                font = pygame.font.SysFont(None, 20)
                labels = ["Input", "Hidden1", "Hidden2", "Output"]
                for i, label in enumerate(labels):
                    text = font.render(label, True, WHITE)
                    screen.blit(text, (nn_x_offset + i * 120, 20))

                # Draw connections (simplified)
                for i in range(min(20, len(hidden1))):
                    for j in range(min(20, len(hidden2))):
                        pygame.draw.line(screen, (100, 100, 100),
                                       (nn_x_offset + 15, 50 + i * 25),
                                       (nn_x_offset + 105, 50 + j * 25), 1)

                for i in range(min(20, len(hidden2))):
                    for j in range(len(output)):
                        pygame.draw.line(screen, (100, 100, 100),
                                       (nn_x_offset + 135, 50 + i * 25),
                                       (nn_x_offset + 345, 200 + j * 50), 1)

                pygame.display.update()
                clock.tick(FPS)

            agent.decay_epsilon()
            if episode % agent.target_update == 0:
                agent.update_target_net()

            # Save model every 50 episodes
            if episode % 50 == 0 and episode > 0:
                agent.save_model('snake_dql.pth')
                print(f"Model saved at episode {episode}")

        agent.save_model('snake_dql.pth')
        print("Training complete. Model saved.")
        pygame.quit()
        return
    else:
        # Playing mode
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption('AI Snake Game')

        snake = Snake()
        food = Food()
        agent = DQLAgent()
        agent.epsilon = 0  # No exploration
        try:
            agent.load_model('snake_dql.pth')
            print("Model loaded.")
        except:
            print("No model found, using random.")

        # Game state for control menu
        game_state = {'paused': False}

        # Initialize control menu (will be updated in main loop)
        control_menu = ControlMenu(agent, game_state)

        # Initialize neural network visualizer
        visualizer = NeuralNetworkVisualizer(agent)
        current_state = agent.get_state(snake, food)

        score = 0
        game_started = True

        while True:
            screen.fill(BLACK)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    visualizer.stop()
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_p:  # Pause/Resume with P key
                        game_state['paused'] = not game_state['paused']
                    elif event.key == pygame.K_s:  # Save model with S key
                        try:
                            agent.save_model('snake_dql.pth')
                            print("Model saved successfully!")
                        except Exception as e:
                            print(f"Failed to save model: {str(e)}")
                    elif event.key == pygame.K_l:  # Load model with L key
                        try:
                            agent.load_model('snake_dql.pth')
                            print("Model loaded successfully!")
                        except Exception as e:
                            print(f"Failed to load model: {str(e)}")

            if game_started and not game_state['paused']:
                # AI makes decision
                current_state = agent.get_state(snake, food)
                action = agent.select_action(current_state)
                directions = [UP, DOWN, LEFT, RIGHT]
                best_direction = directions[action]
                snake.turn(best_direction)

                # Move snake and check for collisions
                move_successful = snake.move()

                if not move_successful:
                    # Collision detected (wall or self)
                    visualizer.stop()
                    game_over(screen, score)
                    return

                if snake.get_head_position() == food.position:
                    snake.length += 1
                    score += 1
                    food.randomize_position(set(snake.positions))

            snake.draw(screen)
            food.draw(screen)

            draw_text(screen, f"Score: {score}", WHITE, 10, 10)
            draw_text(screen, f"Length: {snake.length}", WHITE, 10, 40)
            draw_text(screen, "DQL Mode", BLUE, SCREEN_WIDTH - 100, 10)
            draw_text(screen, "P: Pause/Resume | S: Save | L: Load", WHITE, 10, SCREEN_HEIGHT - 30)

            if game_state['paused']:
                draw_text(screen, "PAUSED", RED, SCREEN_WIDTH // 2 - 50, SCREEN_HEIGHT // 2)

            pygame.display.update()

            # Update neural network visualization
            visualizer.visualize_once(current_state)

            clock.tick(FPS)

def game_over(screen, score):
    draw_text(screen, "Game Over", RED, SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 - 50)
    draw_text(screen, f"Final Score: {score}", WHITE, SCREEN_WIDTH // 2 - 80, SCREEN_HEIGHT // 2 - 10)
    draw_text(screen, "Press Q to Quit or C to Play Again", WHITE, SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2 + 30)
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_c:
                    main()

# Directions
UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)

if __name__ == '__main__':
    import sys
    train_mode = len(sys.argv) > 1 and sys.argv[1] == 'train'
    main(train=train_mode)
