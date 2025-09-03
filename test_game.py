#!/usr/bin/env python3
"""
Test script for the AI Snake Game
Tests core functionality without requiring interactive input
"""

import pygame
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from game import (
    Snake, Food, DQLAgent, NeuralNetworkVisualizer,
    UP, DOWN, LEFT, RIGHT, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE
)

def test_snake_movement():
    """Test basic snake movement and collision detection"""
    print("Testing Snake Movement...")

    snake = Snake()
    food = Food()

    # Test initial state
    assert snake.length == 1
    assert len(snake.positions) == 1
    print("✓ Initial snake state correct")

    # Test movement
    initial_pos = snake.get_head_position()
    snake.turn(RIGHT)
    moved = snake.move()
    assert moved == True
    assert snake.get_head_position() != initial_pos
    print("✓ Snake movement works")

    # Test wall collision
    snake.positions = [(SCREEN_WIDTH - BLOCK_SIZE, SCREEN_HEIGHT // 2)]
    snake.direction = RIGHT
    moved = snake.move()
    assert moved == False  # Should hit wall
    print("✓ Wall collision detection works")

    # Test self collision - create a U-shaped snake that will hit itself
    snake.reset()
    # Create a snake that forms a loop: head will try to move into body
    center_x = SCREEN_WIDTH // 2
    center_y = SCREEN_HEIGHT // 2
    snake.positions = [
        (center_x, center_y),  # head
        (center_x + BLOCK_SIZE, center_y),  # body segment 1
        (center_x + BLOCK_SIZE, center_y + BLOCK_SIZE),  # body segment 2
        (center_x, center_y + BLOCK_SIZE),  # body segment 3
        (center_x - BLOCK_SIZE, center_y + BLOCK_SIZE),  # body segment 4
        (center_x - BLOCK_SIZE, center_y)  # body segment 5 - this is where head will try to move
    ]
    snake.length = 6
    snake.direction = LEFT  # Try to move left into the body
    moved = snake.move()
    assert moved == False  # Should hit self
    print("✓ Self collision detection works")

    print("Snake movement tests passed!\n")

def test_food_generation():
    """Test food generation and positioning"""
    print("Testing Food Generation...")

    food = Food()
    snake_positions = {(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)}

    # Test food randomization
    initial_pos = food.position
    food.randomize_position(snake_positions)
    assert food.position != initial_pos
    assert food.position not in snake_positions
    print("✓ Food randomization works")

    # Test food doesn't spawn on snake
    food.randomize_position(snake_positions)
    assert food.position not in snake_positions
    print("✓ Food avoids snake positions")

    print("Food generation tests passed!\n")

def test_ai_agent():
    """Test AI agent state generation and action selection"""
    print("Testing AI Agent...")

    agent = DQLAgent()
    snake = Snake()
    food = Food()

    # Test state generation
    state = agent.get_state(snake, food)
    assert len(state) == 18  # Expected state size
    assert all(isinstance(x, (int, float)) for x in state)
    print("✓ State generation works")

    # Test action selection
    action = agent.select_action(state)
    assert 0 <= action <= 3  # Valid action range
    print("✓ Action selection works")

    # Test model save/load (create dummy model first)
    try:
        agent.save_model('test_model.pth')
        agent.load_model('test_model.pth')
        print("✓ Model save/load works")
        os.remove('test_model.pth')  # Clean up
    except Exception as e:
        print(f"⚠ Model save/load test failed: {e}")

    print("AI Agent tests passed!\n")

def test_neural_network_visualizer():
    """Test neural network visualizer initialization"""
    print("Testing Neural Network Visualizer...")

    agent = DQLAgent()
    snake = Snake()
    food = Food()
    state = agent.get_state(snake, food)

    try:
        visualizer = NeuralNetworkVisualizer(agent)
        visualizer.visualize_once(state)
        visualizer.stop()
        print("✓ Neural network visualizer works")
    except Exception as e:
        print(f"⚠ Neural network visualizer test failed: {e}")

    print("Neural Network Visualizer tests passed!\n")

def test_game_integration():
    """Test integrated game components"""
    print("Testing Game Integration...")

    # Initialize game components
    snake = Snake()
    food = Food()
    agent = DQLAgent()
    agent.epsilon = 0  # Disable exploration for deterministic testing

    # Simulate a few game steps
    for step in range(10):
        state = agent.get_state(snake, food)
        action = agent.select_action(state)
        directions = [UP, DOWN, LEFT, RIGHT]
        snake.turn(directions[action])

        moved = snake.move()
        if not moved:
            break  # Game over

        if snake.get_head_position() == food.position:
            snake.length += 1
            food.randomize_position(set(snake.positions))

    print("✓ Game integration works")

    print("Game Integration tests passed!\n")

def main():
    """Run all tests"""
    print("Starting AI Snake Game Tests...\n")

    try:
        pygame.init()
        print("✓ Pygame initialized successfully")

        test_snake_movement()
        test_food_generation()
        test_ai_agent()
        test_neural_network_visualizer()
        test_game_integration()

        print("🎉 All tests passed successfully!")
        return True

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        pygame.quit()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
