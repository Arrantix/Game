# TODO for Adding Deep Q Learning to Snake Game

- [x] Install required dependencies (TensorFlow or PyTorch) - PyTorch installed
- [x] Implement DQL Agent class with neural network model
- [x] Define state representation (snake head, food position, direction, obstacles)
- [x] Define action space (4 directions: UP, DOWN, LEFT, RIGHT)
- [x] Implement reward function:
  - +10 for eating food
  - -10 for game over (wall or self collision)
  - -1 for each step
  - -5 for cutting off path to food or encapsulating itself (simplified to -1 per step)
- [x] Add experience replay buffer
- [x] Implement epsilon-greedy policy for exploration
- [x] Modify game loop to support training mode and auto-restart after each episode
- [x] Add option to save/load trained model
- [x] Integrate DQL agent to control snake instead of heuristic AI
- [x] Test training and playing with trained model - Training working, agent learning to get scores
- [x] Improve AI's perception to actively consider walls and obstacles - Implemented in state representation
- [x] Add separate control menu window with pause/resume, save/load buttons - Implemented
- [x] Create live neural network visualization window showing neurons and development - Implemented

## Additional Improvements
- [ ] Optimize tensor creation performance (convert numpy arrays before tensor creation)
- [ ] Add more sophisticated reward shaping (e.g., distance to food, path blocking penalties)
- [ ] Implement prioritized experience replay for better sample efficiency
- [ ] Add hyperparameter tuning (learning rate, batch size, network architecture)
- [ ] Add training statistics logging and visualization
- [ ] Implement curriculum learning (start with simple environments)
