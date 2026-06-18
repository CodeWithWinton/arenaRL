import os
import arenarl
import numpy as np
from pathlib import Path
from arenarl.utils.plotting import plot_learning_curve

def q_learning(env_id="GridWorld-v1", episodes=500, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01):
    env = arenarl.make(env_id, grid_size=6, num_obstacles=5, random_obstacles=False)
    
    # Q-table
    q_table = np.zeros((env.observation_space.n, env.action_space.n))
    
    print(f"Training Q-learning agent on {env_id} for {episodes} episodes...")
    for episode in range(episodes):
        state, info = env.reset(seed=42) # Use same seed to keep obstacles fixed during training
        done = False
        
        while not done:
            # Epsilon-greedy action
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state])
                
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update Q-table
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            td_error = td_target - q_table[state][action]
            q_table[state][action] += alpha * td_error
            
            state = next_state
            
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
    return env

if __name__ == "__main__":
    # Ensure sample_data directory exists
    sample_dir = Path("sample_data")
    sample_dir.mkdir(exist_ok=True)
    
    # Run Q-learning to generate meaningful data
    trained_env = q_learning(episodes=1000)
    
    # Export metrics
    json_path = sample_dir / "gridworld_metrics.json"
    csv_path = sample_dir / "gridworld_metrics.csv"
    plot_path = sample_dir / "gridworld_learning_curve.png"
    
    trained_env.export_data(str(json_path))
    trained_env.export_data(str(csv_path))
    print(f"Exported metrics to {json_path} and {csv_path}")
    
    # Plot learning curve
    metrics = trained_env.get_metrics()
    plot_learning_curve(metrics, save_path=str(plot_path), title="Q-Learning on GridWorld-v1")
