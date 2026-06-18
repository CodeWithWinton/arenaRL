import arenarl
import time

def simulate_maze():
    print("=" * 50)
    print("  ARENARL — MAZE NAVIGATION DEMO")
    print("=" * 50)
    print()

    # Create a 9x9 maze
    env = arenarl.make("Maze-v1", maze_size=9)
    state, info = env.reset(seed=42)
    
    print("Initial Maze:")
    print(env.render())
    print("\nAgent ('A') needs to reach the Exit ('E').")
    print("Let's take a few random steps to see what happens...\n")
    
    # Take 10 random steps
    for step in range(1, 11):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        
        action_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}[action]
        print(f"Step {step}: Action = {action_name:<5} | Reward = {reward:>5.2f} | Position = {info['agent_pos']}")
        
        if terminated:
            print("\nGoal Reached!")
            break
            
    print("\nMaze after 10 random steps:")
    print(env.render())
    
    print("\nMetrics so far:")
    print(env.get_metrics())

if __name__ == "__main__":
    simulate_maze()
