import arenarl
import time

def simulate_snake():
    print("=" * 50)
    print("  ARENARL — SNAKE DEMO")
    print("=" * 50)
    print()

    # Create a 8x8 Snake board
    env = arenarl.make("Snake-v1", grid_size=8)
    state, info = env.reset(seed=123)
    
    print("Initial Snake Game:")
    print(env.render())
    print("\nSnake ('▣' head, '▢' body) needs to eat the Apple ('🍎').")
    print("Taking a few random steps...\n")
    
    # Take up to 15 random steps
    for step in range(1, 15):
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        
        action_name = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}[action]
        print(f"Step {step}: Action = {action_name:<5} | Reward = {reward:>5.2f} | Length = {info['snake_length']}")
        
        if terminated:
            print("\nGame Over! (Hit wall or self)")
            break
            
    print("\nFinal State:")
    print(env.render())
    
    print("\nMetrics:")
    print(env.get_metrics())

if __name__ == "__main__":
    simulate_snake()
