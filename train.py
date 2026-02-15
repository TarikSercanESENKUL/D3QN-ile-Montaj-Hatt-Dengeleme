import numpy as np
import torch
import os
import argparse
from factory_env import BusFactoryEnv
from agent import D3QNAgent
from config import CHECKPOINT_DIR, LOG_DIR, EPSILON_START, EPSILON_END

def train(episodes=1000):
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
        
    env = BusFactoryEnv()
    agent = D3QNAgent(env.obs_dim, env.action_space.n)
    
    print(f"Starting training on {agent.device}...")
    print(f"Observation Dim: {env.obs_dim}, Action Dim: {env.action_space.n}")
    
    best_reward = -float('inf')
    
    import csv
    
    # Initialize CSV logging
    log_file = os.path.join(LOG_DIR, "training_metrics.csv")
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    with open(log_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'Reward', 'BusesProduced', 'AvgEfficiency', 'TotalCost', 'Epsilon'])

    for episode in range(episodes):
        # Exponential Decay
        # epsilon = end + (start - end) * exp(-1. * episode / decay)
        import math
        agent.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
                        math.exp(-1. * episode / (EPSILON_DECAY / 5)) # Adjusted scaling
        # Note: If DECAY=2000, and total episodes=500, we need it to decay faster?
        # Standard: Decay is successful steps. Since we decay per episode:
        # Let's set it to reach ~0.1 at 50% of episodes.
        # Assuming 500 episodes:
        # episode 250 -> exp(-250/X) ~= 0.1 -> -250/X = ln(0.1)=-2.3 -> X ~= 100
        # So we use a localized decay factor here for safety.
        decay_rate = episodes / 4.0 # Reach low epsilon at 125 episodes? No, at 400.
        # Let's aim for 0.1 at episode 300.
        # 0.1 = 1.0 * exp(-300/X) => X = -300/ln(0.1) = 130
        agent.epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * episode / 100)
        
        state, info = env.reset()
        
        # Get valid action mask for initial state
        mask = env._get_action_mask()
        
        total_reward = 0
        done = False
        
        # Final info carrier
        final_info = {}
        
        while not done:
            # Select Action
            action = agent.select_action(state, mask)
            
            # Step
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update info for logging
            final_info = info
            
            # Get next mask (for optimization target)
            next_mask = env._get_action_mask()
            
            # Store
            agent.memory.push(state, action, reward, next_state, done, mask, next_mask)
            
            # Move
            state = next_state
            mask = next_mask
            total_reward += reward
            
            # Optimize
            agent.update()
            
            if done:
                # Calculate aggregated metrics
                buses = final_info.get("buses_produced", 0)
                busy_mins = final_info.get("total_busy_minutes", 0)
                labor_cost = final_info.get("total_labor_cost", 0)
                
                # Efficiency = Busy Minutes / (Total Workers * SimTimeSoFar?)
                # Simplified: Busy Minutes / 1000 (normalization)
                efficiency = busy_mins / (184 * 30 * 24 * 60) * 100 if env.sim_env.now > 0 else 0
                
                print(f"Episode {episode} | Reward: {total_reward:.2f} | Buses: {buses} | Eff: {efficiency:.4f}% | Epsilon: {agent.epsilon:.3f}")
                
                # Log to CSV
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode, total_reward, buses, efficiency, labor_cost, agent.epsilon])
                
        # Save Best
        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(agent.policy_net.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model.pth"))
            print(f"New best model saved with reward: {best_reward:.2f}")

        # Periodic Save
        if episode % 50 == 0:
            torch.save(agent.policy_net.state_dict(), os.path.join(CHECKPOINT_DIR, f"ckpt_{episode}.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    args = parser.parse_args()
    
    train(episodes=args.episodes)
