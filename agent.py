import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from config import LEARNING_RATE, GAMMA, BATCH_SIZE, BUFFER_SIZE, EPSILON_START, EPSILON_END, EPSILON_DECAY, TARGET_UPDATE

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # Dueling streams
        self.value_stream = nn.Linear(128, 1)
        self.advantage_stream = nn.Linear(128, output_dim)
        
    def forward(self, state, mask=None):
        x = self.fc(state)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Q = V + (A - mean(A))
        q_vals = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        # Action Masking mechanism
        if mask is not None:
            # Set invalid actions to a very large negative number
            # Mask is 1 for valid, 0 for invalid.
            # Convert 0 to -inf
            inverted_mask = (1.0 - mask) * -1e9
            q_vals = q_vals + inverted_mask
            
        return q_vals

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done, mask, next_mask):
        self.buffer.append((state, action, reward, next_state, done, mask, next_mask))
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, mask, next_mask = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done), np.array(mask), np.array(next_mask))
        
    def __len__(self):
        return len(self.buffer)

class D3QNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayBuffer(BUFFER_SIZE)
        
        self.steps_done = 0
        self.epsilon = EPSILON_START

    def select_action(self, state, mask):
        self.steps_done += 1
        # Epsilon is now updated explicitly in train.py for linear decay per episode
        # self.epsilon = ... 
                       
        if random.random() < self.epsilon:
            # Select random VALID action
            valid_indices = np.where(mask == 1)[0]
            if len(valid_indices) == 0:
                return 0 # Fallback if everything masked (shouldn't happen)
            return np.random.choice(valid_indices)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_t, mask_t)
                return q_values.argmax().item()

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        state, action, reward, next_state, done, mask, next_mask = self.memory.sample(BATCH_SIZE)
        
        state_t = torch.FloatTensor(state).to(self.device)
        action_t = torch.LongTensor(action).unsqueeze(1).to(self.device)
        reward_t = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state_t = torch.FloatTensor(next_state).to(self.device)
        done_t = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        # mask is NOT used for policy loss calc usually, but next_mask IS used for target
        
        # Current Q
        q_values = self.policy_net(state_t) # No mask, or mask? Usually just gathering the taken action.
        q_value = q_values.gather(1, action_t)
        
        # Target Q (Double DQN)
        # 1. Select best action using Policy Net (with mask!)
        next_mask_t = torch.FloatTensor(next_mask).to(self.device)
        next_q_policy = self.policy_net(next_state_t, next_mask_t)
        next_action = next_q_policy.argmax(1, keepdim=True)
        
        # 2. Evaluate that action using Target Net
        next_q_target = self.target_net(next_state_t, next_mask_t) # Mask applies consistent penalty
        next_q_value = next_q_target.gather(1, next_action)
        
        expected_q_value = reward_t + (1 - done_t) * GAMMA * next_q_value
        
        loss = nn.MSELoss()(q_value, expected_q_value)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.steps_done % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
