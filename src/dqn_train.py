#def a dqn agent and use the gymasium api to train the model
#LunarLander-v3
#   Action Space  Discrete(4)
#   0: do nothing
#   1: fire left orientation engine
#   2: fire main engine
#   3: fire right orientation engine
# Observation Space Box([ -2.5 -2.5 -10. -10. -6.2831855 -10. -0. -0. ]
# the coordinates of the lander in x & y, 
# its linear velocities in x & y, its angle, its angular velocity, 
# and two booleans that represent whether each leg is in contact with the ground or not.
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import torch.optim as optim
import random
import os
import gymnasium as gym
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class DQNNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNNet,self).__init__()
        hidden_size = 128
        self.linear1 = nn.Linear(state_size, 2*hidden_size)
        self.linear2 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.linear3 = nn.Linear(2*hidden_size, hidden_size)
        self.linear4 = nn.Linear(hidden_size, action_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.1)

    def forward(self, state_size):
        x = F.relu(self.linear1(state_size))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = self.linear4(x)
        return x

class PrioritizedBuffer:
    def __init__(self, capacity, alpah = 0.6, beta = 0.4, beta_incre = 0.001):
        self.capacity = capacity
        self.alpah  = alpah
        self.beta = beta
        self.beta_incre = beta_incre
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.pos = 0
        self.max_priority = 1.0

    def __len__(self):
        return len(self.buffer)
    
    def push(self, state, action, reward, nextstate, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, nextstate, done))
        else:
            self.buffer[self.pos] = (state, action, reward, nextstate, done)
            
        self.priorities[self.pos] = self.max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
            
        prob = priorities ** self.alpah
        prob /= prob.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=prob)#smaple according to prob
        samples = [self.buffer[i] for i in indices]
        self.beta  = min(1, self.beta+self.beta_incre)
        weights = (len(self.buffer) * prob[indices] ** (-self.beta))
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5
            self.max_priority = max(self.max_priority, priority)
    
class Agent:
    def __init__(self, state_size, action_size, log_dir=None):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if log_dir is None:
           log_dir = f"runs/lunar_lander_{datetime.now().strftime('%Y%m%d_%H:%M')}"
        self.writer = SummaryWriter(log_dir=log_dir)
        
        self.q_net = DQNNet(state_size, action_size).to(self.device)

        self.target_net = DQNNet(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.AdamW(self.q_net.parameters(), lr=5e-4, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=10000)

        self.memory = PrioritizedBuffer(100000)

        self.batch_size = 120
        self.gamma = 0.99
        self.tau = 0.005  # 软更新系数
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.998
        self.learning_steps = 0
        self.update_frequency = 4

        self.losses = deque(maxlen=100)
        self.q_values = deque(maxlen=100)
        self.td_errors = deque(maxlen=100)
        self.grad_norms = deque(maxlen=100)

        self.episode_rewards = []

    def act(self, state, training=True):
        if training and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return np.argmax(q_values.cpu().data.numpy())
    
    def remember(self, state, action, reward, nextstate, done):
        self.memory.push(state, action, reward, nextstate, done)

    def soft_update_target_net(self):
        for target_param, local_param in zip(self.target_net.parameters(), 
                                           self.q_net.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)
            
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return None, None
        samples, indices, weights = self.memory.sample(self.batch_size)
        states, actions, rewards, nextstates, dones = map(np.stack, zip(*samples))
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        nextstates = torch.FloatTensor(nextstates).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
    
        current_q_values = self.q_net(states).gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_actions = self.q_net(nextstates).max(1)[1]
            next_q_values = self.target_net(nextstates).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))

        td_errors = (target_q_values - current_q_values).abs().squeeze().detach().cpu().numpy()
        loss = (weights * F.smooth_l1_loss(current_q_values.squeeze(), target_q_values.squeeze(), reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        
        self.optimizer.step()
        self.scheduler.step()
        self.memory.update_priorities(indices, td_errors)

        self.soft_update_target_net()

        self.losses.append(loss.item())
        self.q_values.append(current_q_values.mean().item())
        self.td_errors.append(td_errors.mean())
        self.grad_norms.append(grad_norm.item())
        return loss.item(), current_q_values.mean().item()


    def save_model(self, filename):
      save_dict = {
        'q_net_state_dict': self.q_net.state_dict(),
        'target_net_state_dict': self.target_net.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'scheduler_state_dict': self.scheduler.state_dict(),
        'epsilon': float(self.epsilon),
        'episode_rewards': [float(r) for r in self.episode_rewards],
        'training_steps': self.learning_steps
      }
      torch.save(save_dict, filename)
      print(f"模型已保存: {filename}")

    def load_model(self, filename):
     if not os.path.exists(filename):
        print(f"模型文件不存在: {filename}")
        return False
    
     try:
        checkpoint = torch.load(filename, map_location=self.device, weights_only=True)
     except Exception as e:
        print(f"加载失败: {e}")
        try:
            print("尝试使用weights_only=False加载...")
            checkpoint = torch.load(filename, map_location=self.device, weights_only=False)
        except Exception as e2:
            print(f"加载模型失败: {e2}")
            return False
    
     try:
        self.q_net.load_state_dict(checkpoint['q_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.learning_steps = checkpoint.get('training_steps', 0)
        
        print(f"模型加载成功: {filename}")
        return True
     except Exception as e:
        print(f"加载模型状态失败: {e}")
        return False
    
    def log_network_parameters(self, episode):
        """记录神经网络参数到TensorBoard"""
        with torch.no_grad():  # 不计算梯度，只记录参数
        # 遍历网络的所有参数
            for name, param in self.q_net.named_parameters():
                if param.requires_grad:  # 只记录需要梯度的参数
                    if 'weight' in name:
                    # 记录权重直方图
                        self.writer.add_histogram(f'Parameters/Weights/{name}', param, episode)
                    # 记录权重统计量
                        self.writer.add_scalar(f'Parameters/Weights/{name}_mean', param.mean(), episode)
                        self.writer.add_scalar(f'Parameters/Weights/{name}_std', param.std(), episode)
                        self.writer.add_scalar(f'Parameters/Weights/{name}_max', param.max(), episode)
                        self.writer.add_scalar(f'Parameters/Weights/{name}_min', param.min(), episode)
                    
                    elif 'bias' in name:
                    # 记录偏置直方图
                        self.writer.add_histogram(f'Parameters/Biases/{name}', param, episode)
                    # 记录偏置统计量
                        self.writer.add_scalar(f'Parameters/Biases/{name}_mean', param.mean(), episode)
                        self.writer.add_scalar(f'Parameters/Biases/{name}_std', param.std(), episode)

    def log_gradients(self, episode):
        """记录梯度信息"""
        for name, param in self.q_net.named_parameters():
            if param.grad is not None:  # 只有在训练后才有梯度
                if 'weight' in name:
                    self.writer.add_histogram(f'Gradients/Weights/{name}', param.grad, episode)
                    self.writer.add_scalar(f'Gradients/Weights/{name}_mean', param.grad.mean(), episode)
                elif 'bias' in name:
                    self.writer.add_histogram(f'Gradients/Biases/{name}', param.grad, episode)
                    self.writer.add_scalar(f'Gradients/Biases/{name}_mean', param.grad.mean(), episode)

    def TensorBoard_Analysis(self, episode, total_reward, steps, loss, avg_q_value):
        self.log_gradients(episode)
        self.log_network_parameters(episode)
        self.writer.add_scalar('Training/Episode Reward', total_reward, episode)
        self.writer.add_scalar('Training/Episode Length', steps, episode)
        self.writer.add_scalar('Training/Epsilon', self.epsilon, episode)

        if loss is not None:
            self.writer.add_scalar('Training/Loss', loss, episode)
            self.writer.add_scalar('Training/Average Q Value', avg_q_value, episode)
            self.writer.add_scalar('Training/Learning Rate', 
                                 self.scheduler.get_last_lr()[0], episode)
        
        if episode % 10 == 0 and len(self.losses) > 0:
            self.writer.add_scalar('Metrics/Average Loss', np.mean(self.losses), episode)
            self.writer.add_scalar('Metrics/Average Q Value', np.mean(self.q_values), episode)
            self.writer.add_scalar('Metrics/Average TD Error', np.mean(self.td_errors), episode)
            self.writer.add_scalar('Metrics/Average Grad Norm', np.mean(self.grad_norms), episode)
            self.writer.add_scalar('Metrics/Buffer Size', len(self.memory), episode)

        if len(self.episode_rewards) >= 100:
            moving_avg = np.mean(self.episode_rewards[-100:])
            self.writer.add_scalar('Training/Moving Average (100 episodes)', moving_avg, episode)

    def close(self):
        self.writer.close()
 
    

def train_Agent():
    #build env
    env = gym.make('LunarLander-v3')
    validation_env = gym.make('LunarLander-v3')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = Agent(state_size, action_size)
    max_episodes = 1500
    target_score = 200
    consecutive_success = 0
    required_consecutive = 10

    start_time = time.time()
    for episode in range(max_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_losses = []
        episode_q_values = []
        
        while True:
            action = agent.act(state)
            nextstate, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            agent.remember(state, action, reward, nextstate, done)
            
            if steps % agent.update_frequency == 0 and len(agent.memory) >= agent.batch_size:
                loss, q_value = agent.train()
                if loss is not None:
                    episode_losses.append(loss)
                    episode_q_values.append(q_value)
            
            state = nextstate
            total_reward += reward
            steps += 1
            
            if done:
                break

        agent.update_epsilon()
        agent.episode_rewards.append(total_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0
        avg_q_value = np.mean(episode_q_values) if episode_q_values else 0

        agent.TensorBoard_Analysis(episode, total_reward, steps, avg_loss, avg_q_value)#tensorboard
        if episode % 10 == 0:
            moving_avg = np.mean(agent.episode_rewards[-100:]) if len(agent.episode_rewards) >= 100 else np.mean(agent.episode_rewards)
            elapsed_time = time.time() - start_time
            print(f'Episode {episode:4d} | '
                  f'Reward: {total_reward:7.2f} | '
                  f'Moving Avg: {moving_avg:7.2f} | '
                  f'Epsilon: {agent.epsilon:.3f} | '
                  f'Loss: {avg_loss:.4f} | '
                  f'Time: {elapsed_time:.1f}s')
        
        if len(agent.episode_rewards) >= 100:
            recent_avg = np.mean(agent.episode_rewards[-100:])
            if recent_avg >= target_score:
                consecutive_success += 1
                if consecutive_success >= required_consecutive:
                    print(f'\n训练完成! 在 {episode} 回合达到目标分数!')
                    print(f'最近100回合平均分数: {recent_avg:.2f}')
                    agent.save_model('model.pth')
                    break
            else:
                consecutive_success = 0
    total_time = time.time() - start_time
    print(f'\n训练总时间: {total_time:.1f} 秒')
    agent.close()
    env.close()
    validation_env.close()
    
    return agent

def test_agent(render=True, model_path='model.pth'):
    env = gym.make('LunarLander-v3', render_mode='human' if render else None)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = Agent(state_size, action_size)
    
    # 尝试加载模型，如果失败就使用新智能体
    if os.path.exists(model_path):
        success = agent.load_model(model_path)
        if not success:
            print("模型加载失败，使用新初始化的智能体")
    else:
        print("未找到模型文件，使用新初始化的智能体")
    
    agent.epsilon = 0.01  #测试时使用很小的探索率
    
    test_episodes = 5
    test_scores = []
    
    for episode in range(test_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action = agent.act(state, training=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        test_scores.append(total_reward)
        print(f'测试回合 {episode + 1}: 分数 = {total_reward:.2f}')
    
    avg_score = np.mean(test_scores)
    print(f'\n测试结果: 平均分数 = {avg_score:.2f} (±{np.std(test_scores):.2f})')
    
    env.close()
    
    return test_scores
if __name__ == "__main__":
    print("=" * 60)
    print("           LunarLander DQN 训练开始")
    print("=" * 60)
    
    trained_agent = train_Agent()

    print("\n" + "=" * 60)
    print("           开始测试训练好的智能体")
    print("=" * 60)
    
    test_scores = test_agent(render=True)
