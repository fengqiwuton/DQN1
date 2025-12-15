import gymnasium as gym
import numpy as np
from collections import defaultdict
from gymnasium import spaces
from typing import Tuple, Optional
import random
from gymnasium.error import DependencyNotInstalled
import pygame
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# develop a simple maze and train an agent to solve the problem
VIEWPORT_W = 400
VIEWPORT_H = 400

class mazeEnv(gym.Env):
    """
    action space: 0:up 1:down 2:left 3:right
    observation space: 相对坐标 + 局部迷宫拓扑
    """
    metadata = {'render_modes':['human', 'rgb_array'],'render_fps':4}
    
    def __init__(self, 
                 mazesize:Tuple[int,int]=(7,7),
                 render_mode:Optional[str] = None,
                 view_range:int=5):
        super().__init__()
        self.mazesize = mazesize
        self.row, self.col = mazesize
        self.view_range = view_range  # 智能体能看到周围多远的范围
        
        # 动作空间：4个方向
        self.action_space = spaces.Discrete(4)
        
        # 状态空间：
        # 1. 相对坐标 (2维)
        # 2. 局部迷宫拓扑 (view_range*2+1)^2 维
        local_grid_size = (view_range*2 + 1) ** 2
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(2 + local_grid_size,),  # 相对坐标 + 局部迷宫
            dtype=np.float32
        )
        
        self.startpos = (1, 1)
        self.goalpos = (self.row-2, self.col-2)
        self.human = list(self.startpos)
        self.render_mode = render_mode
        self.window: pygame.Surface = None
        self.clock = None
        self.step_count = 0
        self.max_steps = self.row * self.col * 2
        self.create_maze()
        
        # 颜色定义
        self.color = {
            'background': (255, 255, 255),
            'obstacle': (100, 100, 100),
            'path': (255, 255, 255),
            'start': (144, 238, 144),
            'goal': (240, 128, 128),
            'human': (70, 130, 180)
        }

    def create_maze(self):
        # 0:obstacle, 1:path
        maze = [[0 for _ in range(self.col)] for _ in range(self.row)]
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        
        def dfs(x, y):
            maze[y][x] = 1  # 标记为路径
            random.shuffle(directions)  # 随机选择方向
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.col - 1 and 0 < ny < self.row - 1 and maze[ny][nx] == 0:
                    maze[y + dy // 2][x + dx // 2] = 1  # 打通墙壁
                    dfs(nx, ny)
        
        dfs(1, 1)
        self.maze = maze
        
        # 确保起点和终点是通路
        self.maze[self.startpos[0]][self.startpos[1]] = 1
        self.maze[self.goalpos[0]][self.goalpos[1]] = 1

    def get_local_maze_with_position(self):
        """获取带位置信息的局部迷宫（用于调试）"""
        local_maze = []
        for dx in range(-self.view_range, self.view_range + 1):
            row = []
            for dy in range(-self.view_range, self.view_range + 1):
                nx, ny = self.human[0] + dx, self.human[1] + dy
                
                if 0 <= nx < self.row and 0 <= ny < self.col:
                    if (nx, ny) == (self.human[0], self.human[1]):
                        row.append('A')  # Agent
                    elif (nx, ny) == self.goalpos:
                        row.append('G')  # Goal
                    elif self.maze[nx][ny] == 0:
                        row.append('X')  # Wall
                    else:
                        row.append(' ')  # Path
                else:
                    row.append('B')  # Border
            local_maze.append(row)
        return local_maze
    
    def get_observation(self):
        """获取状态观察：相对坐标 + 局部迷宫拓扑"""
        # 1. 相对坐标：当前位置到目标位置的相对坐标（归一化）
        rel_x = (self.goalpos[0] - self.human[0]) / self.row
        rel_y = (self.goalpos[1] - self.human[1]) / self.col
        
        # 2. 局部迷宫拓扑
        local_grid = []
        for dx in range(-self.view_range, self.view_range + 1):
            for dy in range(-self.view_range, self.view_range + 1):
                nx, ny = self.human[0] + dx, self.human[1] + dy
                
                if 0 <= nx < self.row and 0 <= ny < self.col:
                    # 在边界内
                    cell_type = self.maze[nx][ny]
                else:
                    # 超出边界，视为障碍物
                    cell_type = 0
                
                local_grid.append(cell_type)
        
        # 组合成状态向量
        observation = np.array([rel_x, rel_y] + local_grid, dtype=np.float32)
        return observation


    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # 重新生成迷宫（可选）
        self.create_maze()
        
        self.human = list(self.startpos)
        self.step_count = 0
        
        observation = self.get_observation()
        info = {
            'position': tuple(self.human),
            'goal': self.goalpos,
            'local_maze': self.get_local_maze_with_position()
        }
        
        if self.render_mode == 'human':
            self.render()
            
        return observation, info
    
    def step(self, action:int):
        self.step_count += 1
        old_pos = self.human.copy()
    
        # 执行动作
        if action == 0:  # 上
            self.human[0] = max(0, self.human[0] - 1)
        elif action == 1:  # 下
            self.human[0] = min(self.row - 1, self.human[0] + 1)
        elif action == 2:  # 左
            self.human[1] = max(0, self.human[1] - 1)
        elif action == 3:  # 右
            self.human[1] = min(self.col - 1, self.human[1] + 1)
    
        # 检查是否撞墙
        hit_wall = False
        if self.maze[self.human[0]][self.human[1]] == 0:
            self.human = old_pos  # 退回原位
            hit_wall = True
    
        # 计算奖励
        reward = 0
        terminated = False
        truncated = False
    
        # 到达目标
        if (self.human[0] == self.goalpos[0] and self.human[1] == self.goalpos[1]):
            reward = 100.0  # 大幅增加成功奖励
            terminated = True
        else:
            # 距离奖励：鼓励靠近目标
            old_distance = np.linalg.norm(np.array(old_pos) - np.array(self.goalpos))
            new_distance = np.linalg.norm(np.array(self.human) - np.array(self.goalpos))
            distance_reward = (old_distance - new_distance) * 5.0  # 增加权重
        
            # 步数惩罚（减少）
            step_penalty = -0.01  # 减少步数惩罚
        
            # 撞墙惩罚
            wall_penalty = -0.5 if hit_wall else 0.0  # 减少撞墙惩罚
        
            reward = distance_reward + step_penalty + wall_penalty
    
        # 检查是否超时
        truncated = self.step_count >= self.max_steps
    
        # 获取新状态
        observation = self.get_observation()
    
        info = {
            'position': tuple(self.human),
            'goal': self.goalpos,
            'old_position': tuple(old_pos),
            'hit_wall': hit_wall,
            'distance_to_goal': np.linalg.norm(np.array(self.human) - np.array(self.goalpos)),
            'local_maze': self.get_local_maze_with_position(),
            'step': self.step_count
        }
    
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[box2d]"`'
            ) from e
        
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))   
        
        if self.clock is None:
            self.clock = pygame.time.Clock()
        
        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        surf.fill(self.color['background'])
        
        # 计算单元格大小
        cell_width = VIEWPORT_W / self.col
        cell_height = VIEWPORT_H / self.row
        
        # 绘制迷宫
        for i in range(self.row):
            for j in range(self.col):
                x = j * cell_width
                y = i * cell_height
                
                if (i, j) == tuple(self.startpos):
                    color = self.color['start']
                elif (i, j) == self.goalpos:
                    color = self.color['goal']
                elif self.maze[i][j] == 0:
                    color = self.color['obstacle']
                else:
                    color = self.color['path']
                
                pygame.draw.rect(surf, color, (x, y, cell_width, cell_height))
                
                # 绘制局部视野范围
                if abs(i - self.human[0]) <= self.view_range and abs(j - self.human[1]) <= self.view_range:
                    pygame.draw.rect(surf, (255, 255, 200, 100), 
                                   (x, y, cell_width, cell_height), 1)
        
        # 绘制智能体
        human_x = self.human[1] * cell_width + cell_width // 2
        human_y = self.human[0] * cell_height + cell_height // 2
        pygame.draw.circle(surf, self.color['human'], 
                          (int(human_x), int(human_y)), 
                          min(cell_height, cell_width) // 3)
        
        # 绘制视野方向
        # 可以在智能体周围绘制一个小箭头表示方向
        
        if self.render_mode == 'human':
            self.window.blit(surf, (0, 0))
            
            # 添加状态信息显示
            font = pygame.font.Font(None, 24)
            info_text = f"Position: {self.human}  Goal: {self.goalpos}  Steps: {self.step_count}"
            text_surface = font.render(info_text, True, (0, 0, 0))
            self.window.blit(text_surface, (10, 10))
            
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()
            
        elif self.render_mode == 'rgb_array':
            rgb_array = pygame.surfarray.array3d(surf)
            return np.transpose(rgb_array, (1, 0, 2))
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

class ReplayMemory:
    def __init__(self, batch_size, max_size=10000):
        self.state_cap = []
        self.action_cap = []
        self.reward_cap = []
        self.value_cap = []
        self.done_cap = []
        self.log_prob_cap = []
        self.batch_size = batch_size
        self.max_size = max_size

    def add_memo(self, state, action, reward, value, done, log_prob):
        self.state_cap.append(state)
        self.action_cap.append(action)
        self.reward_cap.append(reward)
        self.value_cap.append(value)
        self.done_cap.append(done)
        self.log_prob_cap.append(log_prob)
        
        if len(self.state_cap) > self.max_size:
            self.state_cap.pop(0)
            self.action_cap.pop(0)
            self.reward_cap.pop(0)
            self.value_cap.pop(0)
            self.done_cap.pop(0)
            self.log_prob_cap.pop(0)

    def get_all_data(self):
        """获取所有数据用于PPO更新"""
        return {
            'states': np.array(self.state_cap),
            'actions': np.array(self.action_cap),
            'rewards': np.array(self.reward_cap),
            'values': np.array(self.value_cap),
            'dones': np.array(self.done_cap),
            'log_probs': np.array(self.log_prob_cap)
        }

    def clear_memo(self):
        self.state_cap.clear()
        self.action_cap.clear()
        self.reward_cap.clear()
        self.value_cap.clear()
        self.done_cap.clear()
        self.log_prob_cap.clear()

class PPOAgent:
    def __init__(self, state_dim, action_dim, batch_size=64):
        self.lr_actor = 3e-4
        self.lr_critic = 3e-4
        self.gamma = 0.99
        self.lamb = 0.95  # GAE lambda
        self.epoch = 10   # 优化epoch数
        self.clip_range = 0.2  # PPO clip范围
        self.batch_size = batch_size
        self.entropy_coef = 0.01

        # 网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        
        # 优化器
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.lr_critic)
        
        # 经验缓冲区
        self.replay_buffer = ReplayMemory(batch_size)

    def get_action(self, state):
        """选择动作并返回动作、log概率和状态价值"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            action_probs = self.actor(state_tensor)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state_tensor)
        
        return action.item(), log_prob.item(), value.item()

    def compute_gae(self, rewards, values, dones):
        """计算GAE优势函数和回报"""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        gae = 0
        next_value = 0  # 最后一个时间步之后的价值为0
        
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = 0 if dones[t] else values[t]
            else:
                next_value = values[t + 1] if not dones[t] else 0
            
            # TD误差
            delta = rewards[t] + self.gamma * next_value - values[t]
            
            # GAE
            gae = delta + self.gamma * self.lamb * (1 - dones[t]) * gae
            advantages[t] = gae
            
            # 计算回报
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns

    def compute_actor_critic_output(self, states, actions):
        """计算策略和价值的输出"""
        states_tensor = torch.FloatTensor(states).to(device)
        actions_tensor = torch.LongTensor(actions).to(device)
        
        # 策略输出
        action_probs = self.actor(states_tensor)
        dist = torch.distributions.Categorical(action_probs)
        log_probs = dist.log_prob(actions_tensor)
        entropy = dist.entropy()
        
        # 价值输出
        values = self.critic(states_tensor)
        
        return {
            'log_probs': log_probs,
            'entropy': entropy,
            'values': values,
            'dist': dist
        }

    def normalize_advantages(self, advantages):
        """归一化优势函数"""
        return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    def update(self):
        """执行PPO更新"""
        # 1. 收集所有数据
        data = self.replay_buffer.get_all_data()
        n_samples = len(data['states'])
        
        if n_samples < self.batch_size:
            return
        
        # 2. 计算旧策略的log prob和回报
        with torch.no_grad():
            # 计算旧的价值
            states_tensor = torch.FloatTensor(data['states']).to(device)
            old_values = self.critic(states_tensor).squeeze().cpu().numpy()
            
            # 计算GAE和回报
            advantages, returns = self.compute_gae(
                data['rewards'], 
                old_values, 
                data['dones']
            )
            
            old_log_probs = torch.FloatTensor(data['log_probs']).to(device)
        
        # 转换为张量
        states_tensor = torch.FloatTensor(data['states']).to(device)
        actions_tensor = torch.LongTensor(data['actions']).to(device)
        returns_tensor = torch.FloatTensor(returns).to(device)
        advantages_tensor = torch.FloatTensor(advantages).to(device)
        
        # 归一化优势函数
        advantages_tensor = self.normalize_advantages(advantages_tensor)
        
        # 3. 多轮优化（类似参考代码的for k in range(n_optimization_epochs)）
        policy_train_stats = defaultdict(list)
        critic_train_stats = defaultdict(list)
        
        for epoch in range(self.epoch):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            
            # 分批处理（类似参考代码的mini-batch updates）
            n_batches = int(np.ceil(float(n_samples) / self.batch_size))
            
            for batch_idx in range(n_batches):
                # 获取批次索引
                start_idx = batch_idx * self.batch_size
                end_idx = start_idx + self.batch_size
                batch_indices = indices[start_idx:end_idx]
                
                # 获取批次数据
                batch_states = states_tensor[batch_indices]
                batch_actions = actions_tensor[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                
                # 4. 计算新策略的输出（类似参考代码的policy_output, critic_output）
                actor_critic_output = self.compute_actor_critic_output(
                    batch_states, 
                    batch_actions
                )
                
                new_log_probs = actor_critic_output['log_probs']
                entropy = actor_critic_output['entropy'].mean()
                values_pred = actor_critic_output['values'].squeeze()
                
                # 5. 计算策略损失
                # 计算概率比
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO裁剪目标函数
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 
                                    1.0 - self.clip_range, 
                                    1.0 + self.clip_range) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
                
                # 6. 计算价值损失
                values_pred = values_pred.squeeze()
                batch_returns = batch_returns.squeeze()
                
                if values_pred.dim() == 0:
                    values_pred = values_pred.unsqueeze(0)
                if batch_returns.dim() == 0:
                    batch_returns = batch_returns.unsqueeze(0)
                
                critic_loss = F.mse_loss(values_pred, batch_returns)
                
                # 7. 执行梯度步骤
                # 更新Actor
                self.actor_optim.zero_grad()
                policy_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor_optim.step()
                
                # 更新Critic
                self.critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic_optim.step()
                
                # 记录训练统计
                policy_train_stats['policy_loss'].append(policy_loss.item())
                policy_train_stats['entropy'].append(entropy.item())
                critic_train_stats['value_loss'].append(critic_loss.item())
        
        # 8. 清空缓冲区
        self.replay_buffer.clear_memo()
        
        # 9. 记录训练统计
        if len(policy_train_stats['policy_loss']) > 0:
            avg_policy_loss = np.mean(policy_train_stats['policy_loss'])
            avg_entropy = np.mean(policy_train_stats['entropy'])
            avg_value_loss = np.mean(critic_train_stats['value_loss'])
            
            print(f"PPO Update Stats: "
                  f"Policy Loss: {avg_policy_loss:.4f}, "
                  f"Value Loss: {avg_value_loss:.4f}, "
                  f"Entropy: {avg_entropy:.4f}")
    
    def save_policy(self, path="ppo_model_final.pth"):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optim_state_dict': self.actor_optim.state_dict(),
            'critic_optim_state_dict': self.critic_optim.state_dict(),
        }, path)
        print(f"模型保存到: {path}")
    
    def load_policy(self, path="ppo_model_final.pth"):
        """加载模型"""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optim.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.critic_optim.load_state_dict(checkpoint['critic_optim_state_dict'])
        print(f"模型加载自: {path}")

# 训练函数保持不变，但修改update_interval
def train_maze(env, episodes=2000, max_steps=100, batch_size=128, update_interval=20):
    """训练PPO智能体解决迷宫"""
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    print(f"状态维度: {state_dim}")
    print(f"动作维度: {action_dim}")
    
    agent = PPOAgent(state_dim, action_dim, batch_size)
    
    episode_rewards = []
    successes = []
    best_reward = -float('inf')
    
    print("\n开始训练...")
    print("=" * 50)
    
    start_time = time.time()
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < max_steps:
            action, log_prob, value = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.add_memo(
                state, action, reward, value, done, log_prob
            )
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        successes.append(1 if terminated else 0)
        
        # 定期更新（收集足够数据后）
        if episode % update_interval == 0 and len(agent.replay_buffer.state_cap) >= batch_size:
            agent.update()
        
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else episode_reward
            success_rate = np.mean(successes[-50:]) * 100 if len(successes) >= 50 else 0
            
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_reward:7.2f} | "
                  f"Avg Reward (50): {avg_reward:7.2f} | "
                  f"Success: {('Yes' if terminated else 'No'):3s} | "
                  f"Success Rate: {success_rate:5.1f}%")
    
    training_time = time.time() - start_time
    
    print("\n" + "=" * 50)
    print(f"训练完成!")
    print(f"总训练时间: {training_time:.2f} 秒")
    print(f"最佳奖励: {best_reward:.2f}")
    print(f"最终成功率: {np.mean(successes[-50:])*100:.1f}%" if len(successes) >= 50 else "训练数据不足")
    
    # 保存最终模型
    agent.save_policy("ppo_maze_model_final.pth")
    
    return agent

def test_maze(env, agent, episodes=10, render=True):
    """测试训练好的智能体"""
    total_rewards = []
    total_lengths = []
    successes = []
    
    print("\n开始测试...")
    print("=" * 50)
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done and episode_length < 500:
            if render:
                env.render()
                time.sleep(0.05)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return total_rewards, successes
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_probs = agent.actor(state_tensor)
                action = torch.argmax(action_probs, dim=-1).item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            state = next_state
            episode_reward += reward
            episode_length += 1
        
        total_rewards.append(episode_reward)
        total_lengths.append(episode_length)
        successes.append(1 if terminated else 0)
        
        print(f"Test Episode {episode:2d} | "
              f"Reward: {episode_reward:7.2f} | "
              f"Length: {episode_length:3d} | "
              f"Success: {('Yes' if terminated else 'No'):3s}")
    
    print("\n" + "=" * 50)
    print(f"测试完成!")
    print(f"平均奖励: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    print(f"平均步数: {np.mean(total_lengths):.1f} ± {np.std(total_lengths):.1f}")
    print(f"成功率: {np.mean(successes)*100:.1f}%")
    env.close()
    return total_rewards, successes

# ========== 主程序入口 ==========
if __name__ == "__main__":
    
    
    # 参数设置
    MAZE_SIZE = (11, 11)
    VIEW_RANGE = 3
    EPISODES = 1500
    BATCH_SIZE = 64
    MAX_STEPS = 200
    
    # 创建训练环境
    train_env = mazeEnv(mazesize=MAZE_SIZE, render_mode=None, view_range=VIEW_RANGE)
    
    print("=" * 50)
    print(f"迷宫环境: {MAZE_SIZE[0]}x{MAZE_SIZE[1]}")
    print(f"视野范围: {VIEW_RANGE}")
    print(f"总回合数: {EPISODES}")
    print(f"批次大小: {BATCH_SIZE}")
    print("=" * 50)
    
    # 训练智能体
    agent = train_maze(
        env=train_env,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        update_interval=10
    )
    # 测试智能体
    test_env = mazeEnv(mazesize=MAZE_SIZE, render_mode='human', view_range=VIEW_RANGE)
    test_maze(test_env, agent, episodes=10, render=True)