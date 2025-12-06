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
import pickle
import os

# develop a simple maze and train an agent to solve the problem
VIEWPORT_W = 400
VIEWPORT_H = 400
class mazeEnv(gym.Env):
    """
    action space: 0:up 1:down 2:left 3:right
    observation space: coordinates(x,y)
    """
    metadata = {'render_modes':['human', 'rgb_array'],'render_fps':4}
    def __init__(self, 
                 mazesize:Tuple[int,int]=(7,7),
                 render_mode:Optional[str] = None):
        super().__init__()
        self.mazesize = mazesize
        self.row, self.col = mazesize
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.row, self.col),
            shape=(2,),
            dtype=np.int32
        )
        
        self.startpos = (1,1)
        self.goalpos = (self.row-2, self.col-2)
        self.human = (self.startpos)
        self.render_mode = render_mode
        self.window: pygame.Surface = None
        self.clock = None
        self.step = 0
        self.max_steps = self.row*self.col
        self.create_maze()
        self.color = {
            'background': (255,255,255),
            'obstacle': (100,100,100),
            'path': (255,255,255),
            'start': (144, 238, 144),
            'goal': (240, 128,128),
            'human': (70,130,180)
        }

    def create_maze(self):
        # 0:obstacle
        maze = [[0 for i in range(self.col)] for j in range(self.row)]
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        def dfs(x, y):
            maze[y][x] = 1 # 标记为路径
            random.shuffle(directions) # 随机选择方向
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 < nx < self.col - 1 and 0 < ny < self.row - 1 and maze[ny][nx] == 0:
                    maze[y + dy // 2][x + dx // 2] = 1 # 打通墙壁
                    dfs(nx, ny)
        dfs(1, 1)        
        self.maze = maze
        

    def reset(self, options = None):
        #self.create_maze()
        self.human = list(self.startpos)
        self.step = 0
        observation = np.array(self.human, dtype=np.int32)

        if self.render_mode == 'human':
            self.render()
        return observation, {}
    
    def steping(self, action:int):
        reward = 0
        old = self.human.copy()
        if action == 0:
            self.human[0] = max(0, self.human[0] - 1)
        elif action == 1:
            self.human[0] = min(self.row - 1, self.human[0]+ 1)
        elif action == 2:
            self.human[1] = max(0, self.human[1] - 1)
        elif action == 3:
            self.human[1] = min(self.col, self.human[1] + 1)
        
        if self.maze[self.human[0]][self.human[1]] == 0:
            self.human = old
            penalty = -1
        else:
            penalty = -0.4
        
        terminated = (self.human[0] == self.goalpos[0] and self.human[1] == self.goalpos[1])
        if terminated:
            reward = 100
        else:
            old_dis = np.linalg.norm(np.array(old) - np.array(self.goalpos))
            new_dis = np.linalg.norm(np.array(self.human) - np.array(self.goalpos))
            reward += (old_dis - new_dis)*0.5
            reward += penalty
        
        truncated = self.step >= self.max_steps
        observation = np.array(self.human, dtype=np.int32)
        if self.render_mode == 'human':
            self.render()
        return observation, reward, terminated, truncated, {}
    
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
            self.window = pygame.display.set_mode((VIEWPORT_W,VIEWPORT_H))   
        if self.clock is None:
            self.clock = pygame.time.Clock()
        surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))
        surf.fill(self.color['background'])
        cell_width = VIEWPORT_W/self.mazesize[1]
        cell_height = VIEWPORT_H/self.mazesize[0]
        for i in range(self.row):
            for j in range(self.col):
                x = j * cell_width
                y = i * cell_height
                if (i,j) == self.startpos:
                    color = self.color['start']
                elif (i,j) == self.goalpos:
                    color = self.color['goal']
                elif 0  == self.maze[i][j]:
                    color = self.color['obstacle']
                else:
                    color =self.color['path']
                pygame.draw.rect(surf, color, (x,y,cell_width,cell_height))

        human_x = self.human[1]*cell_width + cell_width//2
        human_y = self.human[0]*cell_height + cell_height//2

        pygame.draw.circle(surf, self.color['human'], (human_x,human_y),min(cell_height,cell_width)//3)
        
        if self.render_mode == 'human':
            self.window.blit(surf, (0,0))
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.flip()
        elif self.render_mode == 'rgb_array':
            rgb_array = pygame.surfarray.array3d(surf)
            return np.transpose(rgb_array, (1,0,2))
    
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
    
    def test_env(self,env):
        obs, info = env.reset()
        for step in range(100):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.steping(action)
            env.render()
            if env.unwrapped.window is not None:
            # 检查是否有退出事件（针对pygame等图形库）
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        env.close()
                        return
            if terminated:
                break
            elif truncated:
                break
        env.close()


class mazeAgent:
    def __init__(
        self,
        env
    ):
        learning_rate = 0.1
        initial_epsilon = 1
        final_epsilon = 0.01
        epsilon_decay = 0.999
        discount_factor:float = 0.95 
        self.env = env
        self.action_size = 4

        self.Q = defaultdict(lambda:np.zeros(self.action_size))
        self.lr = learning_rate
        self.discount = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        #tensorboard
        self.writer = SummaryWriter(log_dir=f'runs/maze_qlearning_{time.strftime("%Y%m%d_%H:%M")}')
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.episode_q_values = []
        self.steps_done = 0
        
    def get_action(self, state_key, training = True):
        state = tuple(state_key)
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_values = self.Q[state]
            max_q = np.max(q_values)
            return random.choice(np.where(q_values == max_q)[0])
        
    def update_q(self, state_key, action, reward, next_statekey, done):
        state = tuple(state_key)
        next_state = tuple(next_statekey)
        current_q  = self.Q[state][action]
        if done:
            target_q  = reward
        else:
            next_q_max = float(np.max(self.Q[next_state]))
            target_q = float(reward) + float(self.discount) * next_q_max
        td_error = target_q - current_q
        self.Q[state][action] = current_q + self.lr*td_error
        return abs(td_error)
    
    def train(self, episodes, steps = 1000):
        self.episode_rewards = []
        self.episode_lengths = []
        self.successes = []
        self.episode_q_values = []
        self.steps_done = 0
        for episode in range (episodes):
            state, _= self.env.reset()
            total_reward = 0
            step = 0
            success = False
            td_errors = []
            for step in range(steps):
                action  = self.get_action(state, training=True)
                nextstate,reward,terminated,truncated,_ = self.env.steping(action)
                td_errors.append(self.update_q(state,action,reward,nextstate,terminated))
                state = nextstate
                total_reward += reward
                step += 1
                self.steps_done += 1
                if terminated:
                    success = True
                    break
                if truncated:
                    break
            self.epsilon = max(self.final_epsilon, self.epsilon_decay*self.epsilon)
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(step)
            self.successes.append(1 if success else 0)
            if td_errors:
                self.episode_q_values.append(np.mean(td_errors))
            self.writer.add_scalar('Training/Episode_Reward', total_reward, episode)
            self.writer.add_scalar('Training/Episode_Length', step, episode)
            self.writer.add_scalar('Training/Epsilon', self.epsilon, episode)
            self.writer.add_scalar('Training/States_Explored', len(self.Q), episode)
            self.writer.add_scalar('Training/Success', 1 if success else 0, episode)
            if td_errors:
                avg_td_error = np.mean(td_errors)
                self.writer.add_scalar('Training/Avg_TD_Error', avg_td_error, episode)
            
            if (episode + 1) % 50 == 0 or episode == 0 or episode == episodes - 1:
                # 计算最近50个episode的平均值
                window = min(50, len(self.episode_rewards))
                recent_rewards = self.episode_rewards[-window:]
                recent_lengths = self.episode_lengths[-window:]
                recent_successes = self.successes[-window:]
                
                avg_reward = np.mean(recent_rewards)
                avg_length = np.mean(recent_lengths)
                success_rate = np.mean(recent_successes) * 100
                
                # 记录移动平均值到TensorBoard
                self.writer.add_scalar('Training/Avg_Reward_50', avg_reward, episode)
                self.writer.add_scalar('Training/Avg_Length_50', avg_length, episode)
                self.writer.add_scalar('Training/Success_Rate_50', success_rate, episode)
                print(f"Episode {episode+1:4d}/{episodes}: "
                      f"奖励={total_reward:6.2f}, "
                      f"步数={step:3d}, "
                      f"成功={'✓' if success else '✗'}, "
                      f"探索率={self.epsilon:.3f}, "
                      f"Q表大小={len(self.Q):4d}, "
                      f"近{window}轮平均奖励={avg_reward:.2f}, "
                      f"成功率={success_rate:.1f}%")
        


    def save_model(self, filename):
        save_dict = {
            'Q_table': dict(self.Q),
            'epsilon': self.epsilon,
            'lr': self.lr,
            'discount': self.discount,
            'epsilon_decay': self.epsilon_decay,
            'final_epsilon': self.final_epsilon,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'successes': self.successes,
            'steps_done': self.steps_done
    }
    
        with open(filename, 'wb') as f:
            pickle.dump(save_dict, f)
    
        print(f" 模型已保存: {filename}")
        return True

    def load_model(self, filename):
        if not os.path.exists(filename):
            print(f" 模型文件不存在: {filename}")
            return False
    
        try:
            with open(filename, 'rb') as f:
                checkpoint = pickle.load(f)
                
        
            # 恢复Q表
            self.Q.clear()
            self.Q.update(checkpoint['Q_table'])
        
            # 恢复参数
            self.epsilon = checkpoint['epsilon']
            self.lr = checkpoint['lr']
            self.discount = checkpoint['discount']
            self.epsilon_decay = checkpoint['epsilon_decay']
            self.final_epsilon = checkpoint['final_epsilon']
        
            # 恢复训练统计
            self.episode_rewards = checkpoint['episode_rewards']
            self.episode_lengths = checkpoint['episode_lengths']
            self.successes = checkpoint['successes']
            self.steps_done = checkpoint['steps_done']
        
            print(f"   模型加载成功: {filename}")
            print(f"   Q表大小: {len(self.Q)}")
            print(f"   训练episodes: {len(self.episode_rewards)}")
            return True
        
        except Exception as e:
            print(f"加载失败: {e}")
            return False

if __name__ == "__main__":
    '''
    #固定的迷宫训练
    env = mazeEnv(mazesize=(19, 19),render_mode='rgb_array')
    agent = mazeAgent(env)
    agent.train(episodes=700)
    agent.save_model("maze_model.pkl")
    env.close()  
    '''  




        
