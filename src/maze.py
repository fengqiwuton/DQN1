import gymnasium as gym
import numpy as np
from collections import defaultdict
from gymnasium import spaces
from typing import Tuple, Optional
import random
from gymnasium.error import DependencyNotInstalled
import pygame

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

    def create_maze(self, obstacle_density = 0.3):
        # 0:obstacle
        maze = [[1 for i in range(self.col)] for j in range(self.row)]
        for x in range(self.col):
           maze[0][x] = 0
           maze[self.row-1][x] = 0
        for y in range(self.row):
           maze[y][0] = 0
           maze[y][self.col-1] = 0
        for y in range(1, self.col-1):
            for x in range(1, self.row-1):
                if random.random() < obstacle_density:
                    maze[y][x] = 0
        maze[self.startpos[0]][self.startpos[1]] = 1
        maze[self.goalpos[0]][self.goalpos[1]] = 1
        for y in range(1, self.col-1):
            for x in range(1, self.row-1):
                if maze[x][y] == 0:
                    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                    path_count = sum(1 for nx, ny in neighbors if maze[nx][ny] == 1)
                    if path_count == 0:
                        dx, dy = random.choice([(1, 0), (-1, 0), (0, 1), (0, -1)])
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.row and 0 <= ny < self.col:
                            maze[nx][ny] = 1        
        self.maze = maze

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
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
            penalty = -0.5
        else:
            penalty =0
        
        terminated = (self.human[0] == self.goalpos[0] and self.human[1] == self.goalpos[1])
        if terminated:
            reward = 10
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
    
    def test_env(self):
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

if __name__ == "__main__":
    env = mazeEnv(mazesize=(8,7),
            render_mode='human')
    env.test_env()

    
        


        


 






class mazeAgent:
    def __init__(
        self,
        env:gym.Env,
        learning_rate:float,
        initial_epsilon:float,
        final_epsilon:float,
        epsilon_decay:float,
        discount_factor:float = 0.95,
    ):
        self.env = env
        self.Q = defaultdict(lambda:np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount = discount_factor

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.train_error = []

        
