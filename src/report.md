# 强化学习智能体训练项目汇报

## 1. 项目概述

### 1.1 项目目标
本项目旨在实现一个强化学习智能体，使其能够完成像迷宫或lunarlander一类的简单游戏。依照不同的强化学习算法，在应对简单固定的迷宫时采用Q-learning算法，通过最小化时序差分误差优化最终目标。对于lunarlander，采用了Gymnasium中的LunarLander-v3作为训练的环境，采取双DQN的强化学习算法训练agent.

### 1.2 技术架构
- **环境框架**: Gymnasium
- **渲染引擎**: PyGame
- **算法核心**: Q-learning, DQN
- **可视化工具**: TensorBoard

## 2. 环境设计 (mazeEnv类)

### 2.1 状态空间
```python
self.observation_space = spaces.Box(
    low=0,
    high=max(self.row, self.col),
    shape=(2,),  
    dtype=np.int32
)
#Agent所处的位置
```
### 2.2 动作空间
```python
self.action_space = spaces.Discrete(4)
#0:up 1:down 2:left 3:right
```
### 2.3 迷宫生成
```python
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
#深度优先搜索算法随机生成迷宫        
```
### 2.4 奖励函数
```python
if self.maze[self.human[0]][self.human[1]] == 0:
            self.human = old
            penalty = -1#撞墙惩罚
        else:
            penalty = 0
        
        terminated = (self.human[0] == self.goalpos[0] and self.human[1] == self.goalpos[1])
        if terminated:
            reward = 100#成功奖励
        else:
            old_dis = np.linalg.norm(np.array(old) - np.array(self.goalpos))
            new_dis = np.linalg.norm(np.array(self.human) - np.array(self.goalpos))
            reward += (old_dis - new_dis)*0.5#靠近目的地奖励
            reward += penalty
```
## 3. 强化学习概念（Agent构成）
### 3.1 核心要素
- **Agent**: 决策主体，选择动作来最大化累积奖励。
- **Environment**: Agent所处的情景。
- **State**: 环境在某一时刻的特征描述。
- **Action**: Agent在当前可以选择的行为。
- **Reward**: 环境对Agent作出的Action产生的反馈。
### 3.2 Q-Learning

- **实现原理**：
Q-learning是一种基于值迭代的时序差分学习算法，使用贝尔曼方程更新Q值：

   $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

### 3.3 DQN
DQN结合了Q-learning和深度神经网络，使用神经网络近似Q函数。
###
采取了优先经验回放：
- 优先级计算: $priority = |TD\ error| + \epsilon$
- 采样概率: $P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}$
- 重要性采样权重: $w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$

