#!/usr/bin/env python3
import gymnasium as gym
import torch
import numpy as np
import random
import yaml
import argparse
import os
from collections import deque

from models.dqn import DQNAgent
from models.utils import create_env, setup_tensorboard, evaluate_agent

def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def train(config):
    """训练函数"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建环境和智能体
    env = create_env()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"环境: {config['env_name']}")
    print(f"状态空间: {state_size}, 动作空间: {action_size}")
    
    agent = DQNAgent(state_size, action_size, config)
    
    # 设置TensorBoard
    writer = setup_tensorboard()
    
    # 训练参数
    scores = []
    recent_scores = deque(maxlen=100)
    best_score = -float('inf')
    
    print("开始训练...")
    print("=" * 60)
    
    for episode in range(config['episodes']):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        episode_loss = 0
        episode_q_values = 0
        loss_count = 0
        
        for step in range(config['max_steps']):
            # 选择动作
            action = agent.act(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 存储经验
            agent.remember(state, action, reward, next_state, done)
            
            # 经验回放
            loss, q_value = agent.replay()
            if loss > 0:
                episode_loss += loss
                episode_q_values += q_value
                loss_count += 1
            
            total_reward += reward
            state = next_state
            steps += 1
            
            if done:
                break
        
        # 软更新目标网络
        agent.update_target_network(hard=False)
        
        # 计算统计量
        avg_loss = episode_loss / loss_count if loss_count > 0 else 0
        avg_q_value = episode_q_values / loss_count if loss_count > 0 else 0
        
        scores.append(total_reward)
        recent_scores.append(total_reward)
        mean_recent_score = np.mean(recent_scores)
        
        # 记录到TensorBoard
        writer.add_scalar('Training/Score', total_reward, episode)
        writer.add_scalar('Training/Average_Score_100', mean_recent_score, episode)
        writer.add_scalar('Training/Steps', steps, episode)
        writer.add_scalar('Training/Loss', avg_loss, episode)
        writer.add_scalar('Training/Epsilon', agent.epsilon, episode)
        writer.add_scalar('Training/Q_Value', avg_q_value, episode)
        writer.add_scalar('Training/Memory_Size', len(agent.memory), episode)
        
        # 定期评估
        if episode % config['eval_every'] == 0:
            eval_env = create_env()
            mean_score, std_score = evaluate_agent(agent, eval_env, n_episodes=5)
            eval_env.close()
            
            writer.add_scalar('Evaluation/Mean_Score', mean_score, episode)
            writer.add_scalar('Evaluation/Std_Score', std_score, episode)
            
            print(f"评估回合 {episode}: 平均得分 = {mean_score:.2f} ± {std_score:.2f}")
        
        # 保存最佳模型
        if total_reward > best_score:
            best_score = total_reward
            agent.save('models/best_model.pth')
            writer.add_scalar('Training/Best_Score', best_score, episode)
        
        # 定期保存检查点
        if episode % config['save_every'] == 0:
            agent.save(f'models/checkpoint_{episode}.pth')
        
        # 打印进度
        if episode % 10 == 0:
            print(f"回合 {episode:4d}/{config['episodes']} | "
                  f"得分: {total_reward:7.2f} | "
                  f"平均得分: {mean_recent_score:7.2f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"步数: {steps:3d} | "
                  f"记忆: {len(agent.memory):5d}")
        
        # 检查是否解决环境
        if mean_recent_score >= 200 and len(recent_scores) == 100:
            print(f"🎉 环境在 {episode} 回合解决!")
            agent.save('models/solved_model.pth')
            break
    
    # 保存最终模型
    agent.save('models/final_model.pth')
    env.close()
    writer.close()
    
    print("训练完成!")
    print(f"最佳得分: {best_score:.2f}")
    print(f"最后100回合平均得分: {np.mean(list(recent_scores)[-100:]):.2f}")
    
    return scores

def main():
    parser = argparse.ArgumentParser(description='训练DQN智能体玩LunarLander')
    parser.add_argument('--config', type=str, default='configs/default.yaml', 
                       help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, 
                       help='随机种子')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    # 开始训练
    scores = train(config)

if __name__ == "__main__":
    main()