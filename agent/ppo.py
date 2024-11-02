import logging
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from environment.building import Building
from environment.enums import Action

from agent.actor_critic import Actor, Critic
from agent.sequence_encoder import SequenceEncoder

class PPOAgent:
    
    def __init__(
        self, 
        env: Building,
        floor_passenger_encoder_size: int = 128,
        elevator_passenger_encoder_size: int = 128,
        elevator_state_embedding_size: int = 64,
        learning_rate: float = 1e-3, 
        gamma: float = 0.99, 
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        memory_size: int = 10000,
        actor_hidden_size: int = 128,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        
        self.floor_passenger_encoder = SequenceEncoder(2, 40, floor_passenger_encoder_size, 4).to(device)
        self.elevator_passenger_encoder = SequenceEncoder(1, 10, elevator_passenger_encoder_size, 4).to(device)
        
        self.elevator_state_embedding = nn.Linear(3, elevator_state_embedding_size).to(device) # (elevator, position, direction) -> embedding
        
        self.state_size = floor_passenger_encoder_size * env.num_floors + (elevator_passenger_encoder_size + elevator_state_embedding_size) * env.num_elevators
        action_size = len(Action)
        
        
        self.shared_actor = nn.Linear(self.state_size, actor_hidden_size).to(device)
        self.individual_actors = nn.ModuleList([Actor(actor_hidden_size, action_size).to(device) for _ in range(env.num_elevators)])
        self.critic = Critic(self.state_size).to(device)
        
        self.shared_actor_optimizer = optim.Adam(self.shared_actor.parameters(), lr=learning_rate)
        self.individual_actor_optimizers = [optim.Adam(actor.parameters(), lr=learning_rate) for actor in self.individual_actors]
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        
        
        self.gamma = gamma # discount factor
        self.gae_lambda = gae_lambda # lambda for GAE
        self.clip_epsilon = clip_epsilon # clip epsilon for PPO
        self.value_coef = value_coef # value function coefficient
        self.entropy_coef = entropy_coef # entropy coefficient
        self.max_grad_norm = max_grad_norm # max gradient norm
        self.update_epochs = update_epochs # number of update epochs
        
        self.device = device
        print(f"Using device: {self.device}")
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.logger.info("PPO Agent initialized")
        
        self.memory = deque(maxlen=memory_size)
        
    def memory_store(self, state, action, log_prob, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, log_prob, reward, next_state, done))
        
    def embed_state(self, state):
        state_embeddings = []

        for idx, floor_pax_state in state[0]:
            
            floor = torch.tensor(floor_pax_state).to(self.device)
            floor = floor.float()
            floor_embedding = self.floor_passenger_encoder(floor)
            state_embeddings.append(floor_embedding)
        
        for idx, elevator_pax_state, position, direction in state[1]:
            
            elevator_pax_state = torch.Tensor(elevator_pax_state).to(self.device)
            elevator_embedding = self.elevator_passenger_encoder(elevator_pax_state)
            state_embeddings.append(elevator_embedding)
            
            
            elevator_state = torch.Tensor([idx, position, direction]).to(self.device)
            elevator_state_embedding = self.elevator_state_embedding(elevator_state)
            state_embeddings.append(elevator_state_embedding)
        
        for i, embedding in enumerate(state_embeddings):
            state_embeddings[i] = embedding.flatten()

        state_embeddings = torch.cat(state_embeddings).to(self.device)
        
        return state_embeddings
    def get_action(self, state):
        
        state_embeddings = self.embed_state(state)
        shared_features = self.shared_actor(state_embeddings)
        
        
        actions = []
        log_probs = []
        
        for actor in self.individual_actors:
            mean, std = actor(shared_features)
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            action = F.softmax(action, dim=-1)
            actions.append(action)
            log_probs.append(log_prob)
            
        return actions, log_probs, state_embeddings
    
    def update(self):
        """Modified update method to use stored experiences"""
        if len(self.memory) == 0:
            return 0, 0, 0
            
        # Unpack experiences from memory
        states, actions, old_log_probs, rewards, next_states, dones = zip(*self.memory)
        
        # Clear memory after getting experiences
        self.memory.clear()
        
        
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack([torch.Tensor(t) for t in old_log_probs]).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)
            
            advantages = []
            returns = []
            gae = 0
            
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages.append(gae)
                returns.append(gae + values[t])
            
            advantages = torch.tensor(advantages[::-1], dtype=torch.float32).to(self.device)
            returns = torch.tensor(returns[::-1], dtype=torch.float32).to(self.device)
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            self.logger.info(f"Advantages: {advantages.mean().item()}, {advantages.std().item()}")
        
        for _ in range(self.update_epochs):
            
            self.shared_actor_optimizer.zero_grad()
            for optimizer in self.individual_actor_optimizers:
                optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            
            shared_features = self.shared_actor(states)
            total_loss = torch.tensor(0.0).to(self.device)
            
            for i, actor in enumerate(self.individual_actors):
                mean, std = actor(shared_features)
                dist = torch.distributions.Normal(mean, std)
                new_log_probs = dist.log_prob(actions[i]).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs[i])
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
                
                actor_loss = -torch.min(surr1, surr2).mean()
                entropy_loss = -self.entropy_coef * entropy
                
                total_loss += actor_loss + entropy_loss
            
            # Add critic loss
            critic_loss = self.value_coef * F.mse_loss(values.squeeze(1), returns)
            total_loss += critic_loss
            
            total_loss.backward(retain_graph=True)
            
            # Clip gradients and step all optimizers
            nn.utils.clip_grad_norm_(self.shared_actor.parameters(), self.max_grad_norm)
            for actor in self.individual_actors:
                nn.utils.clip_grad_norm_(actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            self.shared_actor_optimizer.step()
            for optimizer in self.individual_actor_optimizers:
                optimizer.step()
            self.critic_optimizer.step()

        return total_loss.item()
    
    def train(self, env: Building, num_episodes: int, max_steps: int):
        """Modified train method to use memory system"""
        best_reward = -float("inf")
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            
            for step in range(max_steps):
                action, log_prob = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                # Store experience
                self.store(state, action, log_prob, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            # Update at the end of each episode
            if len(self.memory) > 0:
                actor_loss, critic_loss, entropy_loss = self.update()
                self.logger.info(f"Episode {episode}: Reward: {episode_reward}, Losses: {actor_loss:.3f}, {critic_loss:.3f}, {entropy_loss:.3f}")
            
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.logger.info(f"New best reward: {best_reward}")
                # Save shared actor and all individual actors
                torch.save(self.shared_actor.state_dict(), "shared_actor.pth")
                for i, actor in enumerate(self.individual_actors):
                    torch.save(actor.state_dict(), f"individual_actor_{i}.pth")
                torch.save(self.critic.state_dict(), "critic.pth")
                
            env.print_building()
                
        self.logger.info(f"Training complete, best reward: {best_reward}")
