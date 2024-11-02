import argparse
import torch
from tqdm import tqdm

from agent.ppo import PPOAgent
from environment.building import Building
from environment.enums import Action

from torch.distributions import Categorical

torch.autograd.set_detect_anomaly(True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--load", action="store_true", help="Load a trained model")
    parser.add_argument("--num_floors", type=int, default=5, help="Number of floors in the building")
    parser.add_argument("--num_elevators", type=int, default=1, help="Number of elevators")
    parser.add_argument("--floor_capacity", type=int, default=10, help="Capacity of each floor")
    parser.add_argument("--elevator_capacity", type=int, default=5, help="Capacity of each elevator")
    parser.add_argument("--spawn_prob", type=float, default=0.1, help="Probability of passenger spawn")
    parser.add_argument("--max_group_size", type=int, default=5, help="Maximum group size of passengers")
    parser.add_argument("--num_episodes", type=int, default=1000, help="Number of training episodes")
    parser.add_argument("--max_steps", type=int, default=500, help="Maximum steps per episode")
    parser.add_argument("--save_interval", type=int, default=100, help="Interval to save the model")
    parser.add_argument("--model_path", type=str, default="ppo_agent.pth", help="Path to save/load the model")
    
    args = parser.parse_args()

    # Initialize Building Environment
    building = Building(
        num_floors=args.num_floors,
        num_elevators=args.num_elevators,
        floor_capacity=args.floor_capacity,
        elevator_capacity=args.elevator_capacity,
        spawn_prob=args.spawn_prob,
        max_group_size=args.max_group_size
    )

    # Initialize PPO Agent
    agent = PPOAgent(
        env=building,
        learning_rate=1e-3,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
        update_epochs=10,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load pretrained model if specified
    if args.load:
        agent.actor.load_state_dict(torch.load(args.model_path))
        agent.critic.load_state_dict(torch.load(args.model_path))
        print(f"Loaded model from {args.model_path}")

    if args.train:
        for episode in range(1, args.num_episodes + 1):
            state = building.reset()
            agent.memory.clear()
            episode_reward = 0
            for step in tqdm(range(args.max_steps)):
                probs, log_prob, state_embeddings = agent.get_action(state)
                
                dists = [Categorical(logits=p) for p in probs]
                actions = [d.sample() for d in dists]
                
                next_state, reward, done, _ = building.step(actions)
                
                
                next_state_embeddings = agent.embed_state(next_state)
                agent.memory_store(state_embeddings, actions, [d.log_prob(a) for d, a in zip(dists, actions)], reward, next_state_embeddings, done)
                
                state = next_state
                episode_reward += reward
                
                # if step % 10 == 0:
                #     building.print_building(step)
                
                if step % 100 == 0 and step != 0:
                    print(f"Updating agent at step {step}")
                    agent.update()
                
                if done:
                    break
            
            # Update the agent after each episode
            agent.update()
            
            # Logging
            print(f"Episode {episode}/{args.num_episodes} - Reward: {episode_reward}")
            
            # Save the model at specified intervals
            if episode % args.save_interval == 0:
                torch.save(agent.actor.state_dict(), args.model_path)
                torch.save(agent.critic.state_dict(), args.model_path)
                print(f"Model saved at episode {episode}")

        print("Training completed!")

if __name__ == "__main__":
    main()
