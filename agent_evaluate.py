# Attempt to use this repository to evaluate a trained agent - not working yet

import torch
from nocturne.envs.wrappers import create_env
from cfgs.config import set_display_window, get_scenario_dict
from examples.imitation_learning.model import ImitationAgent

def evaluate(model_path, cfg, num_episodes=15):
    set_display_window()
    env = create_env(cfg)
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    rewards = []
    for _ in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_tensor, deterministic=True)
                if isinstance(action, tuple):
                    action = action[0]
                action = action.squeeze().cpu().numpy()
            obs, reward, done, info = env.step(action)
            total_reward += reward
        rewards.append(total_reward)
    print("Average reward:", sum(rewards) / len(rewards))

# Usage:
import hydra

@hydra.main(config_path="/home/arthur_ooms/dev/nocturne-dream/cfgs/", config_name="config", version_base="1.3")
def main(cfg):
    scenario_dict = get_scenario_dict(cfg)
    evaluate("/home/arthur_ooms/dev/nocturne-dream/policy_epoch_10.pth", cfg)

if __name__ == "__main__":
    main()