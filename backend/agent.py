import argparse
from itertools import count

import gym
import time
import scipy.optimize
import random
import tkinter as tk
import fileinput
import pickle
import thread
import torch
from models import *
from replay_memory import Memory
from running_state import ZFilter
from torch.autograd import Variable
from trpo import trpo_step
from utils import *
from gym.wrappers import Monitor
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from render_browser import render_browser

STEPS = 200

def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch, gamma=0.995, tau=0.97, l2_reg=1e-3, max_kl=2e-2, damping=1e-1):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(batch.state)
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0),1)
    deltas = torch.Tensor(actions.size(0),1)
    advantages = torch.Tensor(actions.size(0),1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss, get_flat_params_from(value_net).double().numpy(), maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))
                
        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()


    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, max_kl, damping)

def run_batches(batch_size, batch_size=15, gamma=0.995, tau=0.97, l2_reg=1e-3, max_kl=2e-2, damping=1e-1):
    running_state = ZFilter((num_inputs,), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)

    for i_episode in range(0,201):
        memory = Memory()

        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        one = random.randint(0,batch_size-2)
        two = one + 1
        steps_one = []
        steps_two = []
        states_one = []
        states_two = []
        seed_one = None
        seed_two = None
        
        for idx in range(batch_size):
            seed = random.randint(0,1000000)
            env.seed(seed)
            state = env.reset()
            
            if idx == one:
                seed_one = seed
            if idx == two:
                seed_two = seed
            
            state = running_state(state)

            reward_sum = 0
            for t in range(STEPS): # Don't infinite loop while learning
                action = select_action(state)
                action = action.data[0].numpy()
                next_state, _, _, _ = env.step(action)
                reward = reward_net(torch.cat((torch.from_numpy(next_state),torch.from_numpy(action)), 0))
                reward_sum += reward
                if idx == one:
                    states_one.append(state)
                    steps_one.append(action)
                if idx == two:
                    states_two.append(state)
                    steps_two.append(action)

                next_state = running_state(next_state)

                mask = int(t!=STEPS-1)

                memory.push(state, np.array([action]), mask, next_state, reward)

                state = next_state
            num_episodes += 1
            reward_batch += reward_sum

        reward_batch /= num_episodes
        batch = memory.sample()
        update_params(batch, gamma, tau, l2_reg, max_kl, damping)