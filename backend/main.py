import argparse
from itertools import count
from shutil import copyfile
import multiprocess
from multiprocess import Process, Queue, Lock
import os
import threading
import copy
import gym
import time
import scipy.optimize
import random
import tkinter as tk
import fileinput
import pickle
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
from flask import Flask, render_template, jsonify

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--load', type=bool, default=False)
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor (default: 0.995)')
parser.add_argument('--env-name', default="Hopper-v2", metavar='G',
                    help='name of the environment to run')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae (default: 0.97)')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--max-kl', type=float, default=2e-2, metavar='G',
                    help='max kl value (default: 2e-2)')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping (default: 1e-1)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--batch-size', type=int, default=35, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--render', action='store_true', default=True,
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

env = gym.make(args.env_name)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

env.seed(args.seed)
torch.manual_seed(args.seed)

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)
reward_net = Reward(num_inputs+num_actions)
reward_optim = torch.optim.Adam(reward_net.parameters(), lr=3e-3)
reward_loss_func = torch.nn.MSELoss()
human_choices = []
recordings = []
lock = threading.Lock()
recording_lock = threading.Lock()
app = Flask(__name__)
recorded_videos = 0
STEPS = 200
recordings_queue = Queue()
trajectories_queue = Queue()
reward_net_queue = Queue()
episode_number = multiprocess.Value('I', 1)
l = Lock()
displayed_episode = 1
videos_path = "./videos/"

def render_video(env_to_wrap, steps, seed, name, id):
    env_to_wrap.seed(seed)
    env_to_wrap.reset()
    video_recorder = VideoRecorder(env_to_wrap, base_path=f'{videos_path}{name}', enabled=True)
    env_to_wrap.render()
    for step in steps:
        video_recorder.capture_frame()
        env_to_wrap.step(step)
        env_to_wrap.render(mode='rgb_array')
    video_recorder.close()

def process_renderer(recordings_queue, id):
    env_to_wrap = gym.make('Hopper-v2')
    global episode_number
    global recordings
    while(True):
        if recordings_queue.empty():
            continue

        (steps_one, states_one, steps_two, states_two, seed_one, seed_two, id) = (None, None, None, None, None, None, None)
        while not recordings_queue.empty():
            (steps_one, states_one, steps_two, states_two, seed_one, seed_two, id) = recordings_queue.get()
            recordings.append((steps_one, states_one, steps_two, states_two, seed_one, seed_two))
        
        render_video(env_to_wrap,steps_one, seed_one, f'video_one_{id}', id)
        render_video(env_to_wrap,steps_two, seed_two, f'video_two_{id}', id)
        if id == 1:
            copyfile(f'{videos_path}video_one_1.mp4', '../frontend/public/videos/video_one_1.mp4')
            copyfile(f'{videos_path}video_two_1.mp4', '../frontend/public/videos/video_two_1.mp4')
        l.acquire()
        episode_number.value = id
        l.release()
    env_to_wrap.close()

def reward_loss(probability_1_over_2, probability_2_over_1, weights):
    return Variable(-torch.sum(probability_1_over_2*weights + probability_2_over_1*(1-weights)), requires_grad=True)

def save_human_choices(human_choices):
    with open('./human_choices/net.data' + str(i_episode), 'wb') as filehandle:
        pickle.dump(recordings, filehandle)

def update_reward_net(i_episode, human_choice):
    global human_choices
    global recordings
    while not trajectories_queue.empty():
        recordings.append(trajectories_queue.get())

    if len(recordings) < i_episode:
        return False
    print("\t\t\t\t\t\t\t\t\t", len(recordings), i_episode)

    (steps_one, states_one, steps_two, states_two, seed_one, seed_two) = recordings[i_episode-1]

    if human_choice == 1:
        human_choices.append((Variable(torch.cat((torch.DoubleTensor(steps_one),torch.DoubleTensor(states_one)),1), requires_grad=True),
                              Variable(torch.cat((torch.DoubleTensor(steps_two),torch.DoubleTensor(states_two)),1), requires_grad=True),
                              1.))
    elif human_choice == 2:
        human_choices.append((Variable(torch.cat((torch.DoubleTensor(steps_one),torch.DoubleTensor(states_one)),1), requires_grad=True),
                              Variable(torch.cat((torch.DoubleTensor(steps_two),torch.DoubleTensor(states_two)),1), requires_grad=True),
                              0.))
    elif human_choice == 3:
        human_choices.append((Variable(torch.cat((torch.DoubleTensor(steps_one),torch.DoubleTensor(states_one)),1), requires_grad=True),
                              Variable(torch.cat((torch.DoubleTensor(steps_two),torch.DoubleTensor(states_two)),1), requires_grad=True),
                              0.5))
        
    for _ in range(1):
        exp_sums_ones = [torch.exp(reward_net(x)).sum() for x,_,_ in human_choices]
        exp_sums_twos = [torch.exp(reward_net(y)).sum() for _,y,_ in human_choices]
        probability_1_over_2 = torch.DoubleTensor([x/(x+y)*0.9+0.05 for x, y in zip(exp_sums_ones, exp_sums_twos)])
        probability_2_over_1 = torch.DoubleTensor([y/(x+y)*0.9+0.05 for x, y in zip(exp_sums_ones, exp_sums_twos)])
        weights = torch.DoubleTensor([w for _,_,w in human_choices])

        lock.acquire()
        loss = reward_loss(probability_1_over_2, probability_2_over_1, weights)
        reward_optim.zero_grad()
        loss.backward()
        reward_optim.step()
        reward_net_queue.put(copy.deepcopy(reward_net))
        lock.release()
    return True

def select_action(state, policy_net):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action

def update_params(batch):
    global policy_net
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
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

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
            value_loss += param.pow(2).sum() * args.l2_reg
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

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

def save_models(i_episode):
    torch.save(policy_net, './nets/policy_net' + str(i_episode))
    torch.save(value_net, './nets/value_net' + str(i_episode))
    torch.save(reward_net, './nets/reward_net' + str(i_episode))
    with open('./nets/net.data' + str(i_episode), 'wb') as filehandle:
        pickle.dump(recordings, filehandle)
    torch.save(policy_net, './nets/policy_net')
    torch.save(value_net, './nets/value_net')
    torch.save(reward_net, './nets/reward_net')
    with open('./nets/net.data', 'wb') as filehandle:
        pickle.dump(recordings, filehandle)
        
def load_models():
    global recordings
    policy_net = torch.load('./nets/policy_net')
    value_net = torch.load('./nets/value_net')
    reward_net = torch.load('./nets/reward_net')
    with open('./nets/net.data', 'rb') as filehandle:
        recordings = pickle.load(filehandle)

def agent_func():
    global policy_net
    global reward_net
    # global recordings
    local_reward_net = copy.deepcopy(reward_net)

    running_state = ZFilter((num_inputs,), clip=5)
    running_reward = ZFilter((1,), demean=False, clip=10)

    for i_episode in range(0,2001):
        print('doing episode', i_episode, flush=True)
        
        if not reward_net_queue.empty():
            local_reward_net = reward_net_queue.get()

        memory = Memory()

        num_steps = 0
        reward_batch = 0
        num_episodes = 0
        one = random.randint(0,args.batch_size-2)
        two = one + 1
        steps_one = []
        steps_two = []
        states_one = []
        states_two = []
        seed_one = None
        seed_two = None
        
        # while num_steps < args.batch_size:
        print('running batch',end='')
        for idx in range(args.batch_size):
            print('.',end='')
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
                action = select_action(state, policy_net)
                action = action.data[0].numpy()
                next_state, _, _, _ = env.step(action)
                reward = local_reward_net(torch.cat((torch.from_numpy(next_state),torch.from_numpy(action)), 0))
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
        print('')

        reward_batch /= num_episodes
        batch = memory.sample()
        update_params(batch)

        # print('getting lock')
        # got_lock = recording_lock.acquire(blocking=False)
        # print(f'recordings size {len(recordings)}')
        # if got_lock:
        # for x in local_recordings:
        # recordings.append((steps_one, states_one, steps_two, states_two, seed_one, seed_two))
        trajectories_queue.put((steps_one, states_one, steps_two, states_two, seed_one, seed_two))
        if i_episode%1==0:
            recordings_queue.put((steps_one, states_one, steps_two, states_two, seed_one, seed_two, i_episode+1))
        # local_recordings = []
        # recordings.append((steps_one, states_one, steps_two, states_two, seed_one, seed_two))
        # recording_lock.release()
        # else:
        #     print("addind to local recordings", flush=True)
        #     local_recordings.append((steps_one, states_one, steps_two, states_two, seed_one, seed_two))

human_choices_dict = {}

@app.route("/human_choice/<choice>", methods = ['POST'])
def human_choice(choice):
    global displayed_episode 
    global episode_number
    global recordings
    global human_choices_dict
    human_choices_dict[choice] = choice
    if not update_reward_net(displayed_episode, choice):
        return jsonify(displayed_episode=displayed_episode, recordings=len(recordings), status='failed')

    while True:
        l.acquire()
        if episode_number.value == displayed_episode:
            l.release()
        else:
            copyfile(f'{videos_path}video_one_{episode_number.value}.mp4', f'../frontend/public/videos/video_one_{episode_number.value}.mp4')
            copyfile(f'{videos_path}video_two_{episode_number.value}.mp4', f'../frontend/public/videos/video_two_{episode_number.value}.mp4')
            displayed_episode = episode_number.value
            l.release()
            break
    return jsonify(displayed_episode=displayed_episode, recordings=len(recordings), status='success')

@app.route("/displayed_episode", methods = ['GET'])
def get_displayed_episode():
    global displayed_episode
    return jsonify(displayed_episode=displayed_episode)

@app.route("/human_choice/<episode_id>", methods = ['GET'])
def get_human_choice(episode_id):
    global human_choices_dict
    if episode_id in human_choices_dict:
        return jsonify(choice=human_choices_dict[episode_id])
    return jsonify(choice=4)

@app.route("/episode_number")
def get_episode_number():
    global episode_number
    l.acquire()
    x = episode_number.value
    l.release()
    return jsonify(episode_number=x)

def start_server():
    app.run(debug=False)

if __name__ == "__main__":
    # if args.load:
        # print('\n\n\n loading \n\n\n', flush=True)
    # policy_net = torch.load('./nets/policy_net120')
    # value_net = torch.load('./nets/value_net120')
    # reward_net = torch.load('./nets/reward_net120')
    # with open('./nets/net.data', 'rb') as filehandle:
    #     recordings = pickle.load(filehandle)
    Process(target=agent_func).start()
    Process(target=start_server, args=()).start()
    # threading.Thread(target=start_server).start()
    process_renderer(recordings_queue, 1)
