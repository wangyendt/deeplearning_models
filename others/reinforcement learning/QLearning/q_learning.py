#!/usr/bin/env python
# encoding: utf-8

"""
@author: Wayne
@contact: wangye.hope@gmail.com
@software: PyCharm
@file: q_learning
@time: 2020/4/20 16:09
"""

import numpy as np
import pandas  as pd
import time

np.random.seed(1)
N_STATES = 10
ACTIONS = ['left', 'right']
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9
MAX_EPISODES = 13
FRESH_TIME = 0.3


def build_q_table(n_states, actions):
    table = pd.DataFrame(
        np.zeros((n_states, len(actions))), columns=actions
    )
    return table


def choose_action(state, q_table):
    state_actions = q_table.loc[state, :]
    if np.random.uniform() < EPSILON or state_actions.all() == 0:
        action_name = np.random.choice(ACTIONS)
    else:
        action_name = ACTIONS[state_actions.argmax()]
    # print(state_actions,action_name)
    return action_name


def get_env_feedback(S, A):
    if A == 'right':
        if S == N_STATES - 2:
            S_ = 'terminal'
            R = 1
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S
        else:
            S_ = S - 1
    return S_, R


def update_env(S, episode, step_counter):
    env_list = ['-'] * (N_STATES - 1) + ['T']
    if S == 'terminal':
        interaction = f'\rEpisode {episode + 1}: total_steps={step_counter + 1}'
        print(interaction, end='')
        time.sleep(2)
        print('\r                                                          ', end='')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print(f'\r{interaction}', end='')
        time.sleep(FRESH_TIME)


def rl():
    q_table = build_q_table(N_STATES, ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S, episode, step_counter)
        while not is_terminated:
            A = choose_action(S, q_table)
            S_, R = get_env_feedback(S, A)
            q_pred = q_table.loc[S, A]
            if S_ != 'terminal':
                q_target = R + GAMMA * q_table.iloc[S_, :].max()
            else:
                q_target = R
                is_terminated = True
            q_table.loc[S, A] += ALPHA * (q_target - q_pred)
            S = S_
            update_env(S, episode, step_counter + 1)
            step_counter += 1
        print('\r\n', q_table)
    return q_table


if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)
