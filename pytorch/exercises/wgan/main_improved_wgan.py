# encoding: utf-8

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def list_all_files(rootdir, key):
    import os
    _files = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path, key))
        if os.path.isfile(path) and key in path:
            _files.append(path)
    return _files


def load_data():
    global df, df_ind
    root = '.'
    key = '10_'
    files = list_all_files(root, key)
    for f in files:
        yield extract_signal(f)


def extract_signal(f):
    data = pd.read_table(f, header=None, skiprows=1)
    rawdata = np.array(data.iloc[:, 19:20])
    force_flag = np.array(data.iloc[:, 2])
    tds = np.where(np.diff(force_flag) == 1)[0]
    # print(len(tds))
    x_data = np.array([np.diff(rawdata[i - 3:i + 13, :], axis=0).T.flatten() for i in tds])
    x_data = x_data[np.max(x_data, axis=1) > 20]
    x_data = x_data.T
    x_data = np.apply_along_axis(
        lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)), 0, x_data
    )
    x_data = (x_data - 0.5) * 2
    # print(x_data.max(), x_data.min())
    return x_data


def make_data():
    sample = np.random.choice(datas.shape[0], batch_size, False)
    return datas[sample]


def make_noise():
    return np.random.uniform(-1, 1, (batch_size, generator_len))


def train():
    d_optim = torch.optim.Adam(D.parameters(), d_lr, betas=(0.5, 0.9))
    g_optim = torch.optim.Adam(G.parameters(), g_lr, betas=(0.5, 0.9))

    plt.ion()
    wd = []

    for epoch in range(epochs):
        D.train(), G.train()
        for ci in range(critic_iters):
            data_batch = make_data()
            gen_batch = make_noise()
            data_batch, gen_batch = Variable(torch.FloatTensor(data_batch)), \
                                    Variable(torch.FloatTensor(gen_batch))
            d_loss = -torch.mean(D(data_batch)) + torch.mean(D(G(gen_batch))) + calc_gradient_penalty(data_batch,
                                                                                                     G(gen_batch))
            wasserstein_distance = -torch.mean(D(G(gen_batch))) + torch.mean(D(data_batch))
            print(wasserstein_distance.item())

            # d_loss = -torch.mean(torch.log(D(data_batch)) + torch.log(1 - D(G(gen_batch))))
            # g_loss = torch.mean(torch.log(1 - D(G(gen_batch))))

            d_optim.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optim.step()

        data_batch = make_data()
        gen_batch = make_noise()
        data_batch, gen_batch = Variable(torch.FloatTensor(data_batch)), \
                                Variable(torch.FloatTensor(gen_batch))
        g_loss = -torch.mean(D(G(gen_batch)))
        g_optim.zero_grad()
        g_loss.backward()
        g_optim.step()

        if epoch % 50 == 0:
            D.eval(), G.eval()
            plt.clf()
            plt.suptitle('epoch=%d, w-dist=%.6f' % (epoch, wasserstein_distance.item()))
            wd.append(wasserstein_distance.item())
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                gen_diff = G(gen_batch).detach().numpy()
                gen_raw = np.hstack((np.cumsum(gen_diff[:, :int(data_len / 2)], axis=1),
                                     np.cumsum(gen_diff[:, int(data_len / 2):], axis=1)))
                plt.plot(gen_raw[i])
                plt.xlim((0, data_len))
                plt.ylim((-1, 1))
            plt.pause(0.01)
    plt.ioff()
    plt.figure()
    plt.plot(wd)
    plt.show()


def calc_gradient_penalty(x_real, x_gen):
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(x_real.size())
    x_hat = alpha * x_real + (1 - alpha) * x_gen
    D_x = D(x_hat)
    gradients = torch.autograd.grad(
        outputs=D_x,
        inputs=x_hat,
        grad_outputs=torch.ones(D_x.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    # print(gradients)
    gradient_penalty = gp_lambda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    print(gradient_penalty)
    return gradient_penalty


if __name__ == '__main__':
    datas = load_data()
    datas = np.hstack([d for d in datas]).T

    batch_size = 32
    generator_len = 20
    data_len = 15
    epochs = 200000
    d_lr = 0.000001
    g_lr = 0.000001
    gp_lambda = 0.1
    critic_iters = 5

    D = nn.Sequential(
        nn.Linear(data_len, 32),
        # nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(32, 16),
        # nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(16, 4),
        # nn.Dropout(0.5),
        nn.ReLU(),
        nn.Linear(4, 1)
    )

    G = nn.Sequential(
        nn.Linear(generator_len, 30),
        nn.ReLU(),
        nn.Linear(30, 30),
        nn.ReLU(),
        nn.Linear(30, data_len),
        nn.Tanh()
    )

    x = np.tile(np.linspace(-1, 1, data_len), [batch_size, 1])
    # make_data()
    train()
    torch.save(D, 'D.model')
    torch.save(G, 'G.model')

    # D_ = torch.load('D.model')
    # G_ = torch.load('G.model')
    # print(D_, G_)
    # batch_size = 1000
    # gen_data = make_noise()
    # print(gen_data.shape)
    # gen_data = G_(Variable(torch.FloatTensor(gen_data))).detach().numpy()
    # plt.ion()
    # for i in range(gen_data.shape[0]):
    #     plt.cla()
    #     plt.plot(np.cumsum(gen_data[i, :int(data_len / 2)]))
    #     plt.plot(np.cumsum(gen_data[i, int(data_len / 2):]))
    #     # gen_raw = np.hstack((np.cumsum(gen_data[:, :int(data_len / 2)], axis=1),
    #     #                      np.cumsum(gen_data[:, int(data_len / 2):], axis=1)))
    #     # plt.plot(gen_raw[i])
    #     plt.pause(0.2)
