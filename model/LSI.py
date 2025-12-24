import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GATConv, global_mean_pool, GINConv
from torch_geometric.utils import add_self_loops, sort_edge_index
from torch_geometric.data.batch import Batch
from torch.nn.utils.rnn import pad_sequence
from parameters import args
import torch.nn as nn

from logging import getLogger
torch.set_printoptions(
    threshold=float('inf'),  # 不省略任何元素
)

input_dim_ = 3
hidden_dim_ = 128

class LSI_Model(nn.Module):

    def __init__(self,
                 # embedding parameters
                 in_channels_fwd,
                 in_channels_bwd,
                 hidden_channels,
                 out_channels,
                 heads=4,
                 dropout_for_gat=0
                 ):

        super().__init__()
        self.logger = getLogger(name='trainer')
        self.encoder = LSI_Encoder()
        self.decoder = LSI_Decoder()

    def forward(self, pyg_sol, feasible_action, optimal_mark, cmax=None, drop_out=0):
        encoder_operations = self.encoder(pyg_sol)
        # shape: (batch, j, k, hidden_dim)
        
        probs = self.decoder(encoder_operations)


        return probs


########################################
# ENCODER
########################################

class LSI_Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        input_dim = input_dim_
        hidden_dim = hidden_dim_

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )

    def forward(self, data):
        # data.shape: (batch, j, m, feature_dim)

        out = self.layers(data)
        # shape: (batch, j, k, hidden_dim)

        return out

########################################
# DECODER
########################################

class LSI_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_dim = hidden_dim_

        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, encoded_last_node, ninf_mask):
        # encoded_last_node.shape: (batch, B, embedding)
        # ninf_mask.shape: (batch, B, problem)

        action_scores = self.layers(encoded_last_node)

        probs = F.softmax(score_masked, dim=2)
        # shape: (batch, B, problem)

        return probs


if __name__ == '__main__':
    import time
    import random
    from env.generateJSP import uni_instance_gen
    from env.environment import Env

    # j, m, batch_size = {'low': 100, 'high': 101}, {'low': 20, 'high': 21}, 500
    # j, m, batch_size = {'low': 30, 'high': 31}, {'low': 20, 'high': 21}, 64
    # j, m, batch_size = {'low': 10, 'high': 11}, {'low': 10, 'high': 11}, 128
    # j, m, batch_size = {'low': 20, 'high': 21}, {'low': 15, 'high': 16}, 64
    j, m, batch_size = {'low': 10, 'high': 11}, {'low': 10, 'high': 11}, 3
    # [j, m], batch_size = [np.array(
    #     [[15, 15],  # Taillard
    #      [20, 15],
    #      [20, 20],
    #      [30, 15],
    #      [30, 20],
    #      [50, 15],
    #      [50, 20],
    #      [100, 20],
    #      [10, 10],  # ABZ, not include: 20x15
    #      [6, 6],  # FT, not include: 10x10
    #      [20, 5],
    #      [10, 5],  # LA, not include: 10x10, 15x15
    #      [15, 5],
    #      [20, 5],
    #      [15, 10],
    #      [20, 10],
    #      [30, 10],
    #      [50, 10],  # SWV, not include: 20x10, 20x15
    #      # no ORB 10x10, no YN 20x20
    #      ])[:, _] for _ in range(2)], 128

    l = 1
    h = 99
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    init_type = 'fdd-divide-wkr'  # 'spt', 'fdd-divide-mwkr'
    seed = 25  # 6: two paths for the second instance
    np.random.seed(seed)
    backward_option = False  # if true usually do not pass N5 property
    print_step_time = True
    print_time_of_calculating_moves = True
    print_action_space_compute_time = True
    path_finder = 'pytorch'  # 'networkx' or 'pytorch'
    tb_size = -1  # tabu_size
    torch.random.manual_seed(seed)
    random.seed(seed)

    if type(j) is dict and type(m) is dict:  # random range
        insts = [np.concatenate(
            [uni_instance_gen(n_j=np.random.randint(**j), n_m=np.random.randint(**m), low=l, high=h)]
        ) for _ in range(batch_size)]
    else:  # random from set
        insts = []
        for _ in range(batch_size):
            i = random.randint(0, j.shape[0] - 1)
            inst = np.concatenate(
                [uni_instance_gen(n_j=j[i], n_m=m[i], low=l, high=h)]
            )
            insts.append(inst)

    # insts = np.load('../test_data_jssp/tai20x15.npy')
    # print(insts)

    env = Env()
    G, (feasible_a, mark, paths) = env.reset(
        instances=insts,
        init_sol_type=init_type,
        tabu_size=tb_size,
        device=dev,
        mask_previous_action=backward_option,
        longest_path_finder=path_finder
    )

    env.cpm_eval()

    # print(env.instance_size)

    net = LSI_Model(
        in_channels_fwd=3,
        in_channels_bwd=3,
        hidden_channels=128,
        out_channels=128,
        heads=4,
        dropout_for_gat=0
    ).to(dev)

    data = []
    h_embd = None
    g_embd = None
    log_p = None
    ent = None
    for _ in range(5):
        t1_ = time.time()
        print('step {}'.format(_))

        sampled_a, log_p, ent = net(
            pyg_sol=G,
            feasible_action=feasible_a,
            optimal_mark=mark,
            cmax=env.current_objs
        )

        G, reward, (feasible_a, mark, paths) = env.step(
            action=sampled_a,
            prt=print_step_time,
            show_action_space_compute_time=print_action_space_compute_time
        )
        # print(env.current_objs)
        # print(env.incumbent_objs)
        t2_ = time.time()
        print("This iteration takes: {:.4f}".format(t2_ - t1_))
        print()

    env.cpm_eval()

    loss = log_p.mean()
    grad = torch.autograd.grad(loss + torch.tensor(1., requires_grad=True), [param for param in net.parameters()])

    log_p_normal = log_p.clone()
    # print(log_p_normal)

    # parameter after backward with normal log_p
    # import torch.optim as optim
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # print(log_p_normal)
    # loss = log_p_normal.mean()
    # # backward
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # print([param for param in net.parameters()])

    # parameter after backward with mean of dummy log_p and normal log_p, should be equal with that of normal log_p,
    # since dummy log_p affect nothing
    # sampled_a, log_p_dummy, ent_dummy = net(
    #     pyg_sol=G,
    #     feasible_action=[[], [], []],
    #     optimal_mark=mark,
    #     critical_path=paths
    # )
    # import torch.optim as optim
    # optimizer = optim.Adam(net.parameters(), lr=0.0001)
    # loss = torch.cat([log_p_dummy, log_p_normal], dim=-1).sum(dim=-1)
    # # backward
    # optimizer.zero_grad()
    # loss.mean().backward()
    # optimizer.step()
    # print([param for param in net.parameters()])
