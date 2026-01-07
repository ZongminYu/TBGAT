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
# torch.set_printoptions(
#     threshold=float('inf'),  # 不省略任何元素
# )

input_dim_ = 3
output_dim_ = 128
hidden_dim_ = 512
decoder_input_dim_ = 128+3+128+3+128+1+1  # h_u, x_u, h_v, x_v, h_g, P, T

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
        self.logger = getLogger(name='root')
        self.encoder = LSI_Encoder()
        self.policy = LSI_Decoder()

    def forward(self, pyg_sol, feasible_action, optimal_mark, cmax=None, P=None, T=None, action_instance_id=None, drop_out=0):
        x = pyg_sol.x[:, [0, 1, 2]]  # only use three features: dur_fea, est_fea, lst_fea
        edge_index = pyg_sol.edge_index
        batch = pyg_sol.batch
        # x.torch.Size([batch * (j * m + 2), 3])
        # len(feasible_action) = batch


        node_h = self.encoder(x)
        # node_h.torch.Size([batch * (j * m + 2), output_dim_])



        sampled_action, log_prob_padded, entropy_padded = self.move_selector_lsi(action_set = feasible_action,
                         node_h = node_h,
                         optimal_mark = optimal_mark,
                         batch=batch,
                         x=x,
                         action_instance_id=action_instance_id,
                         P=P,
                         T=T)


        return sampled_action, log_prob_padded, entropy_padded
    
    def move_selector_lsi(self,
                         action_set,
                         node_h,
                         optimal_mark,
                         batch=None,
                         x=None,
                         action_instance_id=None,
                         P=None,
                         T=None):

        ## compute action probability
        # get action embedding ready
        action_merged_with_tabu_label = torch.cat([actions[0] for actions in action_set if actions], dim=0)
        actions_merged = action_merged_with_tabu_label[:, :2]
        tabu_label = action_merged_with_tabu_label[:, [2]]
        

        h_g = global_mean_pool(node_h, batch)  # global embedding, shape: (batch, output_dim_)
        # h_g torch.Size([3, output_dim_])
        hg = h_g[action_instance_id]
        # hg.shape: (all_actions_num, output_dim_)

        u = actions_merged[:, 0]
        v = actions_merged[:, 1]
        
        h_u = node_h[u]
        x_u = x[u]
        h_v = node_h[v]
        x_v = x[v]

        # self.logger.info("into move_selector_lsi")
        # self.logger.info("h_u.shape: {}".format(h_u.shape))
        # self.logger.info("x_u.shape: {}".format(x_u.shape))
        # self.logger.info("h_v.shape: {}".format(h_v.shape))
        # self.logger.info("x_v.shape: {}".format(x_v.shape))
        # self.logger.info("h_g.shape: {}".format(h_g.shape))
        # self.logger.info("P.shape: {}".format(P.shape))
        # self.logger.info("T.shape: {}".format(T.shape))

        # h_u.shape: torch.Size([17, 128])
        # x_u.shape: torch.Size([17, 3])
        # h_v.shape: torch.Size([17, 128])
        # x_v.shape: torch.Size([17, 3])
        # hg.shape: torch.Size([17, 128])
        # P.shape: torch.Size([17])
        # T.shape: torch.Size([17])


        action_h = torch.cat(
            [
                node_h[u],       # [all_actions_num, 128]
                x[u],            # [all_actions_num, 3]
                node_h[v],       # [all_actions_num, 128]
                x[v],            # [all_actions_num, 3]
                hg,              # [all_actions_num, 128]
                P.unsqueeze(-1), # [all_actions_num] → [all_actions_num, 1]
                T.unsqueeze(-1)  # [all_actions_num] → [all_actions_num, 1]
            ],
            dim=-1  # 按最后一维（列维度）拼接
        )

        if not args.embed_tabu_label:
            action_h = action_h[:, :-1]

        # self.logger.info("action_h.shape: {}".format(action_h.shape))
        # self.logger.info("action_h: {}".format(action_h))
        # compute action score
        action_count = [actions[0].shape[0] for actions in action_set if actions]  # if no action then ignore
        # self.logger.info("action_count: {}".format(action_count))
        action_score = self.policy(action_h)
        # self.logger.info("action_score.shape: {}".format(action_score.shape))
        _max_count = max(action_count)
        actions_score_split = list(torch.split(action_score, split_size_or_sections=action_count))
        padded_score = pad_sequence(actions_score_split, padding_value=-torch.inf).transpose(0, -1).transpose(0, 1)

        # sample actions
        pi = F.softmax(padded_score, dim=-1)
        dist = Categorical(probs=pi)
        action_id = dist.sample()
        padded_action = pad_sequence(
            [actions[0][:, :2] for actions in action_set if actions],
        ).transpose(0, 1)
        sampled_action = torch.gather(
            padded_action, index=action_id.repeat(1, 2).view(-1, 1, 2), dim=1
        ).squeeze(dim=1)
        # self.logger.info(feasible_action)
        # self.logger.info(action_id)
        # self.logger.info(sampled_action)

        # greedy action
        # action_id = torch.argmax(pi, dim=-1)

        # compute log_p and policy entropy regardless of optimal sol
        log_prob = dist.log_prob(action_id)
        entropy = dist.entropy()

        # compute padded log_p, where optimal sol has 0 log_0, since no action, otherwise cause shape bug
        log_prob_padded = torch.zeros(
            size=optimal_mark.shape,
            device=action_h.device,
            dtype=torch.float
        )
        log_prob_padded[~optimal_mark, :] = log_prob.squeeze()

        # compute padded ent, where optimal sol has 0 ent, since no action, otherwise cause shape bug
        entropy_padded = torch.zeros(
            size=optimal_mark.shape,
            device=action_h.device,
            dtype=torch.float
        )
        entropy_padded[~optimal_mark, :] = entropy.squeeze()

        # self.logger.info("sampled_action.shape: {}".format(sampled_action.shape))
        # self.logger.info("sampled_action: {}".format(sampled_action))
        # self.logger.info("log_prob_padded.shape: {}".format(log_prob_padded.shape))
        # self.logger.info("log_prob_padded: {}".format(log_prob_padded))
        # self.logger.info("entropy_padded.shape: {}".format(entropy_padded.shape))
        # self.logger.info("entropy_padded: {}".format(entropy_padded))
        return sampled_action, log_prob_padded, entropy_padded

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
            nn.Linear(hidden_dim, output_dim_)
        )

    def forward(self, x):
        # data.shape: (batch, j, m, feature_dim)
        
        node_h = self.layers(x)
        # shape: (batch, j, k, hidden_dim)

        return node_h

########################################
# DECODER
########################################

class LSI_Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        hidden_dim = hidden_dim_

        self.layers = nn.Sequential(
            nn.Linear(decoder_input_dim_, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    def forward(self, h_uv):
        """
        参数说明：
        h_uv: 所有候选action的特征，shape=(all_actions_num, decoder_input_dim_)
        """
        action_scores = self.layers(h_uv)  
        # shape: (all_actions_num, 1)
        return action_scores

if __name__ == '__main__':
    import time
    import random
    from env.generateJSP import uni_instance_gen
    from env.LSI_environment import Env

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
    # self.logger.info(insts)

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

    # self.logger.info(env.instance_size)

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
