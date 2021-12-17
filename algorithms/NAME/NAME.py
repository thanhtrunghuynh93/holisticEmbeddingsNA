from algorithms.network_alignment_model import NetworkAlignmentModel

from evaluation.metrics import get_statistics
from algorithms.NAME.embedding_model import NAME_MODEL, StableFactor, CombineModel, Combine2Model
from input.dataset import Dataset
from utils.graph_utils import load_gt
import torch.nn.functional as F
import torch.nn as nn
from algorithms.NAME.utils import *
from algorithms.NAME.losses import *

import torch
import numpy as np
import networkx as nx
import random 
import numpy as np
from time import time

import argparse
import os
import time
import sys

from simple_classify.gumbel import gumbel_softmax
from simple_classify.models import Model

# IMPORT PALE
from algorithms.PALE.embedding_model import PaleEmbedding
from algorithms.PALE.mapping_model import PaleMappingLinear
from torch.autograd import Variable
from tqdm import tqdm
import copy


class NAME(NetworkAlignmentModel):
    """
    NAME model for networks alignment task
    """
    def __init__(self, source_dataset, target_dataset, args):
        super(NAME, self).__init__(source_dataset, target_dataset)
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset

        self.alphas = [args.alpha0, args.alpha1, args.alpha2]
        self.args = args
        # full dict is just used in the process of investigating and need to be DELETE later
        self.full_dict = load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        self.gt_train = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')

    
    def align(self):
        source_A_hat, target_A_hat, source_feats, target_feats = self.get_elements()
        source_adj = self.source_dataset.get_adjacency_matrix()
        target_adj = self.target_dataset.get_adjacency_matrix()

        new_source_A_hat, new_source_feats, new_deg, new_edges, new_adj, id2idx_augment = self.graph_augmentation(self.source_dataset,target_adj)

        new_adj = Variable(torch.FloatTensor(new_adj), requires_grad = False)
        source_adj = Variable(torch.FloatTensor(source_adj), requires_grad = False)
        target_adj = Variable(torch.FloatTensor(target_adj), requires_grad = False)
        if self.args.cuda:
            new_adj = new_adj.cuda()
            source_adj = source_adj.cuda()
            target_adj = target_adj.cuda()

        new_source_info = {'num_nodes': len(new_source_feats), 'deg': new_deg, 'edges': new_edges, 'adj': new_adj}
        source_info = {'num_nodes': len(source_feats), 'deg': self.source_dataset.get_nodes_degrees(), 'edges': self.source_dataset.get_edges(), 'adj': source_adj}
        target_info = {'num_nodes': len(target_feats), 'deg': self.target_dataset.get_nodes_degrees(), 'edges': self.target_dataset.get_edges(), 'adj': target_adj}

        test_dict = copy.deepcopy(self.full_dict)
        train_dict = copy.deepcopy(self.gt_train)
        self.full_dict = {u:v for (u,v) in id2idx_augment.items() if random.uniform(0,1) > 0.1}
        self.gt_train = {u:v for (u,v) in id2idx_augment.items() if u not in list(self.full_dict.keys())}
        S_NAME, S_pale, S_mincut = self.get_multi_align(source_A_hat, new_source_A_hat, source_feats, new_source_feats, source_info, new_source_info)
        combine_model = CombineModel()
        combine_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, combine_model.parameters()), lr=self.args.lr)

        for epochs in tqdm(range(200)):
            combine_optimizer.zero_grad()
            loss = combine_model.loss(S_NAME, S_pale, S_mincut, id2idx_augment)
            loss.backward()
            combine_optimizer.step()
            print("Loss: {:.4f}".format(loss.item()))
        
        print(combine_model.thetas)
        S_pale = S_pale.detach().cpu().numpy()
        S_NAME = S_NAME.detach().cpu().numpy()
        S_mincut = S_mincut.detach().cpu().numpy()
        self.full_dict = test_dict
        self.gt_train = train_dict
        S_NAME, S_pale, S_mincut = self.get_multi_align(source_A_hat, target_A_hat, source_feats, target_feats, source_info, target_info)
        S = combine_model(S_NAME, S_pale, S_mincut)
        print('-'*100)
        S = S.detach().cpu().numpy()
        return S


    def get_multi_align(self, source_A_hat, target_A_hat, source_feats, target_feats, source_info, target_info):
        """
        step1: align by GCN
        step2: find stables nodes
        step3: run pale with stabels nodes
        step4: run mincut with stabels nodes


        """
        NAME_start = time()
        NAME = NAME_MODEL(
            activate_function = self.args.act,
            num_GCN_blocks = self.args.num_GCN_blocks,
            input_dim = self.args.input_dim,
            output_dim = self.args.embedding_dim,
            num_source_nodes = len(source_A_hat),
            num_target_nodes = len(target_A_hat),
            source_feats = source_feats,
            target_feats = target_feats
        )


        if self.args.cuda:
            NAME = NAME.cuda()

        NAME.train()

        structural_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, NAME.parameters()), lr=self.args.lr)

        NAME_S = self.train_NAME(NAME, source_A_hat, target_A_hat, structural_optimizer, self.args.threshold)

        gt_train = self.gt_train
        print('NAME running time: {:.4f}'.format(time() - NAME_start))
        # Step2: PALE
        start = time()
        self.args.cuda = True
        source_info['adj'] = source_info['adj'].cuda()
        target_info['adj'] = target_info['adj'].cuda()

        source_pale, target_pale = self.learn_pale_embs(source_info, target_info)

        pale_map = PaleMappingLinear(
                                    embedding_dim=self.args.embedding_dim,
                                    source_embedding=source_pale,
                                    target_embedding=target_pale,
                                    )

        pale_S = self.map_source_target_pale(pale_map, list(gt_train.keys()), gt_train, source_pale, target_pale)
        print('pale running time: {:.4f}'.format(time() - start))
        # Step3: Mincut
        mincut_start = time()
        source_adj = source_info['adj']
        target_adj = target_info['adj']
        source_mincut, target_mincut = self.learn_mincut_embs(source_adj, target_adj)
        pale_map2 = PaleMappingLinear(
                                    embedding_dim=self.args.num_parts,
                                    source_embedding=source_mincut,
                                    target_embedding=target_mincut,
                                    )

        mincut_S = self.map_source_target_pale(pale_map2, list(gt_train.keys()), gt_train, source_mincut, target_mincut)
        print('mincut running time: {:.4f}'.format(time() - mincut_start))
        NAME_S = torch.FloatTensor(NAME_S)
        pale_S = torch.FloatTensor(pale_S)
        mincut_S = torch.FloatTensor(mincut_S)
        # self.args.cuda = False

        return NAME_S, pale_S, mincut_S


    def learn_mincut_embs(self, adj_source, adj_target):
        source_mincut = self.learn_mincut(len(adj_source), adj_source)
        target_mincut = self.learn_mincut(len(adj_target), adj_target)

        return source_mincut, target_mincut


    def learn_mincut(self, num_nodes, adj):
        model = Model(num_nodes, self.args.num_parts)
        if self.args.cuda:
            model = model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=self.args.mincut_lr)

        temp=self.args.temp

        for epoch in tqdm(range(self.args.mincut_epochs)):
            model.zero_grad()
            nodes = None
            super_adj = model(adj, nodes, temp=temp, hard=self.args.hard, beta=self.args.beta)
            loss, ncut_loss, balance_loss = model.loss(super_adj, nodes, balance_node=self.args.balance_node, lam=self.args.lam, w2v_lam = self.args.w2v_lam, new=self.args.new)

            if loss!=loss: import pdb;pdb.set_trace()
            total_loss = loss
            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), self.args.clip)
            optimizer.step()
            if model.params.max() != model.params.max():import pdb;pdb.set_trace()

            if epoch % 500==0:
                if self.args.anneal:
                    temp = min(self.args.min_temp, self.args.temp * np.exp(-0.00003*epoch))
                try:
                    print("loss:", loss.item())
                    print("ncut_loss: ", ncut_loss.item())
                    print("balance_loss: ", balance_loss.item())
                except:
                    import pdb; pdb.set_trace()

        
        params = model.params.max(dim=1)[1].cpu().detach().numpy()
        embedding = gumbel_softmax(model.params, temp=10, hard=False, beta=1)
        embedding_onehot = np.zeros(embedding.shape)
        for i in range(len(embedding_onehot)):
            embedding_onehot[i][params[i]] = 1
        # embedding_onehot[:, params] = 1
        embedding_onehot = torch.FloatTensor(embedding_onehot)
        if self.args.cuda:
            embedding_onehot = embedding_onehot.cuda()
        return embedding_onehot

    
    def map_source_target_pale(self, pale_map, source_train_nodes, gt_train, source_pale, target_pale):
        """
        source_train_nodes: Numpy array
        gt_train: dictionary
        """
        if self.args.cuda:
            pale_map = pale_map.cuda()

        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, pale_map.parameters()), lr=self.args.pale_map_lr)

        pale_map_batchsize = len(source_train_nodes) // 4
        n_iters = len(source_train_nodes) // pale_map_batchsize
        assert n_iters > 0, "batch_size is too large"
        if(len(source_train_nodes) % pale_map_batchsize > 0):
            n_iters += 1
        total_steps = 0
        n_epochs = self.args.pale_map_epochs
        for epoch in range(1, n_epochs + 1):
            # for time evaluate
            start = time.time()
            print('Epochs: ', epoch)
            np.random.shuffle(source_train_nodes)
            for iter in range(n_iters):
                source_batch = source_train_nodes[iter*pale_map_batchsize:(iter+1)*pale_map_batchsize]
                target_batch = [gt_train[x] for x in source_batch]
                source_batch = torch.LongTensor(source_batch)
                target_batch = torch.LongTensor(target_batch)
                if self.args.cuda:
                    source_batch = source_batch.cuda()
                    target_batch = target_batch.cuda()
                optimizer.zero_grad()
                loss = pale_map.loss(source_batch, target_batch)
                loss.backward()
                optimizer.step()
            
                total_steps += 1
            print("mapping_loss: {:.4f}".format(loss.item()))
            self.mapping_epoch_time = time.time() - start

        source_pale_map = pale_map(source_pale)
        self.S_pale = torch.matmul(source_pale_map, target_pale.t())
        self.S_pale = self.S_pale.detach().cpu().numpy()

        return self.S_pale


    def learn_pale_embs(self, source_info, target_info):

        num_source_nodes = source_info['num_nodes']
        source_deg = source_info['deg']
        source_edges = source_info['edges']

        num_target_nodes = target_info['num_nodes']
        target_deg = target_info['deg']
        target_edges = target_info['edges']

        #source_edges, target_edges = self.extend_edge(source_edges, target_edges)
        
        print("Done extend edges")
        source_pale = self.learn_pale(num_source_nodes, source_deg, source_edges) #, 's')
        target_pale = self.learn_pale(num_target_nodes, target_deg, target_edges) #, 't')
        return source_pale, target_pale


    def learn_pale(self, num_nodes, deg, edges):
        pale_model = PaleEmbedding(
                                    n_nodes = num_nodes,
                                    embedding_dim = self.args.embedding_dim,
                                    deg= deg,
                                    neg_sample_size = 10,
                                    cuda = self.args.cuda,
                                    )
        if self.args.cuda:
            pale_model = pale_model.cuda()

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, pale_model.parameters()), lr=self.args.pale_emb_lr)
        embedding = self.train_pale_emb(pale_model, edges, optimizer)
        return embedding
    

    def train_pale_emb(self, embedding_model, edges, optimizer):
        num_edges = len(edges)
        n_iters = num_edges // self.args.pale_emb_batchsize
        assert n_iters > 0, "batch_size is too large!"
        if(num_edges % self.args.pale_emb_batchsize > 0):
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        n_epochs = self.args.pale_emb_epochs
        for epoch in range(1, n_epochs + 1):
            # for time evaluate
            start = time.time()
            if self.args.log:
                print("Epoch {0}".format(epoch))
            np.random.shuffle(edges)
            loss, loss0, loss1 = 0, 0, 0
            for iter in range(n_iters):
                batch_edges = torch.LongTensor(edges[iter*self.args.pale_emb_batchsize:(iter+1)*self.args.pale_emb_batchsize])
                if self.args.cuda:
                    batch_edges = batch_edges.cuda()
                optimizer.zero_grad()
                loss, loss0, loss1 = embedding_model.loss(batch_edges[:, 0], batch_edges[:,1])
                loss.backward()
                optimizer.step()
                total_steps += 1
            if self.args.log:
                print(
                        "train_loss=", "{:.5f}".format(loss.item()),
                        "true_loss=", "{:.5f}".format(loss0.item()),
                        "neg_loss=", "{:.5f}".format(loss1.item()),
                        "time", "{:.5f}".format(time.time()-start)
                    )
            # for time evaluate
            self.embedding_epoch_time = time.time() - start
            
        embedding = embedding_model.get_embedding()
        embedding = embedding.cpu().detach().numpy()
        embedding = torch.FloatTensor(embedding)
        if self.args.cuda:
            embedding = embedding.cuda()

        return embedding


    def get_elements(self):
        """
        Compute Normalized Laplacian matrix
        Preprocessing nodes attribute
        """
        source_A_hat, _ = Laplacian_graph(self.source_dataset.get_adjacency_matrix())
        target_A_hat, _ = Laplacian_graph(self.target_dataset.get_adjacency_matrix())
        if self.args.cuda:
            source_A_hat = source_A_hat.cuda()
            target_A_hat = target_A_hat.cuda()

        source_feats = self.source_dataset.features
        target_feats = self.target_dataset.features

        if source_feats is None:
            source_feats = np.zeros((len(self.source_dataset.G.nodes()), 1))
            target_feats = np.zeros((len(self.target_dataset.G.nodes()), 1))
        
        for i in range(len(source_feats)):
            if source_feats[i].sum() == 0:
                source_feats[i, -1] = 1
        for i in range(len(target_feats)):
            if target_feats[i].sum() == 0:
                target_feats[i, -1] = 1
        if source_feats is not None:
            source_feats = torch.FloatTensor(source_feats)
            target_feats = torch.FloatTensor(target_feats)
            if self.args.cuda:
                source_feats = source_feats.cuda()
                target_feats = target_feats.cuda()
        # Norm2 normalization
        source_feats = F.normalize(source_feats)
        target_feats = F.normalize(target_feats)
        # features is okey, just A_hat, A_hat is okey too
        return source_A_hat, target_A_hat, source_feats, target_feats


    def graph_augmentation(self, dataset, target_adj, type_aug='remove_edges'):
        """
        Generate small noisy graph from original graph
        :params dataset: original graph
        :params type_aug: type of noise added for generating new graph
        """
        t_nodes = target_adj.shape[0]
        t_edges = int(target_adj.sum())/2

        edges = dataset.get_edges()
        adj = dataset.get_adjacency_matrix()
        padder = t_nodes - adj.shape[0]
        if padder > 0:

            adj = nx.to_numpy_matrix(dataset.G)
            edges = list(dataset.G.edges())
            del_edges = random.sample(list(dataset.G.edges()), len(list(dataset.G.edges()))/5)
            for edge in del_edges:
                adj[dataset.id2idx[edge[0]], dataset.id2idx[edge[1]]] = 0
                adj[dataset.id2idx[edge[1]], dataset.id2idx[edge[0]]] = 0
            edges = [(dataset.id2idx[i[0]], dataset.id2idx[i[1]]) for i in edges if i not in del_edges]
            nodes = list(dataset.G.nodes())
            feats = np.copy(dataset.features)
            feats = torch.FloatTensor(feats)
            sidx2idx = {i:i for i in range(len(nodes))}
            deg = adj.sum(axis=1).flatten()
            new_adj_H, _ = Laplacian_graph(adj)
            if self.args.cuda:
                feats = feats.cuda()
                new_adj_H = new_adj_H.cuda()
            deg = np.asarray(deg)[0]
            return new_adj_H, feats, deg, edges, adj, sidx2idx
            
        else:
            nodes = [list(dataset.G.nodes())[0]]
            index = 0
            count = 1
            while count < t_nodes:
                try:
                    nodes.extend([i for i in dataset.G.neighbors(nodes[index]) if i not in nodes])
                    count = len(nodes)
                    index += 1
                except:
                    break
            if nx.number_connected_components(dataset.G) > 1:
                import random
                nodes = random.sample(list(dataset.G.nodes()), t_nodes)
            new_G = dataset.G.subgraph(nodes)
            edges = list(new_G.edges())
            source_idx = dataset.id2idx
            sidx2idx = {source_idx[node]:i for i,node in enumerate(nodes)} # dict {index of source graph: index of augment graph}
            id2idx = {node:i for i,node in enumerate(nodes)}
            new_G = nx.Graph()
            new_G.add_nodes_from(nodes)
            new_G.add_edges_from(edges)
            adj = nx.to_numpy_matrix(new_G)
            feats = np.copy(dataset.features)
            new_feats = np.zeros((len(nodes), feats.shape[1]))
            s_idx2id = {u:v for v,u in source_idx.items()}
            for i,feat in enumerate(feats):
                if s_idx2id[i] in nodes:
                    new_feats[sidx2idx[i]] = feat
            print(new_feats)
            num_eds = adj.sum()/2 - t_edges
            if num_eds > 0:
                while num_eds > 1:
                    edge = random.choice(edges)
                    # if adj.sum(axis = 0)[0,id2idx[edge[0]]] > 1 and adj.sum(axis = 0)[0,id2idx[edge[1]]] > 1:
                    # new_adj = adj + 0
                    adj[id2idx[edge[0]], id2idx[edge[1]]] = 0
                    adj[id2idx[edge[1]], id2idx[edge[0]]] = 0
                    # adj = new_adj + 0
                    edges.remove(edge)
                    num_eds -= 1
                new_feats = torch.FloatTensor(new_feats)
                deg = adj.sum(axis=1).flatten()
                new_adj_H, _ = Laplacian_graph(adj)
                if self.args.cuda:
                    new_feats = new_feats.cuda()
                    new_adj_H = new_adj_H.cuda()
                deg = np.asarray(deg)[0]
                edges = [(id2idx[e[0]],id2idx[e[1]]) for e in edges]
                print('augment nodes: {}, augment edges: {}'.format(adj.shape[0], adj.sum()/2))
                return new_adj_H, new_feats, deg, edges, adj, sidx2idx
            else:
                stat = 2*num_eds / (adj.shape[0]**2 - adj.sum())
                edges = [(id2idx[e[0]],id2idx[e[1]]) for e in edges]
                for i in range(adj.shape[0]):
                    for j in range(i, adj.shape[0]):
                        if adj[i,j] == 0 and random.uniform(0,1) < stat:
                            adj[i,j] = 1
                            adj[j,i] = 1
                            edges.append((i,j))
                new_feats = torch.FloatTensor(new_feats)
                deg = adj.sum(axis=1).flatten()
                new_adj_H, _ = Laplacian_graph(adj)
                if self.args.cuda:
                    new_feats = new_feats.cuda()
                    new_adj_H = new_adj_H.cuda()
                deg = np.asarray(deg)[0]
                print('augment nodes: {}, augment edges: {}'.format(adj.shape[0], adj.sum()/2))
                return new_adj_H, new_feats, deg, edges, adj, sidx2idx


    def linkpred_loss(self, embedding, A):
        pred_adj = torch.matmul(F.normalize(embedding), F.normalize(embedding).t())

        if self.args.cuda:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]).cuda())), dim = 1)
        else:
            pred_adj = F.normalize((torch.min(pred_adj, torch.Tensor([1]))), dim = 1)
        #linkpred_losss = (pred_adj - A[index]) ** 2
        linkpred_losss = (pred_adj - A) ** 2
        linkpred_losss = linkpred_losss.sum() / A.shape[1]
        return linkpred_losss
    

    def gcn_semisup_training(self, NAME, source_A_hat, target_A_hat, optimizer, gamma, k):
        train = np.asarray([[x,y] for (x,y) in self.gt_train.items()])

        for epoch in range(self.args.NAME_epochs):
            optimizer.zero_grad()    
            if self.args.log:
                print("GAlign learning epoch: {}".format(epoch))
            source_outputs = NAME(source_A_hat, 's')
            target_outputs = NAME(target_A_hat, 't')
            # output = torch.cat((source_output, target_output), dim = 0)
            left = train[:,0]
            right = train[:,1]
            left_x1 = source_outputs[-1][left]
            right_x1 = target_outputs[-1][right]
            sup_loss1 = (left_x1 - right_x1) ** 2
            sup_loss1 = sup_loss1.mean()
            left_x2 = source_outputs[-2][left]
            right_x2 = target_outputs[-2][right]
            sup_loss2 = (left_x2 - right_x2) ** 2
            sup_loss2 = sup_loss2.mean()
            sup_loss = sup_loss1 + sup_loss2
            unsup_loss_source = self.linkpred_loss(source_outputs[-1], source_A_hat) + self.linkpred_loss(source_outputs[-2], source_A_hat)
            unsup_loss_target = self.linkpred_loss(target_outputs[-1], target_A_hat) + self.linkpred_loss(target_outputs[-2], target_A_hat)
            unsup_loss = unsup_loss_source + unsup_loss_target
            loss = sup_loss + unsup_loss
            loss.backward()
            #loss = unsup_loss
            print('recent loss: {:.4f}, sup: {}, unsup: {:.4f}'.format(loss, sup_loss, unsup_loss))
            optimizer.step()
        NAME.eval()
        return NAME


    def refine(self, NAME, source_A_hat, target_A_hat, threshold):
        refinement_model = StableFactor(len(source_A_hat), len(target_A_hat), self.args.cuda)
        if self.args.cuda: 
            refinement_model = refinement_model.cuda()
        S_max = None
        source_outputs = NAME(refinement_model(source_A_hat, 's'), 's')
        target_outputs = NAME(refinement_model(target_A_hat, 't'), 't')
        acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas)
        self.NAME_S = S

        source_edges = self.source_dataset.get_edges()
        target_edges = self.target_dataset.get_edges()
        edgess = [source_edges.tolist(), target_edges.tolist()]
        # score = self.compute_score(edgess, S)
        score = np.mean(S.max(axis = 1))
        acc_max = 0
        alpha_source_max = None
        alpha_target_max = None
        # if 1:
        if score > refinement_model.score_max:
            refinement_model.score_max = score
            alpha_source_max = refinement_model.alpha_source
            alpha_target_max = refinement_model.alpha_target
            acc_max = acc
            S_max = S
        print("Acc: {}, score: {:.4f}".format(acc, score))
        source_candidates, target_candidates = [], []            
        alpha_source_max = refinement_model.alpha_source + 0
        alpha_target_max = refinement_model.alpha_target + 0
        for epoch in range(self.args.refinement_epochs):
            if self.args.log:
                print("Refinement epoch: {}".format(epoch))
            source_candidates, target_candidates, len_source_candidates, count_true_candidates = self.get_candidate(source_outputs, target_outputs, threshold)
            
            refinement_model.alpha_source[source_candidates] *= 1.1
            refinement_model.alpha_target[target_candidates] *= 1.1
            source_outputs = NAME(refinement_model(source_A_hat, 's'), 's')
            target_outputs = NAME(refinement_model(target_A_hat, 't'), 't')
            acc, S = get_acc(source_outputs, target_outputs, self.full_dict, self.alphas)
            # score = self.compute_score(edgess, S)
            score = np.mean(S.max(axis = 1))
            if score > refinement_model.score_max:
                refinement_model.score_max = score
                alpha_source_max = refinement_model.alpha_source + 0
                alpha_target_max = refinement_model.alpha_target + 0
                acc_max = acc
                S_max = S
            if self.args.log:
                print("Acc: {}, score: {:.4f}, score_max {:.4f}".format(acc, score, refinement_model.score_max))
            if epoch == self.args.refinement_epochs - 1:
                print("Numcandidate: {}, num_true_candidate: {}".format(len_source_candidates, count_true_candidates))
        print("Done refinement!")
        print("Acc with max score: {:.4f} is : {}".format(refinement_model.score_max, acc_max))
        refinement_model.alpha_source = alpha_source_max
        refinement_model.alpha_target = alpha_target_max
        self.NAME_S = S_max
        self.log_and_evaluate(NAME, refinement_model, source_A_hat, target_A_hat)
        source_candidates = source_candidates.detach().cpu().numpy()
        target_candidates = target_candidates.detach().cpu().numpy()
        # return source_candidates, target_candidates, self.NAME_S
        return self.NAME_S


    def train_NAME(self, NAME, source_A_hat, target_A_hat, structural_optimizer, threshold):
        NAME = self.gcn_semisup_training(NAME, source_A_hat, target_A_hat, structural_optimizer, gamma = 3, k = 5)
        print("Done structural training")
        source_A_hat = source_A_hat.to_dense()
        target_A_hat = target_A_hat.to_dense()
        NAME_S = self.refine(NAME, source_A_hat, target_A_hat, threshold)
        return NAME_S / 3


    def get_similarity_matrices(self, source_outputs, target_outputs):
        """
        Construct Similarity matrix in each layer
        :params source_outputs: List of embedding at each layer of source graph
        :params target_outputs: List of embedding at each layer of target graph
        """
        list_S = []
        for i in range(len(source_outputs)):
            source_output_i = source_outputs[i]
            target_output_i = target_outputs[i]
            S = torch.mm(F.normalize(source_output_i), F.normalize(target_output_i).t())
            list_S.append(S)
        return list_S
        

    def log_and_evaluate(self, embedding_model, refinement_model, source_A_hat, target_A_hat):
        embedding_model.eval()
        source_outputs = embedding_model(refinement_model(source_A_hat, 's'), 's')
        target_outputs = embedding_model(refinement_model(target_A_hat, 't'), 't')
        print("-"* 100)
        source_care = torch.LongTensor(list(self.full_dict.keys()))
        target_care = torch.LongTensor(list(self.full_dict.values()))
        for i in range(len(source_outputs)):
            source_i = source_outputs[i][source_care].detach().cpu().numpy()
            target_i = target_outputs[i][target_care].detach().cpu().numpy()
            np.save("numpy_emb/source_layer{}".format(i), source_i)
            np.save("numpy_emb/target_layer{}".format(i), target_i)
        return source_outputs, target_outputs
    

    def get_candidate(self, source_outputs, target_outputs, threshold):
        List_S = self.get_similarity_matrices(source_outputs, target_outputs)[1:]
        source_candidates = []
        target_candidates = []
        count_true_candidates = 0
        if len(List_S) < 2:
            print("The current model doesn't support refinement for number of GCN layer smaller than 2")
            return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates)

        num_source_nodes = len(self.source_dataset.G.nodes())
        num_target_nodes = len(self.target_dataset.G.nodes())
        for i in range(min(num_source_nodes, num_target_nodes)):
            node_i_is_stable = True
            for j in range(len(List_S)):
                if List_S[j][i].argmax() != List_S[j-1][i].argmax():
                    node_i_is_stable = False 
                    break
            if node_i_is_stable:
                tg_candi = List_S[-1][i].argmax()
                
                source_candidates.append(i)
                target_candidates.append(tg_candi)
                try:
                    if self.full_dict[i] == tg_candi:
                        count_true_candidates += 1
                except:
                    continue

        return torch.LongTensor(source_candidates), torch.LongTensor(target_candidates), len(source_candidates), count_true_candidates

