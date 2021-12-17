import pdb
import time

import torch
import torch.nn as nn
from torch.autograd import Variable

import networkx as nx
import numpy as np
from emb_model import PaleEmbedding
import argparse
from tqdm import tqdm
import os
import shutil
import pickle
import json
import random
from tensorboardX import SummaryWriter
from gumbel import gumbel_softmax
from models import Model
from train_utils import prepare_data, evaluate
from utils.node2vec import fixed_unigram_candidate_sampler
from mapping_model import PaleMappingLinear
import torch.nn.functional as F

def arg_parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--graphdir',
            help='Directory containing graph data')
    parser.add_argument('--task',
            help='classify, linkpred, wordsim or sentence')
    parser.add_argument('--name-suffix', dest='name_suffix',
            help='')

    parser.add_argument('--vocab-size', dest='vocab_size', type=int, default=5000,
            help='Size of vocabulary, used for wordsim and sentence task, default=5000')
    parser.add_argument('--min-count', dest='min_count', type=int, default=100,
            help='Minimum count, used for wordsim and sentence task, default=100')
    parser.add_argument('--neg-samples', dest='neg_samples', type=int, default=1,
            help='Negative sampling, used for wordsim and sentence task, default=1')
    parser.add_argument('--skip-window', dest='skip_window', type=int, default=5,
            help='Skipping window, used for wordsim and sentence task, default=5')

    parser.add_argument('--sen-train', dest='sen_train',
            help='Training file for sentence classification')
    parser.add_argument('--sen-test', dest='sen_test',
            help='Testing file for sentence classification')

    parser.add_argument('--num-parts', dest='num_parts', type=int, default=128,
            help='Number of partitions, default=128')
    parser.add_argument('--train-perc', dest='train_percent', type=float, default=0.5,
            help='Ratio of number of labels for training, default=0.5')

    parser.add_argument('--new', action="store_true",
            help='Using new min cut loss')
    parser.add_argument('--lambda', dest='lam', type=float, default=1,
            help='Weight of the min-cut. 1-lam will be the weight of balance cut, default=1')
    parser.add_argument('--w2v-lambda', dest='w2v_lam', type=float, default=0.0,
            help='Weight of node2vec loss, default=0.0')
    parser.add_argument('--balance_node',action="store_true",
            help='Use only adj_cross')

    parser.add_argument('--anneal', action="store_true",
            help='Annealing temperature')
    parser.add_argument('--min-temp', dest='min_temp', type=float, default=0.1,
            help='Minimum value of temperature when using temp annealing, default=0.1')
    parser.add_argument('--temp', dest='temp', type=float, default=1,
            help='Temperature for gumbel sinkhorn, default=1')
    parser.add_argument('--hard',action="store_true",
            help='Hard assignment of gumbel softmax')
    parser.add_argument('--beta', type=float, default=1,
            help='Beta param of gumbel softmax, default=1')

    parser.add_argument('--weight_decay_type',
            help='elem or vector')
    parser.add_argument('--loss_type',
            help='n2v or mincut')
    parser.add_argument('--epochs', dest='num_epochs', type=int, default=3000,
            help='Number of epochs to train, default=3000.')

    parser.add_argument('--neg_sample_size', type=int, default=20,
            help='Number of negative samples, default=20.')
    parser.add_argument('--batch',action="store_true",
            help='Batch training or not')
    parser.add_argument('--batch_size', type=int, default=1024,
            help='Batch size, default=1024.')
    parser.add_argument('--seed', type=int, default=123,
            help='Random seed, default=123.')

    parser.add_argument('--lr', dest='lr', type=float, default=0.001,
            help='Learning rate, default=0.001.')
    parser.add_argument('--weight_decay', type=float, default=0,
            help='Weight decay, default=0.')
    parser.add_argument('--clip', dest='clip', type=float, default=2.0,
            help='Gradient clipping, default=2.0.')

    parser.add_argument('--save-best',dest='save_best', action="store_true",
            help='Save params of model')
    
    parser.add_argument('--edge_path', type=str)
    parser.add_argument('--old', action='store_true')
    parser.add_argument('--id2idx', type=str)
    parser.add_argument('--emb_path', type=str)
    parser.add_argument('--cluster_name', type=str)
    parser.add_argument('--num_nodes_each', type=int, default=100)
    parser.add_argument('--dense_inner', type=float, default=0.5)
    parser.add_argument('--dense_outer', type=float, default=0.01)
    parser.add_argument('--num_cluster', type=int, default=5)
    parser.add_argument('--emb_model', type=str, default='pale')

    return parser.parse_args()


def train_embedding(embedding_model, edges, optimizer):
    emb_batchsize = 512
    emb_epochs = 100
    n_iters = len(edges) // emb_batchsize
    assert n_iters > 0, "batch_size is too large!"
    if(len(edges) % emb_batchsize > 0):
        n_iters += 1
    print_every = int(n_iters/4) + 1
    total_steps = 0
    n_epochs = emb_epochs
    for epoch in range(1, n_epochs + 1):
        # for time evaluate
        start = time.time()
        print("Epoch {0}".format(epoch))
        np.random.shuffle(edges)
        for iter in range(n_iters):
            batch_edges = torch.LongTensor(edges[iter*emb_batchsize:(iter+1)*emb_batchsize])
            batch_edges = batch_edges.cuda()
            start_time = time.time()
            optimizer.zero_grad()
            loss, loss0, loss1 = embedding_model.loss(batch_edges[:, 0], batch_edges[:,1])
            loss.backward()
            optimizer.step()
            if total_steps % print_every == 0:
                print("Iter:", '%03d' %iter,
                            "train_loss=", "{:.5f}".format(loss.item()),
                            "true_loss=", "{:.5f}".format(loss0.item()),
                            "neg_loss=", "{:.5f}".format(loss1.item()),
                            "time", "{:.5f}".format(time.time()-start_time)
                        )
            total_steps += 1
        
    embedding = embedding_model.get_embedding()
    embedding = embedding.cpu().detach().numpy()
    np.savetxt("emb.tsv", embedding, delimiter='\t')
    return embedding



def mapp(source_embedding, target_embedding):
    num_epochs = 100
    train_percent = 0.2
    num_train = int(len(source_embedding) * train_percent)
    train_index = torch.LongTensor(np.random.choice(list(range(len(source_embedding))), num_train)).cuda()
    import pdb
    pdb.set_trace()

    emb_dim = source_embedding.shape[1]
    source_embeddings = torch.FloatTensor(source_embedding).cuda()
    target_embeddings = torch.FloatTensor(target_embedding).cuda()

    source_embeddings = F.normalize(source_embeddings)
    target_embeddings = F.normalize(target_embeddings)

    map_model = PaleMappingLinear(emb_dim, source_embeddings, target_embeddings) 
    map_model = map_model.cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, map_model.parameters()), lr=0.01)

    # source_batch = source_embedding[train_index]
    # target_batch = target_embedding[train_index]

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss = map_model.loss(train_index, train_index)
        loss.backward()
        optimizer.step()
        # if total_steps % print_every == 0 and total_steps > 0:
        print("train_loss=", "{:.5f}".format(loss.item()))

    source_after_map = map_model.forward(source_embeddings)

    source_embeddings = source_embeddings.detach().cpu().numpy()
    target_embeddings = target_embeddings.detach().cpu().numpy()
    source_after_map = source_after_map.detach().cpu().numpy()

    return source_embeddings, target_embeddings, source_after_map


def main(args):
    model_name = "{}_lr{}_temp{}_lam_{}_w2vlam_{}_beta_{}_{}{}".\
        format(args.name_suffix, args.lr, args.temp, args.lam, args.w2v_lam,\
            args.beta, args.weight_decay_type, args.weight_decay)

    if not os.path.isdir('runs'):
        os.mkdir('runs')

    if os.path.isdir("runs/"+model_name):
        shutil.rmtree("runs/"+model_name)

    writer = SummaryWriter("runs/"+model_name)

    if args.old and 0:
        data = prepare_data(args)
        edges = np.array([list(i) for i in list(zip(*data['G'].edges()))]).astype(int)
        edges = edges.T
        degrees = nx.adjacency_matrix(data['G']).toarray().sum(1)
        num_nodes = nx.number_of_nodes(data['G'])
        adj = nx.adjacency_matrix(data['G']).toarray()
        adj = Variable(torch.FloatTensor(nx.adjacency_matrix(data['G']).toarray()), requires_grad=False)

    elif 0:
        id2idx = json.load(open(args.id2idx))
        edges = np.load(args.edge_path)
        num_nodes = len(id2idx)
        adj = np.zeros((num_nodes, num_nodes))
        for i in range(len(edges)):
            adj[[edges[i][0], edges[i][1]]] = 1
            adj[[edges[i][1], edges[i][0]]] = 1
        adj = Variable(torch.FloatTensor(adj), requires_grad = False)

        print("Number of nodes: {}, number of edges: {}".format(num_nodes, len(edges)))
    else:
        def gen_cluster_graph(num_nodes_each, dense_inner, dense_outer, num_cluster):
            Graphs = []
            minn = 0
            for i in range(num_cluster):
                graph_i = nx.generators.random_graphs.fast_gnp_random_graph(num_nodes_each, dense_inner)
                mapping = {i: i + minn for i in range(num_nodes_each)}
                graph_i = nx.relabel_nodes(graph_i, mapping)
                Graphs.append(graph_i)
                minn += num_nodes_each
            
            # for i in range(num_cluster):
            G = nx.Graph()
            for gr in Graphs:
                G.add_nodes_from(list(gr.nodes()))
                G.add_edges_from(list(gr.edges()))
            
            print("Before connect: ")
            print("Number of nodes: {}".format(len(G.nodes())))
            print("Number of edges: {}".format(len(G.edges())))
            nonedges = list(nx.non_edges(G))
            nonedges = np.array([edge for edge in nonedges if np.abs(edge[0] - edge[1]) > 100])
            to_add_index = np.random.choice(np.arange(len(nonedges)), int(dense_outer * len(nonedges)))

            edges_to_add = nonedges[to_add_index]

            G.add_edges_from(edges_to_add)

            print("After connect: ")
            print("Number of nodes: {}".format(len(G.nodes())))
            print("Number of edges: {}".format(len(G.edges())))

            return G


        graph = gen_cluster_graph(args.num_nodes_each, args.dense_inner, args.dense_outer, args.num_cluster)
        adj = nx.adjacency_matrix(graph).todense()
        adj = Variable(torch.FloatTensor(adj), requires_grad = False)

        num_nodes = adj.shape[0]

    if args.emb_model == "pale":
        deg = adj.sum(dim=1).flatten().detach().cpu().numpy()
        embedding_model = PaleEmbedding(
                                        n_nodes = num_nodes,
                                        embedding_dim = 10,
                                        deg= deg,
                                        neg_sample_size = 10,
                                        cuda = True,
                                        )
        # if self.cuda:
        embedding_model = embedding_model.cuda()
        edges = np.array(list(graph.edges))
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=0.01)
        embedding = train_embedding(embedding_model, edges, optimizer)


    embeddings = []
    for vinhsuhi in range(2):
        if os.path.exists('source_emb.npy') and os.path.exists('target_emb.npy'):
            embeddings = [np.load('source_emb.npy'), np.load('target_emb.npy')]
            break
        model = Model(num_nodes, args.num_parts)
        
        if torch.cuda.is_available():
            model = model.cuda()
            adj = adj.cuda()

        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, model.parameters()), lr=args.lr)

        temp=args.temp
        smallest_loss = 1e20
        results = None
        best_at = 0

        for epoch in tqdm(range(args.num_epochs)):
            model.zero_grad()
            if args.loss_type == 'mincut':
                if args.batch:
                    nodes = np.sort(np.random.choice(np.arange(adj.shape[0]), size=args.batch_size, replace=False))
                else:
                    nodes = None
                super_adj = model(adj,nodes,temp=temp, hard=args.hard, beta=args.beta)
                loss, ncut_loss, balance_loss = model.loss(super_adj, nodes, balance_node=args.balance_node, lam=args.lam, w2v_lam = args.w2v_lam, new=args.new)

            if args.weight_decay_type == 'elem':
                l2_loss = args.weight_decay * (model.params**2).sum()
            elif args.weight_decay_type == 'vector':
                l2_loss = args.weight_decay * (((model.params**2).sum(1)-1)**2).sum()
            else:
                l2_loss = 0
            if loss!=loss: import pdb;pdb.set_trace()
            total_loss = loss + l2_loss

            total_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()
            if model.params.max() != model.params.max():import pdb;pdb.set_trace()

            if epoch %500==0:
                if args.anneal:
                    temp = min(args.min_temp, args.temp * np.exp(-0.00003*epoch))
                try:
                    print("loss:", loss.item(), "l2_loss:", l2_loss.item())
                    print("ncut_loss: ", ncut_loss.item())
                    print("balance_loss: ", balance_loss.item())
                except:
                    import pdb; pdb.set_trace()
                embedding = model.params.detach().cpu().numpy()
                writer.add_scalar("loss", loss.item(), epoch)
                writer.add_scalar("l2_loss", l2_loss.item(), epoch)

                if loss.item() <= smallest_loss:
                    smallest_loss = loss.item()
                    best_at = epoch
                    if args.save_best:
                        torch.save(model.params, "runs/{}/{}.pt".\
                            format(model_name, model_name))
        
        params = model.params.max(dim=1)[1].cpu().detach().numpy()
        embedding = gumbel_softmax(model.params, temp=10, hard=False, beta=1)
        embedding = embedding.detach().cpu().numpy()
        if vinhsuhi == 0:
            np.save('source_emb.npy', embedding)
        else:
            np.save('target_emb.npy', embedding)
        embeddings.append(embedding)
        writer.close()

    # source_embedding = embeddings[0]
    # target_embedding = embeddings[1]
    """
    source_embedding, target_embedding, source_after_map = mapp(embeddings[0], embeddings[1])
    source_target = np.concatenate((source_embedding, target_embedding), axis=0)
    source_target_map = np.concatenate((source_after_map, target_embedding), axis=0)

    with open('label.tsv', 'w', encoding='utf-8') as file:
        for i in range(len(source_embedding)):
            file.write('1\n')
        for i in range(len(source_embedding)):
            file.write('2\n')
    simi = source_after_map.dot(target_embedding.T)
    simi2 = source_embedding.dot(target_embedding.T)
    np.savetxt('source_target.tsv', source_target, delimiter='\t')
    np.savetxt('source_target_map.tsv', source_target_map, delimiter='\t')

    print("Acc: {:.4f}".format(evaluate_align(simi)))
    print("Acc before: {:.4f}".format(evaluate_align(simi2)))
    """

    cluster1 = embeddings[0].argmax(axis=1).reshape(len(embeddings[0]))
    cluster2 = embeddings[1].argmax(axis=1).reshape(len(embeddings[1]))
    # new_embedding_source
    list11 = cluster1[:100]
    list21 = cluster2[:100]
    list12 = cluster1[100:200]
    list22 = cluster2[100:200]
    list13 = cluster1[200:300]
    list23 = cluster2[200:300]
    list14 = cluster2[300:]
    list24 = cluster2[300:]


    def compare_list(list1, list2):
        unique1, counts1 = np.unique(list1, return_counts=True)
        unique2, counts2 = np.unique(list2, return_counts=True)
        key1 = unique1[np.argmax(counts1)]
        key2 = unique2[np.argmax(counts2)]
        return {key1: key2}

    dict1 = compare_list(list11, list21)
    dict2 = compare_list(list12, list22)
    dict3 = compare_list(list13, list23)
    dict4 = compare_list(list14, list24)

    source = embeddings[0]
    target = embeddings[1]

    new_source = np.zeros_like(source)
    new_target = np.zeros_like(target)

    for i in range(len(new_source)):
        new_source[i][cluster1[i]] = 1
        new_target[i][cluster2[i]] = 1

    embeddings = [new_source, new_target]

    source_embedding, target_embedding, source_after_map = mapp(embeddings[0], embeddings[1])
    source_target = np.concatenate((source_embedding, target_embedding), axis=0)
    source_target_map = np.concatenate((source_after_map, target_embedding), axis=0)

    with open('label.tsv', 'w', encoding='utf-8') as file:
        for i in range(len(source_embedding)):
            file.write('1\n')
        for i in range(len(source_embedding)):
            file.write('2\n')
    simi = source_after_map.dot(target_embedding.T)
    simi2 = source_embedding.dot(target_embedding.T)
    np.savetxt('source_target.tsv', source_target, delimiter='\t')
    np.savetxt('source_target_map.tsv', source_target_map, delimiter='\t')

    print("Acc: {:.4f}".format(evaluate_align(simi)))
    print("Acc before: {:.4f}".format(evaluate_align(simi2)))
    import pdb
    pdb.set_trace()


    # new_source = np.zeros_like(source)
    # new_target = np.zeros_like(target)
    """
    for key, value in dict1.items():
        new_source[:, value] = source[:, key]
    
    for key, value in dict2.items():
        new_source[:, value] = source[:, key]

    for key, value in dict3.items():
        new_source[:, value] = source[:, key]
    
    for key, value in dict4.items():
        new_source[:, value] = source[:, key]


    source_target = np.concatenate((source, target), axis=0)
    source_target_map = np.concatenate((new_source, target), axis=0)
    np.savetxt('source_target.tsv', source_target, delimiter='\t')
    np.savetxt('source_target_map.tsv', source_target_map, delimiter='\t')

    simi_new = new_source.dot(target.T)
    import pdb
    pdb.set_trace()
    print("ACC: {:.4f}".format(evaluate_align(simi_new)))
    """


def evaluate_align(simi):
    count = 0
    for i in range(len(simi)):
        if np.argmax(simi[i]) == i:
            count += 1
    return count / len(simi)


if __name__ == "__main__":

    args = arg_parse()

    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)

    main(args)

