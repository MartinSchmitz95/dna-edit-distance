import argparse
import torch
import os
import pickle
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth, AffinityPropagation, AgglomerativeClustering, FeatureAgglomeration
from sklearn.metrics.cluster import adjusted_rand_score

import numpy as np


def to_one_hot(data): # returns list of tuples (2D-tensor, string)
    one_hot_strands = torch.zeros(len(data), 4, 125)

    for s, strand in enumerate(data):
        for i, nucleotide in enumerate(strand):
            if i<=124:
                if nucleotide == 'A':
                    one_hot_strands[s,0,i] = 1
                if nucleotide == 'T':
                    one_hot_strands[s,1,i] = 1
                if nucleotide == 'C':
                    one_hot_strands[s,2,i] = 1
                if nucleotide == 'G':
                    one_hot_strands[s,3,i] = 1
    return one_hot_strands

def get_data(clusters_path):
    #clusters_path = os.path.join(data, 'test_Clusters.txt')
    # centers_path = os.path.join(data, 'Centers.txt')
    # clusters_path = os.path.join(data, 'Clusters.txt')
    with open(clusters_path) as f:
        dataset_txtfile = f.readlines()

    clusters = []
    all_strands = []
    cluster_id = -1
    for l in dataset_txtfile:
        if l[0] == '=' or l[1] == 'L' :
            cluster_id += 1
            continue
        end = l.find('\\')
        dna = l[1:end]
        clusters.append(cluster_id)
        all_strands.append(dna)
    return all_strands, clusters

def create_embs(data_path, all_strands, model_path, filename):
    model = torch.load(model_path)
    one_hots = to_one_hot(all_strands)
    res = model.compute_embs(one_hots).detach().numpy()
    pickle.dump(res, open(os.path.join(data_path, f'{filename}.p'), 'wb'))

def create_files(args, cluster_file, small_emb_file, large_emb_file):

    all_strands, clusters = get_data(args.data_samples)
    pickle.dump(clusters, open(os.path.join(args.data_path, f'{cluster_file}.p'), 'wb'))
    create_embs(args.data_path, all_strands, args.model_small, f"{small_emb_file}")
    create_embs(args.data_path, all_strands, args.model_large, f"{large_emb_file}")
    print("created dataset.")


def global_clusters(args, small_emb_file, n_processors):
    with open(os.path.join(args.data_path,f'{small_emb_file}.p'), 'rb') as handle:
        embs = pickle.load(handle)
    clusters = KMeans(n_clusters=n_processors, max_iter=3).fit(embs).labels_
    return clusters

def local_clusters(args, cluster_indices, large_emb_file, n_processors):
    with open(os.path.join(args.data_path,f'{large_emb_file}.p'), 'rb') as handle:
        embs = pickle.load(handle)

    global_clusters = [[] for _ in range(n_processors)]
    global_indices = [[] for _ in range(n_processors)]  # eval

    cluster_indices = cluster_indices.tolist()
    for i in range (len(cluster_indices)):
        c = cluster_indices[i]
        global_clusters[c].append(embs[i])
        global_indices[c].append(i)  # eval
    max_cluster_index = 0

    for i in range (n_processors):
        embs_in_cluster = np.vstack(global_clusters[i])
        #print(embs_in_cluster)
        #clustering = MeanShift(bandwidth=0.63).fit(embs_in_cluster).labels_

        #clustering = AffinityPropagation().fit(embs_in_cluster).labels_
        #clustering = AgglomerativeClustering(distance_threshold=0.1).fit(embs_in_cluster).labels_
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.5).fit(embs_in_cluster).labels_
        #bandwidth = estimate_bandwidth(embs_in_cluster, quantile=0.2, n_samples=500)
        #print(max(clustering)+1, len(set(clustering)))
        print(f'partition {i} local clusters: {max(clustering)}, strands: {len(embs_in_cluster)}')

        # all for evaluation
        clustering += max_cluster_index
        max_cluster_index = max(clustering)
        clustering = clustering.tolist()
        indices_lookup = global_indices[i]  # list of original indices where the data comes from in the origin dataset
        for e in range(len(clustering)):
            cluster_indices[indices_lookup[e]] = clustering[e]
    return cluster_indices

def simple_clusters(emb_file):
    with open(os.path.join(args.data_path,f'{emb_file}.p'), 'rb') as handle:
        embs = pickle.load(handle)
    #clusters = KMeans(n_clusters=50).fit(embs).labels_
    clusters = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6).fit(embs).labels_

    #clusters = FeatureAgglomeration().fit(embs).labels_

    #clusters = AgglomerativeClustering(n_clusters=103).fit(embs).labels_
    #clusters = MeanShift(bandwidth=0.54).fit(embs).labels_
    #clusters = AffinityPropagation().fit(embs).labels_

    #print(f'result for {(1 + 0.1*i)}')
    return clusters

def evaluate_clustering(cluster_file, l_clusters):

    with open(os.path.join(args.data_path,f'{cluster_file}.p'), 'rb') as handle:
        cluster_gt = pickle.load(handle)
    ars = adjusted_rand_score(cluster_gt, l_clusters)
    #print(cluster_gt)
    print(max(l_clusters), max(cluster_gt))
    print(f'Adjusted random score: {ars}')

def parallel_clustering(args, n_processors, cluster_file, small_emb_file, large_emb_file):

    g_clusters = global_clusters(args, small_emb_file, n_processors)
    l_clusters = local_clusters(args, g_clusters, large_emb_file, n_processors)
    evaluate_clustering(cluster_file, l_clusters)

def simple_clustering(args, cluster_file, emb_file):

    s_clusters = simple_clusters(emb_file)
    evaluate_clustering(cluster_file, s_clusters)

def get_args():
    parser = argparse.ArgumentParser(description="Config for Clustering")
    #parser.add_argument("--data_gt", type=str, default='clustered-nanopore-reads-dataset/test_Centers', help="dataset")
    #parser.add_argument("--data_samples", type=str, default='clustered-nanopore-reads-dataset/test_split_2/test_Clusters.txt', help="dataset")
    #parser.add_argument("--data_samples", type=str, default='data/synth_data_DNA_storage/cat_naive_P0.06_N10/UnderlyingClusters.txt', help="dataset")
    #parser.add_argument("--data_samples", type=str, default='cat_naive_P0.01_N10/UnderlyingClusters.txt', help="dataset")
    parser.add_argument("--data_samples", type=str, default='data/synth_data_DNA_storage/cat_naive_P0.03_N10/NoisyStrands.txt', help="dataset")

    parser.add_argument("--data_path", type=str, default='out_data', help="-")
    parser.add_argument("--model_small", type=str, default='models/emb8.pt', help="model")  # 'shuffled_dna_strings'
    parser.add_argument("--model_large", type=str, default='models/emb128.pt', help="model")  # 'shuffled_dna_strings'

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_args()
    n_processors = 128
    create_files(args, "synth_6_full", "synth_p3_8_embs", "synth_p3_128_embs")
    #create_files(args, "real_full", "real_full_8_embs", "real_full_128_embs")

    parallel_clustering(args, n_processors, "synth_6_full", "synth_p3_8_embs", "synth_p3_128_embs")
    #simple_clustering(args, "synth01_real_clusters", "synth01_64_embs")
    #simple_clustering(args, "real_full", "real_full_128_embs")


    # 64
    # score: 0.778
    #result for 2.1
