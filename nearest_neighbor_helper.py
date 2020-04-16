import matplotlib.pyplot as plt
import numpy as np

def get_nearest_neighbor_idx(query_idx, feature_dict, n=100):
    """ Get the index of n nearest neighbors to query 
    according to the feature dictionary
    The first image is the query image """
    neighbors = {}
    q_feat = feature_dict[query_idx]
    for idx, feat in feature_dict.items():
        neighbors[idx] = np.dot(feat, q_feat)
    neighbors = sorted(neighbors.items(), key=lambda x:x[1], reverse=True)
    return neighbors[:n]
  
def distance_distribution_for_id(query, feature_dict, dataset):
    """ Get the distance histogram for 10,20,50,and 100 nearest
    images for a given inputted query """
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    axs = axs.flatten()
    
    for i, n in enumerate([10,20,50,100]):
        neighbors = get_nearest_neighbor_idx(query, feature_dict, n=n)
        distances = [dist for i,dist in neighbors if i!=query]
        axs[i].hist(distances, bins=15)
        axs[i].set_title('nearest {}'.format(n))
    fig.suptitle("q_idx {}".format(query))

    
def nearest_neighbors_at_epoch(q_id, epoch):
    epoch_name = "/ckpt_epoch_{}.pth".format(epoch)
    my_path = folder_path + model_name + epoch_name
    
    path_to_feat = pickle_path + "/{}_val_to_feat_epoch{}".format(dataset, epoch)
    dval_feat = load_obj(path_to_feat)
    
    nearest_idx = get_nearest_neighbor_idx(q_id, dval_feat, n=10)
    nearest_img = [split_img(val_dataset[idx][0]) for idx,score in nearest_idx]
    return nearest_img

def quick_all_distance_distribution(feature_dict, dataset):
    """ Get an analysis of general distances distributions.
    
    For 100 randomly sampled queries, we get the distance for
    20, 100, 500, and all nearest neighbors and create a histogram 
    out if it.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12,8))
    axs = axs.flatten()
    dist20, dist100, dist500, distAll = [],[],[],[]
    
    for i in range(100):
        query = random.randint(0,len(val_dataset)-1)
        neighbors = get_nearest_neighbor_idx(query, feature_dict, n=len(feature_dict))
        neighbors = [dist for i,dist in neighbors if i!=query]
        dist20.extend(neighbors[:20])
        dist100.extend(neighbors[:100])
        dist500.extend(neighbors[:500])
        distAll.extend(neighbors)
        
    dists = [dist20, dist100, dist500, distAll]
    for i, n in enumerate([20, 100, 500, len(feature_dict)]):
        axs[i].hist(dists[i], bins=25)
        axs[i].set_title('nearest {}'.format(n))
    fig.suptitle("all")