from numpy.core.records import array
from numpy.lib.function_base import quantile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import torchvision.transforms as transforms
from scipy.spatial.distance import pdist, squareform
from scipy.stats import norm
from loguru import logger
import torch.nn.functional as F
from model_loader import load_model
from evaluate import mean_average_precision
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from torch.nn import Parameter
def write_csv(path, data_row):
    with open(path, 'a+') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(data_row)

def train(train_dataloader,
          query_dataloader,
          retrieval_dataloader,
          train_dataloader_wag,
          multi_labels,
          code_length,
          num_features,
          max_iter,
          arch,
          lr,
          device,
          verbose,
          topk,
          threshold,
          eta_1,
          eta_2,
          temperature,
          evaluate_interval,
          ):
    """
    Training model.

    Args
        train_dataloader(torch.evaluate.data.DataLoader): Training data loader.
        query_dataloader(torch.evaluate.data.DataLoader): Query data loader.
        retrieval_dataloader(torch.evaluate.data.DataLoader): Retrieval data loader.
        multi_labels(bool): True, if dataset is multi-labels.
        code_length(int): Hash code length.
        num_features(int): Number of features.
        max_iter(int): Number of iterations.
        arch(str): Model name.
        lr(float): Learning rate.
        device(torch.device): GPU or CPU.
        verbose(bool): Print log.
        evaluate_interval(int): Interval of evaluation.
        snapshot_interval(int): Interval of snapshot.
        topk(int): Calculate top k data points map.
        checkpoint(str, optional): Paht of checkpoint.

    Returns
        None
    """
    # Model, optimizer, criterion
    model = load_model(arch, code_length)
    model.to(device)
    base_params = list(map(id, model.model.parameters()))
    logits_params = filter(lambda p: id(p) not in base_params, model.parameters())
    params = [
        {"params": logits_params, "lr": lr},
        {"params": model.model.parameters(), "lr": lr * 0.01},
    ]
    optimizer = torch.optim.SGD(params, momentum=0.9, weight_decay=1e-5)
    criterion = SEM_CON_Loss()

    # Extract features
    for i in range(1):
        features, _ = extract_features(model, train_dataloader_wag, num_features, device, verbose)
        logger.info('Get the deep feature')
        # S, W combine the affinity and the penlty graph
        S, W = generate_similarity_weight_matrix(features, threshold, num_class=70)
        S = S.to(device)
        W= W.to(device)       
        S_1 = S
        S_2 = S
        W_1 = W
        W_2 = W
        

        # Training
        model.train()
        for epoch in range(max_iter):
            n_batch = len(train_dataloader)
            bt_loss_acc = 0
            quan_loss_acc = 0
            for i, (data, data_aug,_, index) in enumerate(train_dataloader):


                data = data.to(device)
                batch_size = data.shape[0]
                data_aug = data_aug.to(device)

                optimizer.zero_grad()

                v= model(data)
                v_aug= model(data_aug)

                c = F.normalize(v,dim=0).T @ F.normalize(v_aug,dim=0)


                on_diag = torch.diagonal(c).add_(-1).pow_(2).mean()
                off_diag = off_diagonal(c).pow(2).mean()
                bcr_loss = on_diag + eta * off_diag


                H = v @ v.t()/code_length
                H_aug = v_aug @ v_aug.t()/code_length
                targets_1 = S_1[index, :][:, index]
                targets_2 = S_2[index, :][:, index]
                weights_1 = W_1[index, :][:, index]
                weights_2 = W_2[index, :][:, index]

                ssp_loss = criterion(H_aug, weights_2, targets_2) +  criterion(H, weights_1, targets_1)
                quan_loss = - torch.mean(torch.sum(F.normalize(v,dim=-1)*F.normalize(torch.sign(v),dim=-1), dim=1)) 

                loss = ssp_loss + eta_1* bcr_loss + eta_2 * quan_loss 

                loss.backward()
                optimizer.step()

                # Evaluate
            if (epoch % evaluate_interval == evaluate_interval-1) or (epoch==0):
                mAP = evaluate(model,
                                query_dataloader,
                                retrieval_dataloader,
                                code_length,
                                device,
                                topk,
                                multi_labels,
                                )
                logger.info('[iter:{}/{}][map:{:.4f}]'.format(
                    epoch+1,
                    max_iter,
                    mAP,
                ))
                torch.save({'iteration': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            }, os.path.join('cifar_checkpoint', 'resume_{}_{}.t'.format(code_length,epoch)))

    # Evaluate and save 
    mAP = evaluate(model,
                   query_dataloader,
                   retrieval_dataloader,
                   code_length,
                   device,
                   topk,
                   multi_labels,
                   )
    torch.save({'iteration': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join('checkpoints', 'resume_{}.t'.format(code_length)))
    logger.info('Training finish, [iteration:{}][map:{:.4f}]'.format(epoch+1, mAP))


def evaluate(model, query_dataloader, retrieval_dataloader, code_length, device, topk, multi_labels):
    model.eval()

    # Generate hash code
    query_code = generate_code(model, query_dataloader, code_length, device)
    retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

    # One-hot encode targets
    if multi_labels:
        onehot_query_targets = query_dataloader.dataset.get_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_targets().to(device)
    else:
        onehot_query_targets = query_dataloader.dataset.get_onehot_targets().to(device)
        onehot_retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets().to(device)
    

    # Calculate mean average precision
    mAP = mean_average_precision(
        query_code,
        retrieval_code,
        onehot_query_targets,
        onehot_retrieval_targets,
        device,
        topk,
    )
    
    model.train()

    return mAP


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code.

    Args
        model(torch.nn.Module): CNN model.
        dataloader(torch.evaluate.data.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.

    Returns
        code(torch.Tensor): Hash code.
    """
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, _, index in dataloader:
            data = data.to(device)
            outputs= model(data)
            code[index, :] = outputs.sign().cpu()

    return code


def extract_features(model, dataloader, num_features, device, verbose):
    """
    Extract features.
    """
    model.eval()
    model.set_extract_features(True)
    features_1 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    features_2 = torch.zeros(dataloader.dataset.data.shape[0], num_features)
    with torch.no_grad():
        N = len(dataloader)
        for i, (data_1, data_2 ,_, index) in enumerate(dataloader):
            if verbose:
                logger.debug('[Batch:{}/{}]'.format(i+1, N))
            data_1 = data_1.to(device)
            data_2 = data_2.to(device)
            features_1[index, :] = model(data_1).cpu()
            features_2[index, :] = model(data_2).cpu()

    model.set_extract_features(False)
    model.train()
    return features_1, features_2

class SEM_CON_Loss(nn.Module):
    def __init__(self):
        super(SEM_CON_Loss, self).__init__()

    def forward(self, H, W, S):
        loss = (W * S.abs() * (H - S).pow(2)).sum() / (H.shape[0] ** 2)
        return loss


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


    


    
def generate_similarity_weight_matrix(features, threshold, num_class):
    """
    Generate similarity and confidence matrix.

    Args
        features(torch.Tensor): Features.
        alpha, beta(float): Hyper-parameters.

    Returns
        S(torch.Tensor): Similarity matrix.
    """
    # Cosine similarity
    cos_dist = squareform(pdist(features.numpy(), 'cosine'))

    # Construct similarity matrix
    S = (cos_dist <= threshold) * 1.0 + (cos_dist > threshold ) * -1.0
    
    # weight according to similarity

    # find the up and down extreme
    # Find maximum count of cosine distance
    max_cnt, max_cos = 0, 0
    interval = 1. / 100
    cur = 0
    for i in range(100):
        cur_cnt = np.sum((cos_dist > cur) & (cos_dist < cur + interval))
        if max_cnt < cur_cnt:
            max_cnt = cur_cnt
            max_cos = cur
        cur += interval

    # Split features into two parts
    flat_cos_dist = cos_dist.reshape((-1, 1))
    left = flat_cos_dist[np.where(flat_cos_dist <= max_cos)[0]]
    right = flat_cos_dist[np.where(flat_cos_dist > max_cos)[0]]

    # Reconstruct gaussian distribution
    left = np.concatenate([left, 2 * max_cos - left])
    right = np.concatenate([2 * max_cos - right, right])

    # Model data using gaussian distribution
    left_mean, left_std = norm.fit(left)
    right_mean, right_std = norm.fit(right)

    def weight_norm(x, threshold):
        weight_f = (x>threshold)* (norm.cdf((x-right_mean)/right_std)-norm.cdf((threshold-right_mean)/right_std))/(1-norm.cdf((threshold-right_mean)/right_std)) + \
         (x<=threshold) * (norm.cdf((threshold-left_mean)/left_std)-norm.cdf((x-left_mean)/left_std) )/ (norm.cdf((threshold-left_mean)/left_std))
        return weight_f
    
    weight_1 = np.clip(weight_norm(cos_dist, threshold), 0, 1)


    # weight according to clustering
    features_norm = (features.T/ np.linalg.norm(features,axis=1)).T
    kmeans = KMeans(n_clusters=num_class, random_state=0, init='k-means++').fit(features_norm)
    A = kmeans.labels_[np.newaxis, :] #label vector
    weight_2 =  ((((A - A.T) == 0)-1/2)*2* S +1)/2
    W = weight_1 * weight_2
    return torch.FloatTensor(S), torch.FloatTensor(W)