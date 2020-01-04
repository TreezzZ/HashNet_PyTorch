import time
import torch
import torch.nn as nn
import torch.optim as optim

from models.model_loader import load_model
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.evaluate import mean_average_precision
from loguru import logger


def train(
        train_dataloader,
        query_dataloader,
        retrieval_dataloader,
        arch,
        code_length,
        device,
        lr,
        max_iter,
        alpha,
        topk,
        evaluate_interval,
    ):
    """
    Training model.

    Args
        train_dataloader, query_dataloader, retrieval_dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        arch(str): CNN model name.
        code_length(int): Hash code length.
        device(torch.device): GPU or CPU.
        lr(float): Learning rate.
        max_iter(int): Number of iterations.
        alpha(float): Hyper-parameters.
        topk(int): Compute top k map.
        evaluate_interval(int): Interval of evaluation.

    Returns
        checkpoint(dict): Checkpoint.
    """
    # Load model
    model = load_model(arch, code_length).to(device)
    
    # Create criterion, optimizer, scheduler
    criterion = HashNetLoss(alpha)
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=5e-4,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        max_iter,
        lr/100,
    )

    # Initialization
    running_loss = 0.
    best_map = 0.
    training_time = 0.

    # Training
    for it in range(max_iter):
        tic = time.time()
        model.training = True
        for data, targets, index in train_dataloader:
            data, targets, index = data.to(device), targets.to(device), index.to(device)
            optimizer.zero_grad()

            # Create similarity matrix
            S = (targets @ targets.t() > 0).float()
            outputs = model(data)
            loss = criterion(outputs, S)

            running_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()
        training_time += time.time() - tic

        # Evaluate
        if it % evaluate_interval == evaluate_interval - 1:
            model.training = False
            # Generate hash code
            query_code = generate_code(model, query_dataloader, code_length, device)
            retrieval_code = generate_code(model, retrieval_dataloader, code_length, device)

            query_targets = query_dataloader.dataset.get_onehot_targets()
            retrieval_targets = retrieval_dataloader.dataset.get_onehot_targets()

            # Compute map
            mAP = mean_average_precision(
                query_code.to(device),
                retrieval_code.to(device),
                query_targets.to(device),
                retrieval_targets.to(device),
                device,
                topk,
            )
            
            # Log
            logger.info('[iter:{}/{}][loss:{:.2f}][map:{:.4f}][time:{:.2f}]'.format(
                it+1,
                max_iter,
                running_loss / evaluate_interval,
                mAP,
                training_time,
            ))
            running_loss = 0.

            # Checkpoint
            if best_map < mAP:
                best_map = mAP

                checkpoint = {
                    'model': model.state_dict(),
                    'qB': query_code.cpu(),
                    'rB': retrieval_code.cpu(),
                    'qL': query_targets.cpu(),
                    'rL': retrieval_targets.cpu(),
                    'map': best_map,
                }

    return checkpoint


def generate_code(model, dataloader, code_length, device):
    """
    Generate hash code

    Args
        dataloader(torch.utils.data.dataloader.DataLoader): Data loader.
        code_length(int): Hash code length.
        device(torch.device): Using gpu or cpu.

    Returns
        code(torch.Tensor): Hash code.
    """
    model.eval()
    with torch.no_grad():
        N = len(dataloader.dataset)
        code = torch.zeros([N, code_length])
        for data, _, index in dataloader:
            data = data.to(device)
            hash_code = model(data)
            code[index, :] = hash_code.sign().cpu()

    model.train()
    return code


class HashNetLoss(nn.Module):
    """
    HashNet loss function.

    Args
        alpha(float): Sigmoid hyper-parameter.
    """
    def __init__(self, alpha):
        super(HashNetLoss, self).__init__()
        self.alpha = alpha

    def forward(self, H, S):
        # Compute balance weights
        sim_pos = (S == 1)
        dissim_pos = (S == 0)
        w1 = S.numel() / sim_pos.sum().float()
        w0 = S.numel() / dissim_pos.sum().float()
        W = torch.zeros(S.shape, device=H.device)
        W[sim_pos] = w1
        W[dissim_pos] = w0

        # Inner product
        theta = H @ H.t()

        loss = (W * (torch.log(1 + torch.exp(self.alpha * theta)) - self.alpha * S * theta)).mean()
        #loss = (torch.log(1 + torch.exp(self.alpha * theta) - self.alpha * S * theta)).mean()

        return loss

