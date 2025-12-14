import torch
from numpy import *
import builtins
import time
import torch.nn as nn
import torch.nn.init as init
from networks import SharedNetwork, SplitNetwork
from train_transformer_class import optimizer_config

def mve_loss(mean_pred, var_pred, target, beta=None):
    """Compute beta-NLL loss
    :param beta: Parameter from range [0, 1] controlling relative weighting between data points, 
                 where `0` corresponds to high weight on low error points and `1` to an equal weighting.
    """
    loss = 0.5 * ((target - mean_pred) ** 2 / var_pred + var_pred.log())
    if beta:
        loss *= (var_pred.detach() ** beta)
    return torch.mean(loss.sum(axis=-1))


def load_model(network=None, rank=None, name=None, model_config=None, trial_name=None):
    if not name:
        model = network(model_config["src_vocab_size"], 
                        model_config["d_model"], 
                        model_config["num_heads"], 
                        model_config["num_layers"],
                        model_config["d_ff"], 
                        model_config["max_seq_length"], 
                        model_config["dropout"], 
                        model_config["num_classes"], 
                        rank).to(rank)
        print('\nNumber of trainable parameters:', builtins.sum(p.numel() for p in model.parameters()), "\n")
    else:
        model = torch.load(f'./transformer/{trial_name}/model/{name}.pth', map_location=rank).eval()
        print(f'\nNumber of model parameters {name}:', builtins.sum(p.numel() for p in model.parameters()))
    return model


def determine_network_type(model):
    if isinstance(model, SharedNetwork):
        network_type = "shared_network"
    elif isinstance(model, SplitNetwork):
        network_type = "split_network"
    return network_type


def create_optimizers(model, network_type, init_lr):
    # Call correct network config
    config = optimizer_config[network_type]

    mean_params, var_params, other_params, other_params_mean, other_params_var  = [], [], [], [], []
    
    if network_type== "split_network" and config["num_optimizers"] == 6:
        for layer_name in config["encoder_layers"] + config["cnn_layers"]:
            if "mean" in layer_name:
                other_params_mean += list(getattr(model, layer_name).parameters())
            elif "variance" in layer_name:
                other_params_var += list(getattr(model, layer_name).parameters())
    else:
        for layer_name in config["encoder_layers"] + config["cnn_layers"]:
            if "mean" in layer_name:
                mean_params += list(getattr(model, layer_name).parameters())
            elif "variance" in layer_name:
                var_params += list(getattr(model, layer_name).parameters())
            else:
                other_params += list(getattr(model, layer_name).parameters())

    for layer_name in config["fc_layers"]:
        if "mean" in layer_name:
            mean_params += list(getattr(model, layer_name).parameters())
        elif "variance" in layer_name:
            var_params += list(getattr(model, layer_name).parameters())

    optimizers = {
        "shared_backbone_optimizer": torch.optim.Adam(other_params, lr=init_lr) if other_params else None,
        "split_backbone_optimizer_mean": torch.optim.Adam(other_params_mean, lr=init_lr) if other_params_mean else None,
        "split_backbone_optimizer_var": torch.optim.Adam(other_params_var, lr=init_lr) if other_params_var else None,
        "mean_optimizer": torch.optim.Adam(mean_params, lr=init_lr, weight_decay=config["weight_decay_mean"]),
        "var_optimizer": torch.optim.Adam(var_params, lr=init_lr, weight_decay=config["weight_decay_var"]),
    }

    return optimizers


def initialize_variance_weights(model, network_type):
    config = optimizer_config[network_type]

    for layer_name in config["encoder_layers"]:
        if "variance" in layer_name:
            layer = getattr(model, layer_name)
            if config["zero_weights_and_freeze_var_params_encoder"]:
                for sub_layer in layer:
                    init.zeros_(sub_layer.self_attn.W_q.weight)
                    init.zeros_(sub_layer.self_attn.W_k.weight)
                    init.zeros_(sub_layer.self_attn.W_v.weight)
                    init.zeros_(sub_layer.self_attn.W_o.weight)
                    for param in sub_layer.parameters():
                        param.requires_grad = False

    for layer_name in config["cnn_layers"]:
        if "variance" in layer_name:
            layer = getattr(model, layer_name)
            if config["zero_weights_and_freeze_var_params_cnn"]:
                for sub_layer in layer:
                    if isinstance(sub_layer, nn.Conv1d):
                        init.zeros_(sub_layer.weight)
                        if sub_layer.bias is not None:
                            init.zeros_(sub_layer.bias)
                        for param in sub_layer.parameters():
                            param.requires_grad = False

    for layer_name in config["fc_layers"]:
        if "variance" in layer_name:
            if config["zero_weights_and_freeze_var_params_fc"]:
                layer = getattr(model, layer_name)
                init.zeros_(layer[-1].weight)
                init.zeros_(layer[-1].bias)
                for param in layer.parameters():
                    param.requires_grad = False


def update_training_phase(model, network_type):
    config = optimizer_config[network_type]
    for layer_type in ["encoder_layers", "cnn_layers", "fc_layers"]:
        for layer_name in config[layer_type]:
            if "variance" in layer_name:
                for param in getattr(model, layer_name).parameters():
                    param.requires_grad = True


def clip_gradients(model, network_type):
    config = optimizer_config[network_type]
    if config["clip_grad"]:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad_max_norm"])

def to_rank(rank, *batch):
    return [x.to(rank) for x in batch]


def scale_up(standardize, scaler, variance_pred, mean_pred, actual, rank):
    if standardize:
            variance_pred = variance_pred * (scaler.std ** 2)
            mean_pred = torch.from_numpy(scaler.inverse_transform(mean_pred.cpu().detach().numpy())).to(rank)
            actual = torch.from_numpy(scaler.inverse_transform(actual.cpu().detach().numpy())).to(rank)
    return variance_pred, mean_pred, actual


def print_total_time_taken(cp_1, cp_2):
    cp_3 = time.time()
    print('\nTrain Time Taken:',round(float(cp_2-cp_1), 2))
    print('Test Time Taken:',round(float(cp_3-cp_2), 2))