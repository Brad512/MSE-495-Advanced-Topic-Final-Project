import torch
import numpy as np
import torch.nn as nn
from numpy import *
import os
import time
from dataclasses import dataclass
from networks import SharedNetwork, SplitNetwork
from Dataloader import make_dataset
from Validate import *
from utils import *

## Validate File Code Metrics Toggles (Easy Access)
print_epochs = 10            #how often to print validate data (epoch % print_epochs)
print_plots = 200           #how often to plot validate  data (epoch % print_plots)



## TUNE PARAMETERS
trial_name = "Outpts_Shared_Model_2" #name the trial folder that stores all the outputs for the model
network = SharedNetwork #SharedNetwork or SplitNetwork
model_names = ['best_mve_valid', 'best_r2_mean_valid', 'best_r2_var_valid', 'best_r2mean_r2var']

@dataclass
class TrainingConfig:
    num_epochs: int = 600
    init_lr: float = 3e-5
    warmup_epochs: int = 0.1 * num_epochs
    standardize: bool = True                #standardize data
    set_var_bias: bool = True               #sets fc.variance bias as log(avg_mse_loss) during warmup_epochs
    beta: float = None                      #if beta is not None, warmup_epochs=0
    k_folds: int = None                     #replace with int for cross-validation

optimizer_config = {
    "split_network": {
        "encoder_layers": ["encoder_layers_mean", "encoder_layers_variance"],
        "cnn_layers": ["cnn_mean", "cnn_variance"],
        "fc_layers": ["fc_mean", "fc_variance"],
        "num_optimizers": 6, #2 or 6 (1 optimizer/network or 3 optimizers/network)
        "weight_decay_mean": 2,
        "weight_decay_var": 1e-4,
        "zero_weights_and_freeze_var_params_encoder": True,
        "zero_weights_and_freeze_var_params_cnn": True,
        "zero_weights_and_freeze_var_params_fc": True,
        "clip_grad": True,
        "clip_grad_max_norm": 5.0
    },
    "shared_network": {
        "encoder_layers": ["encoder_layers"],
        "cnn_layers": ["cnn1"],
        "fc_layers": ["fc_mean", "fc_variance"],
        "weight_decay_mean": 1e-4,
        "weight_decay_var": 1e-4,
        "zero_weights_and_freeze_var_params_fc": True,
        "clip_grad": True,
        "clip_grad_max_norm": 5.0
    }
}




## Main Code

def initialize(network, rank, init_lr, name=None, max_len=None, test_train_label=None, trial_name=None):
    # Model Configuration
    model_config = {
        "src_vocab_size": 6,
        "d_model": 32,
        "num_heads": 8,
        "num_layers": 1,
        "d_ff": 2048,
        "max_seq_length": max_len,
        "dropout": 0.1,
        "num_classes": 1
    }
    model = load_model(network=network, rank=rank, name=name, model_config=model_config, trial_name=trial_name).to(rank)   
    criterion = nn.MSELoss()
    network_type = determine_network_type(model)
    optimizers = create_optimizers(model, network_type, init_lr)
    if test_train_label == "train":
        initialize_variance_weights(model, network_type)

    return model, criterion, network_type, optimizers
        
## Training loop
def train(network, num_epochs, init_lr, model_names, warmup_epochs, beta, fold, cv_results, standardize=True, set_var_bias=True, trial_name=None, print_epochs=None, print_plots=None):
    test_train_label = "train"
    need_grad = False ## True while test to store the gradient of output wrt attention
    rank = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_loader, valid_loader, _, valid_size, max_len, _, scaler  = make_dataset(fold=fold, standardize=standardize)
    model, criterion, network_type, optimizers = initialize(network, rank, init_lr, max_len=max_len, test_train_label=test_train_label, trial_name=trial_name)
    largest_mve, largest_r2, largest_r2_variance, prev_valid_r2, prev_valid_r2_variance = -1000, -1000, -1000, -1000, -1000

    for epoch in range(num_epochs):
        avg_loss, avg_loss_mse_bias, avg_loss_mse, avg_loss_mse_var_valid, avg_loss_mse_valid = 0, 0, 0, 0, 0
        for i, (i_x, i_classes, i_seq, i_actual, i_s2_actual) in enumerate(train_loader):
            i_x, i_seq, i_classes, i_actual, i_s2_actual = to_rank(rank, i_x, i_seq, i_classes, i_actual, i_s2_actual) #.type(dtype=torch.float32)
            src_mask = None

            ## Forward pass
            mean_pred, variance_pred, _, _ = model(i_classes, src_mask, i_seq, need_grad)
            loss = mve_loss(mean_pred, variance_pred, i_actual, beta)
            avg_loss_mse_bias = (avg_loss_mse_bias * i + criterion(mean_pred, i_actual).item()) / (i + 1)

            # Scale predictions
            variance_pred_scaled, mean_pred_scaled, i_actual_scaled = scale_up(standardize, scaler, variance_pred, mean_pred, i_actual, rank)

            # Calculate loss metrics
            avg_loss = (avg_loss * i + mve_loss(mean_pred_scaled, variance_pred_scaled, i_actual_scaled, beta).item()) / (i + 1)
            avg_loss_mse = (avg_loss_mse*i + criterion(mean_pred_scaled, i_actual_scaled).item())/(i+1)

            # Backward pass
            for name, optimizer in optimizers.items():
                if optimizer is not None and name != "var_optimizer":
                    optimizer.zero_grad()
            if epoch < warmup_epochs:
                loss.backward()
                clip_gradients(model, network_type)
            else:
                optimizers["var_optimizer"].zero_grad()
                loss.backward()
                clip_gradients(model, network_type)
                optimizers["var_optimizer"].step()
            for name, optimizer in optimizers.items():
                if optimizer is not None and name != "var_optimizer":
                    optimizer.step()

        # Set Variance Bias during Warm-up Period
        if epoch < warmup_epochs and set_var_bias:
            model.fc_variance[-1].bias.data.fill_(np.log(avg_loss_mse_bias))
        elif epoch > warmup_epochs:
            update_training_phase(model, network_type) # Unfreeze variance weights for later training

        ## Validation Set
        largest_mve, largest_r2, largest_r2_variance, prev_valid_r2, prev_valid_r2_variance, fold_metrics = validate(model, valid_loader, valid_size, scaler, rank, need_grad, standardize, \
                                                                  beta, criterion, epoch, warmup_epochs, test_train_label, largest_mve, largest_r2, prev_valid_r2, \
                                                                  largest_r2_variance, prev_valid_r2_variance, model_names, avg_loss, avg_loss_mse, avg_loss_mse_var_valid, avg_loss_mse_valid, \
                                                                  trial_name=trial_name, print_epochs=print_epochs, print_plots=print_plots)

        if fold:
            cv_results.append(min(fold_metrics))  # Store best validation loss for each fold
    if fold:
        print(f"Cross-validation MVE Validation Loss Results: {np.mean(cv_results):.4f} ± {np.std(cv_results):.4f}")

## Testing loop
def test(network, init_lr, model_names, standardize=True, trial_name=None):
    for name in model_names:
        if os.path.isfile(f'./transformer/{trial_name}/model/{name}.pth'):
            need_grad= False
            rank = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            _, _, test_loader, _, _, valid_size, scaler  = make_dataset(standardize=standardize)
            model, criterion, _, _ = initialize(network, rank, init_lr, name=name, trial_name=trial_name)
            avg_loss, avg_loss_mse, avg_loss_mse_var, avg_loss_mse_valid = 0, 0, 0, 0
            validate(model, test_loader, valid_size, scaler, rank, need_grad, standardize, criterion=criterion, model_names=name, avg_loss=avg_loss, \
                     avg_loss_mse=avg_loss_mse, avg_loss_mse_var_valid_or_test=avg_loss_mse_var, avg_loss_mse_valid=avg_loss_mse_valid, trial_name=trial_name)

if __name__=='__main__':
    os.makedirs(f'./transformer/{trial_name}', exist_ok=True)
    config = TrainingConfig()
    cp_1 = time.time()
    #training loop
    folds = config.k_folds if config.k_folds is not None else 1
    cv_results=[]
    for fold in range(folds):
        fold = None if config.k_folds is None else fold
        train(network, config.num_epochs, config.init_lr, model_names, config.warmup_epochs, config.beta, \
              fold, cv_results=cv_results, standardize=config.standardize, set_var_bias=config.set_var_bias, \
              trial_name=trial_name, print_epochs=print_epochs, print_plots=print_plots)
    cp_2 = time.time()
    #testing loop
    test(network, config.init_lr, model_names, config.standardize, trial_name=trial_name)
    print_total_time_taken(cp_1, cp_2)