import torch
import numpy as np
from numpy import *
import os
from sklearn.metrics import mean_absolute_error, r2_score
from all_plots import tensorboard_metrics, generate_all_plots
from utils import *

def validate(model, valid_or_test_loader, valid_size, scaler, rank, need_grad, standardize, beta=None, criterion=None, epoch=None, warmup_epochs=None, \
             test_train_label=None, largest_mve=None, largest_r2=None, prev_valid_r2=None, largest_r2_variance=None, prev_valid_r2_variance=None, model_names=None, \
             avg_loss=None, avg_loss_mse=None, avg_loss_mse_var_valid_or_test=None, avg_loss_mse_valid = None, trial_name=None, print_epochs=None, print_plots=None):
    fold_metrics = []
    with torch.no_grad():
        predicted_label = torch.zeros((valid_size, 1))
        predicted_label_var = torch.zeros((valid_size, 1))
        actual_label = torch.zeros((valid_size, 1))
        actual_label_var = torch.zeros((valid_size, 1))
        count_valid, count_valid_var, mve_valid = 0, 0, 0
        cross_val_losses = []
        for j, (j_x,j_classes, j_seq, j_actual, j_s2_actual) in enumerate(valid_or_test_loader):
            j_x, j_seq, j_classes, j_actual, j_s2_actual = to_rank(rank, j_x, j_seq, j_classes, j_actual, j_s2_actual) #.type(dtype=torch.float32)
            src_mask = None
        # Forward pass
            mean_pred, variance_pred, _, _ = model(j_classes, src_mask, j_seq, need_grad)
        # Scale predictions
            variance_pred, mean_pred, j_actual = scale_up(standardize, scaler, variance_pred, mean_pred, j_actual, rank)
        # Compute validation loss (Scaled)
            mve_valid = (mve_valid * j + mve_loss(mean_pred, variance_pred, j_actual, beta).item()) / (j + 1)
            avg_loss_mse_valid = (avg_loss_mse_valid*j + criterion(mean_pred, j_actual).item())/(j+1)
            avg_loss_mse_var_valid_or_test = (avg_loss_mse_var_valid_or_test*j + criterion(variance_pred, j_s2_actual).item())/(j+1)
        # Collect predictions for mean - R2 calculation
            size = mean_pred.size(0)
            predicted_label[count_valid:count_valid + size, :] = mean_pred
            actual_label[count_valid:count_valid + size, :] = j_actual
            count_valid += size
        # Collect predictions for variance - R2 calculation
            size_var = variance_pred.size(0)
            predicted_label_var[count_valid_var:count_valid_var + size_var, :] = variance_pred
            actual_label_var[count_valid_var:count_valid_var + size_var, :] = j_s2_actual
            count_valid_var += size_var
        # Cross-Validation Metrics
            cross_val_losses.append(mve_loss(mean_pred, variance_pred, j_actual, beta).item())

    # Reshape
        predicted_label = predicted_label.cpu().numpy().reshape((-1, 1))
        actual_label = actual_label.cpu().numpy().reshape((-1, 1))
        predicted_label_var = predicted_label_var.cpu().numpy().reshape((-1, 1))
        actual_label_var = actual_label_var.cpu().numpy().reshape((-1, 1))
    # Calculate metrics on validation set
        predicted_label_stdev = np.sqrt(np.mean(predicted_label_var))
        actual_label_stdev = np.sqrt(np.mean(actual_label_var))
        valid_r2 = r2_score(actual_label, predicted_label)
        valid_r2_variance = r2_score(actual_label_var, predicted_label_var)

        if epoch is not None and epoch > warmup_epochs or epoch == None:
            if np.ptp(predicted_label_var) == 0:
                valid_pcc_variance = float('nan')
            else:
                valid_pcc_variance = np.corrcoef(actual_label_var.T, predicted_label_var.T)[0,1]
        else:
            valid_pcc_variance = float('nan')
        if np.ptp(predicted_label) == 0:
            mae = float('nan')
        else:
            mae = mean_absolute_error(actual_label, predicted_label)
        fold_metrics.append(np.mean(cross_val_losses))

    # Round metrics
        mve_valid = round(mve_valid, 2)
        avg_loss = round(avg_loss, 2)
        avg_loss_mse = int(round(avg_loss_mse, 0))
        avg_loss_mse_valid = int(round(avg_loss_mse_valid, 0))
        avg_loss_mse_var_valid_or_test = round(avg_loss_mse_var_valid_or_test, 0)
        valid_r2 = round(valid_r2, 2)
        valid_r2_variance = round(valid_r2_variance, 2)
        valid_pcc_variance = round(valid_pcc_variance, 2)
        mae = round(float(mae), 2)
        predicted_label_stdev = round(float(predicted_label_stdev), 2)
        actual_label_stdev = round(float(actual_label_stdev), 2)

    if test_train_label == "train":
        largest_mve, largest_r2, largest_r2_variance, prev_valid_r2, prev_valid_r2_variance = train_print_save(model, valid_size, epoch, warmup_epochs, test_train_label, largest_mve, largest_r2, prev_valid_r2, largest_r2_variance, prev_valid_r2_variance, model_names, valid_r2, valid_r2_variance, actual_label, predicted_label, actual_label_var, predicted_label_var, actual_label_stdev, predicted_label_stdev, valid_pcc_variance, avg_loss, avg_loss_mse, avg_loss_mse_var_valid_or_test, mve_valid, avg_loss_mse_valid, trial_name, print_epochs, print_plots)
        return largest_mve, largest_r2, largest_r2_variance, prev_valid_r2, prev_valid_r2_variance, fold_metrics
    else:
        test_print_save(model_names, actual_label, predicted_label, actual_label_var, predicted_label_var, valid_pcc_variance, valid_size, valid_r2, valid_r2_variance, mae, mve_valid, avg_loss_mse, avg_loss_mse_var_valid_or_test, predicted_label_stdev, actual_label_stdev, trial_name)
        return

def train_print_save(model, valid_size, epoch, warmup_epochs, test_train_label, largest_mve, largest_r2, prev_valid_r2, \
                     largest_r2_variance, prev_valid_r2_variance, model_names, valid_r2, valid_r2_variance, actual_label, predicted_label, \
                     actual_label_var, predicted_label_var, actual_label_stdev, predicted_label_stdev, \
                     valid_pcc_variance, avg_loss, avg_loss_mse, avg_loss_mse_var_valid_or_test, mve_valid, avg_loss_mse_valid, \
                     trial_name=None, print_epochs=None, print_plots=None):
    # Log the losses and R^2 score
    tensorboard_metrics(epoch, avg_loss, avg_loss_mse, avg_loss_mse_var_valid_or_test, mve_valid, valid_r2, valid_r2_variance, valid_pcc_variance, avg_loss_mse_valid)

    if not os.path.exists(f'./transformer/{trial_name}/model'):
        os.makedirs(f'./transformer/{trial_name}/model')

    # save the model if it has the best MVE, R2 mean, or R2 variance
    if mve_valid > largest_mve and epoch > warmup_epochs:
        torch.save(model, f'./transformer/{trial_name}/model/{model_names[0]}.pth')
        print(f"saved: {model_names[0]}  (epoch {(epoch+1)})")
        largest_mve = mve_valid
    if valid_r2 > largest_r2 and epoch > warmup_epochs:
        torch.save(model, f'./transformer/{trial_name}/model/{model_names[1]}.pth')
        print(f"saved: {model_names[1]}  (epoch {(epoch+1)})")
        largest_r2 = valid_r2
    if valid_r2_variance > largest_r2_variance and epoch > warmup_epochs and valid_r2_variance > .75:
        torch.save(model, f'./transformer/{trial_name}/model/{model_names[2]}.pth')
        print(f"saved: {model_names[2]}  (epoch {(epoch+1)})")
        largest_r2_variance = valid_r2_variance
    if valid_r2_variance > prev_valid_r2_variance and valid_r2 >= prev_valid_r2 and epoch > warmup_epochs and valid_r2_variance >= 0:
        torch.save(model, f'./transformer/{trial_name}/model/{model_names[3]}.pth')
        print(f"saved: {model_names[3]}  (epoch {(epoch+1)})")
        prev_valid_r2 = valid_r2
        prev_valid_r2_variance = valid_r2_variance

    # Print plots at certain epochs
    if epoch == 24 or epoch==49 or epoch == 99 or (epoch+1) %print_plots == 0:
        generate_all_plots(actual_label, predicted_label, actual_label_var, predicted_label_var, valid_pcc_variance, valid_size, epoch, test_train_label, trial_name=trial_name)
    
    # Print training data
    if epoch == 0 or (epoch+1) %print_epochs == 0:
        print(f'Done epoch {epoch+1}, MVE Train: {avg_loss}, MVE Valid: {mve_valid}, MSE Train: {avg_loss_mse}, MSE Valid: {avg_loss_mse_valid}, MSE Valid Var: {avg_loss_mse_var_valid_or_test}')
        print(f'Valid R2_Mean: {valid_r2}, Valid R2_Var: {valid_r2_variance}, PCC: {valid_pcc_variance}, Pred Stdev: {predicted_label_stdev}, Actual Stdev: {actual_label_stdev}')
        print()

    return largest_mve, largest_r2, largest_r2_variance, prev_valid_r2, prev_valid_r2_variance


def test_print_save(model_names, actual_label, predicted_label, actual_label_var, predicted_label_var, valid_pcc_variance, valid_size, \
                    valid_r2, valid_r2_variance, mae, mve_valid, avg_loss_mse, avg_loss_mse_var_valid_or_test, predicted_label_stdev, \
                    actual_label_stdev, trial_name=None):

    generate_all_plots(actual_label, predicted_label, actual_label_var, predicted_label_var, valid_pcc_variance, \
                       valid_size, epoch=None, test_train_label="test", model_names=model_names, trial_name=trial_name)

    print(f"""
    {('-' * 40)}
    Test R2_Mean | {valid_r2}
    Test R2_Var  | {valid_r2_variance}
    MAE          | {mae}
    PCC          | {valid_pcc_variance}
    MVE          | {mve_valid}
    MSE Mean     | {avg_loss_mse}
    MSE Var      | {avg_loss_mse_var_valid_or_test} 
    Pred Stdev   | {predicted_label_stdev} 
    Actual Stdev | {actual_label_stdev}
    {('-' * 40)}""")

    if not os.path.exists(f'./transformer/{trial_name}/predicted_test_values'):
        os.makedirs(f'./transformer/{trial_name}/predicted_test_values')

    np.save(f'./transformer/{trial_name}/predicted_test_values/pred_mean_{model_names}', predicted_label)
    np.save(f'./transformer/{trial_name}/predicted_test_values/pred_variance_{model_names}', predicted_label_var)
    np.save(f'./transformer/{trial_name}/predicted_test_values/actual_mean_{model_names}', actual_label)
    np.save(f'./transformer/{trial_name}/predicted_test_values/actual_variance_{model_names}', actual_label_var)