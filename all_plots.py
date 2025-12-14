import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import date

writer = SummaryWriter(f"Training starting on:{date.today()}")
writer = SummaryWriter(comment="transformer model")

# Log the losses and R2 score
def tensorboard_metrics(epoch, avg_loss, avg_loss_mse, avg_loss_mse_var, valid_loss, valid_r2, valid_r2_variance, valid_pcc_variance, avg_loss_mse_valid):
    writer.add_scalar("MSE Loss per epoch/train mean", avg_loss_mse, epoch+1)
    writer.add_scalar("MSE Loss per epoch/train variance", avg_loss_mse_var, epoch+1)
    writer.add_scalar("MSE Loss per epoch/valid mean", avg_loss_mse_valid, epoch+1)
    writer.add_scalar("MVE Loss per epoch/train", avg_loss, epoch + 1)
    writer.add_scalar("MVE Loss per epoch/valid", valid_loss, epoch + 1)
    writer.add_scalar("R2 Loss per epoch/valid mean", valid_r2, epoch + 1)
    writer.add_scalar("R2 Loss per epoch/valid variance", valid_r2_variance, epoch + 1)
    writer.add_scalar("Pearson Correlation per epoch/valid variance", valid_pcc_variance, epoch + 1)

def generate_all_plots(actual_label, predicted_label, actual_label_var, predicted_label_var, valid_pcc_variance, \
                       valid_size, epoch, test_train_label, model_names=None, trial_name=None): 
    def filename(plot_type, plot_saved_name, plot_title): 
        directory_path = f'./transformer/{trial_name}/plots/{plot_type}'
        os.makedirs(directory_path, exist_ok=True)
        if test_train_label == "train" and epoch is not None:
            if plot_type == "mean_plots":
                title_name = f'{plot_title} (Epoch {epoch + 1})'
            else:
                title_name = f'{plot_title} (Epoch {epoch + 1}, Variance PCC {valid_pcc_variance})'
            save_fig = f'{directory_path}/{test_train_label}_{plot_saved_name}_epoch_{epoch + 1}.png'
            return title_name, save_fig 
        else:
            if plot_type == "mean_plots":
                title_name = f'{plot_title}'
            else:
                title_name = f'{plot_title}, Variance PCC {valid_pcc_variance}'
            if model_names:
                save_fig = f'{directory_path}/{test_train_label}_{model_names}_{plot_saved_name}.png'
            else:
                save_fig = f'{directory_path}/{test_train_label}_{plot_saved_name}.png'
            return title_name, save_fig

    def normalize(data):
        if np.max(data) == np.min(data):
            return np.zeros_like(data)
        else:
            normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
            return normalized

    ## Plots for actual vs predicted variance
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    # First subplot: Log-Scale Actual vs Predicted Variance
    title_name, save_fig = filename("variance_plots", "var_actual_vs_pred", "Actual vs Predicted Variance")
    ax1.scatter(range(len(actual_label_var)), actual_label_var, color='red', alpha=0.6, label='Actual Variance', s=15)
    ax1.scatter(range(len(predicted_label_var)), predicted_label_var, color='blue', alpha=0.6, label='Predicted Variance', s=15)
    ax1.plot(range(len(actual_label_var)), actual_label_var, color='red', alpha=0.6)
    ax1.plot(range(len(predicted_label_var)), predicted_label_var, color='blue', alpha=0.6)
    ax1.set_yscale('log')
    ax1.set_title(title_name)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Variance (Log Scale)')
    ax1.legend()
    # Second subplot: Normalized Actual vs Predicted Variance
    title_name, _ = filename("variance_plots", "var_normal_actual_vs_pred", "Normalized Actual vs Predicted Variance")
    actual_label_var_norm = normalize(actual_label_var)
    predicted_label_var_norm = normalize(predicted_label_var)
    ax2.scatter(range(len(actual_label_var_norm)), actual_label_var_norm, color='red', alpha=0.6, label='Actual Variance', s=15)
    ax2.scatter(range(len(predicted_label_var_norm)), predicted_label_var_norm, color='blue', alpha=0.6, label='Predicted Variance', s=15)
    ax2.plot(range(len(actual_label_var_norm)), actual_label_var_norm, color='red', alpha=0.6)
    ax2.plot(range(len(predicted_label_var_norm)), predicted_label_var_norm, color='blue', alpha=0.6)
    ax2.set_title(title_name)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Normalized Variance')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_fig)
    plt.close()

    ## Plots for actual vs predicted standard deviation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    # First subplot: Actual vs Predicted Standard Deviation     # First subplot: Log-Scale Actual vs Predicted Standard Deviation
    title_name, save_fig = filename("stdev_plots", "stdev_actual_vs_pred", "Actual vs Predicted Standard Deviation")
    ax1.scatter(range(len(actual_label_var)), np.sqrt(actual_label_var), color='red', alpha=0.6, label='Actual Standard Deviation', s=15)
    ax1.scatter(range(len(predicted_label_var)), np.sqrt(predicted_label_var), color='blue', alpha=0.6, label='Predicted Standard Deviation', s=15)
    ax1.plot(range(len(actual_label_var)), np.sqrt(actual_label_var), color='red', alpha=0.6)
    ax1.plot(range(len(predicted_label_var)), np.sqrt(predicted_label_var), color='blue', alpha=0.6)
    # ax1.set_yscale('log')
    ax1.set_title(title_name)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Standard Deviation')   # ax1.set_ylabel('Standard Deviation (Log Scale)')
    ax1.legend()
    # Second subplot: Normalized Actual vs Predicted Standard Deviation
    title_name, _ = filename("stdev_plots", "stdev_normal_actual_vs_pred", "Normalized Actual vs Predicted Standard Deviation")
    actual_label_stdev_norm = normalize(np.sqrt(actual_label_var))
    predicted_label_stdev_norm = normalize(np.sqrt(predicted_label_var))
    ax2.scatter(range(len(actual_label_stdev_norm)), actual_label_stdev_norm, color='red', alpha=0.6, label='Actual Standard Deviation', s=15)
    ax2.scatter(range(len(predicted_label_stdev_norm)), predicted_label_stdev_norm, color='blue', alpha=0.6, label='Predicted Standard Deviation', s=15)
    ax2.plot(range(len(actual_label_stdev_norm)), actual_label_stdev_norm, color='red', alpha=0.6)
    ax2.plot(range(len(predicted_label_stdev_norm)), predicted_label_stdev_norm, color='blue', alpha=0.6)
    ax2.set_title(title_name)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Normalized Standard Deviation')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_fig)
    plt.close()

    ## Plots for actual vs predicted mean
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    # First subplot: X = Y
    title_name, save_fig = filename("mean_plots", "mean_actual_vs_pred", "Actual vs Predicted Mean: X = Y")
    ax1.scatter(actual_label, predicted_label, color='blue', alpha=0.5, label='(Actual Mean, Predicted Mean)')
    min_val = min(min(actual_label), min(predicted_label))
    max_val = max(max(actual_label), max(predicted_label))
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='X = Y')
    ax1.set_title(title_name)
    ax1.set_xlabel('Actual Mean')
    ax1.set_ylabel('Predicted Mean')
    ax1.legend()
    # Second subplot: Scatterplot
    title_name, _ = filename("mean_plots", "mean_actual_vs_pred", "Actual vs Predicted Mean: Scatterplot")
    ax2.scatter(range(len(actual_label)), actual_label, color='red', alpha=0.6, label='Actual Mean')
    ax2.scatter(range(len(predicted_label)), predicted_label, color='blue', alpha=0.6, label='Predicted Mean')
    ax2.set_title(title_name)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Mean')
    ax2.legend()
    plt.tight_layout()
    plt.savefig(save_fig)
    plt.close()

    ## Plots for actual vs predicted prediction intervals (in a 2-row, 2-column layout)
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    ax1, ax2, ax3 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, :])
    # Compute bounds for actual and predicted data
    upper_bound_actual = actual_label + 1.96 * np.sqrt(actual_label_var)
    lower_bound_actual = actual_label - 1.96 * np.sqrt(actual_label_var)
    upper_bound_pred = predicted_label + 1.96 * np.sqrt(predicted_label_var)
    lower_bound_pred = predicted_label - 1.96 * np.sqrt(predicted_label_var)
    # First subplot (Top-Left): Predicted Mean & Variance
    title_name, save_fig = filename("PI_plots", "prediction_interval", "Predicted Prediction Intervals")  
    ax1.fill_between(range(len(predicted_label)), lower_bound_pred.flatten(), upper_bound_pred.flatten(), color='gray', alpha=0.5, label='Predicted ±1.96σ')
    ax1.scatter(range(len(predicted_label)), predicted_label.flatten(), color='blue', alpha=0.6, label='Predicted Mean')
    ax1.plot(range(len(predicted_label)), predicted_label.flatten(), 'b-', linewidth=1.5)
    ax1.set_title(title_name)
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Value')
    ax1.legend()
    # Second subplot (Top-Right): Actual Mean & Variance
    title_name, _ = filename("PI_plots", "prediction_interval", "Actual Prediction Intervals")  
    ax2.fill_between(range(len(actual_label)), lower_bound_actual.flatten(), upper_bound_actual.flatten(), color='gray', alpha=0.5, label='Actual ±1.96σ')
    ax2.scatter(range(len(actual_label)), actual_label.flatten(), color='red', alpha=0.6, label='Actual Mean')
    ax2.plot(range(len(actual_label)), actual_label.flatten(), 'r-', linewidth=1.5)
    ax2.set_title(title_name)
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Value')
    ax2.legend()
    # Third subplot (Bottom, spanning both columns): Overlayed Comparison
    title_name, _ = filename("PI_plots", "prediction_interval", "Overlayed Comparison: Prediction Intervals")  
    ax3.fill_between(range(len(actual_label)), lower_bound_actual.flatten(), upper_bound_actual.flatten(), color='red', alpha=0.25, label='Actual ±1.96σ')
    ax3.scatter(range(len(actual_label)), actual_label.flatten(), color='red', alpha=0.6, label='Actual Mean', s=20)
    ax3.plot(range(len(actual_label)), actual_label.flatten(), 'r-', linewidth=1.67, label='Actual Mean')
    ax3.fill_between(range(len(predicted_label)), lower_bound_pred.flatten(), upper_bound_pred.flatten(), color='blue', alpha=0.25, label='Predicted ±1.96σ')
    ax3.scatter(range(len(predicted_label)), predicted_label.flatten(), color='blue', alpha=0.6, label='Predicted Mean', s=20)
    ax3.plot(range(len(predicted_label)), predicted_label.flatten(), 'b--', linewidth=1.67, label='Predicted Mean')
    ax3.set_title(title_name)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Value')
    ax3.legend()
    plt.tight_layout()
    plt.savefig(save_fig)
    plt.close()