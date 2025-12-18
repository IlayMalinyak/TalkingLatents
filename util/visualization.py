import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import umap

BOUNDS = {'MAX_TEFF' : 7500, 'MIN_TEFF' : 3000, 'MAX_LOGG' : 5.0, 'MIN_LOGG' : 0, 'MAX_FEH' : 0.5, 'MIN_FEH' : -3}


def _scale(arr, name):
    return arr * (BOUNDS[f'MAX_{name.upper()}'] - BOUNDS[f'MIN_{name.upper()}']) + BOUNDS[f'MIN_{name.upper()}']


def plot_quantiles(true, preds, quantiles, names, savedir='figs'):
    num_quantiles = len(quantiles) // 2
    for i, name in enumerate(names):
        color_list = plt.cm.get_cmap('Dark2', 8)
        print("plotting ", name)
        t = _scale(true[:, i], name)
        sorting_indices = np.argsort(t)
        t = t[sorting_indices]
        for q in range(num_quantiles):
            color = color_list(q)
            conf_interval = (quantiles[-(q + 1)] - quantiles[q]) * 100
            q_pred_down = _scale(preds[:, i, q][sorting_indices], name)
            q_pred_up = _scale(preds[:, i, -(q + 1)][sorting_indices], name) 
            q_up_smooth = savgol_filter(q_pred_down, window_length=20, polyorder=1)
            q_down_smooth = savgol_filter(q_pred_up, window_length=20, polyorder=1)
            plt.plot(t, q_up_smooth, color=color, alpha=0.4, label=f'{conf_interval}% confidence interval')
            plt.plot(t, q_down_smooth, color=color, alpha=0.4)
        q_med = _scale(preds[:, i, num_quantiles][sorting_indices], name)
        plt.scatter(t, q_med, color='orange', alpha=0.4, s=5)
        plt.xlabel(f'True {name}')
        plt.ylabel(f'Predicted {name}')
        plt.legend()
        plt.savefig(f'{savedir}/predictions_{name}.png')
        plt.close()

def plot_decode(true, preds, info, wv=None, num_sampels=10, savedir='figs'):
    for i in range(num_sampels):
        id = info.loc[i, 'obsid']
        fig, axes = plt.subplots(2,1)
        decode_sample = preds[i]
        x = wv[i] if wv is not None else np.arange(0, len(decode_sample))
        axes[0].plot(x, true[i], alpha=0.5, label='Original Sample')
        axes[0].plot(x, decode_sample, alpha=0.3, label='Decoded Sample')
        axes[1].plot(x[1000:2000], preds[i][1000:2000], alpha=0.8, label='Decoded Sample')
        axes[1].plot(x[1000:2000], true[i][1000:2000], alpha=0.7, label='True Sample')
        fig.suptitle(f'id: {id}')
        if wv is not None:
            fig.supxlabel('Wavelegnth (A)')
        fig.supylabel('Flux')
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{savedir}/reconstruction_sample_{i}.png')
        plt.close()


def plot_umap(features, info_df, colors, savedir='figs'):
    reducer_standard = umap.UMAP(n_components=2, random_state=42)  # Added random_state for reproducibility
    u_res = reducer_standard.fit_transform(features)
    for color in colors:
        plt.scatter(u_res[:, 0], u_res[:, 1], c=info_df[color])
        plt.colorbar(label=color)
        plt.xlabel('umap x')
        plt.ylabel('umap y')
        plt.savefig(f'{savedir}/umap_{color}.png')
        plt.close()