from torch.nn import functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import os
import numpy as np

color_1 = (50 / 255, 134 / 255, 143 / 255)
color_2 = (207/ 255, 86/ 255, 86/ 255)

leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def plot_reconstruction(sample, model, patch_size, device, logdir, epoch, name, training_strategy, mask_ratio=0.5):
    with torch.no_grad():

        x = sample['signal'].to(device).unsqueeze(0)
        orig_signal = x.clone()
        x = F.pad(x, (0, 0, 0, patch_size - x.shape[1] % patch_size))

        reconstruct, _, _, mask = model(x)

        if training_strategy == 'next_token_prediction':
            orig_signal = orig_signal[:, :reconstruct.shape[1]]
            orig_signal = orig_signal[:, patch_size:].squeeze()
            reconstruct = reconstruct[:, :-patch_size]
        else:
            mask = mask[:, :-patch_size, :]
            reconstruct = reconstruct[:, :-patch_size, :]

        # try to reconstruct one element at a time
        reconstruct = reconstruct.view(1, -1, orig_signal.shape[-1])
        orig_signal = orig_signal.view(1, -1, orig_signal.shape[-1])

        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(x.shape[-1] // 2, 2)
        gs.update(wspace=0.08, hspace=0.16)

        for i in range(orig_signal.shape[-1]):
            ax = plt.subplot(gs[i % 6, i // 6])
            ax.plot(orig_signal[..., i].cpu().squeeze().numpy(), color=color_1)
            # print('shift_reconstruct shape', shift_reconstruct.shape)
            ax.plot(reconstruct[..., i].cpu().squeeze().numpy(), color=color_2)

            # sometimes the signal is zeroed out by the random drop leads
            # so here i just plot the zeroed out signal with a dashed line to know that channel was zeroed out
            #has_augmentation = False
            #if (x[..., i] != orig_signal[..., i]).any():
            #    ax.plot(x[..., patch_size:reconstruct.shape[1], i].cpu().squeeze().numpy(), color='grey', linestyle='--')
            #    has_augmentation = True
                
            ax.set_title(leads[i])

            # add vertical lines avery patch size
            for j in range(0, x.shape[1], patch_size):
                ax.axvline(j, color='gray', linestyle='--', linewidth=0.5)
                
            if training_strategy == 'masked_token_prediction':
                if mask.shape[0] == 1:
                    ax_mask = mask.squeeze()
                else:
                    ax_mask = mask[i]

                ax.fill_between(
                    list(range(x.shape[1])),
                    orig_signal[..., i].min().item(),
                    orig_signal[..., i].max().item(),
                    where=ax_mask.detach().cpu(),
                    color='red',
                    alpha=0.3,
                    label='mask'
                )
        

            # ax.set_yticks([])
            if i == 0:
                if training_strategy == 'masked_token_prediction':
                    ax.legend(['Original', 'Reconstructed', 'Mask'], loc='upper left')
                else:
                    ax.legend(['Original', 'Reconstructed'], loc='upper left')

            if i == 5 or i == 11:
                ax.set_xticks(np.arange(0, len(x[0]), 360))
                ax.set_xticklabels(np.arange(0, len(x[0]), 360) // 360)
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticks([])

        # mkdir if it does not exist
        os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)

        path = f'{logdir}/epoch_{epoch}/reconstruction_{name}.png'
        plt.savefig(path)
        plt.close()
        return path

    
def plot_generation(sample, model, patch_size, device, logdir, epoch, name):
    with torch.no_grad():
        signal = sample['signal'].to(device).unsqueeze(0)

        signal = signal[:, :signal.shape[1] - signal.shape[1] % patch_size]
        if len(signal.shape) == 2:
            signal = signal.unsqueeze(-1)

        generated = model.generate(signal, length=(2048 // patch_size))

        fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(signal.shape[-1] // 2, 2)
        gs.update(wspace=0.08, hspace=0.16)

        for i in range(signal.shape[-1]):
            ax = plt.subplot(gs[i % 6, i // 6])
            ax.plot(signal[..., i].cpu().squeeze().numpy(), color=color_1)
            ax.plot(
                range(signal.shape[1], signal.shape[1] + generated.shape[1]),
                generated[..., i].cpu().squeeze().numpy(), color=color_2)
            ax.set_title(leads[i],)

            # add vertical lines avery patch size
            for j in range(0, signal.shape[1] + generated.shape[1], patch_size):
                ax.axvline(j, color='gray', linestyle='--', linewidth=0.5)

            # ax.set_yticks([])
            if i == 0:
                ax.legend(['Original', 'Generated'], loc='upper left')

            if i == 5 or i == 11:
                ax.set_xticks(np.arange(0, len(signal[0]), 360))
                ax.set_xticklabels(np.arange(0, len(signal[0]), 360) // 360)
                ax.set_xlabel('Time (s)')
            else:
                ax.set_xticks([])

        # mkdir if it does not exist
        os.makedirs(f'{logdir}/epoch_{epoch}', exist_ok=True)

        path = f'{logdir}/epoch_{epoch}/generation_{name}.png'
        plt.savefig(path)
        plt.close()
        return path


