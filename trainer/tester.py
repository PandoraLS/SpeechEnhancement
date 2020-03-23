# -*- coding: utf-8 -*-
# Author：lisen
# Date：20-1-5 下午10:36

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
from trainer.base_trainer import BaseTrainer
from util.utils import set_requires_grad, compute_STOI, compute_PESQ

plt.switch_backend("agg")


class Tester(BaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 generator,
                 discriminator,
                 generator_optimizer,
                 discriminator_optimizer,
                 additional_loss_function,
                 test_dl):
        super().__init__(config, resume, generator, discriminator, generator_optimizer, discriminator_optimizer,
                         additional_loss_function)
        self.test_dataloader = test_dl

    def _visualize_weights_and_grads(self, model, epoch):
        for name, param in model.named_parameters():
            self.writer.add_histogram("WEIGHT_" + name, param.clone().cpu().data.numpy(), epoch)
            self.writer.add_histogram("GRAD_" + name, param.grad.cpu().numpy(), epoch)

    def test(self):
        # model.eval()
        stoi_clean_and_noisy = []
        stoi_clean_and_enhanced = []
        pesq_clean_and_noisy = []
        pesq_clean_and_enhanced = []

        for i, (noisy, clean, name) in enumerate(self.test_dataloader, start=1):
            assert len(name) == 1, "The batch size of test dataloader should be 1."
            name = name[0]

            noisy = noisy.to(self.device)
            norm_max = torch.max(noisy).item()
            norm_min = torch.min(noisy).item()
            noisy = 2 * (noisy - norm_min) / (norm_max - norm_min) - 1

            assert noisy.dim() == 3
            noisy_chunks = torch.split(noisy, sample_length, dim=2)
            if noisy_chunks[-1].shape[-1] != sample_length:
                # Delete tail
                noisy_chunks = noisy_chunks[:-1]

            enhanced_chunks = []
            for noisy_chunk in noisy_chunks:
                enhanced_chunks.append(self.generator(noisy_chunk).detach().cpu().numpy().reshape(-1))

            enhanced = np.concatenate(enhanced_chunks)
            enhanced = (enhanced + 1) * (norm_max - norm_min) / 2 + norm_min

            noisy = noisy.cpu().numpy().reshape(-1)[:len(enhanced)]
            clean = clean.cpu().numpy().reshape(-1)[:len(enhanced)]
        
    

    def _validation_epoch(self, epoch):

        stoi_clean_and_noisy = []
        stoi_clean_and_enhanced = []
        pesq_clean_and_noisy = []
        pesq_clean_and_enhanced = []

        for i, (noisy, clean, name) in enumerate(self.validation_dataloader, start=1):
            assert len(name) == 1, "The batch size of validation dataloader should be 1."
            name = name[0]

            noisy = noisy.to(self.device)
            norm_max = torch.max(noisy).item()
            norm_min = torch.min(noisy).item()
            noisy = 2 * (noisy - norm_min) / (norm_max - norm_min) - 1

            assert noisy.dim() == 3
            noisy_chunks = torch.split(noisy, sample_length, dim=2)
            if noisy_chunks[-1].shape[-1] != sample_length:
                # Delete tail
                noisy_chunks = noisy_chunks[:-1]

            enhanced_chunks = []
            for noisy_chunk in noisy_chunks:
                enhanced_chunks.append(self.generator(noisy_chunk).detach().cpu().numpy().reshape(-1))

            enhanced = np.concatenate(enhanced_chunks)
            enhanced = (enhanced + 1) * (norm_max - norm_min) / 2 + norm_min

            noisy = noisy.cpu().numpy().reshape(-1)[:len(enhanced)]
            clean = clean.cpu().numpy().reshape(-1)[:len(enhanced)]

            if i <= visualize_audio_limit:
                self.writer.add_audio(f"Speech/{name}_noisy", noisy, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_clean", clean, epoch, sample_rate=16000)
                self.writer.add_audio(f"Speech/{name}_enhanced", enhanced, epoch, sample_rate=16000)

            if i <= visualize_waveform_limit:
                fig, ax = plt.subplots(3, 1)
                for j, y in enumerate([noisy, enhanced, clean]):
                    ax[j].set_title("mean: {:.3f}, std: {:.3f}, max: {:.3f}, min: {:.3f}".format(
                        np.mean(y),
                        np.std(y),
                        np.max(y),
                        np.min(y)
                    ))
                    librosa.display.waveplot(y, sr=16000, ax=ax[j])
                plt.tight_layout()
                self.writer.add_figure(f"Waveform/{name}", fig, epoch)

            if i <= visualize_spectrogram_limit:
                fig, axes = plt.subplots(3, 1, figsize=(6, 6))
                for j, y in enumerate([noisy, clean, enhanced]):
                    mag, _ = librosa.magphase(librosa.stft(y, n_fft=320, hop_length=160, win_length=320))
                    axes[j].set_title(f"mean: {np.mean(mag):.3f}, "
                                      f"std: {np.std(mag):.3f}, "
                                      f"max: {np.max(mag):.3f}, "
                                      f"min: {np.min(mag):.3f}")
                    librosa.display.specshow(librosa.amplitude_to_db(mag), cmap="magma", y_axis="linear", ax=axes[j],
                                             sr=16000, hop_length=160)

                plt.tight_layout()
                self.writer.add_figure(f"Spectrogram/{name}", fig, epoch)

            # Metrics
            stoi_clean_and_noisy.append(compute_STOI(clean, noisy, sr=16000))
            stoi_clean_and_enhanced.append(compute_STOI(clean, enhanced, sr=16000))
            pesq_clean_and_noisy.append(compute_PESQ(clean, noisy, sr=16000))
            pesq_clean_and_enhanced.append(compute_PESQ(clean, enhanced, sr=16000))

        self.writer.add_scalars(f"Metrics/STOI", {
            "clean and noisy": np.mean(stoi_clean_and_noisy),
            "clean and enhanced": np.mean(stoi_clean_and_enhanced)
        }, epoch)
        self.writer.add_scalars(f"Metrics/PESQ", {
            "clean and noisy": np.mean(pesq_clean_and_noisy),
            "clean and enhanced": np.mean(pesq_clean_and_enhanced)
        }, epoch)

        score = (np.mean(stoi_clean_and_enhanced) + self._transform_pesq_range(np.mean(pesq_clean_and_enhanced))) / 2
        return score
