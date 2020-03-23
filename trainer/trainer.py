# -*- coding: utf-8 -*-
# Author：sen
# Date：2020/3/22 10:47

import torch
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from trainer.base_trainer import BaseTrainer
from util.utils import compute_PESQ, compute_STOI
from util.loss import SDRLoss, RMSELoss
from util.stoi_utils import STOICalculator


class GeneralTrainer(BaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 model,
                 optim,
                 loss_fucntion,
                 train_dl,
                 validation_dl):
        super().__init__(config, resume, model, optim, loss_fucntion)
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl

    def _visualize_weights_and_grads(self, model, epoch):
        for name, param in model.named_parameters():
            self.writer.add_histogram("WEIGHT_" + name, param.clone().cpu().data.numpy(), epoch)
            self.writer.add_histogram("GRAD_" + name, param.grad.cpu().numpy(), epoch)

    def _train_epoch(self, epoch):
        for i, (noisy, clean, name) in enumerate(self.train_dataloader, start=1):
            # For visualization
            batch_size = self.train_dataloader.batch_size
            n_batch = len(self.train_dataloader)
            n_iter = n_batch * batch_size * (epoch - 1) + i * batch_size

            noisy = noisy.to(self.device)  # [batch_size, 1, sample_length] eg.[600,1,16384]
            clean = clean.to(self.device)  # [600,1,16384]
            enhanced = self.model(noisy)  # [600, 1, 16384]

            """================ Optimize model ================"""
            self.optimizer.zero_grad()
            loss = self.loss_function(enhanced, clean)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                enhanced = self.model(noisy)

                self.writer.add_scalars(f"模型/损失值", {
                    "模型优化前": loss,
                    "模型优化后": self.loss_function(enhanced, clean)
                }, n_iter)

    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

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
            noisy = 2 * (noisy - norm_min) / (norm_max - norm_min) - 1  # 将音频数据归一化到[-1.0,1.0]

            assert noisy.dim() == 3
            noisy_chunks = torch.split(noisy, sample_length, dim=2)
            if noisy_chunks[-1].shape[-1] != sample_length:
                # Delete tail
                noisy_chunks = noisy_chunks[:-1]

            enhanced_chunks = []
            for noisy_chunk in noisy_chunks:
                enhanced_chunks.append(self.model(noisy_chunk).detach().cpu().numpy().reshape(-1))

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
                    mag, _ = librosa.magphase((librosa.stft(y, n_fft=320, hop_length=160, win_length=320)))
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

stoi_cal = None
class JointTrainer(BaseTrainer):
    def __init__(self,
                 config: dict,
                 resume: bool,
                 model,
                 optim,
                 loss_function,
                 train_dl,
                 validation_dl):
        super().__init__(config, resume, model, optim, loss_function)
        self.train_dataloader = train_dl
        self.validation_dataloader = validation_dl

        # joint loss function
        self.sdr_loss_function = SDRLoss()
        self.rmse_loss_fucntion = RMSELoss()

        # joint loss function factor
        self.joint_loss_config = config["trainer"]["joint"]
        self.joint_stoi_loss_factor = self.joint_loss_config["stoi_factor"]
        self.joint_sdr_loss_factor = self.joint_loss_config["sdr_factor"]
        self.joint_rmse_loss_factor = self.joint_loss_config["rmse_factor"]

    def _visualize_weights_and_grads(self, model, epoch):
        for name, param in model.named_parameters():
            self.writer.add_histogram("WEIGHT_" + name, param.clone().cpu().data.numpy(), epoch)
            self.writer.add_histogram("GRAD_" + name, param.grad.cpu().numpy(), epoch)

    def _calc_loss(self, deg, clean):
        """
        用于计算JointLoss的值
        :param deg: 带噪语音(可以是noisy,也可以是enhanced)
        :param clean: 对应的干净语音
        :return: joint loss
        """
        global stoi_cal
        if stoi_cal is None:
            stoi_cal = STOICalculator(windowsize=512, hop=64, requires_grad=False)
            stoi_cal.cuda()
        loss_all = 0
        if self.joint_stoi_loss_factor != 0.:
            loss_stoi = -1 * stoi_cal(deg, clean)
            loss_all += loss_stoi * self.joint_stoi_loss_factor
        if self.joint_sdr_loss_factor != 0.:
            loss_sdr = self.sdr_loss_function(deg, clean)
            loss_all += loss_sdr * self.joint_sdr_loss_factor
        if self.joint_rmse_loss_factor != 0.:
            loss_rmse = self.rmse_loss_fucntion(deg, clean)
            loss_all += loss_rmse * self.joint_rmse_loss_factor
        return loss_all

    def _train_epoch(self, epoch):
        for i, (noisy, clean, name) in enumerate(self.train_dataloader, start=1):
            # For visualization
            batch_size = self.train_dataloader.batch_size
            n_batch = len(self.train_dataloader)
            n_iter = n_batch * batch_size * (epoch - 1) + i * batch_size

            noisy = noisy.to(self.device)  # [batch_size, 1, sample_length] eg.[600,1,16384]
            clean = clean.to(self.device)  # [600,1,16384]
            enhanced = self.model(noisy)  # [600, 1, 16384]

            """================ Optimize model ================"""
            self.optimizer.zero_grad()

            loss = self._calc_loss(enhanced, clean)
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                self.writer.add_scalars(f"模型/损失值", {
                    "模型优化过程": loss
                }, n_iter)

    def _validation_epoch(self, epoch):
        visualize_audio_limit = self.validation_custom_config["visualize_audio_limit"]
        visualize_waveform_limit = self.validation_custom_config["visualize_waveform_limit"]
        visualize_spectrogram_limit = self.validation_custom_config["visualize_spectrogram_limit"]

        sample_length = self.validation_custom_config["sample_length"]

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
            noisy = 2 * (noisy - norm_min) / (norm_max - norm_min) - 1  # 将音频数据归一化到[-1.0,1.0]

            assert noisy.dim() == 3
            noisy_chunks = torch.split(noisy, sample_length, dim=2)
            if noisy_chunks[-1].shape[-1] != sample_length:
                # Delete tail
                noisy_chunks = noisy_chunks[:-1]

            enhanced_chunks = []
            for noisy_chunk in noisy_chunks:
                enhanced_chunks.append(self.model(noisy_chunk).detach().cpu().numpy().reshape(-1))

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
                    mag, _ = librosa.magphase((librosa.stft(y, n_fft=320, hop_length=160, win_length=320)))
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
