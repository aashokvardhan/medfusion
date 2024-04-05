
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
from torchvision.utils import save_image

from medical_diffusion.loss.perceivers import LPIPS
from medical_diffusion.models.model_base import BasicModel
from medical_diffusion.models.utils.conv_blocks_v2 import *


class DiagonalGaussianDistribution(nn.Module):

    def forward(self, x):
        mean, logvar = torch.chunk(x, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        sample = torch.randn(mean.shape, generator=None, device=x.device)
        z = mean + std * sample

        batch_size = x.shape[0]
        var = torch.exp(logvar)
        kl = 0.5 * torch.sum(torch.pow(mean, 2) + var - 1.0 - logvar)/batch_size

        return z, kl



class VAE(BasicModel):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        spatial_dims = 2,
        emb_channels = 4,
        channel_mul = 64,
        hid_chs =    [ 1, 2,  4, 8],
        embedding_loss_weight=1e-6,
        perceiver = LPIPS,
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
        optimizer=torch.optim.AdamW,
        optimizer_kwargs={'lr':1e-4},
        lr_scheduler= None,
        lr_scheduler_kwargs={},
        loss = torch.nn.L1Loss,
        loss_kwargs={'reduction': 'none'},
        sample_every_n_steps = 1000
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )
        self.sample_every_n_steps=sample_every_n_steps
        self.loss_fct = loss(**loss_kwargs)
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None
        self.perceptual_loss_weight = perceptual_loss_weight
        self.depth = len(hid_chs)
        self.ch = channel_mul

        # ----------- In-Convolution ------------
        ConvBlock = Conv[Conv.CONV, spatial_dims]

        # ----------- Encoder ----------------
        self.encoder = Encoder(ch=channel_mul, ch_mult=hid_chs, num_res_blocks=2, in_channels=in_channels, z_channels= emb_channels)

        self.quant_conv = ConvBlock(2*emb_channels, 2*emb_channels, 1)
        self.post_quant_conv = ConvBlock(emb_channels, emb_channels, 1)

        # ----------- Reparameterization --------------
        self.quantizer = DiagonalGaussianDistribution()

        # ----------- In-Decoder ------------
        self.decoder = Decoder(ch=channel_mul, out_ch=out_channels, ch_mult=hid_chs, num_res_blocks=2, dropout=0.0, resamp_with_conv=True, in_channels=emb_channels,z_channels=emb_channels)

    def encode(self, x):
        h = self.encoder(x)
        z = self.quant_conv(h)
        z, _ = self.quantizer(z)
        return z

    def decode(self, z):
        h = self.post_quant_conv(z)
        dec = self.decoder(h)
        return dec

    def forward(self, x_in):
        # --------- Encoder --------------
        h = self.encoder(x_in)
        z = self.quant_conv(h)

        # --------- Quantizer --------------
        z_q, emb_loss = self.quantizer(z)

        # -------- Decoder -----------
        h = self.post_quant_conv(z)
        dec = self.decoder(h)

        return dec, emb_loss

    def perception_loss(self, pred, target, depth=0):
        if (self.perceiver is not None) and (depth<2):
            self.perceiver.eval()
            return self.perceiver(pred, target)*self.perceptual_loss_weight
        else:
            return 0

    def ssim_loss(self, pred, target):
        return 1-ssim(((pred+1)/2).clamp(0,1), (target.type(pred.dtype)+1)/2, data_range=1, size_average=False,
                        nonnegative_ssim=True).reshape(-1, *[1]*(pred.ndim-1))

    def rec_loss(self, pred, pred_vertical, target):
        interpolation_mode = 'nearest-exact'

        # Loss
        loss = 0
        rec_loss = self.loss_fct(pred, target)+self.perception_loss(pred, target)+self.ssim_loss(pred, target)
        # rec_loss = rec_loss/ torch.exp(self.logvar) + self.logvar # Note this is include in Stable-Diffusion but logvar is not used in optimizer
        loss += torch.sum(rec_loss)/pred.shape[0]


        for i, pred_i in enumerate(pred_vertical):
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)
            rec_loss_i = self.loss_fct(pred_i, target_i)+self.perception_loss(pred_i, target_i)+self.ssim_loss(pred_i, target_i)
            # rec_loss_i = rec_loss_i/ torch.exp(self.logvar_ver[i]) + self.logvar_ver[i]
            loss += torch.sum(rec_loss_i)/pred.shape[0]

        return loss

    def _step(self, batch: dict, batch_idx: int, state: str, step: int, optimizer_idx:int):
        # ------------------------- Get Source/Target ---------------------------
        x = batch['source']
        target = x

        # ------------------------- Run Model ---------------------------
        pred, pred_vertical, emb_loss = self(x)

        # ------------------------- Compute Loss ---------------------------
        loss = self.rec_loss(pred, pred_vertical, target)
        loss += emb_loss*self.embedding_loss_weight

        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss':loss, 'emb_loss': emb_loss}
            logging_dict['L2'] = torch.nn.functional.mse_loss(pred, target)
            logging_dict['L1'] = torch.nn.functional.l1_loss(pred, target)
            logging_dict['ssim'] = ssim((pred+1)/2, (target.type(pred.dtype)+1)/2, data_range=1)
            # logging_dict['logvar'] = self.logvar

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log(f"{state}/{metric_name}", metric_val, batch_size=x.shape[0], on_step=True, on_epoch=True)

        # ----------------- Save Image ------------------------------
        if self.global_step != 0 and self.global_step % self.sample_every_n_steps == 0:
            log_step = self.global_step // self.sample_every_n_steps
            path_out = Path(self.logger.log_dir)/'images'
            path_out.mkdir(parents=True, exist_ok=True)
            # for 3D images use depth as batch :[D, C, H, W], never show more than 16+16 =32 images
            def depth2batch(image):
                return (image if image.ndim<5 else torch.swapaxes(image[0], 0, 1))
            images = torch.cat([depth2batch(img)[:16] for img in (x, pred)])
            save_image(images, path_out/f'sample_{log_step}.png', nrow=x.shape[0], normalize=True)

        return loss