import os
import sys
import yaml
import random
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from tqdm import tqdm
import torchvision.utils as vutils
from datetime import datetime
from torchvision.transforms import Compose, Normalize
from PIL import Image
import numpy as np
import kornia
from safetensors.torch import save_file

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.append(project_root)

from common.data_utils_inst import *

class Loader:
    def __init__(self, config, model, scheduler, accelerator, optimizer=None, lr_scheduler=None, train_loader=None, test_loader=None, device=None):
        self.config = config
        self.model = model
        self.use_ema = config.get('use_ema', False)
        if self.use_ema:
            self.ema = EMA(beta=0.9999)
            self.ema_model = self.create_ema_model(model)

        self.scheduler = scheduler
        self._timesteps = config['model']['scheduler']['timesteps']
        self._inference_timesteps = config['model']['scheduler']['inference_steps']
        self.scheduler.set_timesteps(self._inference_timesteps)

        self.accelerator = accelerator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.image_counter = 0

        self.base_path, self.scaler, self.loss_fn, self.train_epochs, self.num_images_per_class, self.img_size = self.initialize()

    def initialize(self):
        print(self.count_parameters())
        base_path = self.initialize_paths()
        self.save_config(base_path)
        scaler = GradScaler()
        loss_fn = self.set_loss_fn() if self.config['config_type'] == 'train' else None
        train_epochs = self.config['loader']['train']['epoch']
        num_images_per_class = 10
        img_size = self.config['data']['resize'][0]
        
        return base_path, scaler, loss_fn, train_epochs, num_images_per_class, img_size
    
    def create_ema_model(self, model):
        ema_model = type(model)(self.config).to(self.device)
        ema_model.load_state_dict(model.state_dict())
        return ema_model

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_loss_fn(self):
        loss_type = self.config['model']['unet']['loss']
        if loss_type == 'l1':
            return nn.L1Loss()
        elif loss_type == 'huber':
            return nn.HuberLoss()
        elif loss_type == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def initialize_paths(self):
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = f"{self.config['config_type']}_{current_time}"
        print(base_path)
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(base_path + "/image_results", exist_ok=True)
        return base_path

    def save_config(self, base_path):
        output_config = self.config.copy()
        output_config['config_type'] = 'test'
        
        with open(base_path + f'/{self.config["config_type"]}_config.yaml', 'w') as f:
            f.write(f"# {self.count_parameters()}\n")
            yaml.safe_dump(output_config, f, default_flow_style=False, sort_keys=False)

    @staticmethod
    def reverse_transform():
        return Compose([Normalize(mean=(-1, -1, -1), std=(2, 2, 2))])

    def save_image_results(self, epoch, comparison_images, use_ema=False):
        prefix = "ema_" if use_ema else ""
        vutils.save_image(comparison_images, f"{self.base_path}/image_results/{prefix}epoch_{epoch+1}.png", nrow=self.num_images_per_class)

    def save_gif(self, epoch, gif_frames):
        combined_gif_frames = []
        for i in range(len(gif_frames[0])):
            combined_frame = np.concatenate([pair_frames[i] for pair_frames in gif_frames], axis=1)
            combined_gif_frames.append(combined_frame)
        pil_images = [Image.fromarray(frame) for frame in combined_gif_frames]
        gif_path = f"{self.base_path}/image_results/epoch_{epoch+1}_steps.gif"
        pil_images[0].save(
            gif_path, save_all=True, append_images=pil_images[1:], duration=50
        )

    from safetensors.torch import save_file

    def train_save_checkpoint(self, epoch):
        checkpoint_dir = f"{self.base_path}/checkpoints/"

        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.safetensors")
        
        model_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        
        save_file(model_state_dict, checkpoint_path)

    from safetensors.torch import save_file

    def train_save_checkpoint_latest(self, epoch):

        checkpoint_dir = f"{self.base_path}/checkpoints/latest/"

        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, "latest_checkpoint.safetensors")
        
        model_state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        
        save_file(model_state_dict, checkpoint_path)

    def train_save_ema_checkpoint(self, epoch):
        checkpoint_path = f"{self.base_path}/checkpoints/ema_checkpoint_epoch_{epoch+1}.pt"
        torch.save(self.ema_model.state_dict(), checkpoint_path)

    @torch.no_grad()
    def train_generate_images(self, epoch, rgb_nir_pairs, use_canny=False, use_ema=False):
        model = self.ema_model if use_ema else self.model
        model.eval()
        
        latents = []
        original_rgbs = []
        original_nirs = []
        original_inst = []

        reverse_transform_tensor = self.reverse_transform()

        for rgb_img, nir_img, inst_img in rgb_nir_pairs:
            latent = torch.randn((1, nir_img.shape[0], self.img_size, self.img_size), device=self.device)
            rgb_img = rgb_img.unsqueeze(0).to(self.device)
            nir_img = nir_img.unsqueeze(0).to(self.device)
            inst_img = inst_img.unsqueeze(0).to(self.device)
            
            for t in tqdm(self.scheduler.timesteps):
                with torch.no_grad():
                    timestep = torch.tensor([t], device=self.device).expand(latent.shape[0])
                    pred_noise = model(latent, timestep, rgb_img, inst_img)
                    latent = self.scheduler.step(pred_noise, t, latent).prev_sample
            
            latents.append(latent)
            original_rgbs.append(rgb_img)
            original_nirs.append(nir_img)
            original_inst.append(inst_img)

        gathered_latents = self.accelerator.gather(latents)
        gathered_original_rgbs = self.accelerator.gather(original_rgbs)
        gathered_original_nirs = self.accelerator.gather(original_nirs)
        gathered_original_inst = self.accelerator.gather(original_inst)

        if self.accelerator.is_main_process:
            gathered_latents = [latent.to("cpu") for latent in gathered_latents]
            gathered_original_rgbs = [rgb.to("cpu") for rgb in gathered_original_rgbs]
            gathered_original_nirs = [nir.to("cpu") for nir in gathered_original_nirs]
            gathered_original_insts = [inst.to("cpu") for inst in gathered_original_inst]

            concatenated_latents = torch.cat(gathered_latents, dim=0)
            concatenated_original_rgbs = torch.cat(gathered_original_rgbs, dim=0)
            concatenated_original_nirs = torch.cat(gathered_original_nirs, dim=0)
            concatenated_original_insts = torch.cat(gathered_original_insts, dim=0)
            concatenated_original_insts = concatenated_original_insts.repeat(1, 3, 1, 1)

            sample_images = reverse_transform_tensor(concatenated_latents)
            original_images = reverse_transform_tensor(concatenated_original_rgbs)
            original_images2 = reverse_transform_tensor(concatenated_original_nirs)
            #original_images3 = reverse_transform_tensor(concatenated_original_insts)

            if use_canny:
                _, canny_rgb = kornia.filters.Canny()(original_images)
                canny_rgb = canny_rgb.repeat(1, 3, 1, 1)
                comparison_images = torch.cat((original_images, canny_rgb, original_images2, sample_images), dim=0)
            else:
                comparison_images = torch.cat((original_images, original_images2, concatenated_original_insts, sample_images), dim=0)
            self.save_image_results(epoch, comparison_images, use_ema)

    @torch.no_grad()
    def test_generate_images(self, batch_idx, rgb, nir, inst):
        self.model.eval()
        
        reverse_transform_tensor = self.reverse_transform()
        
        batch_size = rgb.shape[0]
        
        rgb = rgb.to(self.device)
        nir = nir.to(self.device)
        inst = inst.to(self.device)
        
        latent_batch = torch.randn((batch_size, nir.shape[1], self.img_size, self.img_size), device=self.device)
        
        frames = []
        
        for t in tqdm(self.scheduler.timesteps, desc=f"Generating batch {batch_idx}"):
            timestep = torch.tensor([t], device=self.device).expand(batch_size)
            pred_noise = self.model(latent_batch, timestep, rgb, inst)
            latent_batch = self.scheduler.step(pred_noise, t, latent_batch).prev_sample

            # if t % 25 == 0 and self.accelerator.is_main_process:
            #     with torch.cuda.amp.autocast(enabled=False):
            #         current_latent = latent_batch.to("cpu")
            #         current_image = reverse_transform_tensor(current_latent)
            #         
            #         frame = transforms.ToPILImage()(current_image.squeeze())
            #         frames.append(frame)
        
        gathered_latents = self.accelerator.gather(latent_batch)
        gathered_original_rgbs = self.accelerator.gather(rgb)
        gathered_original_nirs = self.accelerator.gather(nir)
        
        if self.accelerator.is_main_process:
            gathered_latents = gathered_latents.to("cpu")
            gathered_original_rgbs = gathered_original_rgbs.to("cpu")
            gathered_original_nirs = gathered_original_nirs.to("cpu")
            
            sample_images = reverse_transform_tensor(gathered_latents)
            original_images = reverse_transform_tensor(gathered_original_rgbs)
            original_images2 = reverse_transform_tensor(gathered_original_nirs)
            
            for i in range(batch_size):
                self.image_counter += 1 
                vutils.save_image(original_images[i], f"{self.base_path}/image_results/input_{self.image_counter}.png")
                vutils.save_image(sample_images[i], f"{self.base_path}/image_results/generated_{self.image_counter}.png")
                vutils.save_image(original_images2[i], f"{self.base_path}/image_results/gt_{self.image_counter}.png")

                # frames[0].save(
                #     f"{self.base_path}/image_results/diffusion_process_{self.image_counter}.gif",
                #     save_all=True,
                #     append_images=frames[1:],
                #     duration=75)

    def train_base(self, use_canny=False):
        with open(f"{self.base_path}/epoch_losses.txt", "w") as file:
            for epoch in range(self.train_epochs):
                losses = []
                self.model.train()
                epoch_noise_loss = 0.0
                progress_bar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.train_epochs}]", unit="batch")
            
                for rgb, nir, inst in progress_bar:
                    rgb = rgb.to(self.device)
                    nir = nir.to(self.device)
                    inst = inst.to(self.device)

                    noise = torch.randn_like(nir)
                    timesteps = torch.randint(0, int(self.scheduler.config.num_train_timesteps), (nir.shape[0],), device=self.device)

                    noisy_x = self.scheduler.add_noise(nir, noise, timesteps)

                    self.optimizer.zero_grad()

                    with self.accelerator.accumulate(self.model):
                        pred_noise = self.model(noisy_x, timesteps, rgb, inst)
                        loss = self.loss_fn(pred_noise, noise)
                        losses.append(loss.item())
                        self.accelerator.backward(self.scaler.scale(loss))
                        
                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                        if self.use_ema:
                            self.ema.step_ema(self.ema_model, self.model)

                    self.lr_scheduler.step()

                    epoch_noise_loss += loss.item()
                    progress_bar.set_postfix({"Loss": loss.item()})

                avg_loss = sum(losses[-100:]) / 100 if len(losses) >= 100 else sum(losses) / len(losses)
                avg_epoch_loss = epoch_noise_loss / len(self.train_loader)

                file.write(f"Epoch {epoch+1}: Epoch Loss {avg_epoch_loss:.4f}, Avg Loss: {avg_loss:.4f}\n")
                file.flush()

                self.train_save_checkpoint_latest(epoch)

                if (epoch + 1) % 100 == 0:
                    sample_indices = random.sample(range(len(self.test_loader.dataset)), self.num_images_per_class)
                    sample_rgb_nir_pairs = [self.test_loader.dataset[i] for i in sample_indices]
                    self.train_generate_images(epoch, sample_rgb_nir_pairs, use_canny)
                    if self.use_ema:
                        self.train_generate_images(epoch, sample_rgb_nir_pairs, use_canny, use_ema=True)

                if (epoch + 1) % 500 == 0:
                    self.train_save_checkpoint(epoch)
                    if self.use_ema:
                        self.train_save_ema_checkpoint(epoch)

    def train_base_contour(self):
        with open(f"{self.base_path}/epoch_losses.txt", "w") as file:
            for epoch in range(self.train_epochs):
                losses = []
                self.model.train()
                epoch_noise_loss = 0.0
                progress_bar = tqdm(self.train_loader, desc=f"Epoch [{epoch+1}/{self.train_epochs}]", unit="batch")
            
                for _, nir in progress_bar:
                    nir = nir.to(self.device)

                    noise = torch.randn_like(nir)
                    timesteps = torch.randint(0, int(self.scheduler.config.num_train_timesteps), (nir.shape[0],), device=self.device)

                    noisy_x = self.scheduler.add_noise(nir, noise, timesteps)

                    self.optimizer.zero_grad()

                    with self.accelerator.accumulate(self.model):
                        pred_noise = self.model(noisy_x, timesteps, nir)
                        loss = self.loss_fn(pred_noise, noise)
                        losses.append(loss.item())
                        self.accelerator.backward(self.scaler.scale(loss))

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    self.lr_scheduler.step()
                    epoch_noise_loss += loss.item()
                    progress_bar.set_postfix({"Loss": loss.item()})

                avg_loss = sum(losses[-100:]) / 100 if len(losses) >= 100 else sum(losses) / len(losses)
                avg_epoch_loss = epoch_noise_loss / len(self.train_loader)

                file.write(f"Epoch {epoch+1}: Epoch Loss {avg_epoch_loss:.4f}, Avg Loss: {avg_loss:.4f}\n")
                file.flush()

                self.train_save_checkpoint_latest(epoch)

                if (epoch + 1) % 1 == 0:
                    sample_indices = random.sample(range(len(self.test_loader.dataset)), self.num_images_per_class)
                    sample_rgb_nir_pairs = [self.test_loader.dataset[i] for i in sample_indices]
                    self.train_generate_images(epoch, sample_rgb_nir_pairs, use_canny=True)
                    if self.use_ema:
                        self.train_generate_images(epoch, sample_rgb_nir_pairs, use_canny=True, use_ema=True)
                
                if (epoch + 1) % 50 == 0:
                    self.train_save_checkpoint(epoch)

    def train(self):
        if self.config['config_type'] == 'train' and self.config['model']['unet']['condition']['only_contour_condition'] == False:
            print('train base')
            self.train_base(self.config['model']['unet']['condition']['contour_control'])
            
        elif self.config['config_type'] == 'train' and self.config['model']['unet']['condition']['only_contour_condition'] == True:
            print('train base with only contour')
            self.train_base_contour()

    def test_base(self):
        self.model.eval()
        print("Sampling")
                
        total_processed = 0
        for batch_idx, (rgb, nir, inst) in enumerate(tqdm(self.test_loader, desc="Testing", unit="batch")):
            self.test_generate_images(batch_idx, rgb, nir, inst)
            total_processed += rgb.shape[0]

            print(f"Total images processed: {total_processed}")

        print("Testing completed.")

    def test(self):
        if self.config['config_type'] == 'test' and self.config['model']['unet']['condition']['only_contour_condition'] == False:
            print('test base')
            self.test_base()
        elif self.config['config_type'] == 'test' and self.config['model']['unet']['condition']['only_contour_condition'] == True:
            print('test base with only contour')
            self.test_base()

    def run(self):
        if self.config['config_type'] == 'train':
            self.train()
        elif self.config['config_type'] == 'test':
            self.test()
        else:
            raise ValueError(f"Unsupported config_type: {self.config['config_type']}")
