import os
import sys
import math
import torch.fft as fft

# sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.common_modules import *

import torch
import torch.nn as nn
import torch.nn.functional as F

def Fourier_filter(x, threshold, scale):
    # FFT
    x_freq = fft.fftn(x, dim=(-2, -1))
    x_freq = fft.fftshift(x_freq, dim=(-2, -1))
    
    B, C, H, W = x_freq.shape
    mask = torch.ones((B, C, H, W)).cuda() 

    crow, ccol = H // 2, W //2
    mask[..., crow - threshold:crow + threshold, ccol - threshold:ccol + threshold] = scale
    x_freq = x_freq * mask

    # IFFT
    x_freq = fft.ifftshift(x_freq, dim=(-2, -1))
    x_filtered = fft.ifftn(x_freq, dim=(-2, -1)).real
    
    return x_filtered
    
def get_norm_layer(norm_type, in_ch, num_groups=None):
    if norm_type == 'instance':
        return nn.InstanceNorm2d(in_ch)
    elif norm_type == 'batch':
        return nn.BatchNorm2d(in_ch)
    elif norm_type == 'group':
        return nn.GroupNorm(num_groups, in_ch)
    else:
        raise ValueError(f"Unsupported normalization type: {norm_type}")

class UnetTimeProjection(nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        """
        Convert time steps tensor into embedded vectors using the sinusoidal embedding formula.
        Embedded before entering the Unet.

        Config:
        d_time (int): dimension of time embedding
        """
        d_time = config['model']['unet']['time_embedding']['d_time']
        assert d_time % 2 == 0, "Time embedding dimension must be divisible by 2 (cos, sin)"
        self.d_time = d_time

    def forward(self, 
                time_steps: torch.Tensor):
        """
        Args:
        time_steps (torch.randint): refers to t in x_t

        Returns:
        projected_time (torch.tensor): (batch, d_time)
        """
        device = time_steps.device
        half_dim = self.d_time // 2
        
        # Log space
        log_timescale_increment = math.log(10000) / (half_dim - 1)
        inv_timescales = torch.exp(
            torch.arange(half_dim, device=device) * -log_timescale_increment
        )
        
        # Cosine and sine embeddings
        projected_time = time_steps[:, None] * inv_timescales[None, :]
        projected_time = torch.cat(
            [torch.sin(projected_time), torch.cos(projected_time)], dim=-1
        )
        return projected_time
    
class UnetTimeEmbedding(nn.Module):
    def __init__(self,
                 config):
        super().__init__()
        '''
        Convert emmbedded time steps into latent vectors for putting it into the unet network .

        Args:
         d_time (int): dimension of time embedding
        '''
        self.d_time = config['model']['unet']['time_embedding']['d_time']

        self.linear_1 = nn.Linear(self.d_time, self.d_time * 4)
        self.act = nn.SiLU() # silu to gelu? (using gelu)
        self.linear_2 = nn.Linear(self.d_time * 4, self.d_time * 4)

    def forward(self, 
                projected_time: torch.Tensor):
        '''
        Args:
         projected_time: (batch, d_time)

        return:
         embedded_time: (batch, 4 * d_time)
        '''
        # projected_time: (batch, d_time) -> (batch, d_time * 4)
        embedded_time = self.linear_1(projected_time)
        embedded_time = self.act(embedded_time) 
        embedded_time = self.linear_2(embedded_time)

        #  return: (batch, d_time * 4)
        return embedded_time
    
class Residual(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 num_groups: int,
                 norm_type: str):
        super().__init__()

        self.proj = WeightStandardizedConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.norm = get_norm_layer(norm_type, out_ch, num_groups)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x
    
class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 d_embedded_time: int, 
                 num_groups: int,
                 norm_type: str):
        super().__init__()

        self.residual_time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_embedded_time, 2 * out_ch)
        )

        self.block1 = Residual(in_ch, out_ch, num_groups, norm_type)
        self.block2 = Residual(out_ch, out_ch, num_groups, norm_type)

        if in_ch == out_ch:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self,
                latent: torch.Tensor, 
                embedded_time: torch.Tensor):
        
        # Time projection
        projected_latent_embedded_time = self.residual_time_projection(embedded_time)

        # Add width and height dimension to time
        projected_latent_embedded_time = projected_latent_embedded_time[..., None, None]

        # Split projected time embedding into scale and shift
        scale_shift = projected_latent_embedded_time.chunk(2, dim=1)

        h = self.block1(latent, scale_shift)
        h = self.block2(h)
        
        return h + self.residual_layer(latent)

class SDResidualBlock(nn.Module):
    def __init__(self, 
                 in_ch: int, 
                 out_ch: int, 
                 d_embedded_time: int, 
                 num_groups: int,
                 norm_type: str):
        super().__init__()
        """
        UNet residual block

        Args:
        in_ch (int): channels of input images
        out_ch (int): channels of output images
        d_embedded_time (int): d_time * 4 (Change to 2 * out_ch to get scale and shift)
        num_groups (int): the number of groups
        """
        self.time_projection = nn.Linear(d_embedded_time, 2 * out_ch)

        self.norm = get_norm_layer(norm_type, in_ch, num_groups)
        self.conv_feature = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

        self.norm_merged = get_norm_layer(norm_type, out_ch, num_groups)
        self.conv_merged = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1)

        if in_ch == out_ch:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self,
                latent: torch.Tensor, 
                embedded_time: torch.Tensor):
        """
        Args:
        latent: (batch, in_ch, height, width)
        embedded_time: (batch, d_embedded_time)

        Returns:
        merged (skip connection): (batch, out_ch, height, width)
        """
        feature = self.norm(latent)
        feature = F.silu(feature)
        feature = self.conv_feature(feature)

        embedded_time = F.silu(embedded_time)

        # Time projection
        projected_latent_embedded_time = self.time_projection(embedded_time)

        # Add width and height dimension to time
        projected_latent_embedded_time = projected_latent_embedded_time[..., None, None]

        # Split projected time embedding into scale and shift
        scale_shift = projected_latent_embedded_time.chunk(2, dim=1)
        scale, shift = scale_shift

        # Apply scale and shift
        merged = feature * (scale + 1) + shift

        merged = self.norm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)

        return merged + self.residual_layer(latent)

class CrossAttentionBlock(nn.Module):
    def __init__(self, 
                 d_embed: int, 
                 n_heads: int, 
                 d_condition: int, 
                 num_groups: int,
                 num_condition_groups: int,
                 norm_type: str):
        super().__init__()
        '''
        Unet attention block
        in_ch = n_heads * d_embed 

         Args:
          n_heads (int): the number of heads
          d_embed (int): dimension of attention 
          d_condition (int): dimension of condition 
          num_groups (int): the number of groups
        '''
        if d_condition % n_heads != 0:
            self.cond_change = True
            self.condition_conv = nn.Conv2d(d_condition, d_condition * n_heads, 1, 1, 0)
            self.num_condition_groups = d_condition
        else:
            self.cond_change = False
            self.num_condition_groups = num_condition_groups

        self.in_ch = n_heads * d_embed
        self.in_ch_c = d_condition * n_heads

        self.group_norm = get_norm_layer(norm_type, self.in_ch, num_groups)
        self.conv_input = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=1)

        self.group_norm_c = get_norm_layer(norm_type, self.in_ch_c, self.num_condition_groups)
        self.conv_input_c = nn.Conv2d(self.in_ch_c, self.in_ch_c, kernel_size=3, padding=1)

        #self.layer_norm_1 = nn.LayerNorm(self.in_ch)
        self.attention_1 = MultiHeadCrossAttention(self.in_ch, n_heads, d_condition, in_proj_bias=False)
        
        self.linear_geglu_1 = nn.Linear(self.in_ch, self.in_ch * 4)
        self.linear_geglu_2 = nn.Linear(self.in_ch * 4, self.in_ch)

        self.conv_output = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=1)

    def forward(self,
                x: torch.Tensor, 
                condition: torch.Tensor):
        '''
        Args:
         x: (batch, in_ch, height, width)
         condition: (batch, in_ch, height, width)

        return:
         x + residue: (batch, in_ch, height, width)
        '''

        if self.cond_change == True:
            condition = self.condition_conv(condition)

        residue_long = x

        # x: (batch, in_ch, height, width) -> (batch, in_ch, height, width)
        x = self.conv_input(x)
        x = self.group_norm(x)

        # condition: (batch, in_ch_c, height, width) -> (batch, in_ch_c, height, width)
        condition = self.conv_input_c(condition)
        condition = self.group_norm_c(condition)
        
        batch, in_ch, height, width = x.size()
        batch_c, in_ch_c, height_c, width_c = condition.size()

        # x: (batch, in_ch, height, width) -> (batch, height*width, in_ch)
        x = x.view(batch, in_ch, height*width).contiguous()
        x = x.transpose(-1, -2).contiguous()

        # condition: (batch, in_ch, height, width) -> (batch, in_ch, height, width)
        condition = condition.view(batch_c, in_ch_c, height_c*width_c).contiguous()
        condition = condition.transpose(-1, -2).contiguous()

        # Normalization + self attention and skip connection
        residue_short = x

        # x: (batch, seq_len, in_ch) -> (batch, seq_len, in_ch)
        #x = self.layer_norm_1(x)
        x = self.attention_1(x, condition)
        x += residue_short

        # Normalization + FFN with GeGLU and skip connection
        x = self.linear_geglu_1(x)
        x = self.linear_geglu_2(x)

        # x: (batch, seq_len, in_ch) -> (batch, in_ch, height, width)
        x = x.transpose(-1, -2).contiguous()
        x = x.view(batch, in_ch, height, width).contiguous()

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

class SDAttentionBlock(nn.Module):
    def __init__(self, 
                 d_embed: int, 
                 n_heads: int, 
                 d_condition: int, 
                 num_groups: int,
                 num_condition_groups: int,
                 norm_type: str):
        super().__init__()
        '''
        Unet attention block
        in_ch = n_heads * d_embed 

         Args:
          n_heads (int): the number of heads
          d_embed (int): dimension of attention 
          d_condition (int): dimension of condition 
          num_groups (int): the number of groups
        '''
        if d_condition % n_heads != 0:
            self.cond_change = True
            self.condition_conv = nn.Conv2d(d_condition, d_condition * n_heads, 1, 1, 0)
            self.num_condition_groups = d_condition
            self.in_ch_c = d_condition * n_heads
        else:
            self.cond_change = False
            self.num_condition_groups = num_condition_groups
            self.in_ch_c = d_condition

        self.in_ch = n_heads * d_embed
    
        self.group_norm = get_norm_layer(norm_type, self.in_ch, num_groups)
        self.conv_input = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=1)

        self.group_norm_c = get_norm_layer(norm_type, self.in_ch_c, self.num_condition_groups)
        self.conv_input_c = nn.Conv2d(self.in_ch_c, self.in_ch_c, kernel_size=3, padding=1)

        self.layer_norm_1 = nn.LayerNorm(self.in_ch)
        self.attention_1 = MultiHeadSelfAttention(self.in_ch, n_heads, in_proj_bias=False)
        self.layer_norm_2 = nn.LayerNorm(self.in_ch)
        self.attention_2 = MultiHeadCrossAttention(self.in_ch, n_heads, self.in_ch_c, in_proj_bias=False)
        
        self.layer_norm_3 = nn.LayerNorm(self.in_ch)

        self.linear_geglu_1 = nn.Linear(self.in_ch, 4 * self.in_ch * 2)
        self.linear_geglu_2 = nn.Linear(self.in_ch * 4, self.in_ch)

        self.conv_output = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=3, padding=1)

    def forward(self,
                x: torch.Tensor, 
                condition: torch.Tensor):
        '''
        Args:
         x: (batch, in_ch, height, width)
         condition: (batch, in_ch, height, width)

        return:
         x + residue: (batch, in_ch, height, width)
        '''

        if self.cond_change == True:
            condition = self.condition_conv(condition)

        residue_long = x

        # x: (batch, in_ch, height, width) -> (batch, in_ch, height, width)
        x = self.group_norm(x)
        x = self.conv_input(x)

        # condition: (batch, in_ch_c, height, width) -> (batch, in_ch_c, height, width)
        condition = self.group_norm_c(condition)
        condition = self.conv_input_c(condition)
        
        batch, in_ch, height, width = x.size()
        batch_c, in_ch_c, height_c, width_c = condition.size()

        # x: (batch, in_ch, height, width) -> (batch, height*width, in_ch)
        x = x.view(batch, in_ch, height*width).contiguous()
        x = x.transpose(-1, -2).contiguous()

        # condition: (batch, in_ch, height, width) -> (batch, in_ch, height, width)
        condition = condition.view(batch_c, in_ch_c, height_c*width_c).contiguous()
        condition = condition.transpose(-1, -2).contiguous()

        # Normalization + self attention and skip connection
        residue_short = x

        # x: (batch, seq_len, in_ch) -> (batch, seq_len, in_ch)
        x = self.layer_norm_1(x)
        x = self.attention_1(x)
        x += residue_short

        # Normalization + cross attention and skip connection
        residue_short = x

        # x: (batch, seq_len, in_ch) -> (batch, seq_len, in_ch)
        x = self.layer_norm_2(x)
        x = self.attention_2(x, condition)
        x += residue_short

        # Normalization + FFN with GeGLU and skip connection
        residue_short = x

        # x: (batch, seq_len, in_ch) -> (batch, seq_len, in_ch * 4), (batch, seq_len, in_ch * 4)
        x = self.layer_norm_3(x)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)

        # Element-wise product
        x = x * F.gelu(gate)

        # x: (batch, seq_len, in_ch * 4) -> (batch, seq_len, in_ch)
        x = self.linear_geglu_2(x)
        x += residue_short

        # x: (batch, seq_len, in_ch) -> (batch, in_ch, height, width)
        x = x.transpose(-1, -2).contiguous()
        x = x.view(batch, in_ch, height, width).contiguous()

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long
    
class UnetOutputLayer(nn.Module):
    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 num_groups: int):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups, in_ch)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, padding=0)

    def forward(self, x):
        '''
        Args:
         x: (batch, in_channels, height, width)

        return:
         x: (batch, out_channels, height, width)
        '''

        x = self.group_norm(x)
        x = self.act(x)
        x = self.conv(x)
        return x

class Switch(nn.Sequential):
    def forward(self, x, latent_embedded_time, condition = None):
        for layer in self:
            if isinstance(layer, (SDResidualBlock, ResidualBlock)):
                x = layer(x, latent_embedded_time)
            elif isinstance(layer, (SDAttentionBlock, CrossAttentionBlock)):
                x = layer(x, condition)
            else:
                x = layer(x)
        return x
