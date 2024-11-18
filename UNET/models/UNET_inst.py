import sys
import os
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

#sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from UNET.models.unet_modules import *
from UNET.models.unet_core import *

def get_unet_model(config: Dict[str, Any]) -> nn.Module:
    model = UNET(config)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    return model

class EMA:
    def __init__(self, beta: float):
        self.beta = beta
        self.step = 0

    def update_model_average(self, ema_model: nn.Module, current_model: nn.Module):
        for current_params, ema_params in zip(current_model.parameters(), ema_model.parameters()):
            old_weight, up_weight = ema_params.data, current_params.data
            ema_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model: nn.Module, model: nn.Module, step_start_ema: int = 2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model: nn.Module, model: nn.Module):
        ema_model.load_state_dict(model.state_dict())

class UNET(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.validate_config()
        self.init_components()
        self.set_forward_method()

    def validate_config(self):
        required_keys = ['model', 'config_type']
        for key in required_keys:
            if key not in self.config:
                raise KeyError(f"Config is missing required key: {key}")

    def init_components(self):
        self.init_basic_components()
        self.init_conditions()
        self.init_convolutions()
        self._adjust_decoder_block_args()
        self.unet = UnetCore(self.config)  # UnetCore

    def init_basic_components(self):
        self.init_conv = nn.Conv2d(*self.config['model']['unet']['init_conv'])
        self.time_projection = UnetTimeProjection(self.config)
        self.time_embedding = UnetTimeEmbedding(self.config)

    def init_conditions(self):
        model_config = self.config['model']['unet']
        self.skip_connection_type = model_config['skip_connection_type']
        self.config_type = self.config['config_type']
        self.num_classes = model_config['condition']['num_classes']
        self.d_time = model_config['time_embedding']['d_time']
        self.use_inst = model_config['use_inst']
        self.inst_num = model_config['inst_num']
        self.class_cond = model_config['condition']['class_condition']
        self.class_condition_type = model_config['condition']['class_condition_type']
        self.image_cond = model_config['condition']['image_condition']
        self.only_contour_cond = model_config['condition']['only_contour_condition']

        if self.class_cond:
            self.class_embedding = nn.Embedding(self.num_classes, self.d_time)

    def init_convolutions(self):
        if self.image_cond:
            self.init_image_condition_convs()
        if self.only_contour_cond:
            self.init_contour_condition_convs()

    def init_image_condition_convs(self):
        model_config = self.config['model']['unet']
        self.cond_image_input_channel = 3
        self.cond_image_target_channel = self.get_first_encoder_channel(model_config['encoder_block_args'])
        
        self.cond_image_input_conv = nn.Conv2d(self.cond_image_input_channel, self.cond_image_target_channel, kernel_size=1, stride=1, padding=0)
        concat_channels = self.cond_image_target_channel * 2 + (self.inst_num + 3 if self.use_inst else 0) # rgb 3 add
        self.cond_image_concat_conv = nn.Conv2d(concat_channels, self.cond_image_target_channel, kernel_size=3, stride=1, padding=1)

    def init_contour_condition_convs(self):
        import kornia
        self.only_contour_extractor = kornia.filters.Canny()
        self.cond_contour_target_channel = self.get_first_encoder_channel(self.config['model']['unet']['encoder_block_args'])
        self.cond_concat_conv = nn.Conv2d(self.cond_contour_target_channel + 1, self.cond_contour_target_channel, kernel_size=1, stride=1, padding=0)

    def get_first_encoder_channel(self, encoder_block_args: List[Any]) -> int:
        first_arg = encoder_block_args[0][0]
        return first_arg[0] if isinstance(first_arg, list) else first_arg

    def set_forward_method(self):
        if self.class_cond and self.class_condition_type == 'add':
            self.forward = self.forward_with_class_condition
        elif not self.image_cond and self.class_cond and self.class_condition_type == 'concat':
            self._modify_config_for_class_condition_2()
            self.forward = self.forward_with_class_condition_2
        elif self.image_cond and not self.only_contour_cond:
            self.forward = self.forward_with_image_condition
        elif not self.image_cond and self.only_contour_cond:
            self.forward = self.forward_with_only_contour_condition
        elif self.image_cond and self.class_cond:
            self.forward = self.forward_with_image_class_condition
        else:
            self.forward = self.forward_without_condition

    def _modify_config_for_class_condition_2(self):
        first_encoder_arg = self.config['model']['unet']['encoder_block_args'][0][0]
        if isinstance(first_encoder_arg, list):
            first_encoder_arg[0] += self.d_time
        else:
            self.config['model']['unet']['encoder_block_args'][0][0] += self.d_time
        if hasattr(self, 'unet'):
            self.unet.update_config(self.config)

    def _hot_encode_inst(self, inst_map: torch.Tensor) -> torch.Tensor:
        device = inst_map.device
        size = inst_map.size()
        oneHot_size = (size[0], self.inst_num, size[2], size[3])
        input_inst = torch.FloatTensor(torch.Size(oneHot_size)).zero_().to(device)
        inst_map = torch.clamp(inst_map, 0, self.inst_num - 1)
        input_inst = input_inst.scatter_(1, inst_map.data.long().to(device), 1.0)
        return input_inst

    def _concat_inst_rgb(self, input_inst, image_cond):
        input_inst = self.prepare_image_condition(input_inst, image_cond.shape[2:])
        concat_inst = torch.concat([input_inst, image_cond], dim=1)
        return concat_inst

    def _get_output_channels(self, args):
        if isinstance(args[-1], list):
            if len(args[-1]) == 1:  # if down or up sample
                return args[-1][-1] 
            else:
                return args[-1][1]
        else:
            return args[-1]

    def _adjust_decoder_block_args(self):
        if self.skip_connection_type == 'concat' and self.config_type == 'train':
            print('adjust decoder block args')
            encoder_output_channels = [self._get_output_channels(args) for args in self.config['model']['unet']['encoder_block_args']]
            
            decoder_block_types = self.config['model']['unet']['decoder_block_types']
            decoder_block_args = self.config['model']['unet']['decoder_block_args']

            for i in range(len(decoder_block_args)):
                if (decoder_block_types[i] != 'upsample'):
                    block_args = decoder_block_args[i]
                    skip_channels = encoder_output_channels[-i-1]

                    if isinstance(block_args[0], list):
                        block_args[0][0] += skip_channels
                    else:
                        block_args[0] += skip_channels
            
            self.config['model']['unet']['decoder_block_args'] = decoder_block_args
        else:
            return
        
    def prepare_class_condition(self, class_cond: torch.Tensor) -> torch.Tensor:
        if class_cond.dtype != torch.float:
            class_cond = F.one_hot(class_cond, num_classes=self.num_classes).float()
        return class_cond.view(-1, self.num_classes)

    def prepare_image_condition(self, image_cond: torch.Tensor, target_size: torch.Size) -> torch.Tensor:
        if image_cond.shape[2:] != target_size:
            return F.interpolate(image_cond, size=target_size, mode='nearest')
        return image_cond

    def forward_with_class_condition(self, latent: torch.Tensor, time: torch.Tensor, class_cond: torch.Tensor) -> torch.Tensor:
        latent = self.init_conv(latent)
        projected_time = self.time_projection(time)
        
        class_cond = self.prepare_class_condition(class_cond)
        embedded_class = torch.matmul(class_cond, self.class_embedding.weight)
        
        projected_time += embedded_class.view(-1, self.d_time)
        embedded_time = self.time_embedding(projected_time)

        return self.unet(latent, embedded_time)

    def forward_with_class_condition_2(self, latent: torch.Tensor, time: torch.Tensor, class_cond: torch.Tensor) -> torch.Tensor:
        latent = self.init_conv(latent)
        projected_time = self.time_projection(time)
        embedded_time = self.time_embedding(projected_time)

        batch, _, height, width = latent.shape
        class_cond = self.class_embedding(class_cond)
        class_cond = class_cond.view(batch, class_cond.shape[1], 1, 1).expand(batch, class_cond.shape[1], height, width)

        latent = torch.cat((latent, class_cond), dim=1)

        return self.unet(latent, embedded_time)

    def forward_with_image_class_condition(self, latent: torch.Tensor, time: torch.Tensor, image_cond: torch.Tensor, class_cond: torch.Tensor) -> torch.Tensor:
        latent = self.init_conv(latent)
        projected_time = self.time_projection(time)

        class_cond = self.prepare_class_condition(class_cond)
        embedded_class = torch.matmul(class_cond, self.class_embedding.weight)
        projected_time += embedded_class.view(-1, self.d_time)
        embedded_time = self.time_embedding(projected_time)

        image_cond = self.prepare_image_condition(image_cond, latent.shape[2:])
        embedded_image_cond = self.cond_image_input_conv(image_cond)

        latent = self.cond_image_concat_conv(torch.cat((latent, embedded_image_cond), dim=1))

        return self.unet(latent, embedded_time, image_cond, class_cond)

    def forward_with_image_condition(self, latent: torch.Tensor, time: torch.Tensor, image_cond: torch.Tensor, inst_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        latent = self.init_conv(latent)
        projected_time = self.time_projection(time)
        embedded_time = self.time_embedding(projected_time)

        image_cond = self.prepare_image_condition(image_cond, latent.shape[2:])
        embedded_image_cond = self.cond_image_input_conv(image_cond)

        concat_image = torch.cat((latent, embedded_image_cond), dim=1)

        if self.use_inst and inst_map is not None:
            input_inst = self._hot_encode_inst(inst_map)
            concat_inst = self._concat_inst_rgb(input_inst, image_cond)
        
            # input_inst_dummy = torch.zeros_like(input_inst)
            # concat_inst = self._concat_inst_rgb(input_inst_dummy, image_cond)

            concat_image = torch.cat((concat_image, concat_inst), dim=1)
            #concat_image = self.concat_image_conv(concat_image)

        latent = self.cond_image_concat_conv(concat_image)

        return self.unet(latent, embedded_time, image_cond) # image cond or inst cond

    def forward_with_only_contour_condition(self, latent: torch.Tensor, time: torch.Tensor, image_cond: torch.Tensor) -> torch.Tensor:
        latent = self.init_conv(latent)
        projected_time = self.time_projection(time)
        embedded_time = self.time_embedding(projected_time)

        _, contour = self.only_contour_extractor(image_cond)
        contour = self.prepare_image_condition(contour, latent.shape[2:])

        latent = torch.cat((latent, contour), dim=1)
        latent = self.cond_concat_conv(latent)
        return self.unet(latent, embedded_time, contour)

    def forward_without_condition(self, latent: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        latent = self.init_conv(latent)
        projected_time = self.time_projection(time)
        embedded_time = self.time_embedding(projected_time)
        return self.unet(latent, embedded_time)
