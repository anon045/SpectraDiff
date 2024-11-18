import sys
import os
import numpy as np

from safetensors.torch import load_file

#sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.data_utils_inst import *
from UNET.models.UNET_inst import *
from Loader.loader_inst_v2 import *
from accelerate import Accelerator, DistributedDataParallelKwargs

config_path = './flir_sample/flir_sample_config.yaml'
config = load_config(config_path)
loaders = build_loader_inst(config)

print('test_loader len:', len(loaders['test']))

# Acclereator
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
#device = accelerator.device
device = 'cuda:0'
torch.cuda.set_device(device) # change allocation of current GPU

model = UNET(config).to(device)
safetensor_path = './flir_sample/FLIR.safetensors'
state_dict = load_file(safetensor_path)
model.load_state_dict(state_dict)
print(model)

optimizer = create_optimizer(config, model.parameters())
scheduler = create_time_scheduler(config)

model, scheduler, test_loader = accelerator.prepare(model, scheduler, loaders['test'])

loader = Loader(config, model, scheduler, accelerator, test_loader=test_loader, device=device)
loader.run()
