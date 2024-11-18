import sys
import os
import numpy as np

from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.optimization import get_scheduler

# sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(project_root)

from common.data_utils_inst import *
from UNET.models.UNET_inst import *
from Loader.loader_inst import *
from accelerate import Accelerator, DistributedDataParallelKwargs

config_path = '/home/hj/incheol/inst_diffusion/UNET/old2/train_20241004_120833_fmb_b/train_config.yaml'
config = load_config(config_path)
epochs = config['loader']['train']['epoch']
loaders = build_loader_inst(config)

print('train_loader len:', len(loaders['train']))
print('test_loader len:', len(loaders['test']))

# Acclereator
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
#device = accelerator.device
device = 'cuda:0'
torch.cuda.set_device(device) # change allocation of current GPU

model = UNET(config).to(device)
model = load_safetensor(config, model)
safetensor_path = '/home/hj/incheol/inst_diffusion/UNET/trainer/train_20241018_002654/checkpoints/latest/latest_checkpoint.safetensors'
state_dict = load_file(safetensor_path)
model.load_state_dict(state_dict)
print(model)

optimizer = create_optimizer(config, model.parameters())
scheduler = create_time_scheduler(config)

lr_scheduler = get_scheduler(
    'cosine',
    optimizer=optimizer,
    num_warmup_steps=len(loaders['train']) * 3,
    num_training_steps=len(loaders['train']) * epochs,
)

model, optimizer, scheduler, lr_scheduler, train_loader, test_loader = accelerator.prepare(model, optimizer, scheduler, lr_scheduler, loaders['train'], loaders['test'])

loader = Loader(config, model, scheduler, accelerator, optimizer, lr_scheduler, train_loader, test_loader, device)
loader.run()
