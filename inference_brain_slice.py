import glob
import numpy as np
import os
import torch

from omegaconf import OmegaConf
from tqdm import tqdm

from ldm.util import instantiate_from_config
import matplotlib.pyplot as plt

resume_path = './infer_model/brain_slice'

paths = resume_path.split("/")
idx = len(paths)-paths[::-1].index("infer_model")+1
logdir = "/".join(paths[:idx])

logdir = resume_path.rstrip("/")
ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")

base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))[1]
configs = OmegaConf.load(base_configs)
model = instantiate_from_config(configs.model)
model.init_from_ckpt(ckpt)
model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device", device)
model = model.to(device)

save_path = 'slice_generation'
save_path = os.path.join(logdir, save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

save_num = 10
for i in tqdm(range(save_num)):
    pos_id = np.random.uniform()
    pos_id = torch.Tensor([pos_id]).to(model.device)

    c = model.get_learned_conditioning(pos_id)

    samples, _ = model.sample_log(cond=c, batch_size=1, ddim=True, eta=1., ddim_steps=200)

    res = model.decode_first_stage(samples)
    res[res>1.0] = 1.0
    res[res<-1.0] = -1.0
    res = (res+1)/2
    res = res[0,0,...].detach().cpu().numpy()
    data_path = os.path.join(save_path, str(f'{pos_id.item():.2f}_gen.png'))
    plt.imsave(data_path, res, cmap='gray')
