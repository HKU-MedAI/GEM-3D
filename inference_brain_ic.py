import glob
import nibabel as nib
import numpy as np
import os
import torch

from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from ldm.util import instantiate_from_config

resume_path = './infer_model/brain'

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

config = OmegaConf.load('./configs/latent-diffusion/inference_brain.yaml')
data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()

save_path = 'ic'
save_path = os.path.join(logdir, save_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)

val_dataset = data.datasets['validation']
batch_size = 1
valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
val_num = len(val_dataset)
save_gt = True

for idx, data in tqdm(enumerate(valloader)):
    if idx >= val_num:
        break

    ref_id = np.random.randint(0, len(val_dataset))
    ref_data = val_dataset[ref_id]
    ref_data = torch.from_numpy(ref_data['volume_data']).unsqueeze(0)
    ref_slice_num = ref_data.shape[1]

    name = data['name']
    gt = data['volume_data']
    data_seg = data['volume_seg']
    slice_num = gt.shape[1]

    seg = rearrange(data_seg, 'b z h w c -> b z c h w')
    seg = seg.to(memory_format=torch.contiguous_format).float()
    seg = seg.to(model.device)

    start_slice = np.random.randint(0, min(ref_slice_num, slice_num))
    ref_slice = ref_data[:, int(start_slice/slice_num*ref_slice_num), :, :, :]
    ref_slice = rearrange(ref_slice, 'b h w c -> b c h w')
    ref_slice = ref_slice.repeat(1,3,1,1)
    ref_slice = ref_slice.to(memory_format=torch.contiguous_format).float()
    ref_slice = ref_slice.to(model.device)
    encoder_posterior = model.encode_first_stage(ref_slice)
    z_ref_0 = model.get_first_stage_encoding(encoder_posterior).detach()
    z_ref = z_ref_0.clone()

    result = torch.zeros((batch_size, slice_num, 4, 64, 64)).cuda()
    window_length = 16
    h = 1
    # sample in bi-directional with conditional diffusion model
    upper_iters = (slice_num-start_slice-h) // (window_length-h)+1 if (slice_num-start_slice-h)%(window_length-h) != 0 else (slice_num-start_slice-h) // (window_length-h)
    for i in range(upper_iters):
        if i == upper_iters-1:
            seg_i = seg[:, -window_length:]
        else:
            seg_i = seg[:, start_slice+i*window_length-i*h:start_slice+(i+1)*window_length-i*h]

        seg_i = rearrange(seg_i, 'b z c h w -> (b z) c h w')
        seg_i = seg_i.repeat(1,3,1,1)
        encoder_posterior = model.encode_first_stage(seg_i)
        seg_i = model.get_first_stage_encoding(encoder_posterior).detach()

        z_ref = z_ref.unsqueeze(1).repeat(1,window_length,1,1,1)
        z_ref = rearrange(z_ref, 'b z c h w -> (b z) c h w')

        c = torch.cat([seg_i, z_ref], dim=1)

        if i == 0:
            samples_i, _ = model.sample_log(cond=c, batch_size=z_ref.shape[0], ddim=True, eta=1., ddim_steps=200)
        else:
            samples_i, _ = model.sample_log(cond=c, batch_size=z_ref.shape[0], ddim=True, eta=1., ddim_steps=200, 
                                            previous=x_minus1)

        samples_i = rearrange(samples_i, '(b z) c h w -> b z c h w', z=window_length)
    
        if i == upper_iters-1:
            result[:, -window_length+h:] = samples_i[:,h:,...]
        else:
            if i == 0:
                result[:, start_slice:start_slice+window_length] = samples_i
            else:
                result[:, start_slice+i*window_length-i*h+h:start_slice+(i+1)*window_length-i*h] = samples_i[:, h:]
            z_ref = result[:,start_slice+(i+1)*window_length-i*h-h,...]
            x_minus1 = samples_i[:, -h:,...]

    z_ref = z_ref_0.clone()

    # sample the other direction, note the direction
    lower_iters = (start_slice-h) // (window_length-h)+1 if (start_slice-h)%(window_length-h) != 0 else (start_slice-h) // (window_length-h)
    for i in range(lower_iters):
        if i == lower_iters-1:
            seg_i = seg[:, :window_length]
        else:
            seg_i = seg[:, start_slice-(i+1)*window_length+i*h+1:start_slice-i*window_length+i*h+1]
        
        seg_i = rearrange(seg_i, 'b z c h w -> (b z) c h w')
        seg_i = seg_i.repeat(1,3,1,1)
        encoder_posterior = model.encode_first_stage(seg_i)
        seg_i = model.get_first_stage_encoding(encoder_posterior).detach()

        z_ref = z_ref.unsqueeze(1).repeat(1,window_length,1,1,1)
        z_ref = rearrange(z_ref, 'b z c h w -> (b z) c h w')

        c = torch.cat([seg_i, z_ref], dim=1)

        if i == 0:
            samples_i, _ = model.sample_log(cond=c, batch_size=z_ref.shape[0], ddim=True, eta=1., ddim_steps=200)
        else:
            samples_i, _ = model.sample_log(cond=c, batch_size=z_ref.shape[0], ddim=True, eta=1., ddim_steps=200, 
                                            previous=x_minus1, previous_reverse=True)

        samples_i = rearrange(samples_i, '(b z) c h w -> b z c h w', z=window_length)

        if i == lower_iters-1:
            result[:, :window_length-h] = samples_i[:, :window_length-h,...]
        else:
            if i == 0:
                result[:, start_slice-window_length+1:start_slice+1] = samples_i
            else:
                result[:, start_slice-(i+1)*window_length+i*h+1:start_slice-i*window_length+i*h-h+1] = samples_i[:, :-h]
            z_ref = result[:,start_slice-(i+1)*window_length+i*h+1,...]
            x_minus1 = samples_i[:, :h,...]

    result = rearrange(result, 'b z c h w -> (b z) c h w')
    x_result = torch.zeros((result.shape[0],3,512,512))
    
    dec_unit = 16
    num_dec_iter = slice_num // dec_unit + 1 if slice_num % dec_unit != 0 else slice_num // dec_unit
    for i in range(num_dec_iter):
        if i == num_dec_iter - 1:
            x_result[-dec_unit:] = model.decode_first_stage(result[-dec_unit:])
        x_result[i*dec_unit:(i+1)*dec_unit] = model.decode_first_stage(result[i*dec_unit:(i+1)*dec_unit])
    x_result[x_result>1.0] = 1.0
    x_result[x_result<-1.0] = -1.0
    x_result = (x_result+1)/2
    x_result = rearrange(x_result, '(b z) c h w -> b z c h w', z=slice_num)
    x_result = x_result[0,:,0,...].detach().cpu().numpy()

    x_result = x_result.transpose(1,2,0)
    x_result = np.rot90(x_result, k=1, axes=(0,1))
    data_path = os.path.join(save_path, str(f'{name[0]}_output.nii'))
    data_nii = nib.Nifti1Image(x_result, np.identity(4))
    nib.save(data_nii, data_path)

    if save_gt:
        gt = (gt+1)/2
        gt = gt[0,:,:,:,0].detach().cpu().numpy()
        gt = gt.transpose(1,2,0)
        gt = np.rot90(gt, k=1, axes=(0,1))
        data_path = os.path.join(save_path, str(f'{name[0]}_gt.nii'))
        data_nii = nib.Nifti1Image(gt, np.identity(4))
        nib.save(data_nii, data_path)

        data_seg = (data_seg+1)/2 * 3
        data_seg = data_seg[0,:,:,:,0].detach().cpu().numpy()
        data_seg = data_seg.transpose(1,2,0)
        data_seg = np.rot90(data_seg, k=1, axes=(0,1))
        data_path = os.path.join(save_path, str(f'{name[0]}_seg.nii'))
        data_nii = nib.Nifti1Image(data_seg, np.identity(4))
        nib.save(data_nii, data_path)
