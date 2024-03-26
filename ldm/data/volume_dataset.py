import os
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset


class volume_base(Dataset):
    def __init__(self,
                 data_root,
                 data_name,
                 sample=True,
                 sample_length=16,
                 data_repeat=1
                 ):

        self.data_root = data_root
        self.data_name = data_name
        self.sample = sample
        self.sample_length = sample_length

        data_root = os.path.join(data_root, data_name + '/nnUNetPlans_3d_fullres')
        self.list = [os.path.join(data_name + '/nnUNetPlans_3d_fullres',f) for f in os.listdir(data_root) if f.endswith('.npz')]
        self._length = len(self.list)
        self._data_repeat = data_repeat

    def __len__(self):
        return self._length * self._data_repeat

    def __getitem__(self, i):
        i = i % self._length
        
        npz_file = os.path.join(self.data_root, self.list[i])
        data_npz = np.load(npz_file)['data']
        seg_npz = np.load(npz_file)['seg']

        volume_data = data_npz[0, :, :, :]
        volume_seg = seg_npz[0, :, :, :]

        volume_data = (volume_data - np.min(data_npz)) / (np.max(data_npz)-np.min(data_npz))
        if 'Abdomen' in self.data_name:
            max_seg = 4
        elif 'Brain' in self.data_name:
            max_seg = 3
        volume_seg = volume_seg / max_seg

        volume_data = resize(volume_data, (volume_data.shape[0],512,512))
        volume_seg = resize(volume_seg, (volume_seg.shape[0],512,512), order=0, anti_aliasing=False)

        volume_data = volume_data[:, :, :, None].astype(np.float32)
        volume_seg = volume_seg[:, :, :, None].astype(np.float32)

        if self.sample:
            start_id = np.random.randint(0, data_npz.shape[1]-self.sample_length)
            volume_data = volume_data[start_id:start_id+self.sample_length]
            volume_seg = volume_seg[start_id:start_id+self.sample_length]

        rand_idx = np.random.randint(0, volume_data.shape[0])
        volume_ref = volume_data[rand_idx:rand_idx+1]

        example = {}
        example['name'] = self.list[i].split('/')[-1].split('.')[0]
        example['volume_data'] = volume_data * 2 - 1
        example['volume_seg'] = volume_seg * 2 - 1
        example['volume_ref'] = volume_ref * 2 - 1
 
        return example

class volume_train(volume_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class volume_val(volume_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __len__(self):
        return 2 if super().__len__() // 10000 < 2 else super().__len__() // 10000

class volume_test(volume_base):
    def __init__(self, **kwargs):
        super().__init__(sample=False, **kwargs)
    