import os
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset


class slice_base(Dataset):
    def __init__(self,
                 data_root,
                 data_name,
                 data_repeat=1
                 ):

        self.data_root = data_root

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

        volume_data = data_npz[0, :, :, :]
        volume_data = (volume_data - np.min(data_npz)) / (np.max(data_npz)-np.min(data_npz))

        pos_id = np.random.randint(0, data_npz.shape[1])
        slice_data = volume_data[pos_id]
        slice_data = resize(slice_data, (512,512))
        slice_data = slice_data[:, :, None].astype(np.float32)

        example = {}
        example['name'] = self.list[i].split('/')[-1].split('.')[0]
        example['pos_id'] = pos_id / data_npz.shape[1]
        example['slice_data'] = slice_data * 2 - 1
 
        return example

class slice_train(slice_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class slice_val(slice_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def __len__(self):
        return 2 if super().__len__() // 10000 < 2 else super().__len__() // 10000

class slice_test(slice_base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    