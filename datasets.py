from imports import (osp, Dataset, joblib, np, glob)


class BALSER(Dataset):
    """
        Python interface for the Balser (Edixia Automation) custom dataset.
        Can be downloaded here : *_Si jamais on met en ligne la base de données*.
    """
    def __init__(self, data_root, training=True, transform=None):
        self.data_root = data_root
        self.training = training
        self.transform = transform
        #
        if self.training:
            de_file = osp.join(self.data_root, 'training_range.pkl')
            assert osp.isfile(de_file)
            intensity_file = osp.join(self.data_root, 'training_intensity.pkl')
            assert osp.isfile(intensity_file)
        else:
            de_file = osp.join(self.data_root, 'testing_range.pkl')
            assert osp.isfile(de_file)
            intensity_file = osp.join(self.data_root, 'testing_intensity.pkl')
            assert osp.isfile(intensity_file)
        #
        self.de = joblib.load(de_file)
        self.intensity = joblib.load(intensity_file)

    def __len__(self):
        return len(self.de)

    def __getitem__(self, index):
        intensity = self.intensity[index]
        intensity = np.expand_dims(intensity, axis=2)
        intensity = np.broadcast_to(intensity, (intensity.shape[0], intensity.shape[1], 3))
        de = self.de[index]
        mask = np.ones_like(de)
        mask[de == 0] = 0
        sample = {'image': intensity, 'depth': de, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample
##


class NYUV2(Dataset):
    """
        Python interface to serialized NYVU2 dataset, script for serialization can be found here : - Mettre lien repo
        The NYU V2 dataset can be downloaded at : https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html.
    """
    def __init__(self, data_root, training=True, transform=None):
        # assert scene_type in _VALID_SCENE_TYPES
        self.data_root = data_root
        self.training = training
        self.transform = transform
        #
        if self.training:
            de_file = osp.join(self.data_root, 'training_range.pkl')
            assert osp.isfile(de_file)
            intensity_file = osp.join(self.data_root, 'training_intensity.pkl')
            assert osp.isfile(intensity_file)
        else:
            de_file = osp.join(self.data_root, 'testing_range.pkl')
            assert osp.isfile(de_file)
            intensity_file = osp.join(self.data_root, 'testing_intensity.pkl')
            assert osp.isfile(intensity_file)

    def __len__(self):
        return len(self.de)

    def __getitem__(self, index):
        intensity = self.intensity[index]
        de = self.de[index]
        mask = np.ones_like(de)
        mask[de == 0] = 0
        sample = {'image': intensity, 'depth': de, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample
##


class DIODE(Dataset):
    """
        Python interface Serialized version of DIODE Dataset
    """

    def __init__(self, data_root, training=True, transform=None):
        self.data_root = data_root
        self.training = training
        self.transform = transform
        #
        self.de = []
        self.rgb = []
        self.mask = []
        #
        if self.training:
            directory = osp.join(self.data_root, 'Train')
        else:
            directory = osp.join(self.data_root, 'Val')
        print(directory)
        assert osp.isdir(directory)
        #
        rgb_dir = osp.join(directory, 'RGB')
        depth_dir = osp.join(directory, 'Depth')
        mask_dir = osp.join(directory, 'Mask')
        #
        rgb_list = glob.glob(osp.join(rgb_dir, '*'))
        print(len(rgb_list))
        depth_list = glob.glob(osp.join(depth_dir, '*'))
        print(len(depth_list))
        mask_list = glob.glob(osp.join(mask_dir, '*'))
        print(len(mask_list))
        print("Loading elements")
        for de_fname in depth_list:
            de_fpath = osp.join(depth_dir, de_fname)
            entry = joblib.load(de_fpath, 'r+')
            self.de.extend(entry)
        for rgb_fname in rgb_list:
            rgb_fpath = osp.join(rgb_dir, rgb_fname)
            entry = joblib.load(rgb_fpath, 'r+')
            self.rgb.extend(entry)
        for mask_fname in mask_list:
            mask_fpath = osp.join(mask_dir, mask_fname)
            entry = joblib.load(mask_fpath, 'r+')
            self.mask.extend(entry)

    def __len__(self):
        return len(self.de)

    def __getitem__(self, index):
        rgb = self.rgb[index]
        de = self.de[index]
        mask = self.mask[index]
        sample = {'image': rgb, 'depth': de, 'mask': mask}
        if self.transform:
            sample = self.transform(sample)
        return sample
