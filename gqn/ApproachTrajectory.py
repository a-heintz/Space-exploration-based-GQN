import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class ApproachTrajectory(Dataset):
    """
    :param root_dir: location of data on disc
    :param train: whether to use train of test set
    :param transform: transform on images
    :param fraction: fraction of dataset to use
    :param target_transform: transform on viewpoints
    """
    def __init__(self, root_dir, train=True, num_samples = 80000, transform=None, fraction=1.0, target_transform=None):
        super(ApproachTrajectory, self).__init__()
        assert fraction > 0.0 and fraction <= 1.0
        self.root_dir = root_dir
        
        #cur_dir = os.path.join(root_dir, "images")
        #cur_dir_fldr = os.listdir(cur_dir)
        
        #recs = []
        #for fldr in cur_dir_fldr[:]:
          #scene_fldr = os.path.join(cur_dir, fldr)
          #recs.append(scene_fldr)
        #self.records = recs
        #self.records = self.records[:int(len(self.records)*fraction)]
        
        orbit_dir_path = os.path.join(root_dir, "images")
        self.orbits_dir = os.listdir(orbit_dir_path)
        self.records = list(range(num_samples))
        
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        #scene_path = self.records[idx]
        #num_orbit = idx
        orbit_dir = np.random.choice(self.orbits_dir)
        scene_path = os.path.join(self.root_dir, "images", orbit_dir)
        num_orbit = int(orbit_dir[:-3])
        
        orbits_pos = np.load(os.path.join(self.root_dir, "state_files", 'orbits_positions.npy'))
        orbits_att = np.load(os.path.join(self.root_dir, "state_files",'orbits_attitudes.npy'))
        
        rand_image_indices = np.arange(40)
        np.random.shuffle(rand_image_indices)
        rand_image_indices = rand_image_indices[:16]
        
        img_list = []
        viewpoint_list = []
        for i in range(16):
            orbit_state_idx = rand_image_indices[i]
            
            image_name=str(rand_image_indices[i])
            while len(image_name) < 6:
                image_name= "0" + image_name
            img_path = os.path.join(scene_path, (image_name+".png"))
            
            img = Image.open(img_path)
            #img = img.resize((64,64))
            
            background = Image.new('RGB', img.size, (255,255,255))
            background.paste(img, mask=img.split()[3])  # 3 is the alpha channel
            img = background
            
            img = np.asarray(img).transpose(-1, 0, 1) # we have to change the dimensions from width x height x channel (WHC) to channel x width x height (CWH)
            img = img/255
            img = np.asarray(img)
            img_list = img_list + [img]
            
            viewpoint = np.zeros( 7 )
            viewpoint[:3] = orbits_pos[num_orbit, orbit_state_idx, :]
            viewpoint[3:] = orbits_att[num_orbit, orbit_state_idx, :]
            viewpoint_list = viewpoint_list + [viewpoint]
            
        img_list = np.asarray(img_list)
        viewpoint_list = np.asarray(viewpoint_list)
        
        images = torch.from_numpy(img_list)
        viewpoints = torch.from_numpy(viewpoint_list)

        return images.float(), viewpoints.float()
