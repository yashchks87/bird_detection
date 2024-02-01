import torch, torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys 
from dataset.utils import iou_width_height as iou
sys.path.append('../../scripts/')
from dataset.preprocess import helper_dataframe
from sklearn.model_selection import train_test_split


class BirdDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, scales : list, anchors : list, img_size: int = 416):
        self.dataframe = dataframe
        self.images = dataframe['path'].values.tolist()
        self.x_c = dataframe['x_c_updated'].values.tolist()
        self.y_c = dataframe['y_c_updated'].values.tolist()
        self.w = dataframe['w_updated'].values.tolist()
        self.h = dataframe['h_updated'].values.tolist()
        self.labels = dataframe['labels'].values.tolist()

        self.img_size = img_size
        self.scales = scales
        self.anchors = torch.Tensor(anchors[0] + anchors[1] + anchors[2])
        self.num_anchors = self.anchors.shape[0]
        self.num_of_anchors_per_scale = self.num_anchors // 3
        self.norm = torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # This project has each image has only 1 object in it. So we don't need to worry about multiple objects in an image.
        # Reading image and resize images
        img = torchvision.io.read_image(self.images[index])
        img = torchvision.transforms.functional.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        img = img.float()
        img = self.norm(img)

        # getting bounding box co-ordinates
        # Center x and y
        x_c = self.x_c[index] / self.img_size
        y_c = self.y_c[index] / self.img_size
        # Width and height
        w = self.w[index] / self.img_size
        h = self.h[index] / self.img_size
        # Creating target box holding tensors
        target_box = torch.tensor([x_c, y_c, w, h, self.labels[index]])
        # Why 6?
        # [probability weather object is present or not, x, y, w, h, class_prediction]
        targets = [torch.zeros(self.num_anchors // 3, scales, scales, 6) for scales in self.scales]
        iou_anchors = iou(target_box[2:4], self.anchors)
        anchors_indices = iou_anchors.argsort(descending=True, dim=0)
        x, y, w, h, class_label = target_box
        has_anchor = [False] * 3
        for anchor_idx in anchors_indices:
            scale_idx = anchor_idx // self.num_of_anchors_per_scale
            anchor_on_scale = anchor_idx % self.num_of_anchors_per_scale
            scales = self.scales[scale_idx]
            i, j = int(scales * y), int(scales * x)
            anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
            if not anchor_taken and not has_anchor[scale_idx]:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = scales * x - j, scales * y - i
                width_cell, height_cell = (
                    w * scales,
                    h * scales,
                ) # can be greater than 1 as it's relaticve to the cell size
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                has_anchor[scale_idx] = True
            elif not anchor_taken and iou_anchors[anchor_idx] > 0.5:
                targets[scale_idx][anchor_on_scale, i, j, 0] = -1
        return img, tuple(targets)

ANCHORS = [
    [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)],
    [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)],
    [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)],
]


class GetLoaders():
    def __init__(self, 
                 files_path='../../bird_dataset/CUB_200_2011/images.txt', 
                 box_paths='../../bird_dataset/CUB_200_2011/bounding_boxes.txt', 
                 labels_path='../../bird_dataset/CUB_200_2011/image_class_labels.txt', 
                 target_image_size = 416, 
                 path_prefix='../../bird_dataset/CUB_200_2011/images/'):
        self.files_path = files_path
        self.box_paths = box_paths
        self.labels_path = labels_path
        self.path_prefix = path_prefix
        self.target_image_size = target_image_size

    def split_data(self, df, test_size=0.01, random_state=42):
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
        train_df, val_df = train_test_split(train_df, test_size=test_size, random_state=random_state)
        return train_df, val_df, test_df

    def split_datasets(self, test_size=0.01, random_state=42):
        df = helper_dataframe(self.files_path, self.box_paths, self.labels_path, self.target_image_size, self.path_prefix)
        self.df = df
        self.train_df, self.val_df, self.test_df = self.split_data(df, test_size=test_size, random_state=random_state)
    
    def create_torch_datasets(self, scales = [13, 26, 52], anchors = ANCHORS):
        self.train_dataset = BirdDataset(self.train_df, scales=scales, anchors=anchors, img_size=self.target_image_size)
        self.val_dataset = BirdDataset(self.val_df, scales=scales, anchors=anchors, img_size=self.target_image_size)
        self.test_dataset = BirdDataset(self.test_df, scales=scales, anchors=anchors, img_size=self.target_image_size)
    
    def create_dataloaders(self, batch_size = 16, train_shuffle = True, val_shuffle = False, test_shuffle = False, num_workers = 7):
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=train_shuffle, num_workers=num_workers)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=val_shuffle, num_workers=num_workers)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=test_shuffle, num_workers=num_workers)
    
    def get_object(self):
        return self
