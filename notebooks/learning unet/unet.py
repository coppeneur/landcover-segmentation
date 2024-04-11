import matplotlib.pyplot as plt
import numpy as np
import random
from functools import reduce
import itertools
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from collections import defaultdict
import torch.nn.functional as F
import torch
import torch.optim as optim
import albumentations as album
from torch.optim import lr_scheduler
import time
import copy
from torchvision import models
from torchsummary import summary


# def reverse_transform(inp):
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     inp = (inp * 255).astype(np.uint8)
#
#     return inp


class SatelliteDataset(Dataset):
    def __init__(self, df, transform=None, preprocessing=None):
        self.image_paths = df['sat_image_path'].tolist()
        self.mask_paths = df['mask_path'].tolist()
        self.transform = transform
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):

        # read images
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_filepath = self.mask_paths[idx]
        mask = cv2.imread(mask_filepath)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        # one-hot-encode the mask
        # mask = one_hot_encode(mask, self.class_rgb_values).astype('float')

        # if self.transform:
        #     image = self.transform(image)

        # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']
        #
        # # apply preprocessing
        sample = {'image': image, 'mask': mask}
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.transform is not None:
            image = self.transform(image=image)["image"]
            mask = self.transform(mask=mask)["mask"]
        # if self.transform:
        #     sample = self.transform(image=image, mask=mask)
        #     image, mask = sample['image'], sample['mask']
        #     type(image)
        #     type(mask)
        return image, mask


def get_data_loaders():
    # use the same transformations for train/val in this example
    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    ])

    import notebooks.utils as ut

    metadata_df = ut.read_metadata('../data/merge')
    file_name = '../../data/metadata.xlsx'
    metadata_df.to_excel(file_name, index=False, na_rep='N/A', index_label='ID')
    # # Perform 80/20 split for train / val
    # valid_df = metadata_df.sample(frac=0.2, random_state=42)
    # train_df = metadata_df.drop(valid_df.index)


    from sklearn.model_selection import train_test_split

    # Assuming metadata_df is your original dataframe

    # Perform 90/10 split for train / val
    train_df, test_and_val_df = train_test_split(metadata_df, test_size=0.1, random_state=42)

    # Further split the remaining data into training and validation sets
    valid_df, test_df = train_test_split(test_and_val_df, test_size=0.1, random_state=42)

    # Now train_df contains 81% of the data, valid_df contains 9%, and test_df contains 10%

    # Optionally, you can reset the indices if needed
    train_df.reset_index(drop=True, inplace=True)
    valid_df.reset_index(drop=True, inplace=True)
    test_df.reset_index(drop=True, inplace=True)

    # train_set = SatelliteDataset(train_df, transform=transforms.Compose([
    #     A.ToFloat(),
    #     ToTensorV2()
    # ]))
    # # Rescale(256),
    # # RandomCrop(224),
    #
    # val_set = SatelliteDataset(valid_df, transform=transforms.Compose([
    #     A.ToFloat(),
    #     ToTensorV2()
    # ]))

    train_set = SatelliteDataset(train_df, preprocessing=get_preprocessing())
    val_set = SatelliteDataset(valid_df, preprocessing=get_preprocessing())
    test_set = SatelliteDataset(test_df, preprocessing=get_preprocessing())

    image_datasets = {
        'train': train_set, 'val': val_set
    }

    # batch_size = 25
    batch_size = 1

    dataloaders = {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }
    test_loader = {
        'test': DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)
    }

    return dataloaders, test_loader


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    # print(pred.size(), target.size())

    # Assuming pred has shape (batch_size, num_classes, height, width)
    # Target should have shape (batch_size, height, width) with integer class labels

    # Apply softmax along the channel dimension to get probabilities
    pred_softmax = F.softmax(pred, dim=1)
    # target = torch.argmax(target, dim=1)
    # Calculate Cross-Entropy Loss
    ce_loss = F.cross_entropy(pred, target)

    # Calculate Dice Loss
    dice = dice_loss(pred_softmax, target)

    # Combine the losses
    loss = bce_weight * ce_loss + (1 - bce_weight) * dice

    metrics['ce_loss'] += ce_loss.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, dataloaders, optimizer, scheduler, num_epochs=25):
    # dataloaders = get_data_loaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])

                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                # print(inputs.size(), labels.size())

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def run(UNet, df):
    num_class = 3
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = UNet(num_class).to(device)

    summary(model, (3, 512, 512))

    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)
    dataloaders, test_loader = get_data_loaders()
    model = train_model(model, dataloaders, optimizer_ft, exp_lr_scheduler, num_epochs=3)

    model.eval()  # Set model to the evaluation mode

    # trans = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # imagenet
    # ])

    # # Create another simulation dataset for test
    test_loader= test_loader["test"]
    # test_loader = DataLoader(test_dataset, batch_size=3, shuffle=False, num_workers=0)

    # Get the first batch
    inputs, labels = next(iter(test_loader))
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Predict
    pred = model(inputs)
    # The loss functions include the sigmoid function.
    pred = F.sigmoid(pred)
    pred = pred.data.cpu().numpy()
    print(pred.shape)

    # Change channel-order and make 3 channels for matplot
    # input_images_rgb = [reverse_transform(x) for x in inputs.cpu()]

    # Map each channel (i.e. class) to each color
    # target_masks_rgb = [masks_to_colorimg(x) for x in labels.cpu().numpy()]
    # pred_rgb = [masks_to_colorimg(x) for x in pred]
    return inputs, labels, pred
    plot_side_by_side([inputs, labels, pred])

def plot_side_by_side(img_arrays):
    flatten_list = reduce(lambda x,y: x+y, zip(*img_arrays))

    plot_img_array(np.array(flatten_list), ncol=len(img_arrays))

def plot_img_array(img_array, ncol=3):
    print((len(img_array)))
    nrow = len(img_array) // ncol

    f, plots = plt.subplots(nrow, ncol, sharex='all', sharey='all', figsize=(ncol * 4, nrow * 4))

    for i in range(len(img_array)):
        plots[i // ncol, i % ncol]
        plots[i // ncol, i % ncol].imshow(img_array[i])
def plot_errors(results_dict, title):
    markers = itertools.cycle(('+', 'x', 'o'))

    plt.title('{}'.format(title))

    for label, result in sorted(results_dict.items()):
        plt.plot(result, marker=next(markers), label=label)
        plt.ylabel('dice_coef')
        plt.xlabel('epoch')
        plt.legend(loc=3, bbox_to_anchor=(1, 0))

    plt.show()



## from merge-test
def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform
    Args:
        preprocessing_fn (callable): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))

    return album.Compose(_transform)


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, masks = sample['image'], sample['mask']
        # print(f"image size before toTesnor " + image.size())
        # print(f"masks size before toTesnor " + masks.size())
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        # masks = masks.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'mask': torch.from_numpy(masks)}
