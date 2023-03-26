#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import random
import numpy as np
import shutil

from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.autograd import Variable

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, rescale_boxes, non_max_suppression, print_environment_info
from pytorchyolo.utils.datasets import ImageFolder, ListDataset
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS
from pytorchyolo.utils.nps import NPSCalculator
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.utils.augmentations import AUGMENTATION_TRANSFORMS

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

# https://kornia.readthedocs.io/en/latest/_modules/kornia/losses/total_variation.html
from kornia.losses import total_variation

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

def initialize_patch(img_size=416, patch_frac=0.015):
    # Initialize adversarial patch w/ random values
    img_size = img_size * img_size
    patch_size = img_size * patch_frac
    patch_dim = int(patch_size**(0.5)) # sqrt
    patch = torch.rand(1, 3, patch_dim, patch_dim)
    # print("Patch shape = " + str(patch.shape))
    return patch

def get_patch_dummy(patch, data_shape, patch_x, patch_y):
    """
    :param patch: Attack patch to be superimposed on original image
    :type patch: np.array of dim (1, 3, patch_dim, patch_dim)
    :param data_shape: Shape of the original image
    :type data_shape: np.array of rank 4 (batch, channel, dim, dim)
    """
    # Get dummy image which we will place attack patch on.
    dummy = torch.zeros(data_shape)
    
    # Get width or height dimension of patch
    patch_size = patch.shape[-1] # patch.shape == (1, 3, patch_dim, patch_dim)
       
    # Apply patch to dummy image  
    dummy[0][0][patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] = patch[0][0]
    dummy[0][1][patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] = patch[0][1]
    dummy[0][2][patch_x:patch_x+patch_size, patch_y:patch_y+patch_size] = patch[0][2]
    
    return dummy

def get_patch_loc(attack_bbox, patch_side):
    new_h =  int(((attack_bbox[1] + attack_bbox[3]) / 2) - patch_side/2) # y == h
    new_w = int(((attack_bbox[0] + attack_bbox[2]) / 2) - patch_side/2) # x == w
    return new_h, new_w

def patch_on_img(patch_dummy, img, patch_mask, img_mask):
    cutout_img = torch.mul(patch_mask, img) # img with patch area masked out
    adv_img = cutout_img + patch_dummy # img with patch
    return adv_img
    

def detect_directory(model_path, weights_path, img_path, classes, output_path,
                     batch_size=8, img_size=416, n_cpu=8, conf_thres=0.5, nms_thres=0.5):
    data_config = parse_data_config("config/custom.data")
    train_path = data_config["train"]

    dataloader = _create_data_loader(train_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    detect(model, dataloader, output_path, conf_thres, nms_thres)


def detect(model, dataloader, output_path, conf_thres, nms_thres):
    # Delete existing images for sanity
    try:
        shutil.rmtree("adv_output")
    except FileNotFoundError:
        pass

    # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)
    os.makedirs("adv_output", exist_ok=True)

    model.eval()  # Set model to evaluation mode
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Hyperparameters
    dog_class = 16.0
    epoch = 2000
    conf_threshold = 0.25
    lr = 0.05

    # Get dog image to attack
    dog_img = None
    dog_img_targs = None
    for (_, imgs, targets) in tqdm.tqdm(dataloader, desc="Detecting"):
        dog_img = imgs
        dog_img_targs = targets.view((3, 6)).to(device)
        break
    print(dog_img_targs)
    # Configure input
    dog_img = Variable(dog_img.type(Tensor)) # Image values between [0, 1]
    dog_img.to(device)

    # Testing loss computation
    with torch.no_grad():
        model.train()
        outputs = model(dog_img)
        loss, loss_components = compute_loss(outputs, dog_img_targs, model)
        print(loss)
        # model.eval()

    # Get bboxes
    # with torch.no_grad():
    #     detections = model(dog_img)
    #     orig_bboxes, _ = non_max_suppression(detections, conf_thres, nms_thres)
    # print(orig_bboxes)

    # # Grab largest object bbox (or the dog's in this case)
    # for bbox in orig_bboxes[0]:
    #     bbox_class = bbox[-1]
    #     if bbox_class == dog_class:
    #         attack_bbox = bbox
    #         break

    # Initialize patch
    patch = initialize_patch().to(device)
    patch.requires_grad_(True)
    optimizer = torch.optim.Adam([patch], lr=lr, amsgrad=True)

    # Attach patch on largest object
    patch_size = patch.shape[-1]
    # x, y = get_patch_loc(attack_bbox, patch_size)
    x, y = 52, 0
    # x, y = 123, 300 # Around truck
    patch_dummy = get_patch_dummy(patch, dog_img.shape, x, y).to(device)
    img_mask = patch_dummy.clone() # To mask the img
    img_mask[img_mask != 0] = 1.0 # Turn patch values into 1's
    patch_mask = (1 - img_mask).to(device) # To mask the patch

    nps_calculator = NPSCalculator("nps_print.txt", patch_size, device).to(device)

    # adv_img = Variable(adv_img.type(Tensor))
    # adv_img.requires_grad_()
    for e in range(epoch+1):
        patch_dummy = get_patch_dummy(patch, dog_img.shape, x, y).to(device)
        adv_img = patch_on_img(patch_dummy, dog_img, patch_mask, img_mask).to(device)
        output = model(adv_img)
        loss, loss_components = compute_loss(output, dog_img_targs, model)
        # # 1. Object loss
        # grid_obj_conf = output[0][output[0][... , 4] > conf_threshold]
        # conf_sum = torch.log(torch.sum(grid_obj_conf[:,4]))
        # # conf_sum = torch.max(grid_obj_conf[:,4])

        # # 2. Total variation loss
        # tv = total_variation(patch) # total_varation calculates tv per channel
        # tv = torch.sum(tv) / torch.numel(patch)
        # tv_loss = tv * 1.5

        # # 3. Non-printability score loss
        # nps = nps_calculator(patch)
        # nps_loss = nps * 5

        # loss = conf_sum # + tv_loss + nps_loss

        # if e % 50 == 0:
        #     print("------- Epoch {} -------".format(e))
        #     print("Obj score sum:        {:.4f} / {}".
        #         format(conf_sum, len(grid_obj_conf)))
        #     print("Total variation loss: {:.4f} / {:.4f}".
        #         format(tv_loss, tv))
        #     print("NPS loss:             {:.4f} / {:.4f}".
        #         format(nps_loss, nps))

        # if patch.grad is not None:
        #     patch.grad.zero()
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        
        # grad = patch.grad.detach().clone()
        # patch = patch - grad

        patch.data.clamp_(0, 1) # For optimizer
        # patch = torch.clamp(patch, min=0.0, max=1.0) # For manual autograd

        if e % 100 == 0:
            transform = T.ToPILImage()
            img = transform(adv_img[0])
            img.save("adv_output/adv_img_{}.png".format(e))
            img = transform(patch[0])
            img.save("adv_output/adv_patch_{}.png".format(e))

        # print("\n")
    
    output = model(adv_img)
    bboxes, _ = non_max_suppression(output, conf_thres, nms_thres)
    print("---- Training Finished ----")
    print("New BBOX")
    print(bboxes)
    print("Original BBOX")
    print(orig_bboxes)


def _create_data_loader(img_path, batch_size, img_size, n_cpu):
    """Creates a DataLoader for inferencing.

    :param img_path: Path to file containing all paths to validation images.
    :type img_path: str
    :param batch_size: Size of each image batch
    :type batch_size: int
    :param img_size: Size of each image dimension for yolo
    :type img_size: int
    :param n_cpu: Number of cpu threads to use during batch generation
    :type n_cpu: int
    :return: Returns DataLoader
    :rtype: DataLoader
    """
    # dataset = ImageFolder(
    #     img_path,
    #     transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
    dataset = ListDataset(
        img_path,
        img_size=img_size,
        transform=AUGMENTATION_TRANSFORMS)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_cpu,
        pin_memory=True)
    return dataloader


def run():
    print_environment_info()
    parser = argparse.ArgumentParser(description="Detect objects on images.")
    parser.add_argument("-m", "--model", type=str, default="config/yolov3-tiny.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="yolov3-tiny.weights", help="Path to weights or checkpoint file (.weights or .pth)")
    parser.add_argument("-i", "--images", type=str, default="data/samples", help="Path to directory with images to inference")
    parser.add_argument("-c", "--classes", type=str, default="data/coco.names", help="Path to classes label file (.names)")
    parser.add_argument("-o", "--output", type=str, default="output", help="Path to output directory")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="Size of each image batch")
    parser.add_argument("--img_size", type=int, default=416, help="Size of each image dimension for yolo")
    parser.add_argument("--n_cpu", type=int, default=8, help="Number of cpu threads to use during batch generation")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="Object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="IOU threshold for non-maximum suppression")
    args = parser.parse_args()
    print(f"Command line arguments: {args}")

    # Extract class names from file
    classes = load_classes(args.classes)  # List of class names

    detect_directory(
        args.model,
        args.weights,
        args.images,
        classes,
        args.output,
        batch_size=args.batch_size,
        img_size=args.img_size,
        n_cpu=args.n_cpu,
        conf_thres=args.conf_thres,
        nms_thres=args.nms_thres)


if __name__ == '__main__':
    run()
