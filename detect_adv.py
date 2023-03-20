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
from pytorchyolo.utils.datasets import ImageFolder
from pytorchyolo.utils.transforms import Resize, DEFAULT_TRANSFORMS

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
    print("Patch shape = " + str(patch.shape))
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
    """Detects objects on all images in specified directory and saves output images with drawn detections.

    :param model_path: Path to model definition file (.cfg)
    :type model_path: str
    :param weights_path: Path to weights or checkpoint file (.weights or .pth)
    :type weights_path: str
    :param img_path: Path to directory with images to inference
    :type img_path: str
    :param classes: List of class names
    :type classes: [str]
    :param output_path: Path to output directory
    :type output_path: str
    :param batch_size: Size of each image batch, defaults to 8
    :type batch_size: int, optional
    :param img_size: Size of each image dimension for yolo, defaults to 416
    :type img_size: int, optional
    :param n_cpu: Number of cpu threads to use during batch generation, defaults to 8
    :type n_cpu: int, optional
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    """
    dataloader = _create_data_loader(img_path, batch_size, img_size, n_cpu)
    model = load_model(model_path, weights_path)
    detect(model, dataloader, output_path, conf_thres, nms_thres)


def detect(model, dataloader, output_path, conf_thres, nms_thres):
    """Inferences images with model.

    :param model: Model for inference
    :type model: models.Darknet
    :param dataloader: Dataloader provides the batches of images to inference
    :type dataloader: DataLoader
    :param output_path: Path to output directory
    :type output_path: str
    :param conf_thres: Object confidence threshold, defaults to 0.5
    :type conf_thres: float, optional
    :param nms_thres: IOU threshold for non-maximum suppression, defaults to 0.5
    :type nms_thres: float, optional
    :return: List of detections. The coordinates are given for the padded image that is provided by the dataloader.
        Use `utils.rescale_boxes` to transform them into the desired input image coordinate system before its transformed by the dataloader),
        List of input image paths
    :rtype: [Tensor], [str]
    """
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

    img_detections = []  # Stores detections for each image index
    imgs = []  # Stores image paths

    # Get dog image to attack
    dog_img = None
    for (img_paths, input_imgs) in tqdm.tqdm(dataloader, desc="Detecting"):
        dog_img = input_imgs
        break

    # Configure input
    dog_img = Variable(dog_img.type(Tensor)) # Image values between [0, 1]
    dog_img.to(device)

    # Get bboxes
    with torch.no_grad():
        detections = model(dog_img)
        orig_bboxes, _ = non_max_suppression(detections, conf_thres, nms_thres)
    # Grab largest object bbox (or the dog's in this case)
    for bbox in orig_bboxes[0]:
        if bbox[-1] == 16.0:
            attack_bbox = bbox
            break

    # Initialize patch
    patch = initialize_patch()
    # Attach patch on largest object
    x, y = get_patch_loc(attack_bbox, patch.shape[2])
    patch_dummy = get_patch_dummy(patch, dog_img.shape, x, y).to(device)
    img_mask = patch_dummy.clone() # To mask the img
    img_mask[img_mask != 0] = 1.0 # Turn patch values into 1's
    patch_mask = (1 - img_mask).to(device) # To mask the patch
    adv_img = patch_on_img(patch_dummy, dog_img, patch_mask, img_mask)

    adv_img = Variable(adv_img.type(Tensor))
    adv_img.requires_grad_()

    optimizer = torch.optim.Adam([adv_img], lr=1)

    # Training loop
    epoch = 500
    prev_conf_sum = 0
    for e in range(epoch):
        # Get detections
        # adv_img = Variable(adv_img)
        # adv_img.requires_grad_()

        output = model(adv_img)
        detections, grid_obj_conf = non_max_suppression(output, conf_thres, nms_thres)
        detections = detections[0]
        conf_sum = torch.log(torch.sum(grid_obj_conf[:,4]))
        p_tv = total_variation(patch) # total_varation calculates tv per channel
        p_tv = torch.sum(p_tv)

        if e % 50 == 0:
            print("------- Epoch {} -------".format(e))
            print("Objectiveness score sum: {} / {}".format(conf_sum, len(grid_obj_conf)))
            print("Patch total variation loss: {}".format(p_tv))


        # if adv_img.grad is not None:
        #     adv_img.grad.zero()
        optimizer.zero_grad()
        loss = 100 * conf_sum # + 0.5 * p_tv
        loss.backward()
        optimizer.step()
        
        # Mask image area of the patch
        # adv_grad = adv_img.grad.detach().clone()
        # adv_grad = torch.mul(adv_grad, img_mask)
        # # Apply gradient to patch
        # # Note: Subtracting adv_grad trains patch to reduce obj score sum
        # adv_img = adv_img - 0.05 * adv_grad
        # adv_img = torch.clamp(adv_img, min=0.0, max=1.0)

        # Extract patch for saving
        patch = adv_img[0][:, x:x+patch.shape[-1],y:y+patch.shape[-1]]

        if e % 50 == 0:
            transform = T.ToPILImage()
            img = transform(adv_img[0])
            img.save("adv_output/adv_img_{}.png".format(e))
            img = transform(patch)
            img.save("adv_output/adv_patch_{}.png".format(e))

        # if abs(conf_sum - prev_conf_sum) < 0.01:
        #     print("Score difference too small. Breaking out of train.")
        #     break
        # prev_conf_sum = conf_sum
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
    dataset = ImageFolder(
        img_path,
        transform=transforms.Compose([DEFAULT_TRANSFORMS, Resize(img_size)]))
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
    parser.add_argument("-m", "--model", type=str, default="config/yolov3.cfg", help="Path to model definition file (.cfg)")
    parser.add_argument("-w", "--weights", type=str, default="yolov3.weights", help="Path to weights or checkpoint file (.weights or .pth)")
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