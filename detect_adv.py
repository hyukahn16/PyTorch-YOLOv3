#! /usr/bin/env python3

from __future__ import division

import os
import argparse
import tqdm
import shutil
import time

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim

from pytorchyolo.models import load_model
from pytorchyolo.utils.utils import load_classes, non_max_suppression, print_environment_info
from pytorchyolo.utils.datasets import ListDataset
from pytorchyolo.utils.parse_config import parse_data_config
from pytorchyolo.utils.loss import compute_loss
from pytorchyolo.detect import _draw_and_save_output_images

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
    bboxes, img_path = detect(model, dataloader, output_path, conf_thres, nms_thres)
    _draw_and_save_output_images(bboxes, img_path, img_size, "adv_output", classes)


def detect(model, dataloader, output_path, conf_thres, nms_thres):
    # Delete existing images
    delete_old_imgs = True
    if delete_old_imgs:
        try:
            shutil.rmtree("adv_output")
        except FileNotFoundError:
            pass

    # Create output directory, if missing
    os.makedirs(output_path, exist_ok=True)
    os.makedirs("adv_output", exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    transform = T.ToPILImage()
    target_attack = True
    use_optim = True

    # Get dog image to attack
    for _, (img_path, img, targets) in enumerate(dataloader):
        # img values should be between [0, 1]
        dog_img = Variable(img.type(Tensor)).to(device)

        if target_attack:
            # For targeted attack - target class == 50 == brocolli
            targets = torch.tensor(
                [[0.0, 50.0, 0.0, 0.0, 0.12, 0.12]]).to(device)
        else:
            # For untargeted attack
            targets = targets.view((3, 6)).to(device)

    test_train = False
    if test_train:
        with torch.no_grad():
            model.train()
            output = model(dog_img)
            loss, loss_components = compute_loss(output, targets, model)

    # Get bboxes
    test_eval = True
    if test_eval: 
        with torch.no_grad():
            model.eval()
            detections = model(dog_img)
            orig_bboxes, _ = non_max_suppression(detections, conf_thres, nms_thres)
            print(orig_bboxes)

    # Initialize patch
    patch = initialize_patch(patch_frac=0.015).to(device)
    patch.requires_grad_(True)

    # Attach patch to attacking image
    x, y = 0, 0 # x == height from top, y == width from left
    patch_dummy = get_patch_dummy(patch, dog_img.shape, x, y).to(device)
    img_mask = patch_dummy.clone() # To mask the img
    img_mask[img_mask != 0] = 1.0 # Turn patch values into 1's
    patch_mask = (1 - img_mask).to(device) # To mask the patch

    # Training hyperparameters
    epoch = 1000
    lr = 0.1
    save_patch_iter = 100
    save_adv_iter = 0
    print_iter = 50
    time_str = time.strftime("%m%d_%H%M")
    model.train()
    if use_optim:
        optimizer = optim.Adam([patch], lr=lr)
        # gamma = 0.999
        # lr_sched = optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    for e in range(epoch+1):
        if not use_optim:
            patch = Variable(patch.type(Tensor))
            patch.requires_grad_(True)
        
        # Apply patch to original image
        patch_dummy = get_patch_dummy(patch, dog_img.shape, x, y).to(device)
        adv_img = patch_on_img(patch_dummy, dog_img, patch_mask, img_mask).to(device)
        # Forward pass
        output = model(adv_img)
        
        if model.training:
            loss, loss_components = compute_loss(output, targets, model)
            log_loss = torch.log(loss)
            total_loss = log_loss
        else:
            # Filter out grids with objective score less than threshold
            grid_obj_conf = output[0][output[0][... , 4] > conf_thres]
            total_loss = None

        # Optimize
        if use_optim:
            optimizer.zero_grad()
        else:
            if patch.grad is not None:
                patch.grad.zero()

        total_loss.backward(retain_graph=True)

        if use_optim:
            optimizer.step()
            # lr_sched.step()
            # print(lr_sched.get_lr())
        else:
            grad = patch.grad.detach().clone()
            patch = patch + lr * grad
        
        # Clamping
        if use_optim:
            patch.data.clamp_(0, 1) # For optimizer
        else:
            patch = torch.clamp(patch, min=0.0, max=1.0) # For manual autograd

        # Print training info
        if e % print_iter == 0:
            print("------- Epoch {} -------".format(e))
            if model.training:
                print("Loss                : {:3f} / {:3f}".format(
                    log_loss[0], loss[0]))
                print("Loss component      : {}".format(loss_components))

        # Save adversarial image and patch checkpoints
        if e % save_patch_iter == 0:
            img = transform(patch[0])
            img_path = "adv_output/adv_patch_{}_{}.png".format(e, time_str)
            img.save(img_path)
        if save_adv_iter > 0 and e % save_adv_iter == 0:
            img = transform(adv_img[0])
            img_path = "adv_output/adv_img_{}_{}.png".format(e, time_str)
            img.save(img_path)
    
    # Save last adversarial image
    last_adv_img_path = "adv_output/adv_img_last_{}.png".format(time_str)
    img = transform(adv_img[0]).save(last_adv_img_path)

    # Evaluate bbox differences between adv img and original img
    model.eval()
    output = model(adv_img)
    bboxes, _ = non_max_suppression(output, conf_thres, nms_thres)
    print("---- Training Finished ----\n\n\n")
    print("New BBOX")
    print(bboxes)
    print("Original BBOX")
    print(orig_bboxes)

    return bboxes, [last_adv_img_path]


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
    dataset = ListDataset(
        img_path,
        img_size=img_size)
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
