"""Model test."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020, All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 11月 28日 星期六 21:26:34 CST
# ***
# ************************************************************************************/
#
import argparse
import os

import torch

from data import get_data
from model import enable_amp, get_model, model_device, model_load, valid_epoch

if __name__ == "__main__":
    """Test model."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="output/ImageDehaze.pth", help="checkpoint file")
    parser.add_argument('--bs', type=int, default=2, help="batch size")
    args = parser.parse_args()

    # get model
    model = get_model()
    model_load(model, args.checkpoint)

    # CPU or GPU ?
    device = model_device()
    model.to(device)

    enable_amp(model)

    print("Start testing ...")
    test_dl = get_data(trainning=False, bs=args.bs)
    valid_epoch(test_dl, model, device, tag='test')
