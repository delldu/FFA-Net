"""Model predict."""  # coding=utf-8
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
import glob
import os

import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

from model import enable_amp, get_model, model_device, model_load

if __name__ == "__main__":
    """Predict."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default="models/ImageDehaze.pth", help="checkpint file")
    parser.add_argument(
        '--input', type=str, default="dataset/predict/haze/*.jpg", help="input image")
    args = parser.parse_args()

    model = get_model()
    device = model_device()
    model_load(model, args.checkpoint)
    model.to(device)
    model.eval()

    enable_amp(model)

    # totensor = transforms.ToTensor()
    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])
    ])

    toimage = transforms.ToPILImage()

    image_filenames = glob.glob(args.input)
    progress_bar = tqdm(total=len(image_filenames))

    for index, filename in enumerate(image_filenames):
        progress_bar.update(1)

        image = Image.open(filename).convert("RGB")
        input_tensor = totensor(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output_tensor = model(input_tensor).clamp(0, 1.0).squeeze()

        toimage(output_tensor.cpu()).save(
            "dataset/predict/clean/{}".format(os.path.basename(filename)))
