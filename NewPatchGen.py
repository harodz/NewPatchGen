import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange
import torch
import torchvision.transforms.functional as TF
from torch.utils.tensorboard import SummaryWriter

import random
from PIL import Image
import os


logger = logging.getLogger(__name__)


class NewPatchGen():

    def __init__(
        self,
        model: torch.nn.Module,
        num_cat: int = 80,
        patch_shape: Tuple[int, int, int] = (3, 40, 40),
        patch_location: Tuple[int, int] = (0, 0),
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        target_ID: int = 11,  # Target ID = 11 (stop sign index in COCO)
        verbose: bool = True,
        logging_frequency: int = 5,  # log every n iterations
        # range of kernel size for gaussian blur
        ksize_range: Tuple[int, int] = (5, 25),
        rotation_range: Tuple[int, int] = (-15, 15),
        tv_weight: float = 0.1,
        skip_prob: float = 0.001,
        scheduler_step_size: int = 100,
        scheduler_gamma: float = 0.9
    ):
        self.writer = SummaryWriter(log_dir='./tensorboard_logs')

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_cat = num_cat
        self.logging_frequency = logging_frequency
        self.ksize_range = ksize_range
        self.rotation_range = rotation_range
        self.tv_weight = tv_weight
        self.skip_prob = skip_prob
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        # random patch initialization
        self._patch = torch.rand(
            self.patch_shape, device=self.device, dtype=torch.float32)

        # # white patch initialization
        # self._patch = torch.ones(
        #     self.patch_shape, device=self.device, dtype=torch.float32)

        self._patch.requires_grad = True
        self.optimizer = torch.optim.Adam(
            [self._patch], lr=self.learning_rate, amsgrad=True)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=100, gamma=0.9)

        self.verbose = verbose
        self.patch_location = patch_location
        self.target_ID = target_ID
        self.model = model

        self.MaxPro = MaxPro(
            model=self.model, num_cat=self.num_cat, target_ID=self.target_ID).to(self.device)

    def apply_patch(self, x: torch.tensor):
        # Get the height and width of x
        _, _, H, W = x.shape

        # Apply patch:
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + self.patch_shape[1], y_1 + self.patch_shape[2]

        if x_2 > H or y_2 > W:
            raise ValueError(
                "Patch goes beyond the boundaries of the input tensor.")

        # check if patch shape has batch size, if not add it
        if len(self._patch.shape) == 3:
            self._patch = self._patch.unsqueeze(0)

        x[:, :, x_1:x_2, y_1:y_2] = self._patch[0]

        return x

    def _augment_images_with_patch(
            self, x: torch.Tensor):

        patched_x = self.apply_patch(x)

        # Add blur:
        if random.random() >= self.skip_prob:
            ksize = 2 * \
                np.random.randint(self.ksize_range[0], self.ksize_range[1]) + 1
            patched_x = TF.gaussian_blur(patched_x, ksize)

        # Resize to smaller images
        if random.random() >= self.skip_prob:
            new_size = np.random.randint(128, 640)
            patched_x = TF.resize(patched_x, [new_size, new_size])
            patched_x = TF.pad(
                patched_x, [(640 - new_size) // 2, (640 - new_size) // 2])
            patched_x = TF.resize(patched_x, [640, 640])

        # Random Rotation
        if random.random() >= self.skip_prob:
            patched_x = TF.rotate(patched_x, random.randint(
                self.rotation_range[0], self.rotation_range[1]))

        # Perspective Transform
        if random.random() >= self.skip_prob:
            def random_shift(val, shift_range=(-100, 100)):
                return val + random.uniform(*shift_range)

            # Define source and destination points
            src_points = [[0, 0], [639, 0], [639, 639], [
                0, 639]]  # corners of the original image
            dst_points = [
                [random_shift(0), random_shift(0)],
                [random_shift(639), random_shift(0)],
                [random_shift(639), random_shift(639)],
                [random_shift(0), random_shift(639)]
            ]

            patched_x = TF.perspective(
                patched_x, startpoints=src_points, endpoints=dst_points)

        # ColorJitter
        if random.random() >= self.skip_prob:
            brightness_factor = np.random.uniform(0.4, 1.5)
            saturation_factor = np.random.uniform(0.4, 1.5)
            hue_factor = np.random.uniform(-0.1, 0.1)
            contrast_factor = np.random.uniform(0.4, 1.5)

            patched_x = TF.adjust_brightness(patched_x, brightness_factor)
            patched_x = TF.adjust_saturation(patched_x, saturation_factor)
            patched_x = TF.adjust_hue(patched_x, hue_factor)
            patched_x = TF.adjust_contrast(patched_x, contrast_factor)

        # # Add Gaussian Noise
        # if random.random() >= self.skip_prob:
        #     std_intensity = np.random.uniform(0.01, 0.1)
        #     noise = torch.normal(
        #         mean=0., std=std_intensity, size=patched_x.shape)
        #     noise = noise.to(device=self.device)
        #     patched_x = patched_x + noise
        #     # Keep values within [0, 1]
        #     patched_x = torch.clamp(patched_x, 0, 1)

        return patched_x

    def _total_variation_loss(self):
        # calculate total variation loss for the patch
        # https://en.wikipedia.org/wiki/Total_variation_denoising

        tv = torch.mean(torch.abs(self._patch[:, :, :, :-1] - self._patch[:, :, :, 1:])) + \
            torch.mean(
                torch.abs(self._patch[:, :, :-1, :] - self._patch[:, :, 1:, :]))

        return tv

    def generate(self, x):
        # Load image and to tensor
        og_x = torch.from_numpy(x).to(device=self.device)
        x = og_x.detach().clone()

        # check if output folder exists, if not, create one
        isExist = os.path.exists('outputs')
        if not isExist:
            os.makedirs('outputs')

        print('Generating Patch...')

        for i_step in trange(self.max_iter, desc="PatchGen iteration", disable=not self.verbose):

            # create a new batch of transformations
            Transformations = []

            for i_batch in range(self.batch_size):
                new_x = x.detach().clone()
                new_x = self._augment_images_with_patch(new_x)
                Transformations.append(new_x)

            New_batch = torch.cat(Transformations, dim=0)

            max_final_score = torch.mean(self.MaxPro(New_batch))
            tv_loss = self.tv_weight * self._total_variation_loss()

            loss = max_final_score + tv_loss

            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._patch.data.clamp_(0, 1)
            self.scheduler.step()

            self.writer.add_scalar(
                'Total_Loss', loss, global_step=i_step)
            self.writer.add_scalar(
                'Max_Final_Score', max_final_score, global_step=i_step)
            self.writer.add_scalar(
                'Total_Variation_Loss', tv_loss, global_step=i_step)

            self.writer.add_image('Adversarial Patch',
                                  self._patch.squeeze(dim=0), global_step=i_step)

            # save patch every n iterations
            if i_step == 0 or (i_step + 1) % self.logging_frequency == 0:
                # patch_copy = self._patch.detach().clone()
                # patch_copy = TF.to_pil_image(patch_copy[0].cpu())
                # patch_copy.save('outputs/patch_' + str(i_step) + '.png')

                # save New_batch too
                New_batch_copy = New_batch.detach().clone()
                New_batch_copy = TF.to_pil_image(New_batch_copy[0].cpu())
                New_batch_copy.save('outputs/x_' + str(i_step) + '.png')

                # # log learning rate
                # logging.info(
                #     f"Current learning rate: {self.optimizer.param_groups[0]['lr']}")

        self.writer.close()
        save_tensor_to_image(self._patch, 'Patch/patch.png')
        final_patched = self.apply_patch(og_x)
        save_tensor_to_image(final_patched, 'Patch/final_patched.png')


class MaxPro(torch.nn.Module):
    def __init__(self, model, num_cat, target_ID):

        super(MaxPro, self).__init__()
        self.model = model
        self.num_cat = num_cat
        self.target_ID = target_ID

    def forward(self, x):
        outputs = self.model(x)

        if len(outputs) == 3:
            for i in range(len(outputs)):
                # reshape outputs to [batch, grid, 5 + num classes]
                shape = outputs[i].shape
                outputs[i] = outputs[i].reshape(
                    shape[0], shape[1] * shape[2] * shape[3], shape[4])
            all_outputs = torch.cat(outputs, axis=1)
        else:
            all_outputs = outputs[0]

        class_confidence = all_outputs[:, :, 5:5 + self.num_cat]
        class_confidence = class_confidence[:, :, self.target_ID]

        objectness = all_outputs[:, :, 4]

        final_score = objectness * class_confidence
        max_final_score = torch.max(final_score, dim=1)[0]

        return max_final_score


def save_tensor_to_image(tensor, filename):
    tensor = tensor.detach().clone()
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    tensor = TF.to_pil_image(tensor)
    tensor.save(filename)
