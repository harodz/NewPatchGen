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
        sample_size: int = 1,
        learning_rate: float = 5.0,
        max_iter: int = 500,
        batch_size: int = 16,
        target_ID: int = 11,
        verbose: bool = True,
    ):
        self.writer = SummaryWriter(log_dir='./tensorboard_logs')

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_cat = num_cat

        self._patch = torch.rand(
            self.patch_shape, device=self.device, dtype=torch.float32)
        self._patch.requires_grad = True
        self.optimizer = torch.optim.Adam(
            [self._patch], lr=self.learning_rate, amsgrad=True)

        self.verbose = verbose
        self.patch_location = patch_location
        self.sample_size = sample_size
        self.target_ID = target_ID
        self.model = model

        self.max_prob = max_pro(
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

        # Resize to smaller images

        new_size = np.random.randint(320, 640)
        patched_x = TF.resize(patched_x, [new_size, new_size])
        patched_x = TF.pad(
            patched_x, [(640 - new_size) // 2, (640 - new_size) // 2])
        patched_x = TF.resize(patched_x, [640, 640])

        # Adjust Brightness
        brightness_factor = np.random.uniform(0.4, 1)
        patched_x = TF.adjust_brightness(patched_x, brightness_factor)

        # Add blur:

        ksize = 2 * np.random.randint(0, 25) + 1

        patched_x = TF.gaussian_blur(patched_x, ksize)

        # Reduce saturation:

        saturation_factor = np.random.uniform(0.4, 1)
        patched_x = TF.adjust_saturation(patched_x, saturation_factor)

        return patched_x

    def generate(self, x):
        # Load image and to tensor
        og_x = torch.from_numpy(x).to(device=self.device)
        x = og_x.detach().clone()

        # check if output folder exists, if not, create one
        isExist = os.path.exists('outputs')
        if not isExist:
            os.makedirs('outputs')

        for i_step in trange(self.max_iter, desc="RobustDPatch iteration", disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 5 == 0:
                logger.info("Training Step: %i", i_step + 1)

            epi_loss = 0

            # create a new batch of transformations
            Transformations = []

            for i_batch in range(self.batch_size):
                new_x = x.detach().clone()
                new_x = self._augment_images_with_patch(new_x)
                Transformations.append(new_x)

            New_batch = torch.cat(Transformations, dim=0)

            loss = torch.mean(self.max_prob(New_batch))

            epi_loss += loss
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._patch.data.clamp_(0, 1)

            avg_loss = epi_loss / self.batch_size
            self.writer.add_scalar(
                'Average Loss', avg_loss, global_step=i_step)

            # save patch every n iterations
            if (i_step + 1) % 1 == 0:
                patch_copy = self._patch.detach().clone()
                patch_copy = TF.to_pil_image(patch_copy[0].cpu())
                patch_copy.save('outputs/patch_' + str(i_step) + '.png')

                # save New_batch too
                New_batch_copy = New_batch.detach().clone()
                New_batch_copy = TF.to_pil_image(New_batch_copy[0].cpu())
                New_batch_copy.save('outputs/x_' + str(i_step) + '.png')

        self.writer.close()
        save_tensor_to_image(self._patch, 'Patch/patch.png')
        final_patched = self.apply_patch(og_x)
        save_tensor_to_image(final_patched, 'Patch/final_patched.png')


class max_pro(torch.nn.Module):
    def __init__(self, model, num_cat, target_ID):

        super(max_pro, self).__init__()
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
        class_confidence = torch.softmax(class_confidence, dim=2)
        class_confidence = class_confidence[:, :, self.target_ID]

        max_prob = torch.max(class_confidence, dim=1)[0]
        return max_prob


def save_tensor_to_image(tensor, filename):
    tensor = tensor.detach().clone()
    if len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    tensor = TF.to_pil_image(tensor)
    tensor.save(filename)
