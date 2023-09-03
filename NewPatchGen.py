import logging
import math
import random
from typing import Dict, List, Optional, Tuple, Union, TYPE_CHECKING

import numpy as np
from tqdm.auto import trange
import torch
import torchvision.transforms.functional as TF
import random
from PIL import Image

logger = logging.getLogger(__name__)


class NewPatchGen():

    def __init__(
        self,
        estimator: "OBJECT_DETECTOR_TYPE",
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

        super().__init__(estimator=estimator, summary_writer=summary_writer)

        self.device = self.estimator._device

        self.patch_shape = patch_shape
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.num_cat = num_cat

        self._patch = torch.ones(self.patch_shape, device=self.device) * 255
        self._patch.requires_grad = True
        self.optimizer = torch.optim.Adam(
            [self._patch], lr=self.learning_rate, amsgrad=True)

        self.verbose = verbose
        self.patch_location = patch_location
        self.sample_size = sample_size
        self.target_ID = target_ID

    def apply_patch(self, x: torch.tensor):

        # Apply patch:
        x_1, y_1 = self.patch_location
        x_2, y_2 = x_1 + self._patch.shape[0], y_1 + self._patch.shape[1]

        x[:, x_1:x_2, y_1:y_2, :] = self._patch

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
        og_x = torch.from_numpy(x).to(self.device)
        x = og_x.detach().clone()

        # check if output folder exists, if not, create one
        """
        Waiting for copilot
        """

        for i_step in trange(self.max_iter, desc="RobustDPatch iteration", disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                logger.info("Training Step: %i", i_step + 1)

            epi_loss = 0

            # create a new batch of transformations
            Transformations = []

            for i_batch in range(self.batch_size):
                new_x = x.detach().clone()
                new_x = self._augment_images_with_patch(new_x)
                Transformations.append(new_x)

            New_batch = torch.cat(Transformations, dim=0)

            for i_batch in range(self.batch_size):
                loss = torch.mean(self.max_prob(New_batch, 11))
                epi_loss += loss

                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._patch.data.clamp_(0, 255)

            patch_copy = self._patch.detach().clone()
            patch_copy = TF.to_pil_image(patch_copy)
            img = Image.open(patch_copy)
            img.save('outputs/patch_' + str(i_step) + '.png')


class max_pro(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def foward(self):
        outputs = self.estimator.model(x)

        for i in range(len(outputs)):
            # reshape outputs to [batch, grid, 5 + num classes]
            shape = outputs[i].shape
            outputs[i] = outputs[i].reshape(
                shape[0], shape[1] * shape[2] * shape[3], shape[4])
        all_outputs = torch.cat(outputs, axis=1)

        class_confidence = outputs[:, :, 5:5 + self.num_cat]
        class_confidence = torch.softmax(class_confidence, dim=2)
        class_confidence = class_confidence[:, :, target_ID]

        max_prob = torch.max(class_confidence, dim=1)[0]
        return max_prob
