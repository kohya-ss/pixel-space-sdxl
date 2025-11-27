import json
import os
import random
from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import CLIPTokenizer

import pixel_sdxl.text_encoder_utils as text_encoder_utils

MAX_PIXELS_PER_IMAGE = 1024 * 1024
MIN_PIXELS_PER_IMAGE = 768 * 768  # slightly smaller than 1024*1024

CAPTION_DROP_PROBABILITY = 0.1  # probability to drop all captions, for CFG unconditioned training
TAG_DROP_PROBABILITY = 0.25  # probability to drop each tag

# define class for item
class ImageItem:
    def __init__(self, image_path: str, tags: list[str], width: int, height: int):
        self.image_path = image_path
        self.tags = tags
        self.width = width
        self.height = height


class ImageDataset(Dataset):
    def __init__(
        self,
        shared_epoch: Any,
        metadata_files: list[str],
        batch_size: int,
        tokenizer1: CLIPTokenizer,
        tokenizer2: CLIPTokenizer,
        seed: int = 42,
        is_skipping: Any = None,
    ):
        self.shared_epoch = shared_epoch
        self.current_epoch: int = 0  # start from epoch 1
        self.batch_size = batch_size
        self.tokenizer1 = tokenizer1
        self.tokenizer2 = tokenizer2
        self.items: list[ImageItem] = []
        self.seed = seed
        self.random = random.Random(seed)
        self.is_skipping = is_skipping

        skip_count = 0
        for metadata_file in metadata_files:
            print(f"Loading metadata from {metadata_file}...")
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)

                # WSL2の場合は、WindowsのパスをLinuxのパスに変換する
                if os.name == "posix" and "WSL_DISTRO_NAME" in os.environ:
                    print("Detected WSL2 environment, converting Windows paths to WSL paths...")
                    wsl_prefix = "/mnt/"
                    for key in list(metadata.keys()):
                        if len(key) > 2 and key[1] == ":":
                            drive_letter = key[0].lower()
                            path_without_drive = key[2:].replace("\\", "/")
                            linux_path = os.path.join(wsl_prefix, drive_letter, path_without_drive.lstrip("/"))
                            metadata[linux_path] = metadata.pop(key)

                for key in metadata:
                    item_data = metadata[key]
                    tag_str = item_data.get("tags", "")
                    tags = [tag.strip() for tag in tag_str.split(",") if tag.strip()]
                    width = item_data["image_size"][0]
                    height = item_data["image_size"][1]
                    width_bucket = (width // 32) * 32
                    height_floor = (height // 32) * 32
                    if width_bucket * height_floor < MIN_PIXELS_PER_IMAGE:
                        # print(f"Image {key} is too large after flooring to 32-pixel multiples, skipping.")
                        skip_count += 1
                        continue
                    item = ImageItem(
                        image_path=key,
                        tags=tags,
                        width=width,
                        height=height,
                    )
                    self.items.append(item)
        print(f"Total {len(self.items)} items loaded.")
        print(f"Total {skip_count} items skipped due to size.")

        # sort items by image_path to ensure consistent order
        self.items.sort(key=lambda x: x.image_path)

        # make Aspect Ratio Buckets
        need_resize_count = 0
        self.buckets: dict[tuple[int, int], list[ImageItem]] = {}
        for item in self.items:
            need_resize, width_bucket, height_bucket = self.get_bucket_resolution(item.width, item.height)
            if need_resize:
                need_resize_count += 1
            key = (width_bucket, height_bucket)
            if key not in self.buckets:
                self.buckets[key] = []
            self.buckets[key].append(item)
        print(f"Total {len(self.buckets)} aspect ratio buckets created. Total {need_resize_count} items need resizing.")

        # calculate total number of batches
        self.total_batches = 0
        for key in self.buckets:
            num_items = len(self.buckets[key])
            num_batches = (num_items + self.batch_size - 1) // self.batch_size
            self.total_batches += num_batches
        print(f"Total {self.total_batches} batches available.")

        # make a list of (bucket_key, start_index) for each batch
        self.batch_indices: list[tuple[tuple[int, int], int]] = []
        for key in self.buckets:
            num_items = len(self.buckets[key])
            num_batches = (num_items + self.batch_size - 1) // self.batch_size
            for i in range(num_batches):
                start_index = i * self.batch_size
                self.batch_indices.append((key, start_index))
        print(f"Total {len(self.batch_indices)} batch indices created.")

    def get_bucket_resolution(self, width: int, height: int) -> tuple[bool, int, int]:
        if width * height <= MAX_PIXELS_PER_IMAGE:
            width_floor = (width // 32) * 32
            height_floor = (height // 32) * 32
            return False, width_floor, height_floor

        scale = (MAX_PIXELS_PER_IMAGE / (width * height)) ** 0.5
        new_width = width * scale
        new_height = height * scale
        bucket_resolution_candidate = [
            ((int(new_width) // 32) * 32, (int(new_height) // 32) * 32),
            ((int(new_width + 31) // 32) * 32, (int(new_height) // 32) * 32),
            ((int(new_width) // 32) * 32, (int(new_height + 31) // 32) * 32),
            ((int(new_width + 31) // 32) * 32, (int(new_height + 31) // 32) * 32),
        ]
        bucket_resolution_candidate = np.array(bucket_resolution_candidate)

        # select the one with aspect ratio closest to original
        aspect_ratio = width / height
        best_idx = 0
        best_diff = float("inf")
        for i in range(len(bucket_resolution_candidate)):
            w, h = bucket_resolution_candidate[i]
            candidate_aspect_ratio = w / h
            diff = abs(candidate_aspect_ratio - aspect_ratio)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        w, h = bucket_resolution_candidate[best_idx]
        if w > width or h > height:
            # this happens when the original image is already small enough but just exceeds MAX_PIXELS_PER_IMAGE
            # no need to resize
            width_floor = (width // 32) * 32
            height_floor = (height // 32) * 32
            return False, width_floor, height_floor

        return True, *tuple(bucket_resolution_candidate[best_idx])

    def shuffle(self):
        for key in self.buckets:
            self.random.shuffle(self.buckets[key])
        self.random.shuffle(self.batch_indices)  # shuffle the order of batches too

    def reorder_and_drop_tags(self, tags: list[str]) -> list[str]:
        quality_tag = tags[-2] if len(tags) >= 2 else None
        rating_tag = tags[-1] if len(tags) >= 1 else None
        other_tags = tags[:-2] if len(tags) >= 2 else tags[:-1] if len(tags) >= 1 else tags

        # drop tags randomly
        dropped_tags = []
        for tag in other_tags:
            if TAG_DROP_PROBABILITY > 0.0 and random.random() < TAG_DROP_PROBABILITY:
                continue
            dropped_tags.append(tag)
        if quality_tag is not None:
            dropped_tags.append(quality_tag)
        if rating_tag is not None:
            dropped_tags.append(rating_tag)

        # shuffle all tags
        random.shuffle(dropped_tags)
        return dropped_tags

    def __len__(self):
        return self.total_batches

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self.total_batches:
            raise IndexError("Index out of range")
        if self.is_skipping is not None and self.is_skipping.value:
            # if skipping, return an empty batch
            images_tensor = torch.empty((1, 3, 32, 32), dtype=torch.float32)
            input_ids1_tensor = torch.empty((1, 77), dtype=torch.long)
            input_ids2_tensor = torch.empty((1, 77), dtype=torch.long)
            original_sizes_tensor = torch.empty((1, 2), dtype=torch.float32)
            crop_sizes_tensor = torch.empty((1, 2), dtype=torch.float32)
            target_sizes_tensor = torch.empty((1, 2), dtype=torch.float32)
            return (
                images_tensor,
                input_ids1_tensor,
                input_ids2_tensor,
                original_sizes_tensor,
                crop_sizes_tensor,
                target_sizes_tensor,
            )

        epoch = self.shared_epoch.value
        if epoch > self.current_epoch:
            print(f"epoch is incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
            num_epochs = epoch - self.current_epoch
            for _ in range(num_epochs):
                self.current_epoch += 1
                self.random.seed(self.seed + self.current_epoch)
                self.shuffle()
        elif epoch < self.current_epoch:
            print(f"epoch is not incremented. current_epoch: {self.current_epoch}, epoch: {epoch}")
            self.current_epoch = epoch

        bucket_key, start_index = self.batch_indices[idx]
        width, height = bucket_key

        bucket_items = self.buckets[bucket_key]
        batch_items = []
        for i in range(self.batch_size):
            item_index = start_index + i
            if item_index >= len(bucket_items):
                # select random item to fill the batch
                item_index = random.randint(0, len(bucket_items) - 1)
            batch_items.append(bucket_items[item_index])

        images = []
        input_ids1_list = []
        input_ids2_list = []
        original_sizes = []
        crop_sizes = []
        target_sizes = []
        for item in batch_items:
            # load and trim image
            img = Image.open(item.image_path).convert("RGB")
            assert width <= img.width and height <= img.height
            original_width, original_height = img.width, img.height

            # resize to fit within the bucket while maintaining aspect ratio
            if img.width != width or img.height != height:
                scale = max(width / img.width, height / img.height)
                new_width = int(img.width * scale + 0.5)
                new_height = int(img.height * scale + 0.5)
                img = img.resize((new_width, new_height), resample=Image.LANCZOS)
                assert new_width >= width and new_height >= height and (new_width == width or new_height == height)

            img = np.array(img)

            # random horizontal flip
            if random.random() < 0.5:
                img = np.flip(img, axis=1).copy()  # axis=1 is horizontal, need to copy to make it contiguous

            # random crop for rounded sizes
            crop_left = random.randint(0, img.shape[1] - width)
            crop_top = random.randint(0, img.shape[0] - height)
            img = img[crop_top : crop_top + height, crop_left : crop_left + width]

            img_tensor = torch.from_numpy(img).permute(2, 0, 1).to(dtype=torch.float32) / 127.5 - 1.0  # normalize to [-1, 1], C,H,W
            images.append(img_tensor)

            # process tags
            if CAPTION_DROP_PROBABILITY > 0.0 and random.random() < CAPTION_DROP_PROBABILITY:
                dropped_tags = []
            else:
                dropped_tags = self.reorder_and_drop_tags(item.tags)
    
            text = ", ".join(dropped_tags) if len(dropped_tags) > 0 else ""

            input_ids1, input_ids2 = text_encoder_utils.tokenize(self.tokenizer1, self.tokenizer2, text)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)

            original_sizes.append(torch.FloatTensor([original_height, original_width]))
            crop_sizes.append(torch.FloatTensor([crop_top, crop_left]))
            target_sizes.append(torch.FloatTensor([height, width]))

        images_tensor = torch.stack(images, dim=0)  # B,C,H,W
        input_ids1_tensor = torch.cat(input_ids1_list, dim=0)  # B,77
        input_ids2_tensor = torch.cat(input_ids2_list, dim=0)  # B,77
        original_sizes_tensor = torch.stack(original_sizes, dim=0)  # B,2
        crop_sizes_tensor = torch.stack(crop_sizes, dim=0)  # B,2
        target_sizes_tensor = torch.stack(target_sizes, dim=0)  # B,2

        return images_tensor, input_ids1_tensor, input_ids2_tensor, original_sizes_tensor, crop_sizes_tensor, target_sizes_tensor


if __name__ == "__main__":
    import sys
    import glob
    from multiprocessing import Value
    import cv2

    dataset_dir = sys.argv[1]
    metadata_files = glob.glob(f"{dataset_dir}/*.json")
    epoch_value = Value("i", 0)

    tokenizer1, tokenizer2 = text_encoder_utils.get_sdxl_tokenizers()
    dataset = ImageDataset(epoch_value, metadata_files, 8, tokenizer1, tokenizer2)
    print(f"Dataset length: {len(dataset)}")
    for i in range(20):
        images, input_ids1, input_ids2, original_sizes, crop_sizes, target_sizes = dataset[i]
        print(f"Batch {i}: images {images.shape}, input_ids1 {input_ids1.shape}, input_ids2 {input_ids2.shape}")
        print(f" original_sizes: {original_sizes}, crop_sizes: {crop_sizes}, target_sizes: {target_sizes}")

        image = images[0]
        image = ((image.permute(1, 2, 0).numpy() + 1.0) * 127.5).astype(np.uint8)
        cv2_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("image", cv2_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
