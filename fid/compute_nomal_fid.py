# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""Compute standard FID/KID scores using an ImageNet Inception v3 backbone."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import Inception_V3_Weights, inception_v3
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

IMAGE_EXTENSIONS: Tuple[str, ...] = (
	".png",
	".jpg",
	".jpeg",
	".tif",
	".tiff",
	".bmp",
)


@dataclass
class FeatureStats:
	"""Container object that keeps feature statistics for later reuse."""

	mean: np.ndarray
	cov: np.ndarray
	features: Optional[np.ndarray]


def collect_image_paths(directory: Path) -> List[Path]:
	"""Walk a directory tree and collect all supported image files."""

	if not directory.is_dir():
		raise FileNotFoundError(f"Directory not found: {directory}")

	paths = [
		path
		for path in sorted(directory.rglob("*"))
		if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
	]
	if not paths:
		raise RuntimeError(f"No supported image files found in {directory}")
	return paths


class ImageFolderDataset(Dataset):
	"""Dataset that reads images from disk and applies a deterministic transform."""

	def __init__(self, paths: Sequence[Path], transform: transforms.Compose):
		self.paths = list(paths)
		self.transform = transform

	def __len__(self) -> int:
		return len(self.paths)

	def __getitem__(self, index: int) -> torch.Tensor:
		path = self.paths[index]
		with Image.open(path) as img:
			image = img.convert("RGB")
		return self.transform(image)


def build_transform(
	mean: Sequence[float],
	std: Sequence[float],
	image_size: int,
) -> transforms.Compose:
	"""Assemble the preprocessing pipeline expected by torchvision Inception v3."""

	return transforms.Compose(
		[
			transforms.Resize(
				(image_size, image_size),
				interpolation=InterpolationMode.BILINEAR,
			),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		]
	)


def instantiate_inception_v3() -> Tuple[nn.Module, Sequence[float], Sequence[float], str]:
	"""Load an ImageNet pretrained Inception v3 backbone for feature extraction."""

	weights = Inception_V3_Weights.IMAGENET1K_V1
	model = inception_v3(weights=weights, aux_logits=False, transform_input=False)
	model.fc = nn.Identity()
	model.eval()
	mean = list(weights.meta["mean"])
	std = list(weights.meta["std"])
	source = "torchvision.models.inception_v3(IMAGENET1K_V1)"
	return model, mean, std, source


def compute_feature_stats(
	model: nn.Module,
	dataloader: DataLoader,
	device: torch.device,
	store: bool,
	desc: str,
) -> FeatureStats:
	"""Extract features for every image and compute mean/covariance statistics."""

	features_list: List[np.ndarray] = []
	with torch.no_grad():
		for batch in tqdm(dataloader, desc=desc, unit="batch"):
			batch = batch.to(device, non_blocking=True)
			feats = model(batch)
			if feats.ndim == 4:
				feats = nn.functional.adaptive_avg_pool2d(feats, (1, 1))
				feats = torch.flatten(feats, 1)
			feats = feats.cpu().numpy().astype(np.float64)
			features_list.append(feats)

	all_features = np.concatenate(features_list, axis=0)
	mean = np.mean(all_features, axis=0)
	cov = np.cov(all_features, rowvar=False)
	return FeatureStats(mean=mean, cov=cov, features=all_features if store else None)


def compute_fid(
	mu1: np.ndarray,
	sigma1: np.ndarray,
	mu2: np.ndarray,
	sigma2: np.ndarray,
) -> float:
	"""Compute Fréchet Inception Distance between two multivariate Gaussians."""

	diff = mu1 - mu2
	covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
	if not np.isfinite(covmean).all():
		offset = np.eye(sigma1.shape[0]) * 1e-6
		covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
	covmean = np.real_if_close(covmean)
	fid_value = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
	return float(fid_value)


def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
	"""Evaluate a third-degree polynomial kernel for KID computation."""

	gamma = 1.0 / x.shape[1]
	return (gamma * x @ y.T + 1.0) ** degree


def multi_subset_kid(
	real: np.ndarray,
	fake: np.ndarray,
	subset_size: int,
	n_subsets: int,
	seed: int,
) -> Tuple[float, float]:
	"""Compute KID mean and standard error across repeated subsets."""

	if real.shape[0] < subset_size or fake.shape[0] < subset_size:
		subset_size = min(real.shape[0], fake.shape[0])
		if subset_size < 2:
			raise RuntimeError("Not enough samples to compute KID")

	rng = np.random.default_rng(seed)
	estimates = []
	for _ in range(n_subsets):
		real_idx = rng.choice(real.shape[0], subset_size, replace=False)
		fake_idx = rng.choice(fake.shape[0], subset_size, replace=False)
		real_subset = real[real_idx]
		fake_subset = fake[fake_idx]

		k_rr = polynomial_kernel(real_subset, real_subset)
		k_ff = polynomial_kernel(fake_subset, fake_subset)
		k_rf = polynomial_kernel(real_subset, fake_subset)

		m = subset_size
		diag_rr = np.trace(k_rr)
		diag_ff = np.trace(k_ff)
		kid_est = (
			(k_rr.sum() - diag_rr) / (m * (m - 1))
			+ (k_ff.sum() - diag_ff) / (m * (m - 1))
			- 2.0 * k_rf.mean()
		)
		estimates.append(kid_est)

	estimates = np.array(estimates, dtype=np.float64)
	mean = float(estimates.mean())
	std = float(estimates.std(ddof=1) / math.sqrt(n_subsets))
	return mean, std


def parse_args() -> argparse.Namespace:
	"""Parse CLI arguments to configure the measurement."""

	parser = argparse.ArgumentParser(
		description=(
			"Compute standard FID (and optionally KID) using ImageNet Inception v3"
		),
	)
	parser.add_argument("real_dir", type=Path, help="Directory with real images")
	parser.add_argument("gen_dir", type=Path, help="Directory with generated images")
	parser.add_argument(
		"--batch-size",
		type=int,
		default=32,
		help="Batch size for feature extraction",
	)
	parser.add_argument(
		"--num-workers",
		type=int,
		default=4,
		help="Number of dataloader workers",
	)
	parser.add_argument(
		"--device",
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device for inference (cuda or cpu)",
	)
	parser.add_argument(
		"--image-size",
		type=int,
		default=299,
		help="Input resolution expected by Inception v3",
	)
	parser.add_argument(
		"--output-json",
		type=Path,
		default=Path("inception_fid.json"),
		help="Where to save the measurement summary",
	)
	parser.add_argument(
		"--data-volume",
		type=str,
		default=None,
		help=(
			"Optional host:container volume mapping string to record in the summary"
		),
	)
	parser.add_argument(
		"--compute-kid",
		action="store_true",
		help="Also compute Kernel Inception Distance",
	)
	parser.add_argument(
		"--kid-subset-size",
		type=int,
		default=1000,
		help="Subset size used per KID estimate",
	)
	parser.add_argument(
		"--kid-subset-count",
		type=int,
		default=100,
		help="Number of subsets for KID",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for KID sampling",
	)
	return parser.parse_args()


def main() -> None:
	"""Entrypoint for CLI usage."""

	args = parse_args()
	device = torch.device(args.device)

	model, mean, std, weight_source = instantiate_inception_v3()
	model.to(device)

	transform = build_transform(mean, std, args.image_size)

	real_paths = collect_image_paths(args.real_dir)
	gen_paths = collect_image_paths(args.gen_dir)

	real_loader = DataLoader(
		ImageFolderDataset(real_paths, transform),
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=device.type == "cuda",
	)
	gen_loader = DataLoader(
		ImageFolderDataset(gen_paths, transform),
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		pin_memory=device.type == "cuda",
	)

	real_stats = compute_feature_stats(
		model,
		real_loader,
		device,
		store=args.compute_kid,
		desc="Real",
	)
	gen_stats = compute_feature_stats(
		model,
		gen_loader,
		device,
		store=args.compute_kid,
		desc="Generated",
	)

	fid_value = compute_fid(
		real_stats.mean,
		real_stats.cov,
		gen_stats.mean,
		gen_stats.cov,
	)

	now_utc = datetime.now(timezone.utc)
	timestamp_suffix = now_utc.strftime("%Y%m%d_%H%M")

	result = {
		"fid": fid_value,
		"backbone": "inception_v3",
		"weights": weight_source,
		"num_real": len(real_paths),
		"num_generated": len(gen_paths),
		"image_size": args.image_size,
		"normalization_mean": mean,
		"normalization_std": std,
		"timestamp_utc": now_utc.isoformat(),
		"real_dir": str(args.real_dir.resolve()),
		"gen_dir": str(args.gen_dir.resolve()),
	}

	if args.data_volume:
		result["data_volume"] = args.data_volume

	if args.compute_kid:
		if real_stats.features is None or gen_stats.features is None:
			raise RuntimeError("Features not stored; cannot compute KID")
		kid_mean, kid_std = multi_subset_kid(
			real_stats.features,
			gen_stats.features,
			subset_size=args.kid_subset_size,
			n_subsets=args.kid_subset_count,
			seed=args.seed,
		)
		result["kid"] = kid_mean
		result["kid_std"] = kid_std

	output_path = args.output_json
	if output_path.suffix:
		output_path = output_path.with_name(
			f"{output_path.stem}_{timestamp_suffix}{output_path.suffix}"
		)
	else:
		output_path = output_path.with_name(f"{output_path.name}_{timestamp_suffix}")

	output_path.write_text(json.dumps(result, indent=2))

	print("=== Inception FID Results ===")
	print("Backbone       : inception_v3")
	print(f"Weights source : {weight_source}")
	print(f"Real images    : {len(real_paths)}")
	print(f"Generated imgs : {len(gen_paths)}")
	print(f"FID            : {fid_value:.6f}")
	if args.compute_kid:
		print(f"KID (mean)     : {kid_mean:.6f}")
		print(f"KID (std/sqrt(n)) : {kid_std:.6f}")
	print(f"Saved summary  : {output_path}")


if __name__ == "__main__":
	main()
