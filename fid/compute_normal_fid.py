# SPDX-License-Identifier: MIT
#!/usr/bin/env python3
"""ImageNet学習済みInception v3を用いてFID/KIDを算出するユーティリティ。

このモジュールは実画像群と生成画像群の特徴統計量を比較し、Fréchet
Inception Distance (FID) と Kernel Inception Distance (KID) を測定するための
コマンドラインインターフェースを提供する。
"""

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
    """特徴統計量を保持し再利用できるようにするデータコンテナ。

    Attributes:
        mean (np.ndarray): 特徴ベクトルの平均。
        cov (np.ndarray): 特徴ベクトルの分散共分散行列。
        features (Optional[np.ndarray]): 必要に応じて保存された生の特徴行列。
    """

    mean: np.ndarray
    cov: np.ndarray
    features: Optional[np.ndarray]


def collect_image_paths(directory: Path) -> List[Path]:
    """サポート対象画像を再帰的に探索してパス一覧を取得する。

    Args:
        directory (Path): 画像を探索するルートディレクトリ。

    Returns:
        List[Path]: ソート済みの画像ファイルパス一覧。

    Raises:
        FileNotFoundError: 指定ディレクトリが存在しない場合。
        RuntimeError: 画像ファイルが1枚も見つからない場合。
    """

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
    """ディスク上の画像を読み込み決定的な前処理を施すデータセット。

    Args:
        paths (Sequence[Path]): 読み込む画像ファイルのパス列。
        transform (transforms.Compose): 画像に適用する前処理パイプライン。
    """

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
    """Inception v3で期待される前処理パイプラインを構築する。

    Args:
        mean (Sequence[float]): 正規化時に使用するチャネルごとの平均。
        std (Sequence[float]): 正規化時に使用するチャネルごとの標準偏差。
        image_size (int): リサイズ後の正方形画像サイズ。

    Returns:
        transforms.Compose: 前処理を順次適用するコンポーズオブジェクト。
    """

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
    """特徴抽出用にImageNet学習済みInception v3を初期化する。

    Returns:
        Tuple[nn.Module, Sequence[float], Sequence[float], str]:
            モデル本体、正規化平均、正規化標準偏差、学習済み重みの出典。
    """

    weights = Inception_V3_Weights.IMAGENET1K_V1
    model = inception_v3(weights=weights, transform_input=False)
    model.fc = nn.Identity()
    if hasattr(model, "aux_logits"):
        model.aux_logits = False
    model.eval()
    meta = getattr(weights, "meta", {}) or {}
    mean = list(meta.get("mean", (0.485, 0.456, 0.406)))
    std = list(meta.get("std", (0.229, 0.224, 0.225)))
    source = "torchvision.models.inception_v3(IMAGENET1K_V1)"
    return model, mean, std, source


def compute_feature_stats(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    store: bool,
    desc: str,
) -> FeatureStats:
    """全画像の特徴を抽出し平均および共分散を計算する。

    Args:
        model (nn.Module): 特徴抽出に用いるモデル。
        dataloader (DataLoader): 画像のテンソルを供給するデータローダ。
        device (torch.device): 推論に使用するデバイス。
        store (bool): KID計算用に特徴を保持するかどうか。
        desc (str): 進捗バーに表示する説明テキスト。

    Returns:
        FeatureStats: 計算済みの特徴統計量。
    """

    features_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            batch = batch.to(device, non_blocking=True)
            feats = model(batch)
            # 特徴マップとして出力された場合は空間平均でベクトルに変換する。
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
    """2つの多変量ガウス分布間でFréchet距離を算出する。

    Args:
        mu1 (np.ndarray): 実データの平均ベクトル。
        sigma1 (np.ndarray): 実データの共分散行列。
        mu2 (np.ndarray): 生成データの平均ベクトル。
        sigma2 (np.ndarray): 生成データの共分散行列。

    Returns:
        float: FIDスコア。
    """

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        # 数値的不安定性が発生した場合はバイアスを足して安定化する。
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    covmean = np.real_if_close(covmean)
    fid_value = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid_value)


def polynomial_kernel(x: np.ndarray, y: np.ndarray, degree: int = 3) -> np.ndarray:
    """KID計算に用いる3次多項式カーネルを評価する。

    Args:
        x (np.ndarray): サンプル行列A。
        y (np.ndarray): サンプル行列B。
        degree (int): 多項式カーネルの次数。

    Returns:
        np.ndarray: カーネル値の行列。
    """

    gamma = 1.0 / x.shape[1]
    return (gamma * x @ y.T + 1.0) ** degree


def multi_subset_kid(
    real: np.ndarray,
    fake: np.ndarray,
    subset_size: int,
    n_subsets: int,
    seed: int,
) -> Tuple[float, float]:
    """サブセットを繰り返し抽出してKIDの平均と標準誤差を求める。

    Args:
        real (np.ndarray): 実データの特徴行列。
        fake (np.ndarray): 生成データの特徴行列。
        subset_size (int): 1回の推定で使用するサンプル数。
        n_subsets (int): 推定を繰り返すサブセット数。
        seed (int): サンプリングの乱数シード。

    Returns:
        Tuple[float, float]: KID平均値と標準誤差。

    Raises:
        RuntimeError: 十分なサンプルが確保できない場合。
    """

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
    """測定の実行に必要なコマンドライン引数を解析する。

    Returns:
        argparse.Namespace: パース済み引数を格納した名前空間。
    """

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
    """コマンドライン実行時のエントリーポイント。"""

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
