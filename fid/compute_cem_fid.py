#!/usr/bin/env python3
"""CEMで事前学習したResNet50を用いてFIDとKIDを計算するスクリプト."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from scipy import linalg
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

MODEL_CONFIGS: Dict[str, Dict[str, object]] = {
    "cem500k": {
        "pretraining": "mocov2",
        "url": (
            "https://zenodo.org/record/6453140/files/"
            "cem500k_mocov2_resnet50_200ep.pth.tar?download=1"
        ),
        "mean": 0.57287007,
        "std": 0.12740536,
    },
    "cem1.5m": {
        "pretraining": "swav",
        "url": (
            "https://zenodo.org/record/6453160/files/"
            "cem15m_swav_resnet50_200ep.pth.tar?download=1"
        ),
        "mean": 0.575710,
        "std": 0.127650,
    },
}

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
    """特徴統計量を保持するコンテナ.

    Attributes:
        mean (np.ndarray): 特徴ベクトルの平均値.
        cov (np.ndarray): 特徴ベクトルの共分散行列.
        features (Optional[np.ndarray]): 必要に応じて保存された全特徴行列.
    """

    mean: np.ndarray
    cov: np.ndarray
    features: Optional[np.ndarray]


def load_python_module(name: str, path: Path):
    """指定したファイルからPythonモジュールを動的に読み込む.

    Args:
        name (str): 読み込み後に利用するモジュール名.
        path (Path): 読み込み対象となるPythonファイルのパス.

    Returns:
        ModuleType: 読み込まれたモジュールオブジェクト.

    Raises:
        ImportError: モジュール仕様の取得または読み込みに失敗した場合.
    """

    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


def find_repo_root(script_path: Path) -> Path:
    """pretrainingディレクトリを手掛かりにリポジトリルートを探索する.

    Args:
        script_path (Path): 実行中スクリプトの絶対パス.

    Returns:
        Path: pretrainingディレクトリを含む最上位ディレクトリ.
    """

    for candidate in [script_path.parent, *script_path.parents]:
        if (candidate / "pretraining").is_dir():
            return candidate
    return script_path.parent


def get_mocov2_resnet50_class(repo_dir: Path):
    """MoCo v2版ResNet50クラスをリポジトリ内から読み込む.

    Args:
        repo_dir (Path): リポジトリのルートディレクトリ.

    Returns:
        Callable[..., nn.Module]: ResNet50を生成するコンストラクタ.
    """

    module = load_python_module(
        "cem_mocov2_resnet", repo_dir / "pretraining" / "mocov2" / "resnet.py"
    )
    return module.resnet50  # type: ignore[attr-defined]


def get_swav_resnet50_class(repo_dir: Path):
    """SwAV版ResNet50クラスをリポジトリ内から読み込む.

    Args:
        repo_dir (Path): リポジトリのルートディレクトリ.

    Returns:
        Callable[..., nn.Module]: ResNet50を生成するコンストラクタ.
    """

    module = load_python_module(
        "cem_swav_resnet",
        repo_dir / "pretraining" / "swav" / "models" / "resnet.py",
    )
    return module.resnet50  # type: ignore[attr-defined]


def collect_image_paths(directory: Path) -> List[Path]:
    """指定ディレクトリ以下の画像ファイルパスを辞書順に収集する.

    Args:
        directory (Path): 画像を探索するルートディレクトリ.

    Returns:
        List[Path]: 見つかった画像ファイルのパス一覧.

    Raises:
        FileNotFoundError: 指定ディレクトリが存在しない場合.
        RuntimeError: 対応する画像ファイルが1枚も見つからない場合.
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
    """ディスク上の画像群を一括読み込みする軽量データセット.

    Attributes:
        paths (List[Path]): 利用する画像パスのリスト.
        transform (transforms.Compose): 画像読み込み時に適用する前処理.
    """

    def __init__(self, paths: Sequence[Path], transform: transforms.Compose):
        self.paths = list(paths)
        self.transform = transform

    def __len__(self) -> int:
        """データセット内の画像枚数を返す."""

        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        """指定インデックスの画像を読み込み前処理したテンソルを返す.

        Args:
            index (int): 取得する画像のインデックス.

        Returns:
            torch.Tensor: 正規化済みの画像テンソル.
        """

        path = self.paths[index]
        with Image.open(path) as img:
            image = img.convert("RGB")
        return self.transform(image)


def build_transform(
    mean: float,
    std: float,
    image_size: int,
) -> transforms.Compose:
    """学習時と同じ決定的な前処理パイプラインを構築する.

    Args:
        mean (float): グレースケール正規化の平均値.
        std (float): グレースケール正規化の標準偏差.
        image_size (int): リサイズ後の一辺の長さ.

    Returns:
        transforms.Compose: 連続適用される画像前処理.
    """

    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BILINEAR,
            ),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[mean], std=[std]),
        ]
    )


def load_checkpoint(
    url: str,
    download_dir: Optional[Path] = None,
    local_path: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """事前学習済み重みを取得しチェックポイントを読み込む.

    Args:
        url (str): 重みファイルを取得するURL.
        download_dir (Optional[Path]): ダウンロードキャッシュ先のディレクトリ.
        local_path (Optional[Path]): 既存のローカル重みファイルのパス.

    Returns:
        Dict[str, torch.Tensor]: 読み込まれたチェックポイントの状態辞書.

    Raises:
        FileNotFoundError: 指定したローカルファイルが存在しない場合.
        RuntimeError: ダウンロードに失敗した場合.
    """

    if local_path is not None:
        if not local_path.is_file():
            raise FileNotFoundError(f"Weights file not found: {local_path}")
        return torch.load(local_path, map_location="cpu")

    model_dir = None if download_dir is None else str(download_dir)
    try:
        checkpoint = torch.hub.load_state_dict_from_url(
            url,
            map_location="cpu",
            model_dir=model_dir,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to download pretrained weights. Download them manually "
            "from the Zenodo URL and pass the local path via --weights-path."
        ) from exc
    return checkpoint


def find_cached_weights(url: str, repo_root: Path) -> Optional[Path]:
    """既存の重みファイルがローカルに存在する場合はそのパスを返す."""

    parsed = urlparse(url)
    filename = Path(parsed.path).name
    if not filename:
        return None
    candidate = repo_root / "fid" / "weights" / filename
    return candidate if candidate.is_file() else None


def extract_mocov2_backbone(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """MoCo v2のエンコーダ重みからバックボーン部分のみを抽出する.

    Args:
        state_dict (Dict[str, torch.Tensor]): MoCo v2のチェックポイントstate_dict.

    Returns:
        Dict[str, torch.Tensor]: バックボーンに対応するパラメータのみを含む辞書.

    Raises:
        RuntimeError: エンコーダの重みが見つからない場合.
    """

    prefixes = [
        "module.encoder_q.",
        "encoder_q.",
        "module.encoder.",
        "encoder.",
        "",
    ]

    for prefix in prefixes:
        backbone: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            if key.startswith(prefix + "fc"):
                continue
            new_key = key[len(prefix):]
            backbone[new_key] = value
        if backbone:
            return backbone

    raise RuntimeError("No encoder_q weights found in checkpoint")


def extract_swav_backbone(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """SwAVのチェックポイントからバックボーン重みを抽出する.

    Args:
        state_dict (Dict[str, torch.Tensor]): SwAVのチェックポイントstate_dict.

    Returns:
        Dict[str, torch.Tensor]: バックボーン層のみを保持したstate_dict.

    Raises:
        RuntimeError: バックボーン重みが見つからない場合.
    """

    prefixes = ["module.", ""]

    for prefix in prefixes:
        backbone: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            if not key.startswith(prefix):
                continue
            clean = key[len(prefix):]
            if clean.startswith("projection_head"):
                continue
            if clean.startswith("prototypes"):
                continue
            backbone[clean] = value
        if backbone:
            return backbone

    raise RuntimeError("No backbone weights found in checkpoint")


class CEMFeatureExtractor(nn.Module):
    """CEMバックボーンを包み込み特徴ベクトルを直接返すユーティリティ."""

    def __init__(self, backbone: nn.Module, variant: str):
        super().__init__()
        self.backbone = backbone
        self.variant = variant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """特徴抽出の前向き計算を実行する.

        Args:
            x (torch.Tensor): 前処理済み入力画像のバッチ.

        Returns:
            torch.Tensor: プーリング後の特徴ベクトル.
        """

        if self.variant == "cem500k":
            model = self.backbone
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.maxpool(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            return x

        features = self.backbone.forward_backbone(x)
        if features.ndim == 4:
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
        return features


def instantiate_backbone(
    variant: str,
    repo_root: Path,
    weights_path: Optional[Path],
    download_dir: Optional[Path],
) -> Tuple[CEMFeatureExtractor, float, float, str]:
    """指定バリアントのCEMバックボーンを初期化する.

    Args:
        variant (str): 使用するバックボーンの識別子.
        repo_root (Path): リポジトリルートへのパス.
        weights_path (Optional[Path]): ローカルに保存された重みのパス.
        download_dir (Optional[Path]): 重みをダウンロードする場合の保存先.

    Returns:
        Tuple[CEMFeatureExtractor, float, float, str]: 特徴抽出器、正規化平均、標準偏差、
            重み取得元をまとめたタプル.

    Raises:
        RuntimeError: 重みの読み込みに不整合がある場合.
    """

    conf = MODEL_CONFIGS[variant]
    if weights_path is None:
        cached_path = find_cached_weights(conf["url"], repo_root)
        if cached_path is not None:
            weights_path = cached_path

    checkpoint = load_checkpoint(
        url=conf["url"],
        download_dir=download_dir,
        local_path=weights_path,
    )

    state_dict = checkpoint.get("state_dict", checkpoint)
    norms = checkpoint.get("norms")
    mean = float(norms[0]) if norms is not None else float(conf["mean"])
    std = float(norms[1]) if norms is not None else float(conf["std"])

    if conf["pretraining"] == "mocov2":
        resnet50 = get_mocov2_resnet50_class(repo_root)
        backbone_model = resnet50(in_channels=1)
        backbone_model.fc = nn.Identity()
        backbone_weights = extract_mocov2_backbone(state_dict)
    else:
        resnet50 = get_swav_resnet50_class(repo_root)
        backbone_model = resnet50(
            normalize=False,
            output_dim=0,
            hidden_mlp=0,
            nmb_prototypes=0,
        )
        backbone_weights = extract_swav_backbone(state_dict)

    missing, unexpected = backbone_model.load_state_dict(
        backbone_weights,
        strict=False,
    )
    unexpected = set(unexpected)
    missing = [key for key in missing if not key.startswith("fc.")]
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys when loading weights: {unexpected}"
        )
    if missing:
        raise RuntimeError(f"Missing keys when loading weights: {missing}")

    backbone_model.eval()
    feature_extractor = CEMFeatureExtractor(backbone_model, variant)
    feature_extractor.eval()

    source = str(weights_path) if weights_path else str(conf["url"])
    return feature_extractor, mean, std, source


def compute_feature_stats(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    store: bool,
    desc: str,
) -> FeatureStats:
    """モデルで特徴を計算し平均と共分散を推定する.

    Args:
        model (nn.Module): 特徴抽出に使用するモデル.
        dataloader (DataLoader): 入力画像を供給するデータローダー.
        device (torch.device): 推論を実行するデバイス.
        store (bool): KID計算向けに全特徴を保存するかどうか.
        desc (str): tqdmに表示する進捗メッセージ.

    Returns:
        FeatureStats: 計算済みの統計量と必要に応じて特徴行列.
    """

    features_list: List[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc, unit="batch"):
            batch = batch.to(device, non_blocking=True)
            feats = model(batch)
            feats = feats.cpu().numpy().astype(np.float64)
            features_list.append(feats)

    all_features = np.concatenate(features_list, axis=0)
    mean = np.mean(all_features, axis=0)
    cov = np.cov(all_features, rowvar=False)
    return FeatureStats(
        mean=mean,
        cov=cov,
        features=all_features if store else None,
    )


def compute_fid(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
) -> float:
    """2つのガウス分布間のFréchet距離(FID)を計算する.

    Args:
        mu1 (np.ndarray): 実画像側の特徴平均ベクトル.
        sigma1 (np.ndarray): 実画像側の特徴共分散行列.
        mu2 (np.ndarray): 生成画像側の特徴平均ベクトル.
        sigma2 (np.ndarray): 生成画像側の特徴共分散行列.

    Returns:
        float: 算出されたFID値.
    """

    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma1 + offset) @ (sigma2 + offset))
    covmean = np.real_if_close(covmean)
    fid_value = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid_value)


def polynomial_kernel(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3,
) -> np.ndarray:
    """KID計算で用いる3次の多項式カーネルを評価する.

    Args:
        x (np.ndarray): 1組目の特徴ベクトル群.
        y (np.ndarray): 2組目の特徴ベクトル群.
        degree (int): 多項式カーネルの次数.

    Returns:
        np.ndarray: カーネル行列.
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
    """複数サブセットで評価するKIDの不偏推定量を計算する.

    Args:
        real (np.ndarray): 実画像から得た特徴行列.
        fake (np.ndarray): 生成画像から得た特徴行列.
        subset_size (int): 各サブセットに抽出するサンプル数.
        n_subsets (int): サブセットの繰り返し回数.
        seed (int): 乱数シード値.

    Returns:
        Tuple[float, float]: KIDの平均値と標準誤差.

    Raises:
        RuntimeError: 十分なサンプルがなくKIDを計算できない場合.
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
    """コマンドライン引数を解析し名前空間として返す.

    Returns:
        argparse.Namespace: 解析済み引数を格納した名前空間.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Compute FID (and optionally KID) using CEM pretrained ResNet50"
        ),
    )
    parser.add_argument(
        "real_dir",
        type=Path,
        help="Directory with real images",
    )
    parser.add_argument(
        "gen_dir",
        type=Path,
        help="Directory with generated images",
    )
    parser.add_argument(
        "--backbone",
        choices=list(MODEL_CONFIGS.keys()),
        default="cem500k",
        help="Which pretrained backbone to use",
    )
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
        default=224,
        help="Input resolution (matches CEM pretraining)",
    )
    parser.add_argument(
        "--weights-path",
        type=Path,
        default=None,
        help="Path to a local checkpoint downloaded from Zenodo",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=None,
        help="Directory to cache downloaded weights",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("cem_fid.json"),
        help="Where to save the measurement summary",
    )
    parser.add_argument(
        "--data-volume",
        type=str,
        default=None,
        help=(
            "Optional host:container volume mapping string to record "
            "in the summary"
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
    """スクリプトのエントリーポイント."""

    args = parse_args()
    script_path = Path(__file__).resolve()
    repo_root = find_repo_root(script_path)
    device = torch.device(args.device)

    model, mean, std, weight_source = instantiate_backbone(
        variant=args.backbone,
        repo_root=repo_root,
        weights_path=args.weights_path,
        download_dir=args.download_dir,
    )
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
        "backbone": args.backbone,
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
        output_path = output_path.with_name(
            f"{output_path.name}_{timestamp_suffix}"
        )

    output_path.write_text(json.dumps(result, indent=2))

    print("=== CEM FID Results ===")
    print(f"Backbone       : {args.backbone}")
    print(f"Weights source : {weight_source}")
    print(f"Real images    : {len(real_paths)}")
    print(f"Generated imgs : {len(gen_paths)}")
    print(f"FID            : {fid_value:.6f}")
    if args.compute_kid:
        print(f"KID (mean)     : {kid_mean:.6f}")
        print(f"KID (std/sqrt(n)) : {kid_std:.6f}")
    print(f"Saved summary  : {output_path}")
    print("Download weights manually if needed:")
    print(MODEL_CONFIGS[args.backbone]["url"])


if __name__ == "__main__":
    main()
