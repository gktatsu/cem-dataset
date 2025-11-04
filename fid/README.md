# FID / KID 計算ツールの使い方

このディレクトリには FID（Fréchet Inception Distance）および KID（Kernel Inception Distance）を計算するためのユーティリティスクリプトが含まれます。主に次の 2 つのスクリプトを提供しています。

- `compute_cem_fid.py` : CEM 事前学習済み ResNet50（CEM500K / CEM1.5M）を特徴抽出に用いる EM 画像向けの FID/KID 計算スクリプト。
- `compute_normal_fid.py` : torchvision の ImageNet 学習済み Inception v3 を用いる、一般的な FID/KID 計算スクリプト。

以下に依存ライブラリ、各スクリプトの概要・主なオプション・使い方例、差分（前処理の違い）をまとめます。

## 共通の前提（依存ライブラリ）
両スクリプトで必要な Python パッケージ:

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `tqdm`

未インストールの場合は、仮想環境を有効にした上で次を実行してください。

```bash
pip install torch torchvision numpy scipy tqdm
```

## compute_cem_fid.py（CEM ResNet50 を使う）

概要
- `compute_cem_fid.py` は CEM500K (MoCoV2) または CEM1.5M (SwAV) の事前学習済み ResNet50 を特徴抽出器として用い、2 つの EM 画像フォルダ間の FID を算出します。オプションで KID も推定できます。グレースケールの EM 画像を自動で 3 チャンネルに変換し、CEM 事前学習時と同じ前処理（リサイズ・正規化）を適用した上で 2048 次元のグローバル平均プーリング特徴を抽出します。

基本的な使い方

```bash
python fid/compute_cem_fid.py REAL_DIR GEN_DIR [オプション]
```

- `REAL_DIR`: 実画像が入ったディレクトリ
- `GEN_DIR`: 生成画像が入ったディレクトリ

主なオプション（抜粋）

| オプション | 既定値 | 説明 |
|---|---:|---|
| `--backbone {cem500k, cem1.5m}` | `cem500k` | 使用する CEM 事前学習モデル |
| `--batch-size INT` | `32` | 特徴抽出時のバッチサイズ |
| `--num-workers INT` | `4` | DataLoader のワーカープロセス数 |
| `--device` | 自動 (GPU があれば `cuda`) | 推論デバイス |
| `--image-size INT` | `224` | 入力をリサイズするサイズ（CEM と同一） |
| `--weights-path PATH` | なし | 手動でダウンロードした重みを指定する場合に使用 |
| `--download-dir PATH` | なし | 重みのキャッシュ先を指定する場合に使用（`TORCH_HOME` を参照） |
| `--output-json PATH` | `cem_fid.json` | 結果保存先（実行時にタイムスタンプが付与される場合あり） |
| `--compute-kid` | 無効 | 指定すると KID も計算（特徴を保存して推定） |
| `--kid-subset-size INT` | `1000` | KID サブセットあたりのサンプル数 |
| `--kid-subset-count INT` | `100` | KID のサブセット試行回数 |
| `--seed INT` | `42` | KID 用乱数シード |

出力

- コンソールに FID（および KID の場合は平均と標準誤差）を表示します。
- 指定した `--output-json` に測定結果（FID/KID、バックボーン、画像数、正規化パラメータ、UTC タイムスタンプ、入力ディレクトリ等）を JSON として保存します。

備考

- 初回実行時に Zenodo から事前学習重みを自動ダウンロードする実装になっている場合があります（ネットワークが無い環境では手動ダウンロードして `--weights-path` を指定してください）。

## compute_normal_fid.py（ImageNet Inception v3 を使う）

概要
- `compute_normal_fid.py` は torchvision の ImageNet 学習済み Inception v3 を用いて、実画像群と生成画像群の FID（およびオプションで KID）を算出するユーティリティです。標準的な Inception ベースの FID 評価を行います。

基本的な使い方

```bash
python fid/compute_normal_fid.py REAL_DIR GEN_DIR [オプション]
```

主なオプション（抜粋）

| オプション | 既定値 | 説明 |
|---|---:|---|
| `--batch-size INT` | `32` | 特徴抽出時のバッチサイズ |
| `--num-workers INT` | `4` | DataLoader のワーカープロセス数 |
| `--device` | 自動 (GPU があれば `cuda`) | 推論デバイス |
| `--image-size INT` | `299` | Inception v3 が期待する入力解像度（既定 299） |
| `--output-json PATH` | `inception_fid.json` | 結果保存先（タイムスタンプ付きにする挙動あり） |
| `--data-volume STR` | なし | 実行環境メモ（ホスト:コンテナ マウント等）を記録する文字列 |
| `--compute-kid` | 無効 | 指定すると KID も計算（特徴を保存して推定） |
| `--kid-subset-size INT` | `1000` | KID サブセットあたりのサンプル数 |
| `--kid-subset-count INT` | `100` | KID のサブセット試行回数 |
| `--seed INT` | `42` | KID 用乱数シード |

出力

- コンソールに FID（および KID の場合は平均と標準誤差）を表示します。
- 指定した `--output-json` に測定結果（FID/KID、バックボーン名、重み情報、画像数、正規化値、UTC タイムスタンプ、入力ディレクトリ等）を JSON として保存します。

## 前処理の違い（両スクリプトの差分）

- `compute_cem_fid.py` は CEM 用の ResNet50 を使うため、グレースケール EM 画像を 3 チャンネル化し、CEM 事前学習時と同一の正規化（平均・標準偏差）および解像度（224）を用います。出力特徴は ResNet50 のグローバル平均プーリング（2048 次元）です。
- `compute_normal_fid.py` は ImageNet 学習済み Inception v3 を用いるため、RGB 入力（グレースケール→RGB に変換して使用可能）を 299×299 にリサイズし、ImageNet の正規化を適用します。Inception の出力を用いて FID/KID を計算します。

## ベストプラクティス

- 評価対象の画像のみを含むディレクトリを指定してください（スクリプトは再帰的に画像を探索します）。
- GPU 利用時は `--batch-size` を増やすと高速化できますが、メモリに注意してください。
- KID を有効にする場合、`--kid-subset-size` と `--kid-subset-count` の値により計算コストが増加します。適切な値に調整してください。

## Docker 利用例（CEM ResNet50 の例）

```bash
sudo docker run --rm \
  -v /path/to/real_and_fake:/data \
  -v /path/to/weights:/weights \
  -v /path/to/save/results:/results \
  cem-fid \
  /data/real /data/gen \
  --backbone cem500k \
  --weights-path /weights/cem500k_mocov2_resnet50_200ep.pth.tar \
  --output-json /results/cem_fid.json \
  --data-volume /path/to/real_and_fake:/data
```

## 追加の注意

- ネットワークが使えない環境では、事前学習重みを手動でダウンロードして `--weights-path` に渡してください。
- 各スクリプトは実行時のメタ情報（タイムスタンプ、入力ディレクトリ、使用オプションなど）を JSON に保存するため、実験ログとして利用できます。

---

スクリプト本体:

- `fid/compute_cem_fid.py`
- `fid/compute_normal_fid.py`

詳細は各ソースコード内のコマンドライン引数説明を参照してください。
