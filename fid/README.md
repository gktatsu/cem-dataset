# `compute_cem_fid.py` の使い方

## 概要
`compute_cem_fid.py` は、CEM500K (MoCoV2) または CEM1.5M (SwAV) の事前学習済み ResNet50 を特徴抽出器として用い、2 つの電子顕微鏡 (EM) 画像フォルダ間の Fréchet Inception Distance (FID) とオプションで Kernel Inception Distance (KID) を算出するスクリプトです。グレースケール EM 画像を自動で 3 チャンネルに変換し、CEM 事前学習時と同一の前処理を適用してから 2048 次元のグローバルプーリング特徴を抽出します。

## 必要なライブラリ
以下の Python パッケージが必要です。

- `torch`
- `torchvision`
- `numpy`
- `scipy`
- `tqdm`

未インストールの場合は、仮想環境を有効化した上で次を実行してください。

```bash
pip install torch torchvision numpy scipy tqdm
```

## 基本的な使い方

```bash
python fid/compute_cem_fid.py REAL_DIR GEN_DIR [オプション]
```

- `REAL_DIR`: 実写 EM 画像が入ったフォルダへのパス
- `GEN_DIR` : 生成 EM 画像が入ったフォルダへのパス

計算後、FID がコンソールに表示され、同じ内容が JSON ファイルとして保存されます (既定は `cem_fid.json`)。

## 主なオプション
| オプション | 既定値 | 説明 |
|-----------|-------|------|
| `--backbone {cem500k, cem1.5m}` | `cem500k` | 使用する事前学習モデルを選択します。 |
| `--batch-size INT` | `32` | 特徴抽出時のバッチサイズ。GPU メモリに応じて調整してください。 |
| `--num-workers INT` | `4` | DataLoader のワーカープロセス数。I/O がボトルネックの場合に増やします。 |
| `--device {cuda,cpu}` | 利用可能 GPU に応じ自動 | 推論を行うデバイス。CPU で実行する場合は明示的に `--device cpu` を指定します。 |
| `--image-size INT` | `224` | 入力画像をリサイズする一辺の長さ。CEM 事前学習と同じ 224px を推奨します。 |
| `--weights-path PATH` | なし | Zenodo から手動でダウンロードしたチェックポイントを指定します。 |
| `--download-dir PATH` | なし | 重みファイルのキャッシュ先ディレクトリ。既定では `TORCH_HOME` を利用します。 |
| `--output-json PATH` | `cem_fid.json` | 結果を書き出す JSON ファイルパス。既存ファイルは上書きされます。 |
| `--compute-kid` | 無効 | 指定すると KID も計算します (小規模データセットで有用)。 |
| `--kid-subset-size INT` | `1000` | KID 推定で使用するサブセットのサンプル数。 |
| `--kid-subset-count INT` | `100` | KID 推定でサンプリングするサブセットの回数。 |
| `--seed INT` | `42` | KID サンプリングの乱数シード。 |

## 出力
- **コンソール**: FID (および KID を有効化した場合は平均と標準誤差) を整形表示します。
- **JSON**: 指定した `--output-json` に、FID/KID の値、使用したバックボーン、画像枚数、正規化パラメータなどのメタデータを保存します。

## 前処理と特徴抽出の流れ
1. 画像を読み込み、3 チャンネルの RGB へ変換。
2. 224×224 へバイリニアリサイズ後、再び 1 チャンネルへ変換しテンソル化。
3. CEM 事前学習時の平均・標準偏差で正規化。
4. ResNet50 のグローバル平均プーリング出力 (2048 次元) を抽出。
5. 実画像と生成画像の特徴分布から平均・共分散を算出し、FID を計算。
6. `--compute-kid` 指定時は、同特徴を用いて KID をサブセット平均で推定。

## 事前学習済み重み
- 初回実行時に Zenodo から自動ダウンロードします。
  - CEM500K (MoCoV2): `https://zenodo.org/record/6453140/...`
  - CEM1.5M (SwAV): `https://zenodo.org/record/6453160/...`
- ネットワークが制限されている場合は、ブラウザでダウンロードし `--weights-path` にファイルパスを指定してください。

## ベストプラクティス
- フォルダ内にサブフォルダがある場合でも再帰的に画像を探索します。評価したい画像のみが集まったディレクトリを指定してください。
- GPU 実行時は `--batch-size` を増やすことで処理時間を短縮できますが、メモリ不足に注意してください。
- `--compute-kid` はサンプル数が少ない場合に有益ですが、計算コストが増えるため必要に応じて有効化してください。

## 参考
スクリプト本体は `fid/compute_cem_fid.py` にあります。詳細なコマンドライン仕様や実装内容についてはソースコードを参照してください。
