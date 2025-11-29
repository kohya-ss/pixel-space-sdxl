# pixel-sdxl

ピクセル空間で Stable Diffusion XL (SDXL) を学習・推論するための実験的リポジトリです。SDXL の U-Net を活用しつつ、画像を直接 1024x1024 のピクセル空間で扱う構成を持っています。モデル学習に慣れた読者を想定し、必要に応じてスクリプトを書き換えられる前提で概要をまとめています。

## プロジェクト概要
- 16x16 もしくは 32x32 のパッチに分割した画像をパッチエンコーダで圧縮し、SDXL 由来の U-Net でノイズ予測（x0／velocity）を行い、パッチデコーダでフル解像度に戻すピクセル空間 SDXL 実装です。
- 学習済み SDXL をベースに、Encoder/Decoder を蒸留 → Encoder/Decoder 微調整 → 本体（SDXLUNetBody 含む）微調整の順に学習することを推奨しています。
- 重みの保存には PyTorch （optimizer の state を含むため）を利用し、`model_utils.py` 経由でテキストエンコーダ2種とピクセル U-Net をまとめて取り扱えます。最初にfinetuning元の latent SDXL チェックポイントをロードするときには safetensors 形式を利用します。

**Text Encoderの学習コードは書いてありますがテストされていません。必要に応じて適宜修正してください。**

## アーキテクチャ概要
- `sdxl_pixel_unet.py`  
  - 1024x1024 入力をパッチ化（16x16 または 32x32）→ パッチエンコーダで 64x64x640 もしくは 32x32x1280 表現へ圧縮。  
  - `sdxl_unet_base.py` に基づく U-Net 本体（ResNet + クロスアテンション Transformer）で条件付きノイズ推定を実施。  
  - パッチデコーダで 3ch 画像に復元し、アンパッチ化して元解像度を再構成。
- 条件情報  
  - テキスト: SDXL 互換の 2 系統の CLIP トークナイザ／テキストエンコーダ出力を結合。  
  - サイズ: 元解像度・クロップ位置・ターゲット解像度を埋め込み、テキスト条件と結合した ADM 条件 `y` として利用。
- 予測タイプは x0（生成画像）を基本とし、velocity 損失も選択可能です。LPIPS 損失による画質補助もオプションで追加できます。

## データセットとメタデータ
- メタデータ JSON は `image_path -> { "tags": "tag1, tag2, ...", "image_size": [width, height] }` 形式を想定しています。パスは OS に依存せず絶対パス推奨です（WSL2 では自動で Windows パスを `/mnt/` 形式に変換）。
- 画像は 32 の倍数にリサイズ／クロップされ、`MAX_PIXELS_PER_IMAGE=1024*1024` を超える場合はアスペクト比を保ったバケットに縮小されます。極端に小さい画像（768x768 相当未満）はスキップされます。
- DataLoader はアスペクト比バケットごとにバッチを組み、タグはランダムドロップ＋シャッフル後に 2 系統のトークナイザで 77 トークンに整形します。
- メタデータファイルは複数指定可能で、全てのエントリが結合されます。

### メタデータの例
```json
{
  "data/images/img1.jpg": {
    "tags": "1girl, highres, solo, bag, skirt, maple leaf, standing, best quality, general",
    "image_size": [1200, 800]
  },
  "data/images/img2.png": {
    "tags": "2girls, cityscape, night, lights, building, street, best quality, general",
    "image_size": [1920, 1080]
  }
}
```

最後の二つのタグはクオリティ、レーティングという前提でコードが記述されています。これらはドロップされません。必要ならば他のタグも同様に扱うようコードを修正してください。

### 依存環境（目安）
- Python 3.10–3.12、PyTorch (CUDA 推奨)、bitsandbytes、transformers、safetensors、Pillow、numpy、tensorboard、tqdm など。  
  LPIPS 損失を使う場合は `lpips` もインストールしてください。

`pip install -e .` として開発モードでインストールできます。

## 学習スクリプトと例
共通で `--metadata_files` にメタデータ JSON、`--base_resolution` に 32 または 64（パッチサイズ 16/32 が内部で選択）を渡します。チェックポイントは PyTorch で保存／読み込みしますので、拡張子は `.pt` や `.pth` としてください。

注：compileするとlossがおかしくなりましたので、現状では `--compile_model` を指定しないことを推奨します。

### 1. Encoder/Decoder 蒸留（教師: latent SDXL）、任意
`src/pixel_sdxl/train_encoder_decoder_distillation.py`  
- 目的: latent SDXL からパッチ Encoder/Decoder を蒸留してピクセル空間の初期重みを作る。  
- 主要引数: `--teacher_checkpoint`（latent SDXL ckpt）, `--train_encoder`（指定時は Encoder、未指定なら Decoder を更新）, `--lr`, `--steps`, `--grad_accum_steps`, `--uniform_sampling`, `--gradient_checkpointing`, `--compile_model`。  
- 例:
```bash
python -m pixel_sdxl.train_encoder_decoder_distillation ^
  --metadata_files data/meta.json ^
  --checkpoint checkpoints/sdxl_latent_or_pixel_base.safetensors ^
  --teacher_checkpoint checkpoints/sdxl_latent_teacher.safetensors ^
  --save_path save/path/encdec_distill.pth ^
  --base_resolution 64 ^
  --batch_size 1 --steps 2000 --lr 1e-4 --train_encoder
```

Encoder/Decoder の両方を学習したい場合は、上記スクリプトを `--train_encoder` 有無で 2 回実行してください。初回にEncoderを学習するなら、初回はlatent SDXLを、2 回目はEncoderを蒸留済みのチェックポイントを、checkpoint に指定します。 

### 2. Encoder/Decoder のみの調整、推奨
3.の`train.py`で本体の学習率を 0 にして、 Encoder/Decoder のみを微調整します。

### 3. 本体学習（単一 GPU）
`src/pixel_sdxl/train.py`  
- 目的: テキストエンコーダ（任意）と SDXL U-Net 本体を含むピクセル U-Net を学習。  
- 主要引数: `--checkpoint`, `--save_path`, `--steps`, `--initial_steps`, `--grad_accum_steps`, `--lr`, `--lr_sdxl_unet_body`, `--lr_text_encoder1/2`, `--clip_grad_norm`, `--velocity_loss`, `--lpips_lambda`, `--uniform_sampling`, `--sample_every`, `--sample_prompts`, `--gradient_checkpointing`, `--compile_model`.  
- 例:
```bash
python -m pixel_sdxl.train ^
  --metadata_files data/meta.json ^
  --checkpoint save/path/encdec_pretrained.pth ^
  --save_path save/path/full_train.pth ^
  --base_resolution 64 ^
  --batch_size 1 --steps 40000 --grad_accum_steps 2 ^
  --lr 1e-4 --lr_sdxl_unet_body 1e-5 --lr_text_encoder1 0 --lr_text_encoder2 0 ^
  --lpips_lambda 0.05 --sample_every 1000 --sample_prompts "1girl, highres, solo"
```

### 4. 2 GPU 分業（テキストエンコーダ/UNet 分割）
`src/pixel_sdxl/train_multi_gpu.py`  
- テキストエンコーダをセカンダリ GPU、U-Net をメイン GPU に配置するシンプルな 2 GPU 版。テキストエンコーダの微調整は非対応（`lr_text_encoder*` は 0 固定）。
- LPIPS 損失が NaN になる不具合は修正済みです。

### 5. DDP 学習
`src/pixel_sdxl/train_ddp.py`  
- torch.distributed を使った DDP 版。`--backend` に Windows なら `gloo`, Linux/NCCL 環境なら `nccl`。DataLoader 用に `--num_workers`, `--pin_memory` を追加。
- 起動例（単ノード）：`torchrun --nproc_per_node=4 -m pixel_sdxl.train_ddp --metadata_files ...`
- Ctrl+C で中断した場合もチェックポイントを保存しますが、時間がかかると保存が終わる前に強制終了される可能性がありますので、頻回に `--save_interval` を指定することを推奨します。

### 6. EMA 生成
`src/pixel_sdxl/post_hoc_ema.py`  
- 既存 safetensors から事後的に EMA 重みを生成して保存します。

### 7. 推論
`src/pixel_sdxl/inference.py`  
- 主要引数: `--checkpoint`, `--prompt`, `--negative_prompt`, `--height/--width`（パッチサイズの倍数必須）, `--num_steps`, `--schedule {linear,flow_shift}`, `--cfg_scale`, `--flow_shift`, `--precision`.  
- 例:
```bash
python -m pixel_sdxl.inference ^
  --checkpoint save/path/full_train.pth ^
  --prompt "1girl, highres, solo" ^
  --height 1024 --width 1024 --num_steps 30 --cfg_scale 7.5
```

### 参考：実験的な学習設定

- 蒸留
  - 学習率 2e-3、バッチサイズ 2、勾配蓄積 4、uniform サンプリング、x0 損失、ステップ数（勾配累積含まず） 2,000 程度
- Encoder/Decoder 微調整
  - 学習率 2e-3、バッチサイズ 4 、勾配蓄積 8、x0 損失、uniform サンプリング、ステップ数 10,000
- 本体学習
  - 学習率 5e-4、U-Net 本体 5e-5、バッチサイズ 4、勾配蓄積 12、clip_grad_norm 1.0、lpips_lambda 0.05、velocity 損失、logit norm サンプリング、ステップ数 300,000～

## TIPS
- 推奨学習順序: Encoder/Decoder 蒸留は任意 → Encoder/Decoder 追加学習 → 本体学習。
- パッチサイズ: `--base_resolution 64` はパッチ 16、`--base_resolution 32` はパッチ 32 を選びます。学習・推論ともに高さ・幅はパッチサイズの倍数にしてください。
- 時間ステップ: デフォルトは logit-normal サンプリング（JiTの論文と同じ補正を行います）、`--uniform_sampling` で一様。
- LPIPS: `--lpips_lambda` を 0.05–0.2 程度で試すと x0 予測の画質が安定するかもしれませんが、メモリを消費し学習時間もかかります。
- ログとサンプリング: `--sample_every` と `--sample_prompts` で学習中のサンプル画像を定期生成できます。TensorBoard ログは `logs/<timestamp>` に出力されます。
- チェックポイント運用: `--save_interval` でステップ間保存、復帰時に `--no_restore_optimizer` でオプティマイザを無視可能。

## ライセンス
本リポジトリは `LICENSE` の内容に従います。モデルやデータセットの利用規約は各自の責任でご確認ください。
