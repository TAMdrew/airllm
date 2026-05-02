このドキュメントは日本語版です。[English version](README.md)

# 🌬️ AirLLM v3.0

> わずか4GBのVRAMで、あらゆる大規模言語モデル（LLM）をあらゆるGPUで実行。

[![PyPI](https://img.shields.io/pypi/v/airllm)](https://pypi.org/project/airllm/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![Downloads](https://static.pepy.tech/personalized-badge/airllm?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads)](https://pepy.tech/project/airllm)

## 目次

- [AirLLMとは？](#-airllmとは)
- [クイックスタート](#-クイックスタート)
- [対応モデル](#-対応モデル)
- [VRAM要件](#-vram要件)
- [インストール](#-インストール)
- [使用例](#-使用例)
- [設定](#-設定)
- [アーキテクチャ](#-アーキテクチャ)
- [テスト](#-テスト)
- [貢献](#-貢献)
- [ライセンス](#-ライセンス)
- [謝辞](#-謝辞)

## ✨ AirLLMとは？

AirLLMは、**完全にローカルで無料のLLM推論**を可能にします。APIキー、サブスクリプション、クラウドへの依存は一切不要です。モデルは完全にあなたのハードウェア上で実行されます。

AirLLMは、わずか4GBのVRAMを搭載したコンシューマー向けGPUで70B（700億）以上のパラメータを持つLLMの実行を可能にする**レイヤーごとの推論エンジン**です。
モデル全体をGPUメモリにロードするのではなく、AirLLMはトランスフォーマーのレイヤーを1つずつロードし、フォワードパスを実行した後、次のレイヤーをロードする前にそれをオフロードします。

**主な機能:**

- **100%ローカル** — クラウドAPI、サブスクリプション、初回ダウンロード後のインターネット接続は不要です。
- **モデルの劣化なし** — 精度の低下なしにフル精度（FP16）の重みで実行します。
- **オプションの量子化** — 4-bitおよび8-bitのブロック単位の圧縮により、最大3倍高速な推論を実現します。
- **33のアーキテクチャに対応** — Llama 3やGemma 4からDeepSeek R1、Qwen 3まで。
- **macOSサポート** — MLXバックエンドによるApple Siliconでの推論。
- **プリフェッチ** — ディスクI/OとGPU計算をオーバーラップさせ、スループットを向上させます。
- **ドロップインAPI** — HuggingFace `transformers`の`GenerationMixin`インターフェースに準拠しています。

## 🏠 なぜローカルなのか？

AirLLMを使用してローカルでLLMを実行することは、クラウドAPIサービスと比較して大きな利点があります：

|                  | **AirLLM (ローカル)**                      | **クラウドAPI (OpenAI, Anthropicなど)**       |
| ---------------- | ------------------------------------------ | --------------------------------------------- |
| **コスト**       | ハードウェア導入後は無料                   | 月額20〜200ドル以上、またはトークンごとの料金 |
| **プライバシー** | データは決してマシンから外部に出ない       | データはサードパーティのサーバーに送信される  |
| **可用性**       | オフラインで動作、レート制限なし           | インターネットが必要、障害の影響を受ける      |
| **コントロール** | モデルへの完全なアクセス、任意のプロンプト | コンテンツフィルタリング、利用規約            |
| **モデル**       | 33以上のアーキテクチャ、任意のHFモデル     | プロバイダーが提供するものに限定される        |

> **HuggingFace**ライブラリ（`transformers`、`safetensors`、`huggingface-hub`）は完全にローカルで動作します。モデル設定の解析、トークナイゼーション、および一回限りの重みダウンロードに使用されます。クラウドAPI、有料サービス、データの外部送信は一切ありません。初回ダウンロード後は100%オフラインで動作します（`HF_HUB_OFFLINE=1`を設定）。
>
> **サーバーベースのツール**（例：Ollama）とは異なり、AirLLMはデーモンプロセスやサーバーを必要としないネイティブPythonライブラリとして動作します。`import`して実行するだけです。

## 🚀 クイックスタート

```bash
pip install airllm
```

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

input_tokens = model.tokenizer(
    ["What is the capital of France?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

> **注意:** 初回実行時に、モデルは分解され、レイヤーごとに保存されます。
> HuggingFaceのキャッシュディレクトリに十分なディスク空き容量があることを確認してください。

ステップバイステップの解説については、[クイックスタートガイド](docs/QUICKSTART.md)をご覧ください。

## 📋 対応モデル

AirLLM v3.0は、**23のモデルバックエンド**にわたる**33のアーキテクチャ文字列**をサポートしています。
[`AutoModel.from_pretrained()`](air_llm/airllm/auto_model.py:57)は、HuggingFaceの`config.json`から正しいバックエンドを自動的に検出します。

### モデルファミリー

| ファミリー       | バックエンドクラス                                       | アーキテクチャ文字列                                                            | モデル例                        | 最小VRAM      |
| ---------------- | -------------------------------------------------------- | ------------------------------------------------------------------------------- | ------------------------------- | ------------- |
| **Llama**        | [`AirLLMLlama2`](air_llm/airllm/airllm.py:17)            | `LlamaForCausalLM`, `LLaMAForCausalLM`                                          | Llama 3.1 8B/70B/405B           | 4 GB          |
| **Gemma**        | [`AirLLMGemma`](air_llm/airllm/airllm_gemma.py:24)       | `GemmaForCausalLM`                                                              | Gemma 7B                        | 4 GB          |
| **Gemma 2**      | [`AirLLMGemma2`](air_llm/airllm/airllm_gemma.py:40)      | `Gemma2ForCausalLM`                                                             | Gemma 2 9B/27B                  | 4 GB          |
| **Gemma 3**      | [`AirLLMGemma3`](air_llm/airllm/airllm_gemma.py:56)      | `Gemma3ForCausalLM`, `Gemma3ForConditionalGeneration`                           | Gemma 3 4B/12B/27B              | 4 GB          |
| **Gemma 4**      | [`AirLLMGemma4`](air_llm/airllm/airllm_gemma4.py:35)     | `Gemma4ForCausalLM`, `Gemma4ForConditionalGeneration`                           | Gemma 4 E2B/E4B/12B/31B         | 4 GB          |
| **Qwen**         | [`AirLLMQWen`](air_llm/airllm/airllm_qwen.py:19)         | `QWenLMHeadModel`                                                               | Qwen 7B/14B/72B                 | 4 GB          |
| **Qwen 2 / 2.5** | [`AirLLMQWen2`](air_llm/airllm/airllm_qwen2.py:21)       | `Qwen2ForCausalLM`, `Qwen2_5ForCausalLM`                                        | Qwen2.5 7B/72B, QwQ-32B         | 4 GB          |
| **Qwen 3**       | [`AirLLMQwen3`](air_llm/airllm/airllm_qwen3.py:30)       | `Qwen3ForCausalLM`                                                              | Qwen3 8B/32B/72B                | 4 GB          |
| **DeepSeek**     | [`AirLLMDeepSeek`](air_llm/airllm/airllm_deepseek.py:42) | `DeepseekV3ForCausalLM`, `DeepseekV2ForCausalLM`                                | DeepSeek-V3, DeepSeek-R1 (671B) | 16 GB         |
| **Mistral**      | [`AirLLMMistral`](air_llm/airllm/airllm_mistral.py:29)   | `MistralForCausalLM`, `Mistral4ForCausalLM`, `Mistral3ForConditionalGeneration` | Mistral 7B, Mistral Small 3/4   | 4 GB          |
| **Mixtral**      | [`AirLLMMixtral`](air_llm/airllm/airllm_mixtral.py:18)   | `MixtralForCausalLM`                                                            | Mixtral 8x7B, 8x22B             | 4 GB          |
| **GLM-4**        | [`AirLLMGlm4`](air_llm/airllm/airllm_glm4.py:35)         | `GlmForCausalLM`, `Glm4ForCausalLM`, `ChatGLM4Model`                            | GLM-4-9B                        | 4 GB          |
| **ChatGLM**      | [`AirLLMChatGLM`](air_llm/airllm/airllm_chatglm.py:20)   | `ChatGLMModel`                                                                  | ChatGLM3-6B                     | 4 GB          |
| **Phi**          | [`AirLLMPhi`](air_llm/airllm/airllm_phi.py:34)           | `Phi3ForCausalLM`, `PhiForCausalLM`, `Phi4ForCausalLM`                          | Phi-4 14B, Phi-3 Mini/Small     | 4 GB          |
| **Cohere**       | [`AirLLMCohere`](air_llm/airllm/airllm_cohere.py:33)     | `CohereForCausalLM`, `Cohere2ForCausalLM`                                       | Command-R 35B, Command-R+ 104B  | 4 GB          |
| **Zamba 2**      | [`AirLLMZamba2`](air_llm/airllm/airllm_zamba.py:39)      | `Zamba2ForCausalLM`                                                             | Zamba2-2.7B, Zamba2-7B          | 4 GB          |
| **Baichuan**     | [`AirLLMBaichuan`](air_llm/airllm/airllm_baichuan.py:19) | `BaichuanForCausalLM`                                                           | Baichuan2-7B/13B                | 4 GB          |
| **InternLM**     | [`AirLLMInternLM`](air_llm/airllm/airllm_internlm.py:18) | `InternLMForCausalLM`                                                           | InternLM-20B                    | 4 GB          |
| **Llama MLX**    | [`AirLLMLlamaMlx`](air_llm/airllm/airllm_llama_mlx.py)   | _(macOSのみ)_                                                                   | 任意のLlama互換モデル           | Apple Silicon |

プロンプト形式やバリアントごとのVRAMを含む詳細なモデル情報については、[対応モデルリファレンス](docs/SUPPORTED_MODELS.md)を参照してください。

## 💻 VRAM要件

AirLLMのレイヤーごとのアプローチは、VRAM使用量がモデル全体のサイズではなく、**最大の単一レイヤー**に依存することを意味します。
ほとんどの密な（Dense）モデルは4GBのVRAMで動作します。
MoEモデルは、エキスパートが多数存在するレイヤーのため、より多くのVRAMを必要とします。

| VRAM階層   | 密な（Dense）モデル                                                 | MoEモデル                                                  | 備考                                 |
| ---------- | ------------------------------------------------------------------- | ---------------------------------------------------------- | ------------------------------------ |
| **4 GB**   | Llama ≤405B, Gemma ≤31B, Qwen ≤72B, Phi ≤14B, GLM-4, Cohere, Zamba2 | —                                                          | FP16、量子化は不要                   |
| **8 GB**   | すべての密なモデル                                                  | Mixtral 8x7B (4-bit使用時)                                 | ほとんどのモデルで快適なヘッドルーム |
| **12 GB**  | すべての密なモデル                                                  | Mixtral 8x22B (4-bit使用時), Mistral Small 4 (4-bit使用時) | —                                    |
| **16 GB**  | すべての密なモデル                                                  | DeepSeek-V3/R1 (4-bit使用時)                               | DeepSeek MoEに推奨                   |
| **24 GB+** | すべての密なモデル                                                  | DeepSeek-V3/R1 (FP16)                                      | MoEレイヤーはFP16で各約22GB          |

> **ヒント:** `compression='4bit'`を使用すると、レイヤーごとのメモリを約70%削減でき、より小さなGPUでより大きなMoEモデルを実行できるようになります。

## 🔧 インストール

### 標準インストール

```bash
pip install airllm
```

### 量子化サポート付き

```bash
pip install "airllm[quantization]"
```

### macOS MLXサポート付き

```bash
pip install "airllm[mlx]"
```

### 開発用インストール

```bash
git clone https://github.com/lyogavin/airllm.git
cd airllm
pip install -e ".[dev]"
```

### 前提条件

- Python 3.10以上
- PyTorch 2.6以上
- CUDA対応GPU（Linux/Windows）またはApple Silicon Mac
- モデルシャード用の十分なディスク空き容量（通常、初回実行時にモデルサイズの2倍）

## 📖 使用例

### 基本的な推論

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

input_tokens = model.tokenizer(
    ["Explain quantum computing in simple terms."],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=50,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### 4-bit量子化を使用（3倍高速）

```bash
pip install bitsandbytes
```

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    compression="4bit",  # または "8bit"
)

input_tokens = model.tokenizer(
    ["What are the benefits of renewable energy?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=50,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### Gemma 4の実行

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("google/gemma-4-12b-it")

input_tokens = model.tokenizer(
    ["Write a haiku about machine learning."],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=True,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=30,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### Qwen 3の実行

```python
from airllm import AutoModel

model = AutoModel.from_pretrained("Qwen/Qwen3-8B")

input_tokens = model.tokenizer(
    ["Describe the water cycle."],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=50,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### DeepSeek R1の実行

```python
from airllm import AutoModel

# DeepSeek R1 MoE — 16GB以上のVRAMを推奨、または4-bit圧縮を使用
model = AutoModel.from_pretrained(
    "deepseek-ai/DeepSeek-R1",
    compression="4bit",
)

input_tokens = model.tokenizer(
    ["Solve: What is 25 * 37?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"].cuda(),
    max_new_tokens=100,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### macOSでのMLXの使用

```python
from airllm import AutoModel

# Apple Silicon搭載のmacOSでは、AutoModelは自動的にMLXバックエンドを使用します
model = AutoModel.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

input_tokens = model.tokenizer(
    ["What is the meaning of life?"],
    return_tensors="pt",
    return_attention_mask=False,
    truncation=True,
    max_length=128,
    padding=False,
)

output = model.generate(
    input_tokens["input_ids"],  # macOSでは .cuda() は不要です
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True,
)

print(model.tokenizer.decode(output.sequences[0]))
```

### ゲート付きモデル（ダウンロード専用のHFトークン）

一部のモデル（Llama 3、Gemma）は、ダウンロードする前にHuggingFaceでライセンスに同意する必要があります。
トークンは**初期の重みのダウンロードにのみ**使用されます。推論は完全にローカルで実行され、インターネット接続は不要です。

```python
from airllm import AutoModel

model = AutoModel.from_pretrained(
    "meta-llama/Llama-3.1-70B-Instruct",
    hf_token="hf_YOUR_TOKEN_HERE",  # 重みをダウンロードするために一度だけ必要です
)
```

## ⚙️ 設定

すべての設定は[`AutoModel.from_pretrained()`](air_llm/airllm/auto_model.py:57)を通じて渡されます：

| パラメータ                 | 型            | デフォルト      | 説明                                                                                                              |
| -------------------------- | ------------- | --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `compression`              | `str \| None` | `None`          | ブロック単位の量子化のための`"4bit"`または`"8bit"`。`bitsandbytes`が必要です。                                    |
| `hf_token`                 | `str \| None` | `None`          | ゲート付きモデルの重みをダウンロードするためのHuggingFaceトークン（ダウンロード専用で、推論には使用されません）。 |
| `prefetching`              | `bool`        | `True`          | ディスクI/OとGPU計算をオーバーラップさせます。圧縮を使用する場合は自動的に無効になります。                        |
| `layer_shards_saving_path` | `str \| None` | `None`          | 分割されたモデルシャードのカスタムディレクトリ。                                                                  |
| `profiling_mode`           | `bool`        | `False`         | レイヤーごとのタイミング計測を有効にします。                                                                      |
| `delete_original`          | `bool`        | `False`         | ディスク容量を節約するために、分割後に元のHFモデルを削除します。                                                  |
| `device`                   | `str`         | `"cuda:0"`      | 実行のターゲットデバイス。                                                                                        |
| `dtype`                    | `torch.dtype` | `torch.float16` | 重みの精度。                                                                                                      |
| `max_seq_len`              | `int`         | `512`           | モデルのコンテキストウィンドウの最大シーケンス長。                                                                |

### 圧縮の詳細

AirLLMは、ディスクI/Oを削減するために（アクティベーション量子化ではなく）ブロック単位の重み量子化を使用します：

| モード         | 圧縮率 | 速度向上     | 精度への影響 |
| -------------- | ------ | ------------ | ------------ |
| `None` (FP16)  | 1.0×   | ベースライン | なし         |
| `"8bit"`       | 0.50×  | 約2倍高速    | ほぼなし     |
| `"4bit"` (NF4) | 0.28×  | 約3倍高速    | 最小限       |

AirLLMのボトルネックはディスクからGPUへの転送であるため、（アクティベーションではなく）重みのみを圧縮することで、精度を保ちながら速度を劇的に向上させます。

## 🏗️ アーキテクチャ

AirLLMは、クリーンで拡張可能なアーキテクチャ上に構築されています：

```
┌──────────────────────────────────────────────────────────────┐
│                    AutoModel.from_pretrained()                │
│                   ┌─────────────────────┐                    │
│                   │   ModelRegistry      │                    │
│                   │  (30 architectures)  │                    │
│                   └────────┬────────────┘                    │
│                            │ resolves                        │
│         ┌──────────────────┼───────────────────┐             │
│         ▼                  ▼                   ▼             │
│   AirLLMLlama2     AirLLMGemma4       AirLLMDeepSeek        │
│   AirLLMMistral    AirLLMQwen3        AirLLMCohere    ...   │
│         │                  │                   │             │
│         └──────────────────┼───────────────────┘             │
│                            ▼                                 │
│                   AirLLMBaseModel                             │
│              (layer-by-layer forward pass)                    │
│                            │                                 │
│              ┌─────────────┼─────────────┐                   │
│              ▼             ▼             ▼                   │
│         load_layer   move_to_device   offload                │
│         (disk→CPU)   (CPU→GPU)        (GPU→meta)             │
└──────────────────────────────────────────────────────────────┘
```

詳細なアーキテクチャドキュメント：

- [アーキテクチャの概要](docs/ARCHITECTURE.md) — リポジトリの構造と設計上の決定事項。
- [クラス階層](docs/CLASS_HIERARCHY.md) — 継承ツリーとメソッドのオーバーライドポイント。
- [VRAM管理](docs/VRAM_MANAGEMENT.md) — メモリ最適化戦略。
- [モデル統合ガイド](docs/MODEL_INTEGRATION.md) — 新しいモデルバックエンドを追加する方法。
- [APIリファレンス](docs/API_REFERENCE.md) — パブリッククラスとメソッド。

## 🧪 テスト

```bash
# 開発用依存関係をインストール
pip install -e ".[dev]"

# すべてのテストを実行
pytest

# カバレッジ付きで実行
pytest --cov=airllm --cov-report=term-missing

# 特定のテストファイルを実行
pytest air_llm/tests/test_model_registry.py -v
```

テストスイートのカバー範囲：

- [`test_model_registry.py`](air_llm/tests/test_model_registry.py) — レジストリデコレータとルックアップ。
- [`test_automodel.py`](air_llm/tests/test_automodel.py) — AutoModelのディスパッチ。
- [`test_constants.py`](air_llm/tests/test_constants.py) — 定数とプラットフォーム検出。
- [`test_model_layer_config.py`](air_llm/tests/test_model_layer_config.py) — `ModelLayerConfig`データクラス。
- [`test_model_backends.py`](air_llm/tests/test_model_backends.py) — すべてのモデルバックエンドクラス。
- [`test_compression.py`](air_llm/tests/test_compression.py) — 量子化ユーティリティ。
- [`test_persist.py`](air_llm/tests/test_persist.py) — モデル永続化レイヤー。
- [`test_utils.py`](air_llm/tests/test_utils.py) — ユーティリティ関数。

## 🤝 貢献

貢献を歓迎します！
以下のガイドラインについては、[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください：

- 開発環境のセットアップ。
- 新しいモデルバックエンドの追加。
- コードスタイル（Google Python Style、ruff、mypy）。
- テストとCIの実行。
- プルリクエストのプロセス。

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています。詳細については、[LICENSE](LICENSE)ファイルを参照してください。

## 🙏 謝辞

AirLLMは、多くの才能ある研究者やエンジニアの成果の上に構築されています：

- **SimJeg** — AirLLMのレイヤーごとのアプローチにインスピレーションを与えた、元のKaggleコンペティションコード。
  [GitHub](https://github.com/SimJeg) ·
  [Kaggle notebook](https://www.kaggle.com/code/simjeg/platypus2-70b-with-wikipedia-rag) ·
  [Discussion](https://www.kaggle.com/competitions/kaggle-llm-science-exam/discussion/446414)
- **HuggingFace** — `transformers`、`accelerate`、および`safetensors`ライブラリ。
- **NavodPeiris** — CPU推論および非シャードモデルのサポート。

## 📚 AirLLMの引用

```bibtex
@software{airllm2024,
  author = {Gavin Li},
  title = {AirLLM: scaling large language models on low-end commodity computers},
  url = {https://github.com/lyogavin/airllm/},
  version = {3.0.0},
  year = {2024},
}
```

---

⭐ AirLLMが役立つと思ったら、リポジトリにスターを付けてください！

[![Star History Chart](https://api.star-history.com/svg?repos=lyogavin/airllm&type=Timeline)](https://star-history.com/#lyogavin/airllm&Timeline)
