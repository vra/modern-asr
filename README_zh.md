# Modern ASR — 简体中文文档




## 特性

- **🧩 19 款模型** — Whisper、SenseVoice、Qwen、MiMo、FireRedASR、GLM-ASR 等
- **🔌 零配置插件** — 通过 `@register_model` 装饰器添加新模型
- **🚀 运行时热切换** — 无需重启进程即可切换模型
- **🌍 多语言** — 52 种语言，22 种中文方言
- **🎯 多任务** — 转录、翻译、说话人分离、情绪识别、声学事件
- **💻 本地优先** — 完全本地推理，无需 API Key，数据不出本机
- **🍎 Apple Silicon** — 原生 MPS（Metal Performance Shaders）加速
- **📦 自动安装** — 依赖、代码仓库、模型权重首次使用时自动安装
- **🐍 现代 Python** — Pydantic 配置、Rich CLI、ISO 时间戳日志

## 安装

```bash
pip install modern-asr
```

**模型依赖和权重会在第一次使用时自动安装** — 你只需要输入模型名字：

```python
from modern_asr import ASRPipeline

pipe = ASRPipeline("sensevoice-small")
pipe = ASRPipeline("mimo-asr-v2.5")
pipe = ASRPipeline("whisper-small")
```

离线环境预装全部依赖：

```bash
pip install modern-asr[all-models]
```

**可用 extras：** `transformers`、`vllm`、`onnx`、`firered-asr`、`sensevoice`、`fun-asr`、`qwen-asr`、`mimo-asr`、`glm-asr`、`whisper`、`moonshine`、`all-models`、`all-backends`、`all`

**要求：** Python ≥ 3.10

## 支持的模型

| 系列 | 模型 ID | 参数量 | 语言 | Extra |
|------|---------|--------|------|-------|
| **Whisper** (OpenAI) | `whisper-tiny` | 39M | 99+ | `whisper` |
| | `whisper-base` | 74M | 99+ | `whisper` |
| | `whisper-small` | 244M | 99+ | `whisper` |
| | `whisper-medium` | 769M | 99+ | `whisper` |
| | `whisper-large-v3` | 1.5B | 99+ | `whisper` |
| | `whisper-large-v3-turbo` | 809M | 99+ | `whisper` |
| **SenseVoice** (阿里) | `sensevoice-small` | 234M | 中英日韩粤 | `sensevoice` |

| **Qwen3-ASR** (阿里) | `qwen3-asr-0.6b` | 0.6B | 22 种方言 | `qwen-asr` |
| | `qwen3-asr-1.7b` | 1.7B | 22 种方言 | `qwen-asr` |

| **FunASR / Paraformer** (阿里) | `funasr-nano` | 0.8B | 中英 | `fun-asr` |
| | `paraformer-zh` | 0.2B | 中文 | `fun-asr` |
| | `paraformer-large` | 0.7B | 中文 | `fun-asr` |
| **FireRedASR** (小红书) | `fireredasr-aed` | 1.1B | 中文 | `firered-asr` |
| | `fireredasr-llm` | 8.3B | 中文 | `firered-asr` |
| **MiMo-ASR** (小米) | `mimo-asr-v2.5` | 8B | 中文/方言 | `mimo-asr` |
| **MiDasheng** (小米) | `midashenglm-7b` | 7B | 音频理解 | `mimo-asr` |

| **GLM-ASR** (智谱 AI) | `glm-asr-nano-2512` | 1.5B | 中英粤 | `glm-asr` |
| **Granite Speech** (IBM) | `granite-speech-3.3-8b` | 8B | 英文 | `transformers` |
| **Moonshine** (Useful Sensors) | `moonshine-tiny` | 27M | 英文 | `moonshine` |


```bash
# 列出所有可用模型
python -m modern_asr list
```

## 快速开始

```python
from modern_asr import ASRPipeline

# 中文使用 SenseVoice
pipe = ASRPipeline("sensevoice-small")
result = pipe("audio.wav", language="zh")
print(result.text)

# 切换到 Qwen3-ASR 识别方言
pipe.switch_model("qwen3-asr-0.6b")
result = pipe("audio.wav", language="zh")
print(result.text)

# 英文使用 Whisper
pipe.switch_model("whisper-small")
result = pipe("audio.wav", language="en")
print(result.text)
```

## 架构

Modern ASR 采用三层架构：

1. **ASRPipeline** — 统一用户 API，负责输入归一化、任务分发、模型生命周期管理
2. **ASRModel / AudioLLMModel** — 适配层，新模型通常只需 **8 行配置**
3. **Backends** — Transformers、vLLM、ONNX Runtime

### 添加新模型

```python
from modern_asr.core.audio_llm import AudioLLMModel
from modern_asr.core.registry import register_model

@register_model("my-model-1b")
class MyModel1B(AudioLLMModel):
    HF_PATH = "org/MyModel-1B"
    SUPPORTED_LANGUAGES = {"zh", "en"}
    CHUNK_DURATION = 30.0

    @property
    def model_id(self) -> str:
        return "my-model-1b"
```

注册表会在运行时自动发现。仅此而已。

## 文档

完整的 Material for MkDocs 文档：

```bash
mkdocs serve
```

## 贡献

参见 [Contributing Guide](docs/contributing.md) 了解开发环境搭建、代码规范和 PR 清单。

## 许可证

Apache-2.0
