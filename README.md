# lowres_translation

FLORES-200 翻译评估（OpenAI 兼容 API / vLLM）与 CCMatrix 数据下载。

## 目录结构

| 路径 | 说明 |
|------|------|
| `run_evaluation.py` | **唯一评估入口**：读 `evaluation_config.json`，可选自动启动 vLLM |
| `download_flores.py` | 仅下载并解压官方 FLORES-200 tarball |
| `download_ccmatrix.py` | 从 OPUS 下载 CCMatrix 各语言对 zip，并写出预览 jsonl（可用 JSON 配置指定语对） |
| `ccmatrix_download_config.example.json` | CCMatrix 下载语对列表示例（复制为 `ccmatrix_download_config.json` 后修改） |
| `ccmatrix_to_llamafactory.py` | 将指定 CCMatrix 有向语对转为 **LLaMA-Factory** Alpaca 格式 JSONL |
| `evaluation_config.json` | 评估配置（可拷贝 `evaluation_config.example.json` 改名） |
| `src/lowres_translation/` | 评估与下载的实现模块 |

## 环境与依赖

```bash
cd translation_tjw
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-flores200.txt
```

运行前请保证能访问翻译 API（默认 `http://localhost:8005/v1`）。

**Hugging Face**：导入 `lowres_translation` 时会默认设置 **`HF_ENDPOINT=https://hf-mirror.com`**（可被环境变量覆盖）。文档与脚本中的 Hub **页面/ raw 链接**默认使用 **`hf-mirror.com`**。

**本地权重示例**：`"model": "/data/hub/Qwen3.5-9B"` 或 `"model": "models/my-qwen"`（相对项目根），并可选 `"served_model_name": "Qwen/Qwen3.5-9B"`，这样评估请求里仍用短模型名。

仓库中的 **`evaluation_config.json`** 已按 **Qwen3.5-9B** 写好：`model` 为 **`Qwen/Qwen3.5-9B`**，语对为 **英/简中 × 西/印尼/越/泰/菲** 的双向笛卡尔积（共 20 个有向语对），与原先 `preset: zh_en_cross_lowres` 一致；**`limit`: null** 表示每语对使用 **dev 全量**。试跑可把 **`limit`** 改为整数（如 `50`），或对个别语对在 **`pair_sample_limits`** 里单独写条数。

## 下载数据

```bash
# FLORES-200（官方 tar.gz，解压到 ~/.cache/huggingface/flores200/）
python3 download_flores.py

# CCMatrix：按语言对从 OPUS 拉 zip，每对默认预览 50 条 → jsonl
python3 download_ccmatrix.py --out_dir ccmatrix_preview --max_examples 50

# 仅下载配置文件中列出的有向语对（推荐）
# 复制 ccmatrix_download_config.example.json 为 ccmatrix_download_config.json 后编辑 pairs
python3 download_ccmatrix.py \
  --config-file ccmatrix_download_config.json \
  --out_dir ccmatrix_preview \
  --max_examples 50

# 列出 Hub 缓存中的全部可用语对（不写预览 jsonl）；无向 id 与 OPUS zip 名一致
python3 download_ccmatrix.py --list-pairs
# 全部有向语对（每个无向对含 a-b 与 b-a）
python3 download_ccmatrix.py --list-pairs --list-directed
# 写入文件（UTF-8）
python3 download_ccmatrix.py --list-pairs --pairs-output ccmatrix_all_undirected.txt
```

**`ccmatrix_download` 配置文件**（JSON）字段：

| 字段 | 说明 |
|------|------|
| `pairs` | 字符串数组，每项为有向语对 **`源-目标`**（ISO 短码，小写），如 **`en-zh`**（与 CCMatrix / OPUS 一致） |
| `language_pairs` | 可选。与评估配置类似：`[["en","zh"], ...]` 或 `[{"src":"en","tgt":"zh"}, ...]`；与 `pairs` 合并后按顺序去重 |

指定 **`--config-file`** 时，不再从 Hub 拉全量语对列表，**`--limit_configs` 无效**。

### CCMatrix → LLaMA-Factory（Alpaca JSONL）

从 OPUS 流式读取平行句，写出 **`instruction` / `input` / `output`**（与 [LLaMA-Factory data README](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/README.md) 中 **alpaca** 监督微调一致）。`input` 为源语句子，`output` 为目标语句子。

```bash
# 英译中，每向最多 10 万条；按 score 过滤（可选）
python3 ccmatrix_to_llamafactory.py \
  --pairs en-zh \
  --out-dir ./llamafactory_data/ccmatrix_mt \
  --max-samples-per-pair 100000 \
  --min-score 1.15

# 多个有向语对；合并为一个 jsonl 供单一 dataset 使用
python3 ccmatrix_to_llamafactory.py \
  --pairs en-zh,zh-en \
  --merge \
  --max-samples-per-pair 50000
```

- **`--pairs`**：有向语对 **`源-目标`**（ISO 短码，与 CCMatrix 一致），如 `en-zh` 表示英语→中文；`zh-en` 为中文→英语。  
- **`--zip-cache-dir`**：OPUS 下载缓存（默认项目下 `ccmatrix_zip_cache`）。  
- **`--instruction-template`**：可改指令模板，占位符 **`{src_name}` `{tgt_name}` `{src_code}` `{tgt_code}`**。  
- **`--system`**：写入每条样本的 `system` 字段。  
- **`--include-score`**：额外写入 `score` 列（需在 LLaMA-Factory 的 `dataset_info.json` 里用 `columns` 映射才会参与训练，一般可不加）。  

运行后在 **`--out-dir`** 下会生成各语对 **`ccmatrix_<src>_<tgt>_alpaca.jsonl`** 以及 **`dataset_info.snippet.json`**，把片段合并进 LLaMA-Factory 的 **`data/dataset_info.json`**，训练时 **`dataset: ccmatrix_en_zh`**（键名与片段中一致）。

**注意**：不设 `--max-samples-per-pair` 时数据量极大，磁盘与时间消耗很高，建议先试小上限或配合 **`--min-score`**。

CCMatrix 语言对列表默认从镜像拉取 `language_pairs_cache.py`。若需改用其他地址，可设置环境变量 **`CCMATRIX_PAIRS_URL`**（指向该文件的 raw URL）。

### CCMatrix 预览里的 `score`

每条双语句对的 `score` 为 bitext mining 的**匹配质量分数**，一般 **越高越可信**，可用于过滤。数据来自 [OPUS CCMatrix v1](https://opus.nlpl.eu/CCMatrix.php)；数据集说明见镜像站 [yhavinga/ccmatrix](https://hf-mirror.com/datasets/yhavinga/ccmatrix)（与 Hugging Face Hub 同源同步）。

## 评估配置：`evaluation_config.json`

顶层为 JSON 对象，字段如下。

### 通用字段

| 字段 | 类型 | 可取值 / 说明 |
|------|------|----------------|
| `mode` | string | **`batch`**：多语对循环；**`single`**：只评一个语对 |
| `base_url` | string 或 null | OpenAI 兼容 API 根 URL，如 `http://localhost:8005/v1`。为 null 时由 `run_evaluation.py` 根据 `--port` 拼出 `http://localhost:<port>/v1` |
| `model` | string 或 null | **Hub 模型 id**（如 **`Qwen/Qwen3.5-9B`**）或 **本地权重路径**：绝对路径、`~` 开头，或相对 **`translation_tjw` 项目根** 的目录/文件（如 `models/Qwen3.5-9B`）。若路径在磁盘上存在，会展开为绝对路径传给 `vllm serve`。为 null 时：自动起服务则用环境变量 `VLLM_MODEL` 或默认 `Qwen/Qwen3.5-9B`；仅 `--no-serve` 且未指定 model 时由 `models.list()` 取第一个 |
| `served_model_name` | string 或 null | **可选**。从**本地目录**加载时，若希望 OpenAI API 里的 `model` 仍为简短名字（与手动 `vllm serve /path --served-model-name ...` 一致），在此填写该名字；脚本会自动加上 vLLM 的 `--served-model-name`。未填时，请求里的 `model` 与解析后的 `model` 字段一致（本地一般为绝对路径） |
| `split` | string | **`dev`** 或 **`devtest`**（FLORES 划分） |
| `limit` | int 或 null | **默认**每个有向语对评估的样本条数；**null 表示该语对用全量**句子。可被 `pair_sample_limits` 按语对覆盖 |
| `pair_sample_limits` | object | **按语对覆盖样本条数**。键为有向语对字符串，格式与 `config` 相同：`源代码-目标代码`（如 **`eng_Latn-spa_Latn`**）。值为 **整数** 表示该语对条数；为 **null** 表示该语对**全量**（忽略顶层 `limit`）。未列出的语对使用顶层 **`limit`** |
| `random_sample` | bool | 与 `limit` / 各语对实际条数联用：true 为随机采样，false 为取前 N 条 |
| `seed` | int 或 null | 随机采样种子 |
| `metrics` | string 数组 | **`bleu`**、**`comet`** 的组合；未列出的指标不计算（可只填 `bleu` 以跳过 COMET） |
| `max_tokens` | int | 翻译生成的 max_tokens |
| `max_workers` | int | 翻译请求并发线程数 |
| `verbose` | bool | batch 模式下是否打印每个语对的详细日志 |

### `mode`: `batch`

推荐用 **`language_pair_groups`** 或 **`language_pairs`** 声明语对，无需 `preset`。

| 字段 | 类型 | 可取值 / 说明 |
|------|------|----------------|
| `language_pair_groups` | 数组 | 必须是 **恰好两个** 子数组（两组 FLORES 语言代码）；生成 **笛卡尔积** `(g0[i], g1[j])`。与 **`bidirectional`** 联用见下 |
| `language_pairs` | 数组 | 未使用 `language_pair_groups` 时必填（且非空）。每项为 **`["src", "tgt"]`** 或 **`{"src":"...","tgt":"..."}`** |
| `bidirectional` | bool | 对 **`language_pairs`**：为 true 时每个项再增加反向（去重）。对 **`language_pair_groups`**：为 true 时对每个 `(a,b)` 再增加 `(b,a)`（去重）。同一语言不做 `src==tgt` |
| `preset` | string 或 null | **可选**。仅兼容旧配置：非空时忽略 `language_pair_groups` / `language_pairs`，使用内置预设。可选值：**`zh_en_cross_lowres`**（中/英 × 西/印尼/越/泰/菲，双向 20 有向语对） |
| 语对来源优先级 | | 若设置了 **`preset`**，则只用预设；否则 **`language_pair_groups`**（非空）优先于 **`language_pairs`** |
| `replicates` | int | 每个有向语对重复跑几次（>1 时输出文件带 `_run{k}`） |
| `output_dir` | string | 各语对结果 JSON 目录（相对项目根） |
| `summary` | string 或 null | 汇总 JSON 路径；null 默认为 `<output_dir>/summary.json` |

### `mode`: `single`

| 字段 | 类型 | 可取值 / 说明 |
|------|------|----------------|
| `config` | string | 语对，格式 **`源flores代码-目标flores代码`**，如 **`eng_Latn-zho_Hans`** |
| `language_pairs` | | 若不写 `config`，可用仅含一项的 `language_pairs`，效果相同 |
| `limit` / `pair_sample_limits` | | 与 batch 相同：`pair_sample_limits` 的键为当前 `config` 字符串时可单独指定该语对条数 |
| `output` | string 或 null | 结果 JSON 路径；null 默认为 `eval_single/<config>.json` |

### FLORES 语言代码

与 FLORES-200 一致，例如：`eng_Latn`、`zho_Hans`、`spa_Latn`、`ind_Latn`、`vie_Latn`、`tha_Thai`、`tgl_Latn`。完整列表：

```bash
PYTHONPATH=src python3 -m lowres_translation.eval_single --list-languages
```

## 运行评估

```bash
# 使用默认 evaluation_config.json（可与命令行叠加，命令行优先）
python3 run_evaluation.py

# 指定配置文件
python3 run_evaluation.py --config-file my_eval.json

# 已有 vLLM，不自动起服务
python3 run_evaluation.py --no-serve

# 命令行覆盖示例
python3 run_evaluation.py --no-serve --limit 50 --metrics bleu
```

### `run_evaluation.py` 命令行参数

| 参数 | 说明 |
|------|------|
| `--config-file` | 配置文件路径（默认 `./evaluation_config.json`） |
| `--no-serve` | 不启动 vLLM |
| `--port` | vLLM 端口（默认 8005，在未指定 `base_url` 时拼 API 地址） |
| `--base-url` | 覆盖配置中的 API 根 URL |
| `--model` | 覆盖配置：Hub id 或本地路径（规则同配置项 `model`） |
| `--served-model-name` | 覆盖配置中的 `served_model_name` |
| `--tensor-parallel-size` | 自动起 vLLM 时的张量并行数 |
| `--gpus` | 自动起 vLLM 时的 `CUDA_VISIBLE_DEVICES`，逗号分隔 |
| `--mode` | `single` / `batch`，覆盖配置 |
| `--config` | 仅 `single`：语对字符串，如 `eng_Latn-zho_Hans`，覆盖配置中的 `config` |
| `--limit` / `--random-sample` / `--seed` / `--replicates` | 覆盖配置 |
| `--output-dir` / `--summary` / `--output` | 覆盖配置 |
| `--max-workers` / `--max-tokens` | 覆盖配置 |
| `--metrics` | 逗号分隔，如 `bleu` 或 `bleu,comet` |
| `--runs-csv` / `--lang-csv` | 导出 CSV 路径（默认在结果目录下） |
| `--verbose` | 打开详细日志 |

成功结束后会在结果目录生成 `run_results.csv`、`language_scores.csv`（与原先行为一致）。

## 配置示例：`single`（Qwen3.5-9B）

```json
{
  "mode": "single",
  "config": "eng_Latn-zho_Hans",
  "base_url": null,
  "model": "Qwen/Qwen3.5-9B",
  "split": "dev",
  "limit": 100,
  "pair_sample_limits": {},
  "metrics": ["bleu", "comet"],
  "output": "eval_single/eng-zho.json",
  "max_tokens": 512,
  "max_workers": 16
}
```

## 配置示例：`batch` + 笛卡尔积 + 按语对样本数

`language_pair_groups` 为 **`[ 组A, 组B ]`**；`pair_sample_limits` 的键必须是有向语对 **`src-tgt`**（与内部 `config` 一致）。

```json
{
  "mode": "batch",
  "model": "Qwen/Qwen3.5-9B",
  "language_pair_groups": [
    ["eng_Latn", "zho_Hans"],
    ["spa_Latn", "ind_Latn", "vie_Latn"]
  ],
  "bidirectional": true,
  "language_pairs": [],
  "split": "dev",
  "limit": 100,
  "pair_sample_limits": {
    "eng_Latn-spa_Latn": 200,
    "spa_Latn-eng_Latn": null
  },
  "metrics": ["bleu", "comet"],
  "output_dir": "eval_multilingual"
}
```

- 未写在 `pair_sample_limits` 里的有向语对：使用顶层 **`limit`（100）**。  
- **`spa_Latn-eng_Latn`: null**：该语对用 **全量** dev 句。  
- 双向开启时：2×3 个正向 + 反向，共 12 个有向语对（去重）。

## 配置示例：`batch` + 显式 `language_pairs`

```json
{
  "mode": "batch",
  "model": "Qwen/Qwen3.5-9B",
  "language_pair_groups": null,
  "bidirectional": true,
  "language_pairs": [
    ["eng_Latn", "zho_Hans"],
    ["zho_Hans", "fra_Latn"]
  ],
  "split": "dev",
  "limit": 50,
  "pair_sample_limits": {},
  "metrics": ["bleu"],
  "output_dir": "eval_custom",
  "summary": null,
  "replicates": 1,
  "verbose": false
}
```

双向为 true 时：`eng→zho`、`zho→eng`、`zho→fra`、`fra→zho`（去重后）。
