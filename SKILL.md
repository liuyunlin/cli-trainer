---
name: cli-trainer
description: 引导用户在 AI Studio 无代码训练平台完成大模型微调，支持 ERNIE 和开源模型（Qwen/LLaMA）。当用户提到微调、训练模型、SFT、fine-tune、问答对、让模型学会、定制领域助手、专属领域 AI、行业术语、调教模型、业务知识注入、领域适配、医疗问答、客服问答、无代码训练、AiStudio 训练、ERNIE 微调、模型训练、finetune 时，优先触发。

---

# CLI Trainer Skill

帮助用户在 AI Studio 云端完成大模型微调，不需要 GPU、不需要搭环境、不需要写训练代码。

用 `scripts/train.py` 执行所有平台交互，脚本路径通过 `$SKILL_PATH` 引用：

```bash
python3 "$SKILL_PATH/scripts/train.py" --check-data my_data.jsonl
```

---

## !! 绝对禁止事项（最高优先级）

1. **禁止自己猜测或硬写超参数然后提交。** 必须先运行 `--suggest-params`，把推荐值展示给用户，等用户确认再提交。
2. **禁止在用户明确确认前调用 `--submit`。** 展示参数时逐行解释含义，问”需要调整哪些？没问题就提交”，等用户回复。
3. **禁止把推荐参数当作默认训练配置直接提交。** 训练方式默认 `SFT/Full`；除非用户明确要求 `SFT/LoRA`，否则不要传 `--train-type`，也不要提交 `lora_rank`、`lora_alpha`、`lora_dropout` 等 LoRA 专属参数。`--suggest-params --model-type llama` 的输出可能包含 LoRA 字段，展示给用户前必须按实际 `train_type` 过滤并说明。
4. **训练方式变化时必须同步调整超参数。** 从 `SFT/LoRA` 改为 `SFT/Full` 时，必须删除 LoRA 专属参数，并把学习率降到 Full 微调适用范围（通常 `5e-5` 起步，必要时更低）；从 `SFT/Full` 改为 `SFT/LoRA` 时，才可恢复 LoRA 字段和较高学习率。用户提出”学更完整长解释/长回答/保留更多上下文”时，提醒可把 `cutoff_len`/`max_seq_len` 提到 512 或更高，同时说明训练会更慢、更吃显存，OOM 时先降 `per_device_train_batch_size`。

---

## 执行顺序

1. **环境自检** — 首次使用时先跑一次，自动检测 Python 版本、依赖包、网络连通性，requests 缺失会自动安装
2. **鉴权** — 确认有 Access Token，没有就引导去 https://aistudio.baidu.com/account/accessToken 获取
3. **选模型** — 运行 `--list-models` 展示白名单，让用户选，不要让用户自己猜名称
4. **验数据** — 运行 `--check-data` 检查格式，提前发现错误
5. **准备/上传数据集** — 只能访问 AiStudio 新版 Git 仓库型数据集，不支持外部 URL。创建仓库时先确认真实 `gitlogin`；用 Playwright MCP 或 `web-access` skill 创建仓库并拿到真实 `repo_id`；上传前优先删除 `.gitattributes` 中的 JSON/JSONL LFS 规则，上传后记录 `is_lfs` 状态并确认文件大小。
6. **推荐超参** — 运行 `--suggest-params`，展示结果，**等用户确认后才提交**
7. **提交训练** — 运行 `--submit`，拿到 jobId
8. **监控训练** — 提交后运行 `--poll JOB_ID`；任务进入 running 且接口返回 `tensorboardUrl` 后自动打开 Tensorboard
9. **取结果** — 训练完成后给出模型仓库地址和训练摘要

---

## 各步骤关键细节

### 环境自检

```bash
python3 "$SKILL_PATH/scripts/train.py" --env-check
```

检查内容：Python 版本（需 3.8+）、requests 包（缺失时自动 pip install）、网络连通性。
如果 python3 命令本身不存在，告知用户安装 Python 3.8+：macOS 用 `brew install python3`，其他平台参考 https://www.python.org/downloads/

### 通用 [web-access](https://github.com/eze-is/web-access) / Codex 推荐安装 Playwright MCP

自建数据集、网页端创建数据集仓库、必要时初始化 Gitea 或操作登录后网页时，不要只固定使用一种浏览器方案。这里按"通用 web-access / Codex 推荐安装 Playwright MCP"处理：Codex 环境优先安装/启用 Playwright MCP；需要复用用户日常 Chrome 登录态时，再使用 `web-access`。目标是拿到页面详情页真实 `repo_id` 并完成必要的 `.gitattributes` 编辑；优先选择当前可用、已登录、最少阻塞的浏览器自动化能力。

推荐顺序：
1. **Playwright MCP 可用且页面已登录时优先使用。** 适合打开 AI Studio 页面、创建数据集、填写表单、选择协议、编辑 `.gitattributes`、读取 `repo_id`。常用能力：`browser_navigate`、`browser_snapshot`、`browser_click`、`browser_fill_form`、`browser_evaluate`。它不依赖用户 Chrome 的 remote debugging 授权，CDP 授权卡住时尤其适合。
2. **需要用户日常 Chrome 登录态时使用 `web-access`。** 如果 Playwright 未登录、页面需要真实 Chrome cookie/扩展/特殊会话，加载 `web-access` skill 并启动 CDP Proxy。
3. **两者都不可用时再让用户手动创建仓库。** 此时给出最短步骤，并要求用户回传详情页真实 `repo_id`，不要猜。

进入上传/建仓步骤前，若选择 `web-access`，按下面流程检查并准备它。

准备顺序：
1. 如果当前会话可用 `web-access` skill，先加载它并遵循其 `SKILL.md`
2. 如果 skill 元数据没触发，但 `$HOME/.codex/skills/web-access/SKILL.md` 存在，直接把 `WEB_ACCESS_SKILL_PATH` 指向该目录并运行 `scripts/check-deps.sh`
3. 如果目录不存在，先使用 `skill-installer` 安装 [`web-access`](https://github.com/eze-is/web-access)。安装完成后可立即用安装目录下的 `scripts/check-deps.sh`；同时告诉用户重启 AI IDE / 刷新 Agent 会话后才能自动识别新 skill
4. 如果安装失败或 Chrome 远程调试授权不可用，才退回让用户手动在网页创建数据集仓库

检查命令：
```bash
WEB_ACCESS_SKILL_PATH="${WEB_ACCESS_SKILL_PATH:-$HOME/.codex/skills/web-access}"
bash "$WEB_ACCESS_SKILL_PATH/scripts/check-deps.sh"
```

### Token

推荐使用环境变量，但不要把真实 token 直接写进 shell history。交互式输入后再导出：
```bash
read -rsp "AI Studio Access Token: " AISTUDIO_ACCESS_TOKEN
echo
export AISTUDIO_ACCESS_TOKEN
```

验证：`python3 "$SKILL_PATH/scripts/train.py" --verify-token`

如果用户已在本机手工用 `aistudio config/login` 配过 token，脚本无环境变量时自动读取 SDK 缓存；环境变量里的过期 token 会覆盖缓存，遇到 401 先检查环境变量。

**安全要求：不要把真实 token 写进命令行实参、URL、文件、notebook、日志或命令历史。** 例如避免使用 `aistudio upload ... --token YOUR_TOKEN`、`python train.py --api-key YOUR_TOKEN` 或 `https://TOKEN:TOKEN@...`，因为这些值可能被完整 argv、remote URL、终端日志或进程列表暴露。上传数据时优先使用 `aistudio_sdk.hub.upload_folder(..., token=os.environ["AISTUDIO_ACCESS_TOKEN"])`，token 只从当前 shell 环境变量或 SDK 缓存读取。若 token 已经出现在聊天、终端输出或日志中，训练结束后提醒用户立即重新生成。

### 平台运行环境（已验证，2026-05-14）

| 组件 | 版本 | 影响 |
|------|------|------|
| Transformers | **4.49.0** | Qwen3 需要 4.51+，平台**不支持**；Qwen2.5 完全支持 |
| LlamaFactory | 未打印版本号 | 不支持 `deepseek_r1` chat template（所有 DeepSeek-R1 系列均失败）；`template` 超参数被 API 拒绝（code=10007） |
| CUDA Runtime | corex-4.3.8 | - |
| GPU 显存 | ~32GB | Full SFT 7B 时 OOM；LoRA + cutoff=2048 + batch=1 可稳定运行 |


### 选模型

```bash
python3 "$SKILL_PATH/scripts/train.py" --list-models
```

- 建议用白名单里的模型
- 单卡最大 32B 以下，超过会 OOM
- 训练方式默认 `SFT/Full`；只有明确需要 LoRA 时才传 `--train-type`
- ERNIE → 框架 PaddleFormers，trainType 只能 `SFT/Full`，数据格式 src/tgt
- 开源模型 → 框架 LlamaFactory，trainType 推荐 `SFT/Full` 或 `SFT/LoRA`，数据格式支持 Alpaca 和 ShareGPT
- 具体模型的风险提示以 `--list-models` 输出为准

### 数据格式

**ERNIE 格式（src/tgt 值必须是列表）：**
```jsonl
{"src": ["问题"], "tgt": ["回答"]}
```
最常见错误：用了 Alpaca 格式，任务会跑起来但 poller 阶段报"非 ERNIE 格式"。

**LlamaFactory 格式 1：Alpaca（支持 JSONL 或 JSON 数组）：**
```jsonl
{"instruction": "问题", "input": "", "output": "回答"}
```
JSON 数组文件也可检查，常见文件名是 `alpaca_data.json`：
```json
[{"instruction": "问题", "input": "", "output": "回答"}]
```

**LlamaFactory 格式 2：ShareGPT（支持 JSONL 或 JSON 数组）：**
```jsonl
{"conversations": [{"from": "human", "value": "问题"}, {"from": "gpt", "value": "回答"}]}
```
OpenAI `messages` 也属于 ShareGPT 特例，但平台兼容性不确定时，优先转成 `conversations/from/value`。

```bash
python3 "$SKILL_PATH/scripts/train.py" --check-data 数据文件.jsonl
python3 "$SKILL_PATH/scripts/train.py" --check-data alpaca_data.json
python3 "$SKILL_PATH/scripts/train.py" --check-data sharegpt_data.jsonl
```

数据规模参考：50-500 条（验证流程）/ 1k-10k（场景微调）/ 10k+（全面提升）

### 上传数据

没有自己数据时，先读取 `$SKILL_PATH/references/datasets.md`，根据用户目标推荐合适的数据集；再用 `--list-datasets` 展示完整列表。有本地训练文件时，按下面门禁上传。浏览器自动化只负责建仓、读真实 `repo_id`、必要时编辑 `.gitattributes`；SDK/CLI 负责上传，`--verify-upload` 负责最终验收。

1. **准备 web-access/CDP**
   ```bash
   WEB_ACCESS_SKILL_PATH="${WEB_ACCESS_SKILL_PATH:-$HOME/.codex/skills/web-access}"
   bash "$WEB_ACCESS_SKILL_PATH/scripts/check-deps.sh"
   ```
   如果 web-access skill 不在默认目录，从实际 `SKILL.md` 路径解析目录；Chrome 出现远程调试授权时让用户允许，页面未登录时让用户在 Chrome 登录 AI Studio。

2. **创建或确认数据集仓库**
   - 在 `https://aistudio.baidu.com/my/dataset` 创建或打开数据集仓库。新建时英文 ID 用小写字母、数字、下划线；可见性按页面默认公开处理，除非用户明确要求私密或数据含隐私。
   - 创建表单里的开源协议要选一个（例如 `CC0`）；公开数据集必须选择，不选时可能点击创建但没有明显报错。
   - 创建或打开仓库后，从详情页读取完整 `repo_id`，形如 `gitlogin/repo_name`。不要用昵称、展示名、登录用户名或邮箱猜 `gitlogin`。
   - `aistudio upload` / `aistudio_sdk.hub.upload_file` 只上传到已有数据集仓库，不会自动创建 dataset repo。如果上传时报 `preupload` 404，优先检查仓库是否已在网页端创建、`repo_id` 是否来自详情页、token 是否对该仓库有写权限、仓库类型是否为 dataset。
   - 如果传 `--output-repo`，斜杠前半段必须和当前账号可写的 `gitlogin` 匹配。

3. **上传前处理 JSON/JSONL 的 LFS 规则**
   - 预防优先：新仓库创建后、上传训练文件前，进入数据集详情页 → 文件列表 → `.gitattributes` → 编辑，删除训练文件扩展名对应的 LFS 规则，例如 `*.jsonl filter=lfs ...` 或 `*.json filter=lfs ...`，然后保存。
   - ERNIE 的 `src/tgt` JSONL、LlamaFactory 的 Alpaca/ShareGPT JSONL/JSON 都要检查；训练用 JSON/JSONL 推荐作为普通文件上传，便于下载回验和排查。
   - 如果已经上传成 LFS：不要直接判定训练必然失败。已实测部分 `is_lfs:true` 文件也可能被 AI Studio 成功挂载并完成训练；但它会降低可回验性，也会让 `waiting_data` 排查更困难。若任务卡在 `waiting_data`，优先删除旧 LFS 训练文件、移除 `.gitattributes` 中 JSON/JSONL LFS 规则，然后按本上传小节第 5 步 SDK 方案或第 6 步 CLI 方案重新上传。

4. **上传前检查普通文件大小**
   - AI Studio Git 仓库普通文件上传可能拒绝超过 5MB 的 JSON/JSONL；不要为了绕过大小限制默认改走 LFS。优先 compact/crop/split，并保留 manifest；如果最终只能走 LFS，要在回执里明确记录 `is_lfs:true` 风险。
   - 如果训练 JSON/JSONL 超过 5MB，先停止上传并向用户说明取舍。可选处理：压缩固定 prompt、裁剪过长上下文、减少 replay、拆分实验档，或确认平台是否支持该任务的多文件训练。
   - 做过裁剪或 compact 处理时，必须保留 manifest，记录源文件、输出文件、样本数是否变化、截断规则、被截断样本数和原因。不要把 compact 文件伪装成未裁剪的完整数据。

5. **用 SDK 上传数据集文件夹（推荐）**
   - 上传原则见 `references/aistudio_sdk_upload.md`；主规则是：完整 `repo_id`、一仓一数据集、token 只走环境变量、JSON/JSONL 优先不走 LFS。
   - SDK 日志出现 `201` 和 `Commit part 1 successful!` 才算提交成功；STS 分支的已知回退报错不单独视为失败。
   - 上传后必须继续执行本上传小节第 6 步；不要只看 SDK 上传日志。

6. **用 CLI 上传单个训练文件（备选）**
   ```bash
   AISTUDIO_CLI="${AISTUDIO_CLI:-$(python3 -m site --user-base)/bin/aistudio}"
   [ -x "$AISTUDIO_CLI" ] || AISTUDIO_CLI="$(command -v aistudio)"
   LOCAL_FILE="/path/to/train.jsonl"          # 也可以是 alpaca_data.json
   TRAIN_FILE="$(basename "$LOCAL_FILE")"     # 仓库内文件名，提交训练时 --train-file 也用它
   "$AISTUDIO_CLI" upload "$REPO_ID" "$LOCAL_FILE" "$TRAIN_FILE" --repo-type dataset
   ```
   只有确认当前 CLI 不会打印完整 argv 时才使用备选方案；不要附加 `--token TOKEN`，token 仍应走环境变量或 SDK 缓存。
   `REPO_ID` 必须是详情页显示的完整 `repo_id`。如果出现 `preupload` 404，回到本上传小节第 2 步确认仓库已存在且路径无误。
   上传命令成功返回后，必须立刻主动告诉用户：训练文件已上传到哪个 `repo_id`、仓库内文件名是什么、接下来会做 `is_lfs` 和下载回验；不要等到提交训练后才暴露上传问题。

7. **验证上传结果**
   ```bash
   python3 "$SKILL_PATH/scripts/train.py" \
     --verify-upload \
     --train-data "$REPO_ID" \
     --train-file "$TRAIN_FILE" \
     --local-file "$LOCAL_FILE"
   ```
   验证标准：文件大小接近本地文件，下载回来后 `--check-data` 通过；推荐 `is_lfs:false`。如果 `is_lfs:true`，不要直接隐瞒或继续静默提交，必须提示风险：本项目 smoke 实测 LFS 也可能训练成功，但若任务卡 `waiting_data` 应优先修复 LFS。
   验证通过后必须再主动给用户一个上传完成回执，至少包含：
   - 数据集：`REPO_ID`
   - 训练文件：`TRAIN_FILE`
   - LFS 状态：`is_lfs:false`（推荐）或 `is_lfs:true` 风险说明
   - 文件大小：仓库大小与本地大小是否一致或接近
   - 下一步将使用的提交参数：`--train-data "$REPO_ID" --train-file "$TRAIN_FILE"`

8. **卡在 `waiting_data` 时按顺序排查**
   - `REPO_ID` 是否来自详情页，`gitlogin` 是否真实可写
   - `--train-file` 是否和仓库内文件名完全一致
   - 训练文件大小是否接近本地文件；若 `is_lfs:true`，优先修复为普通 JSON/JSONL 后重试
   - 下载回本地后 `--check-data` 是否通过
   - 如果新 commit 和新 `mount Job` 仍循环“正在等待数据集下载完成...”，通常是平台挂载任务卡住；停止反复重传，保留 jobId、repo_id、commitId、mount Job 和上传校验结果给平台排查。

### 推荐超参并确认

```bash
python3 "$SKILL_PATH/scripts/train.py" --suggest-params 数据文件.jsonl --model-type ernie
```

脚本输出推荐值后，逐行解释给用户，明确问："需要调整哪些？没问题就提交。" 等回复再往下走。

**ERNIE（PaddleFormers）主要参数：**

| 参数 | 类型 | 推荐值 | 说明 |
|------|------|-------|------|
| `num_train_epochs` | float | 3 | 训练轮数，数据少可调大到 5 |
| `per_device_train_batch_size` | int | 4 | 每步样本数，OOM 就调小到 2 |
| `learning_rate` | float | 5e-5 | 学习率，新手一般不用改 |
| `max_seq_len` | int | 512 | 最大序列长度，超过截断 |
| `max_steps` | int | -1 | -1 表示由 epochs 控制；不传时平台可能使用默认固定步数，导致 epochs 没有按预期跑满 |
| `warmup_steps` | int | 50 | 预热步数，约总步数的 5-10% |
| `logging_steps` | int | 5 | 每几步打一次日志 |
| `bf16` | bool | true | 混合精度，节省显存 |

**LlamaFactory（开源模型）核心训练参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `learning_rate` | float | `5e-5` | 建议范围 `1e-6`~`5e-4`，小模型（0.5B-3B）推荐 `5e-5`，大模型（7B+）推荐 `1e-5`~`3e-5` |
| `num_train_epochs` | float | `3.0` | 训练轮次，数据量小可适当增大 |
| `per_device_train_batch_size` | int | `2` | 单卡 batch size，0.5B 可用 `2`~`8`，7B+ 建议 `1`~`2` |
| `gradient_accumulation_steps` | int | `8` | 梯度累积步数，等效 batch = `batch_size × accumulation` |
| `cutoff_len` | int | `1024` | 最大序列长度，长文本场景可调至 `2048`/`4096`，显存随之增大 |
| `warmup_ratio` | float | `0.1` | 学习率预热比例 |
| `lr_scheduler_type` | string | `"cosine"` | 学习率调度策略：`cosine`、`linear`、`constant` |
| `max_grad_norm` | float | `1.0` | 梯度裁剪阈值，防止梯度爆炸 |
| `logging_steps` | int | `5` | 日志打印间隔 |
| `save_steps` | int | `50` | checkpoint 保存间隔 |
| `fp16` | bool | `true` | 混合精度 |

**LlamaFactory（SFT/LoRA 时生效）LoRA 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lora_rank` | int | `8` | LoRA 秩，越大表达能力越强但显存越多，常用 `8`/`16`/`32`/`64` |
| `lora_alpha` | int | `16` | 缩放系数，通常设为 `lora_rank` 的 1~2 倍 |
| `lora_dropout` | float | `0` | 防过拟合可设 `0.05`~`0.1` |
| `lora_target` | string | `"all"` | 应用 LoRA 的目标模块，`all` 表示所有线性层 |

训练方式选定后，按以下原则提交参数：`SFT/Full` 只传核心训练参数，不传 LoRA 参数；`SFT/LoRA` 在核心训练参数基础上叠加 LoRA 参数。只要训练方式发生变化，必须重新展示调整后的完整参数给用户确认。

上下文长度要跟目标回答风格联动：短问答/分类/固定格式可用脚本推荐的 `cutoff_len`/`max_seq_len`；如果用户希望模型学习更完整的长解释、长评论或保留热搜详情，优先建议 512。调大长度会增加训练时间和显存占用；出现 OOM 时，先把 `per_device_train_batch_size` 降到 2，再考虑降长度。

### 命名规范

提交任何 CLI 训练任务前都必须主动生成可读名称，避免平台默认产物显示为 `train_xxxxxxxx` 后难以区分。命名应包含模型简称、业务领域或任务类型、数据档位或版本，只使用小写字母、数字、下划线。

推荐格式：
- 任务名 `--name`：`{model_short}_{domain_or_task}_{profile}`，例如 `ernie03b_customer_qa_smoke`、`qwen25_7b_finance_summary_v1`、`llama3_legal_review_balanced`
- 输出仓库 `--output-repo`：`{gitlogin}/{model_short}_{domain_or_task}_{profile}`，例如 `your_gitlogin/ernie03b_customer_qa_smoke`

提交前把建议名称展示给用户确认。若用户没有指定，按训练目标自动起名：从 `base_model` 提取模型简称（如 `ernie03b`、`qwen25_7b`），从数据集、文件名或用户目标提取领域/任务（如 `customer_qa`、`finance_summary`、`legal_review`、`medical_record`），从运行档位或版本提取 `smoke`、`balanced`、`thesis`、`v1`。`--name` 只控制训练任务名，不保证最终模型仓库名；如果不传 `--output-repo`，训练成功后的模型仓库可能仍由平台命名为 `train_xxxxxxxx`。需要最终模型仓库名可读时，必须显式传 `--output-repo`。传入前必须确认该 `gitlogin/repo` 属于当前账号可写命名空间，且没有和不相关的已有模型仓库冲突；不确定时只传 `--name`，训练完成后再补充模型卡片和 README。

**Model Card 标题**（推送到模型仓库 README.md 的 H1）与训练任务名命名逻辑不同，按 `$SKILL_PATH/references/model-card-spec.md` §一 生成，格式为 `【{基座模型名称}】{场景/功能描述}`，场景描述 12 汉字以内。

### 提交

默认只传可读任务名，让平台自动创建模型仓库：

```bash
JOB_NAME="ernie03b_customer_qa_smoke"              # 按本次训练目标替换

python3 "$SKILL_PATH/scripts/train.py" --submit \
  --base-model "PaddlePaddle/ERNIE-4.5-0.3B-PT" \
  --train-data "$REPO_ID" \
  --train-file "$TRAIN_FILE" \
  --name "$JOB_NAME" \
  --params '{"num_train_epochs": 3, "per_device_train_batch_size": 4, "learning_rate": 5e-5, "max_seq_len": 512, "bf16": true}'
```

这个模式只保证任务列表里的名称可读；最终模型仓库仍可能是平台生成的 `train_xxxxxxxx`。如果用户明确要求模型仓库名也可读，使用下一段 `--output-repo` 示例。

如果已经确认目标命名空间可写且仓库名不冲突，再显式指定最终模型仓库：

```bash
JOB_NAME="ernie03b_customer_qa_smoke"              # 按本次训练目标替换
OUTPUT_REPO="your_gitlogin/ernie03b_customer_qa_smoke"

python3 "$SKILL_PATH/scripts/train.py" --submit \
  --base-model "PaddlePaddle/ERNIE-4.5-0.3B-PT" \
  --train-data "$REPO_ID" \
  --train-file "$TRAIN_FILE" \
  --name "$JOB_NAME" \
  --output-repo "$OUTPUT_REPO" \
  --params '{"num_train_epochs": 3, "per_device_train_batch_size": 4, "learning_rate": 5e-5, "max_seq_len": 512, "bf16": true}'
```

注意：
- `--train-data` 必须使用数据集详情页显示的完整 `repo_id`，形如 `gitlogin/repo_name`；不要用昵称、展示名、邮箱或自己拼出来的仓库路径
- `--train-file` 强烈建议指定；不传时平台会自动选择数据集目录下首个 JSON/JSONL，只有仓库里训练文件唯一且明确时才可省略
- `--train-file` 的值如果指定，必须和数据集仓库里的实际文件名完全一致（进数据集详情页 → 文件列表确认），写错会导致 `waiting_data` 静默卡住
- 任务名 `--name` 不能含横杠，只能用字母/数字/下划线
- `--name` 只控制训练任务名；`--output-repo` 才控制最终模型仓库路径。如果省略 `--output-repo`，平台可能生成 `train_xxxxxxxx` 这类不可读仓库名
- `--output-repo` 不是必填项；只有确认目标命名空间可写、仓库名未和无关模型冲突时才传，不确定时先省略
- 每账号最多 30 个模型仓库，满了用 `--output-repo` 复用已有仓库或指定一个已有可写仓库
- 模型产物默认按“公开发布”处理：如果网页端或接口出现公开/私密选项，除非用户明确要求私密或数据/模型含敏感内容，否则选择公开。若平台训练完成后自动生成的模型仓库仍显示私密，第一时间提醒用户到 AI Studio 模型库页面把可见性改为公开，并补充模型卡片和协议
- 可视化参数（`report_to`/`visualdl`）由平台后端自动管理，用户传了反而会报"不支持的参数"错误，无需手动传

### 提交后监控与 Tensorboard

提交成功拿到 jobId 后，优先运行持续轮询：
```bash
python3 "$SKILL_PATH/scripts/train.py" --poll JOB_ID
```

`--poll` 会持续打印状态；任务进入 `running` 且接口返回 `tensorboardUrl` 后，会自动打开一次 Tensorboard。这个 URL 可能在 `waiting_data` 阶段提前生成，但不要在训练真正开始前打开空看板。如果没有自动打开，说明任务尚未进入 running 或 Tensorboard URL 尚未生成，可稍后手动运行：
```bash
python3 "$SKILL_PATH/scripts/train.py" --open-tb JOB_ID
```

状态响应规则：
- `waiting_data`：告诉用户平台正在下载/挂载模型和数据集，通常 1-10 分钟；Tensorboard 地址可能已生成，但要等进入 running 后再打开才有内容
- `pending`：告诉用户任务已排队，等待 GPU 分配
- `running`：报告进度、当前 loss，并用 `--train-summary JOB_ID` 解读 loss 趋势
- `succeeded`：运行 `--train-summary JOB_ID` 汇报最终 loss、模型仓库和测试建议
- `failed/cancelled`：必须主动运行 `--diagnose JOB_ID`，结合 system log 说明失败原因
- `waiting_data` 超过 10 分钟：必须主动运行 `--diagnose JOB_ID`；先按上传门禁核对 `repo_id`、`--train-file`、文件大小和下载回验；若 `is_lfs:true`，先修复 LFS 并重提。若这些都通过且 system log 持续卡在数据集下载，按平台 mount 异常处理，不要让用户重复修同一份数据

### 监控进度

```bash
python3 "$SKILL_PATH/scripts/train.py" --status JOB_ID    # 查一次
python3 "$SKILL_PATH/scripts/train.py" --diagnose JOB_ID  # 主动查状态 + system log + stdout
python3 "$SKILL_PATH/scripts/train.py" --logs JOB_ID      # 查 stdout loss
python3 "$SKILL_PATH/scripts/train.py" --logs JOB_ID --system  # 单独查 system log
```

状态含义：`waiting_data`（下载中，1-10min）→ `pending`（等 GPU，1-5min）→ `running`（训练中）→ `succeeded`

- `--logs` 查 stdout 是最可靠的 loss 监控方式，ERNIE/LlamaFactory 都支持
- `--diagnose` 是异常排查首选，会主动拉 system log；用于 `waiting_data` 超时、`failed`、`cancelled`、平台挂载/调度问题
- `--train-summary` 会从日志解析 loss/lr 并输出训练趋势，训练完成后依然有效
- Tensorboard 不是完全不可用：优先看 `Scalars` 面板。`Scalars` 能显示 loss/lr 等曲线时，说明 event file 和标量日志基本正常；`Time Series` 空白通常是新版 Tensorboard 面板兼容、前端缓存或加载问题，不等价于日志损坏或训练失败。训练结束后入口仍可能 `INACTIVE` 或部分面板不渲染，不要把 Tensorboard 作为唯一验收依据，优先以 raw log、`--train-summary` 和本地导出的 CSV/PNG 曲线为准
- `logUrl` 是鉴权 API，不是公开网页。浏览器直接打开 `/v1/train/jobs/.../master/output.log` 可能返回 `Missing or invalid Authorization header`；请用 `--logs`、`--export-artifacts`，或带 `Authorization: Bearer $AISTUDIO_ACCESS_TOKEN` 的 requests/curl 拉取日志。Tensorboard 能打开不代表 log API 可匿名访问
- `--poll` 会阻塞终端；对话场景不能长期占用终端时，改为周期性运行 `--status`、`--diagnose`、`--logs`、`--train-summary`

### 训练完成后

运行 `--train-summary` 获取 loss/lr 汇报：
```bash
python3 "$SKILL_PATH/scripts/train.py" --train-summary 训练任务ID
```

训练 `succeeded` 后，先完成训练证据归档，再处理模型卡片与可见性。

**训练证据归档清单：**
- `job_detail.json`：最终任务状态、模型仓库、Tensorboard/log URL
- `master_output_raw.log`：stdout 原始训练日志
- `system_raw.log`：系统日志，用于排查数据挂载/调度
- `train_summary.txt`：`--train-summary` 输出
- `loss_curve.csv`：从 raw log 解析出的 step/loss/lr/ppl 等指标
- `loss_curve.png`、`learning_rate_curve.png`、`training_curves.png`：本地曲线图
- `README.md`：记录 jobId、模型仓库、数据集、训练文件、超参、loss 摘要、Tensorboard 是否可用、任何 compact/截断处理

如果 Tensorboard 页面不可读，但 raw log 能解析出 loss/lr，训练证据仍然成立。最终回复用户时应区分“训练任务成功”和“模型效果已通过推理验收”。

训练 `succeeded` 后，**必须自动完成以下两步，不要等用户提醒**：

#### 第一步：通过 git 自动推送 README（已验证可行）

AI Studio 模型仓库支持通过 git 推送。不要把 token 拼到 remote URL；用 `GIT_ASKPASS` 从环境变量读取凭据：

```bash
# 将 REPO_ID 替换为实际仓库路径，如 18610248/train_c8979534
cd /tmp && rm -rf model_readme_tmp
ASKPASS_FILE="$(mktemp)"
cat > "$ASKPASS_FILE" <<'EOF'
#!/usr/bin/env sh
case "$1" in
  *Username*) printf '%s\n' "$AISTUDIO_ACCESS_TOKEN" ;;
  *Password*) printf '%s\n' "$AISTUDIO_ACCESS_TOKEN" ;;
  *) printf '%s\n' "$AISTUDIO_ACCESS_TOKEN" ;;
esac
EOF
chmod 700 "$ASKPASS_FILE"
trap 'rm -f "$ASKPASS_FILE"' EXIT

GIT_ASKPASS="$ASKPASS_FILE" GIT_TERMINAL_PROMPT=0 \
git clone "https://git.aistudio.baidu.com/$REPO_ID.git" model_readme_tmp \
  --no-checkout --depth=1 --filter=blob:none 2>&1 | tail -3

cd /tmp/model_readme_tmp
git sparse-checkout init --cone
git sparse-checkout set README.md
git checkout

# 写入 README（见下方模板，按实际训练信息填充）
cat > README.md << 'EOF'
...内容见下方模板...
EOF

git config user.email "train@aistudio.baidu.com"
git config user.name "AI Studio Train"
git add README.md
git commit -m "docs: 完善模型卡片 README"
GIT_ASKPASS="$ASKPASS_FILE" GIT_TERMINAL_PROMPT=0 git push origin master
```

克隆时用 `--filter=blob:none --no-checkout` + sparse-checkout 只拉 README，跳过 LFS 大文件，速度快且不会因 LFS 报错中断。

**Model Card 内容按 `$SKILL_PATH/references/model-card-spec.md` 生成。** 读取该文件，根据实际训练数据填充各字段；无法从训练配置、日志或数据集元数据获取的字段，直接省略对应行/句/节，不留任何占位符。完整字段规范、命名规则和示例见 spec 文件。

#### 第二步：设置模型元信息标签 + 确认公开状态

git push 只能更新 README 文件内容，**标签（多语言、任务方向、训练框架、基座模型）和公开状态必须通过网页端操作**。

**默认标签选择规则**（根据训练场景选择，不要只选"文本生成"一个）：

| 维度 | 选择规则 |
|------|---------|
| 多语言 | 中文数据 → **中文**；英文数据 → **English**；混合 → 两个都选 |
| 任务方向 | 问答对 / QA / 知识库 → **问答** + **文本生成**<br>对话 / Chat / 角色扮演 → **文本对话** + **文本生成**<br>分类/NER/抽取 → **文本分类** 或 **命名实体识别**<br>写作/摘要 → **文本生成**<br>医疗/法律/金融等专业领域 → 在以上基础上额外添加对应领域标签 |
| 训练框架 | ERNIE 系列 → **ERNIEKit**；Qwen/LLaMA 等 → **LlamaFactory** |
| 基座模型 | 搜索 base_model 名称（如 `ERNIE-4.5-0.3B`），选中匹配项 |

**优先使用 Playwright MCP 自动完成**（已验证可行）：

```
模型空间 Tab → 模型元信息 区域：
- 多语言：点击添加 → 按上表选中对应语言 → 确定
- 任务方向：点击添加 → 按上表选中 1-2 个任务标签 → 确定
- 训练框架：下拉选择 ERNIEKit（ERNIE 模型）或 LlamaFactory（开源模型）
- 基座模型：点击添加 → 搜索 base model 名称 → 选中 → 确定
填写 commit 信息 → 点击"完成编辑"保存
```

Playwright 操作要点（已踩坑）：
- 弹出的多选框不在 accessibility tree 里，必须用坐标点击：先用 `page.evaluate` 找 span 的坐标，再 `page.mouse.click`
- 训练框架是单选 combobox，直接点选项文本即可，无需确定按钮
- 基座模型有搜索框，在弹窗内找到 `input[placeholder="请输入搜索关键词"]` 并区分它和顶部导航搜索框（用坐标或 index 区分）
- 每个多选弹窗确认后，点"完成编辑"时需要填写 commit 信息，否则提交不会生效

**确认公开状态**（必须主动执行）：

导航到 `https://aistudio.baidu.com/modelsdetail/{MODEL_ID}/setModel`，检查右侧"其他设置"区域：
- 显示"当前模型状态为 **公开**" → 无需操作
- 显示私密 → 点击"设为公开"切换

在给用户的最终回复里，提供一个不超过 200 字的**模型简介**建议文本，用户可直接粘贴到星河社区模型简介框。

### LoRA 产物可用性确认

开源模型使用 `SFT/LoRA` 时，训练任务 `succeeded` 不等于产物一定可直接推理。训练结束后必须确认平台是否完成 **LoRA 合并导出**，而不是只上传 adapter。

检查顺序：
1. 查 system/stdout 日志，确认出现类似 `LoRA 合并配置 export_config.yaml 已生成`、导出/上传模型文件完成等信息
2. 查模型仓库文件，确认至少有 `model.safetensors`、`config.json`、`tokenizer.json`、`tokenizer_config.json`、`generation_config.json` 等直接推理所需文件
3. 用 AiStudio API 对最终模型仓库做一次最小调用测试，确认 `errorCode: 0` 且能返回正常文本

示例（避免把 token 展开到 `curl` 命令行参数中）：
```python
import os
import requests

resp = requests.post(
    "https://aistudio.baidu.com/llm/lmapi/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "Authorization": f"token {os.environ['AISTUDIO_ACCESS_TOKEN']}",
    },
    json={
        "model": "gitlogin/model_repo",
        "messages": [{"role": "user", "content": "1+1等于几？只输出答案。"}],
    },
    timeout=60,
)
print(resp.json())
```

判断标准：
- 如果模型仓库只有 LoRA adapter 文件，没有完整权重/配置/tokenizer，不能直接按完整模型调用；需要平台完成合并导出，或另走 adapter 加载流程
- 如果 API 调用成功，说明产物可作为完整模型使用；但可见性需到模型库网页确认，若仍私密则提醒用户改为公开
- 最终答复用户时要区分”训练任务成功”和”模型产物已实测可用”

---

## 常见错误速查

| 错误 | 原因 | 解决 |
|------|------|------|
| `waiting_data` 超过 10 分钟 | 常见根因：仓库/路径不对、文件名写错、LFS 指针、平台挂载异常 | 按序排查：1）确认 `repo_id` 来自详情页；2）核对 `--train-file` 和仓库实际文件名；3）确认文件大小接近本地文件；4）若 `is_lfs:true`，优先修复为普通 JSON/JSONL；5）下载回验 `--check-data`；6）若 system log 仍循环“正在等待数据集下载完成...”，说明可能是平台内部的数据集挂载任务卡住，保留 jobId 给平台排查，或取消后稍后重提 |
| 提交时报数据集权限错误 | 常见不是公开权限问题，而是仓库路径不对、`gitlogin` 不匹配或文件没传成功 | 确认真实 `gitlogin` 和完整 `repo_id`，用新版数据集仓库重新上传 |
| "非 ERNIE 格式" | Alpaca 格式，或 src/tgt 值是字符串非列表 | `{"src": ["问题"], "tgt": ["回答"]}` |
| "类型错误：期望 float，实际 str" | 超参数是字符串 | 去掉引号：`3` 不是 `"3"` |
| "PaddleFormers 仅支持 SFT/Full" | ERNIE 传了 LoRA | `--train-type "SFT/Full"` |
| 任务名报错 | name 含横杠 | 改用下划线 |
| 模型库满 | 超过 30 个仓库 | 删旧仓库或 `--output-repo` 复用 |
| 训练完成后模型上传失败 | `gitlogin` 或 `--output-repo` 命名空间不可写/未初始化 | 用 web-access/CDP 创建一个空数据集仓库初始化命名空间；确认 `--output-repo` 前缀是可写 `gitlogin` 后重新提交 |
| 微调后的模型再训练报错 | 平台白名单只允许官方模型 | 只能从官方基础模型重新训练，迭代时合并数据集 |
| code=401 | Token 过期 | 重新获取 Access Token |
| Qwen3 架构识别失败 | 平台 Transformers 版本旧 | 改用 `ModelHub/Qwen2.5-*` |
| 文件选择不符合预期 | 未传 `--train-file`，平台自动选择了首个 JSON/JSONL；或文件名写错 | 显式加 `--train-file 文件名.jsonl` |

```bash
python3 "$SKILL_PATH/scripts/train.py" --cancel JOB_ID  # 取消卡住的任务
```

---

## 脚本参数速查

```
--env-check                     检查本地环境（Python 版本、依赖、网络）
--verify-token                  验证 Token
--list-models                   列出可用模型（白名单）
--list-datasets                 列出内置推荐数据集
--check-data <file>             检查数据格式
--verify-upload --train-data REPO_ID --train-file F [--local-file FILE]
                                 验证上传结果并打印上传完成回执
--suggest-params <file> [--model-type ernie|llama]  推荐超参数
--submit ...                    提交训练任务
  --base-model MODEL              基底模型
  --train-type TYPE               训练类型（默认 SFT/Full；可选 SFT/LoRA）
  --train-data REPO_ID            数据集仓库路径，必须是详情页真实 repo_id，形如 gitlogin/repo_name
  --train-file FILENAME           数据文件名（如 train.jsonl）
  --params JSON                   超参数 JSON 字符串
  --name NAME                     任务名称（只允许字母、数字、下划线）
  --output-repo GITLOGIN/REPO     模型输出仓库（可选，超过 30 个仓库时复用；命名空间必须可写）
  --max-run-time HOURS            最长运行时间（小时，1-240）
--status <job_id>               查看任务状态
--logs <job_id> [--system]      查看训练日志（stdout loss）
--diagnose <job_id>             主动诊断状态、system log 和 stdout
--train-summary <job_id>         训练完成后汇报 loss/lr 趋势和健康状态
--poll <job_id>                 持续轮询（阻塞终端，对话场景不推荐）
--cancel <job_id>               取消任务

--api-key TOKEN / --env-file FILE  仅本地手工调试；自动化流程优先用环境变量
--base-url URL
```

---

## References 索引

| 文件 | 何时加载 |
|------|---------|
| `references/datasets.md` | 用户没有自己的数据，或问"用什么数据集好"时，先读此文件再推荐 |
| `references/aistudio_sdk_upload.md` | 需要用 SDK 上传数据集时 |
| `references/model-card-spec.md` | 训练 `succeeded` 后生成 README.md（Model Card）时读取，包含字段规范、命名规则和完整示例 |

