#!/usr/bin/env python3
"""AiStudio 无代码训练 CLI

用法：
  train.py --verify-token                         验证 Access Token
  train.py --check-data <file>                   检查数据格式
  train.py --verify-upload --train-data REPO_ID --train-file F [--local-file FILE]
                                                验证数据集上传结果并打印回执
  train.py --suggest-params <file> [--model-type ernie|llama]  推荐超参
  train.py --submit --base-model M --train-data REPO_ID [--train-file F] [--train-type T] [--params JSON]
  train.py --status <job_id>                     查看任务状态
  train.py --poll <job_id> [--interval N]        持续轮询（running 后自动打开 Tensorboard）
  train.py --open-tb <job_id>                    running 后打开 Tensorboard
  train.py --logs <job_id> [--system]            查看日志
  train.py --diagnose <job_id>                   主动诊断状态、system log 和 stdout
  train.py --export-artifacts <job_id> --out DIR 导出训练日志、指标 CSV 和曲线图
  train.py --cancel <job_id>                     取消任务
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import random
import re
import sys
import time
import warnings
import webbrowser
from pathlib import Path
from typing import Any
from urllib.parse import quote

warnings.filterwarnings(
    "ignore",
    message=r"urllib3 v2 only supports OpenSSL 1\.1\.1\+.*",
)

try:
    import requests
except ImportError:
    print("缺少依赖：requests\n安装命令：pip install requests")
    sys.exit(1)

BASE_URL = "https://train.aistudio-app.com"
GIT_BASE_URL = "https://git.aistudio.baidu.com"
TOKEN_ENV_VARS = ("AISTUDIO_ACCESS_TOKEN", "AISTUDIO_API_KEY")
WHITELIST_PATH = Path(__file__).parent.parent / "references" / "model_whitelist.yaml"
AISTUDIO_CLI_TOKEN_PATH = Path.home() / ".cache" / "aistudio" / ".auth" / "token"
COMMON_UPLOAD_WARN_BYTES = 5 * 1024 * 1024

# --------------------------- whitelist ---------------------------- #

def _load_whitelist() -> dict[str, dict[str, str]]:
    """解析 model_whitelist.yaml，不依赖 PyYAML。"""
    if not WHITELIST_PATH.exists():
        return {}
    result: dict[str, dict[str, str]] = {}
    current: str | None = None
    for line in WHITELIST_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not line.startswith(" ") and stripped.endswith(":"):
            current = stripped[:-1]
            result[current] = {}
        elif current and ":" in line:
            k, _, v = line.partition(":")
            result[current][k.strip()] = v.strip().strip('"').strip("'")
    return result


def _whitelist_tool(model: str) -> str:
    """返回模型对应的 train_tool，不在白名单返回空字符串。"""
    return _load_whitelist().get(model, {}).get("train_tool", "")

# ---------------------------- auth ---------------------------- #

def resolve_token(args: argparse.Namespace) -> tuple[str, str]:
    if args.api_key:
        return args.api_key.strip(), "--api-key"

    if args.env_file:
        p = Path(args.env_file)
        if not p.exists():
            die(f"env 文件不存在：{p}")
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition("=")
            if k.strip() in TOKEN_ENV_VARS and v.strip():
                return v.strip().strip('"').strip("'"), f"--env-file {p}"
        die(f"env 文件中未找到 AISTUDIO_ACCESS_TOKEN 或 AISTUDIO_API_KEY")

    for var in TOKEN_ENV_VARS:
        val = os.environ.get(var, "").strip()
        if val:
            return val, f"环境变量 {var}"

    if AISTUDIO_CLI_TOKEN_PATH.exists():
        val = AISTUDIO_CLI_TOKEN_PATH.read_text(encoding="utf-8", errors="replace").strip()
        if val:
            return val, f"AI Studio SDK 缓存 {AISTUDIO_CLI_TOKEN_PATH}"

    die(
        "未找到 Access Token。请通过以下任一方式提供：\n"
        "  export AISTUDIO_ACCESS_TOKEN='your_token'\n"
        "  --api-key 'your_token'\n"
        "  --env-file .aistudio.env\n\n"
        "如果已经用 aistudio config/login 配过，本脚本也会自动读取：\n"
        f"  {AISTUDIO_CLI_TOKEN_PATH}\n\n"
        "获取地址：https://aistudio.baidu.com/account/accessToken"
    )
    return "", ""  # unreachable; die() raises SystemExit


def load_token(args: argparse.Namespace) -> str:
    return resolve_token(args)[0]


def headers(token: str) -> dict:
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


def api(method: str, path: str, token: str, base_url: str, **kwargs) -> dict:
    url = base_url.rstrip("/") + path
    resp: requests.Response
    try:
        resp = requests.request(method, url, headers=headers(token), timeout=30, **kwargs)
    except requests.exceptions.ConnectionError:
        die(f"无法连接到 {base_url}，请检查网络或 --base-url 参数")
        return {}
    except requests.exceptions.Timeout:
        die("请求超时，请稍后重试")
        return {}

    if resp.status_code == 401:
        die("Token 认证失败（401）。请确认 Access Token 是否正确，或重新获取：\nhttps://aistudio.baidu.com/account/accessToken")
    if resp.status_code == 404:
        die(f"资源不存在（404）：{path}")

    body: dict
    try:
        body = resp.json()
    except Exception:
        die(f"API 返回了非 JSON 响应（HTTP {resp.status_code}）：\n{resp.text[:300]}")
        return {}

    if body.get("code") != 0:
        msg = body.get("msg", "未知错误")
        die(_format_api_error(body.get("code"), msg))

    return body.get("data") or {}


def git_contents(repo_id: str, file_path: str, token: str, ref: str = "master") -> dict:
    repo = quote(repo_id.strip().strip("/"), safe="/")
    path = quote(file_path.strip().strip("/"), safe="/")
    url = f"{GIT_BASE_URL}/api/v1/repos/{repo}/contents/{path}"
    try:
        resp = requests.get(
            url,
            headers={"Authorization": f"token {token}"},
            params={"ref": ref},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        die(f"无法连接到 {GIT_BASE_URL}，请检查网络")
        return {}
    except requests.exceptions.Timeout:
        die("请求 Git 仓库内容 API 超时，请稍后重试")
        return {}

    if resp.status_code == 401:
        die("Git 仓库 API 认证失败（401）。请确认 Access Token 是否正确，或重新运行 aistudio config。")
    if resp.status_code == 404:
        die(
            f"未在数据集仓库找到训练文件：{repo_id}/{file_path}\n"
            "请确认 repo_id 来自数据集详情页，且 --train-file 与仓库内文件名完全一致。"
        )

    try:
        body = resp.json()
    except Exception:
        die(f"Git 仓库 API 返回了非 JSON 响应（HTTP {resp.status_code}）：\n{resp.text[:300]}")
        return {}

    if resp.status_code >= 400:
        msg = body.get("message") or body.get("msg") or resp.text[:300]
        die(f"Git 仓库 API 错误（HTTP {resp.status_code}）：{msg}")

    return body


def _format_api_error(code: Any, msg: str) -> str:
    text = str(msg)
    lower = text.lower()
    permission_words = ("permission", "forbidden", "unauthorized", "access denied")
    dataset_words = ("dataset", "trainData", "数据集", "权限", "无权", "访问")

    if any(word in lower for word in permission_words) or any(word in text for word in dataset_words):
        return (
            f"API 错误（code={code}）：{msg}\n\n"
            "如果这里是在提交训练任务时访问数据集失败，优先检查数据集仓库和上传结果：\n"
            "  - 自建数据集先在 AiStudio 网页创建或确认数据集仓库，拿到真实 repo_id；\n"
            "  - `repo_id` 必须形如 gitlogin/repo_name，gitlogin 不能用昵称、展示名或邮箱猜；\n"
            "  - 用 `aistudio upload ... --repo-type dataset` 上传到已有仓库后，确认训练文件存在且 is_lfs=false。\n"
            "若系统日志反复停在 waiting_data，先核对 `--train-file`、is_lfs、文件大小和下载回验；"
            "这些都正常时，更可能是平台内部的数据集挂载任务（mount job）卡住，保留 jobId 给平台排查。"
        )

    return f"API 错误（code={code}）：{msg}"


def die(msg: str) -> None:
    print(f"\n错误：{msg}\n", file=sys.stderr)
    sys.exit(1)


# -------------------------- list models --------------------------- #

def cmd_list_models(_args: argparse.Namespace) -> None:  # noqa: ARG001
    wl = _load_whitelist()
    if not wl:
        print("白名单文件不存在或为空。")
        print(f"预期路径：{WHITELIST_PATH}")
        return

    paddle: list[str] = []
    llama: list[str] = []
    for model, meta in wl.items():
        tool = meta.get("train_tool", "")
        if tool == "paddleformers":
            paddle.append(model)
        elif tool == "llamafactory":
            llama.append(model)

    sep = "-" * 60
    print(f"\n{sep}")
    print("  平台可用模型（星河社区白名单）")
    print(f"{sep}")
    print("""
  说明：只有以下模型才能提交训练，不支持 HuggingFace 或其他
  来源的模型路径。算力限制：单卡最大支持 32B 参数以下的模型。
""")

    print("  【文心 ERNIE】  trainType: SFT/Full  数据格式: src/tgt JSONL")
    for m in paddle:
        meta = wl.get(m, {})
        note = meta.get("note", "")
        print(f"    {m:<48} {note}")

    print(f"\n  【开源模型】  trainType: SFT/Full / SFT/LoRA  数据格式: Alpaca 或 ShareGPT JSONL/JSON")
    for m in llama:
        meta = wl.get(m, {})
        family = m.split("/")[-1].split("-")[0] if "/" in m else m
        note = meta.get("note", family)
        print(f"    {m:<48} {note}")

    print(f"\n{sep}")
    print("  使用示例：")
    print("    --base-model 'PaddlePaddle/ERNIE-4.5-0.3B-PT'   # 默认 SFT/Full")
    print("    --base-model 'ModelHub/Qwen2.5-7B-Instruct'   --train-type 'SFT/LoRA'")
    print(f"{sep}\n")


# ------------------------ list datasets ----------------------- #

# 内置推荐数据集（全部托管在 AiStudio 星河，文件为 train.jsonl）
_BUILTIN_DATASETS = {
    "ernie": [
        {
            "path": "lmtyyz/Multilingual-Thinking",
            "file": "train.jsonl",
            "size": "2 MB",
            "desc": "多语言推理链数据",
            "note": "小体量，适合快速上手",
        },
        {
            "path": "lmtyyz/Nemotron-SFT-Safety-v1",
            "file": "train.jsonl",
            "size": "77 MB",
            "desc": "安全对齐 SFT 数据（Nemotron）",
            "note": "中等体量",
        },
        {
            "path": "lmtyyz/Bespoke-Stratos-17s",
            "file": "train.jsonl",
            "size": "288 MB",
            "desc": "Bespoke-Stratos 推理数据",
            "note": "大体量，训练时间较长",
        },
    ],
    "llama": [
        {
            "path": "lmtyyz/self-cognition",
            "file": "train.jsonl",
            "size": "23 KB",
            "desc": "自我认知（我是谁）",
            "note": "极小，仅验证流程用",
        },
        {
            "path": "lmtyyz/lima",
            "file": "train.jsonl",
            "size": "2.8 MB",
            "desc": "LIMA 高质量对话",
            "note": "小体量，质量高",
        },
        {
            "path": "lmtyyz/alpaca-gpt4-data-zh",
            "file": "train.jsonl",
            "size": "31 MB",
            "desc": "中文 Alpaca GPT-4 指令",
            "note": "中等体量，通用中文指令",
        },
        {
            "path": "lmtyyz/alpaca-gpt4-data-en",
            "file": "train.jsonl",
            "size": "38 MB",
            "desc": "英文 Alpaca GPT-4 指令",
            "note": "中等体量，通用英文指令",
        },
        {
            "path": "lmtyyz/school-math-0.25M",
            "file": "train.jsonl",
            "size": "119 MB",
            "desc": "数学题 0.25M 条",
            "note": "大体量，适合增强数学能力",
        },
        {
            "path": "lmtyyz/ShareGPT-Chinese-zh",
            "file": "train.jsonl",
            "size": "181 MB",
            "desc": "中文多轮对话（ShareGPT）",
            "note": "大体量，中文多轮",
        },
        {
            "path": "lmtyyz/Agent-FLAN",
            "file": "train.jsonl",
            "size": "137 MB",
            "desc": "Agent 工具调用微调数据",
            "note": "大体量，适合训练 Agent 能力",
        },
        {
            "path": "lmtyyz/WizardLM-evol-instruct-V2",
            "file": "train.jsonl",
            "size": "316 MB",
            "desc": "英文进化指令数据（WizardLM）",
            "note": "最大，全面提升英文指令能力",
        },
    ],
}


def cmd_list_datasets(_args: argparse.Namespace) -> None:
    sep = "-" * 62
    print(f"\n{sep}")
    print("  内置推荐数据集（已托管于 AiStudio，可直接用于训练）")
    print(f"{sep}\n")
    print("  说明：这些数据集已公开，无需自己准备数据，适合快速体验。")
    print("        使用时直接将 --train-data 和 --train-file 填入即可。\n")

    print("  【ERNIE 专用】  格式：src/tgt JSONL  trainType: SFT/Full")
    for ds in _BUILTIN_DATASETS["ernie"]:
        path_w = f"{ds['path']:<42}"
        print(f"    {path_w} {ds['size']:>8}  {ds['desc']}  [{ds['note']}]")

    print()
    print("  【开源模型专用】  格式：Alpaca 或 ShareGPT JSONL/JSON  trainType: SFT/Full / SFT/LoRA")
    for ds in _BUILTIN_DATASETS["llama"]:
        path_w = f"{ds['path']:<42}"
        print(f"    {path_w} {ds['size']:>8}  {ds['desc']}  [{ds['note']}]")

    print(f"\n{sep}")
    print("  使用示例（ERNIE + Multilingual-Thinking 数据集）：")
    print("    python3 train.py --submit \\")
    print("      --base-model 'PaddlePaddle/ERNIE-4.5-0.3B-PT' \\")
    print("      --train-data 'lmtyyz/Multilingual-Thinking' \\")
    print("      --train-file 'train.jsonl' \\")
    print("      --params '{\"num_train_epochs\": 1, \"max_steps\": 200}'")
    print()
    print("  使用示例（Qwen2.5 + lima 数据集）：")
    print("    python3 train.py --submit \\")
    print("      --base-model 'ModelHub/Qwen2.5-3B-Instruct' \\")
    print("      --train-type 'SFT/LoRA' \\")
    print("      --train-data 'lmtyyz/lima' \\")
    print("      --train-file 'train.jsonl' \\")
    print("      --params '{\"num_train_epochs\": 1, \"lora_rank\": 8}'")
    print(f"{sep}\n")


# -------------------------- verify token ---------------------- #

def cmd_verify_token(args: argparse.Namespace) -> None:
    token, source = resolve_token(args)
    print("正在验证 Access Token ...")
    print(f"Token 来源：{source}")
    # 用一个无副作用的 GET 端点验证（非 200→401 表示 token 无效）
    url = args.base_url.rstrip("/") + "/v1/train/jobs/__verify_probe__/progress"
    resp: requests.Response
    try:
        resp = requests.get(url, headers=headers(token), timeout=10)
    except Exception as e:
        die(f"连接失败：{e}")
        return

    if resp.status_code == 401:
        print("Token 无效或已过期。")
        if source.startswith("环境变量") and AISTUDIO_CLI_TOKEN_PATH.exists():
            print("提示：当前环境变量会覆盖 AI Studio SDK 缓存；若缓存 token 才是新 token，请先 unset AISTUDIO_ACCESS_TOKEN/AISTUDIO_API_KEY，或显式传 --api-key。")
        print("请重新获取：https://aistudio.baidu.com/account/accessToken")
        sys.exit(1)

    print("Token 有效！")


# -------------------------- check data ------------------------ #

def _is_sharegpt_obj(obj: dict) -> bool:
    return isinstance(obj.get("conversations"), list) or isinstance(obj.get("messages"), list)


def _validate_sharegpt_obj(obj: dict) -> str:
    messages = obj.get("conversations")
    role_key = "from"
    content_key = "value"
    user_roles = {"human", "user"}
    assistant_roles = {"gpt", "assistant"}

    if messages is None:
        messages = obj.get("messages")
        role_key = "role"
        content_key = "content"
        user_roles = {"user"}
        assistant_roles = {"assistant"}

    if not isinstance(messages, list) or not messages:
        return "ShareGPT 格式错误：conversations/messages 必须是非空列表"

    has_user = False
    has_assistant = False
    for i, msg in enumerate(messages, start=1):
        if not isinstance(msg, dict):
            return f"ShareGPT 格式错误：第 {i} 条消息必须是对象"
        role = msg.get(role_key)
        content = msg.get(content_key)
        if not isinstance(role, str) or not role:
            return f"ShareGPT 格式错误：第 {i} 条消息缺少 {role_key}"
        if not isinstance(content, str) or not content:
            return f"ShareGPT 格式错误：第 {i} 条消息缺少 {content_key}"
        if role in user_roles:
            has_user = True
        if role in assistant_roles:
            has_assistant = True

    # 偏好数据可以把回答放在 chosen/rejected 中，conversations 里不一定已有 assistant。
    has_preference_response = isinstance(obj.get("chosen"), (dict, str)) and isinstance(obj.get("rejected"), (dict, str))
    if not has_user:
        return "ShareGPT 格式错误：至少需要一条 human/user 消息"
    if not has_assistant and not has_preference_response:
        return "ShareGPT 格式错误：SFT 至少需要一条 gpt/assistant 回复"
    return ""


def detect_format_obj(obj: object) -> str:
    """返回 'ernie' | 'alpaca' | 'sharegpt' | 'unknown'"""
    if isinstance(obj, dict):
        if "src" in obj and "tgt" in obj:
            return "ernie"
        if ("instruction" in obj or "input" in obj) and ("output" in obj or ("chosen" in obj and "rejected" in obj)):
            return "alpaca"
        if _is_sharegpt_obj(obj):
            return "sharegpt"
    return "unknown"


def detect_format(line: str) -> str:
    """兼容旧调用：从一行 JSON 字符串检测格式。"""
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return "invalid_json"
    return detect_format_obj(obj)


def _load_data_records(path: Path) -> tuple[list[tuple[int, dict, str]], list[tuple[int, str, str]], str, int]:
    """读取 JSONL 或 JSON 数组，返回 (records, errors, kind, total_units)。"""
    text = path.read_text(encoding="utf-8", errors="replace")
    stripped = text.strip()
    if not stripped:
        return [], [], "empty", 0

    records: list[tuple[int, dict, str]] = []
    errors: list[tuple[int, str, str]] = []

    if stripped.startswith("["):
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as e:
            return [], [(1, stripped[:80], f"JSON 数组解析失败：{e}")], "json_array", 0
        if not isinstance(data, list):
            return [], [(1, stripped[:80], "JSON 顶层必须是数组或 JSONL 行")], "json_array", 0
        for idx, item in enumerate(data, start=1):
            preview = json.dumps(item, ensure_ascii=False)[:120]
            if isinstance(item, dict):
                records.append((idx, item, preview))
            else:
                errors.append((idx, preview, "JSON 数组元素必须是对象"))
        return records, errors, "json_array", len(data)

    lines = text.splitlines()
    non_empty = [(i + 1, l) for i, l in enumerate(lines) if l.strip()]
    for lineno, raw in non_empty:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            errors.append((lineno, raw[:80], "JSON 解析失败"))
            continue
        if isinstance(obj, dict):
            records.append((lineno, obj, raw[:120]))
        else:
            errors.append((lineno, raw[:80], "每行必须是 JSON 对象"))
    return records, errors, "jsonl", len(non_empty)


def cmd_check_data(args: argparse.Namespace) -> None:
    path = Path(args.check_data)
    if not path.exists():
        die(f"文件不存在：{path}")

    print(f"正在检查：{path}")
    records, errors, data_kind, total_units = _load_data_records(path)

    if data_kind == "empty":
        die("文件为空，没有可用数据。")

    # 检测格式
    format_counts: dict[str, int] = {}

    for lineno, obj, preview in records:
        fmt = detect_format_obj(obj)
        format_counts[fmt] = format_counts.get(fmt, 0) + 1
        if fmt in ("invalid_json", "unknown"):
            errors.append((lineno, preview[:80], "字段不符合 ERNIE、Alpaca 或 ShareGPT 格式"))
        elif fmt == "ernie":
            # 额外校验：src/tgt 必须是 list，不能是裸字符串
            if not isinstance(obj.get("src"), list) or not isinstance(obj.get("tgt"), list):
                errors.append((lineno, preview[:80], 'ERNIE 格式错误：src/tgt 必须是列表 ["..."]，不能是裸字符串'))
        elif fmt == "sharegpt":
            reason = _validate_sharegpt_obj(obj)
            if reason:
                errors.append((lineno, preview[:80], reason))

    detected = max(format_counts.keys(), key=lambda k: format_counts[k]) if format_counts else "invalid_json"

    print(f"\n{chr(45) * 50}")
    kind_label = "JSON 数组" if data_kind == "json_array" else "JSONL"
    print(f"  文件容器：        {kind_label}")
    print(f"  记录数：          {total_units}")
    print(f"  检测到格式：      {_fmt_label(detected)}")

    if errors:
        print(f"\n  发现 {len(errors)} 处格式错误：")
        for lineno, preview, reason in errors[:10]:
            print(f"    第 {lineno} 行 [{reason}]：{preview}")
        if len(errors) > 10:
            print(f"    ... 还有 {len(errors) - 10} 处错误（只显示前 10 条）")
    else:
        print(f"  格式检查：        全部通过 [OK]")

    # 混合格式警告
    open_formats = [f for f in ("alpaca", "sharegpt") if f in format_counts]
    if "ernie" in format_counts and open_formats:
        print("\n  警告：文件中同时包含 ERNIE 格式和开源模型格式，可能混淆了两种数据集！")
    elif len(open_formats) > 1:
        print("\n  警告：文件中同时包含 Alpaca 和 ShareGPT 格式，建议统一成一种格式后再训练。")

    # 如果 ERNIE 用户用了 Alpaca 格式，提示转换
    if detected == "alpaca":
        print("""
  提示：如果要训练 ERNIE 模型，需要将格式转换为 ERNIE 格式：
    {"src": ["问题或指令"], "tgt": ["期望回答"]}

  快速转换脚本（Python）：
    import json

    def load_alpaca(path):
        text = open(path, encoding="utf-8").read().strip()
        if text.startswith("["):
            return json.loads(text)
        return [json.loads(line) for line in text.splitlines() if line.strip()]

    with open("ernie_data.jsonl", "w", encoding="utf-8") as fout:
        for d in load_alpaca("alpaca_data.json"):
            text = d.get("instruction", "")
            if d.get("input"):
                text += "\\n" + d["input"]
            fout.write(json.dumps({"src": [text], "tgt": [d["output"]]}, ensure_ascii=False) + "\\n")
""")
    elif detected == "sharegpt":
        print("""
  提示：ShareGPT 是 LlamaFactory 开源模型格式，可以直接用于 Qwen/DeepSeek 等开源模型。
  如果要训练 ERNIE 模型，需要先转换为 ERNIE 的 src/tgt JSONL。
""")

    # 样本量建议
    invalid_record_lines = {lineno for lineno, _preview, _reason in errors}
    n = len([1 for lineno, _obj, _preview in records if lineno not in invalid_record_lines])
    print(f"\n  有效样本数：{n}")
    if n < 50:
        print("  建议：样本量偏少（< 50），仅适合验证流程是否跑通，效果不保证。")
    elif n < 1000:
        print("  建议：样本量适合验证特定场景，期望全面提升建议达到 1,000 条以上。")
    elif n < 10000:
        print("  建议：样本量良好，适合特定场景微调。")
    else:
        print("  建议：样本量充足，适合较全面的能力提升训练。")

    print(f"{chr(45) * 50}\n")

    if errors:
        print("存在格式错误，建议修复后再提交训练任务。")
        sys.exit(1)
    elif detected == "alpaca":
        print("Alpaca 格式检查通过。")
        print("-> 训练开源模型（Qwen/DeepSeek 等）：可以直接提交。")
        print("-> 训练 ERNIE 模型：需要先用上面的脚本转换为 src/tgt 格式。")
    elif detected == "sharegpt":
        print("ShareGPT 格式检查通过。")
        print("-> 训练开源模型（Qwen/DeepSeek 等）：可以直接提交。")
        print("-> 训练 ERNIE 模型：需要先转换为 src/tgt 格式。")
    else:
        print("ERNIE 格式检查通过，可以提交训练任务。")


def _fmt_label(fmt: str) -> str:
    return {
        "ernie": "ERNIE 格式（src/tgt）",
        "alpaca": "Alpaca 格式（instruction/input/output）",
        "sharegpt": "ShareGPT 格式（conversations/from/value 或 messages/role/content）",
        "unknown": "未知格式",
        "invalid_json": "JSON 解析失败",
    }.get(fmt, fmt)


# ------------------------ suggest params --------------------- #

def cmd_suggest_params(args: argparse.Namespace) -> None:
    path = Path(args.suggest_params)
    if not path.exists():
        die(f"文件不存在：{path}")

    records, _errors, data_kind, _total_units = _load_data_records(path)
    valid_lines = [json.dumps(obj, ensure_ascii=False) for _idx, obj, _preview in records]
    n = len(valid_lines)
    if not n:
        die("文件中没有可解析的有效训练样本，请先用 --check-data 检查数据格式。")

    lengths = [len(line) for line in valid_lines]
    avg_len = sum(lengths) / len(lengths)
    p50_len = _percentile(lengths, 0.50)
    p90_len = _percentile(lengths, 0.90)
    p95_len = _percentile(lengths, 0.95)
    max_len = max(lengths)
    suggested_seq_len = min(2048, max(128, int(math.ceil(p90_len / 64) * 64)))

    model_type = (args.model_type or "ernie").lower()
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    print(f"\n{chr(45) * 50}")
    print(f"  数据集：{path.name}")
    print(f"  文件容器：{'JSON 数组' if data_kind == 'json_array' else 'JSONL'}")
    print(f"  样本数：{n}")
    print(f"  文件大小：{size_bytes} bytes ({size_mb:.2f} MB)")
    print(f"  字符长度：avg={avg_len:.0f}  p50={p50_len:.0f}  p90={p90_len:.0f}  p95={p95_len:.0f}  max={max_len}")
    print(f"  建议 max_seq_len / cutoff_len：{suggested_seq_len}")
    print(f"  模型类型：{model_type}")
    if _is_json_training_file(path) and size_bytes > COMMON_UPLOAD_WARN_BYTES:
        print("\n  [!] 上传风险：普通 JSON/JSONL 文件超过约 5MB，AI Studio 可能拒绝普通文件上传。")
        print("      不要改用 LFS 训练文件；优先 compact/crop/split，并保留 manifest。")
    print("-" * 50)

    if model_type == "ernie":
        _suggest_ernie(n, suggested_seq_len)
    else:
        _suggest_llama(n, suggested_seq_len)


def _suggest_ernie(n: int, seq_len: int) -> None:
    if n < 500:
        epochs, batch, lr = 5, 4, 5e-5
        note = "数据量较少，适当增加 epochs"
    elif n < 5000:
        epochs, batch, lr = 3, 4, 5e-5
        note = "适中数据量，使用默认配置"
    else:
        epochs, batch, lr = 3, 8, 3e-5
        note = "数据量较大，可适当增大 batch"

    params = {
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch,
        "learning_rate": lr,
        "max_seq_len": seq_len,
        "max_steps": -1,
        "warmup_steps": min(100, max(10, n // 10)),
        "logging_steps": 5,
        "save_steps": 500,
        "bf16": True,
    }
    _print_params(params, note, "PaddleFormers（ERNIE）")


def _suggest_llama(n: int, seq_len: int) -> None:
    if n < 500:
        epochs, batch, lr = 5, 4, 2e-4
        rank = 8
        note = "数据量较少，适当增加 epochs"
    elif n < 5000:
        epochs, batch, lr = 3, 4, 2e-4
        rank = 8
        note = "适中数据量，使用默认 LoRA 配置"
    else:
        epochs, batch, lr = 3, 8, 1e-4
        rank = 16
        note = "数据量较大，可适当增大 lora_rank 和 batch"

    params = {
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch,
        "learning_rate": lr,
        "cutoff_len": seq_len,
        "lora_rank": rank,
        "lora_alpha": rank * 2,
        "lora_dropout": 0.05,
        "warmup_ratio": 0.03,
        "logging_steps": 5,
        "save_steps": 500,
        "fp16": True,
    }
    _print_params(params, note, "LlamaFactory（开源模型）")


def _print_params(params: dict, note: str, framework: str) -> None:
    print(f"\n  框架：{framework}")
    print(f"  说明：{note}")
    print(f"\n  推荐超参数：")
    for k, v in params.items():
        display = json.dumps(v) if isinstance(v, bool) else v
        print(f"    {k}: {display}")
    print(f"\n  JSON 格式（可直接粘贴到 --params）：")
    print(f"  {json.dumps(params, ensure_ascii=False)}")
    print(f"\n  提示：可以调整这些参数，调整后加 --params '{{...}}' 传给 --submit 命令。")
    print(f"  重要：超参数类型必须是数字/布尔，不能加引号。")
    print()


def _percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(ordered[int(pos)])
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def _is_json_training_file(path: Path) -> bool:
    return path.suffix.lower() in {".json", ".jsonl"}


# ------------------------------ submit ----------------------- #

def cmd_submit(args: argparse.Namespace) -> None:
    token = load_token(args)

    # 白名单校验
    tool = _whitelist_tool(args.base_model)
    if not tool:
        wl = _load_whitelist()
        if wl:  # 白名单存在但模型不在里面
            die(
                f"模型 '{args.base_model}' 不在平台白名单中，无法提交训练。\n"
                f"请运行以下命令查看所有可用模型：\n"
                f"  python3 {Path(__file__).name} --list-models"
            )
    # 框架与 trainType 一致性检查
    VALID_TRAIN_TYPES = {"SFT/Full", "SFT/LoRA", "Post-PreTrain"}
    if args.train_type not in VALID_TRAIN_TYPES:
        # 尝试大小写纠正
        normalized = next((t for t in VALID_TRAIN_TYPES if t.lower() == args.train_type.lower()), None)
        hint = f"\n你可能想用：--train-type '{normalized}'" if normalized else ""
        die(
            f"trainType '{args.train_type}' 不合法，支持的类型：{', '.join(sorted(VALID_TRAIN_TYPES))}"
            + hint
        )
    if tool == "paddleformers" and args.train_type != "SFT/Full":
        die(
            f"文心 ERNIE 模型只支持 SFT/Full，不支持 {args.train_type}。\n"
            f"请改为：--train-type 'SFT/Full'"
        )
    if tool == "llamafactory" and args.train_type == "SFT/Full":
        print("提示：开源模型全量微调（SFT/Full）显存消耗较大。")
        print()


    # 构建请求体
    payload: dict[str, Any] = {
        "baseModel": args.base_model,
        "trainType": args.train_type,
        "trainData": [args.train_data],
    }

    if args.train_file:
        payload["trainDataFiles"] = {args.train_data: {"train": args.train_file}}

    if args.name:
        _validate_name(args.name)
        payload["name"] = args.name

    if args.description:
        payload["description"] = args.description

    if args.output_repo:
        payload["modelOutputRepo"] = args.output_repo

    if args.max_run_time:
        payload["maxRunTime"] = args.max_run_time

    hp: dict = {}
    if args.params:
        try:
            hp = json.loads(args.params)
        except json.JSONDecodeError as e:
            die(f"--params 不是合法的 JSON：{e}")
            return

    # 可视化参数（report_to/visualdl）由平台后端自动注入，用户无需传也不能传

    if hp:
        _validate_hyperparams(hp, args.train_type, tool)
        payload["hyperparameters"] = hp

    print(f"\n提交训练任务：")
    print(f"  基底模型：{args.base_model}")
    print(f"  训练类型：{args.train_type}")
    print(f"  数据集：  {args.train_data}/{args.train_file or '（自动选择）'}")
    if payload.get("hyperparameters"):
        print(f"  超参数：  {json.dumps(payload['hyperparameters'])}")
    print()

    data = api("POST", "/v1/train/jobs", token, args.base_url, json=payload)
    job_id = data.get("jobId", "")

    if not job_id:
        die(f"提交成功但未返回 jobId，响应：{data}")

    print(f"任务已提交！\n")
    print(f"  Job ID：{job_id}")
    print(f"\n查看状态：")
    print(f"  python3 {Path(__file__).name} --status {job_id}")
    print(f"\n持续轮询进度：")
    print(f"  python3 {Path(__file__).name} --poll {job_id}")


def _validate_name(name: str) -> None:
    import re
    if not re.match(r'^[a-zA-Z0-9_一-鿿]*$', name):
        die(f"任务名称不能包含横杠或特殊字符，只允许字母、数字、下划线。\n当前：{name}")


_STRING_PARAMS = {"report_to", "save_strategy", "lr_scheduler_type", "optim", "fsdp", "logging_dir", "output_dir"}

# 各框架平台不支持的参数（实测结论）
_LLAMAFACTORY_UNSUPPORTED = {"max_steps", "max_seq_len", "bf16", "warmup_steps"}
_PADDLEFORMERS_UNSUPPORTED = {"cutoff_len", "lora_rank", "lora_alpha", "lora_dropout", "warmup_ratio", "fp16"}


def _validate_hyperparams(hp: dict, train_type: str, tool: str | None = None) -> None:
    errors = []
    for k, v in hp.items():
        if isinstance(v, str) and k not in _STRING_PARAMS:
            errors.append(f"  {k}: \"{v}\"  <- 应该是数字或布尔，不能是字符串")
    if errors:
        die("超参数类型错误，请去掉值的引号：\n" + "\n".join(errors))

    is_llama = tool == "llamafactory" or train_type == "SFT/LoRA"
    is_paddle = tool == "paddleformers"

    if is_llama:
        bad = [k for k in hp if k in _LLAMAFACTORY_UNSUPPORTED]
        if bad:
            die(f"以下参数 LlamaFactory 不支持，提交会被平台拒绝：{bad}\n"
                f"  max_steps -> 改用 num_train_epochs\n"
                f"  max_seq_len -> 改用 cutoff_len\n"
                f"  bf16/warmup_steps -> 改用 fp16/warmup_ratio")
    elif is_paddle:
        bad = [k for k in hp if k in _PADDLEFORMERS_UNSUPPORTED]
        if bad:
            die(f"以下参数 PaddleFormers（ERNIE）不支持，提交会被平台拒绝：{bad}\n"
                f"  cutoff_len -> 改用 max_seq_len\n"
                f"  fp16/warmup_ratio -> 改用 bf16/warmup_steps\n"
                f"  lora_rank/lora_alpha -> ERNIE 不支持 LoRA")


# --------------------------- upload verify -------------------- #

def cmd_verify_upload(args: argparse.Namespace) -> None:
    token = load_token(args)
    info = git_contents(args.train_data, args.train_file, token)

    is_lfs = info.get("is_lfs")
    size = info.get("size")
    sha = info.get("sha", "")
    html_url = info.get("html_url", "")
    local_size: int | None = None

    if args.local_file:
        p = Path(args.local_file)
        if not p.exists():
            die(f"本地文件不存在，无法对比大小：{p}")
        local_size = p.stat().st_size

    print("\n数据集上传校验结果")
    print("-" * 55)
    print(f"  数据集 repo_id： {args.train_data}")
    print(f"  训练文件：       {args.train_file}")
    print(f"  is_lfs：         {str(is_lfs).lower()}")
    print(f"  仓库文件大小：   {size} bytes")
    if local_size is not None:
        print(f"  本地文件大小：   {local_size} bytes")
    if sha:
        print(f"  sha：            {sha}")
    if html_url:
        print(f"  文件页：         {html_url}")
    print("-" * 55)

    if is_lfs is not False:
        msg = (
            "训练文件 is_lfs:true。推荐修复为普通 JSON/JSONL（is_lfs:false），"
            "这样下载回验和 waiting_data 排查更可靠；实测部分 LFS 文件也可能训练成功。"
        )
        if getattr(args, "strict_lfs", False):
            die(
                msg + "\n"
                "请到数据集文件页编辑 .gitattributes，删除训练文件扩展名对应的 LFS 规则，"
                "再删除旧训练文件并重新上传；或确认风险后去掉 --strict-lfs。"
            )
        print(f"[!]  {msg}")
    elif local_size is not None and isinstance(size, int) and local_size != size:
        print("[!]  仓库文件大小与本地文件不一致，请下载回本地后再跑 --check-data 确认内容。")
    else:
        print("[OK] 上传完成，训练文件是普通文件（is_lfs:false）。")

    print("\n提交训练时使用：")
    print(f"  --train-data {args.train_data}")
    print(f"  --train-file {args.train_file}")
    if is_lfs is not False:
        print("  # 注意：当前训练文件 is_lfs:true。若后续卡在 waiting_data，先修复 LFS 后重提。")
    print()


# ------------------------------ env check -------------------- #

def cmd_env_check() -> None:
    import subprocess
    import shutil
    ok = True

    # Python 版本
    v = sys.version_info
    if v >= (3, 8):
        print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")
    else:
        print(f"[X]  Python {v.major}.{v.minor}.{v.micro}  需要 3.8+")
        ok = False

    # requests
    try:
        import requests as _r
        print(f"[OK] requests {_r.__version__}")
    except ImportError:
        print("[X]  requests 未安装，正在尝试自动安装...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "requests", "-q"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("[OK] requests 安装成功")
        else:
            print(f"[X]  requests 安装失败，请手动运行：pip install requests\n{result.stderr.strip()}")
            ok = False

    # aistudio CLI
    cli = shutil.which("aistudio")
    user_base = subprocess.check_output([sys.executable, "-m", "site", "--user-base"], text=True).strip()
    user_cli = Path(user_base) / "bin" / "aistudio"
    if cli:
        print(f"[OK] aistudio CLI {cli}")
    elif user_cli.exists():
        print(f"[!]  aistudio CLI 已安装但不在 PATH：{user_cli}")
        print(f"     可临时使用：AISTUDIO_CLI=\"{user_cli}\"")
    else:
        print("[!]  aistudio CLI 未安装；自建数据集上传前运行：python3 -m pip install --upgrade aistudio-sdk")

    env_tokens = [var for var in TOKEN_ENV_VARS if os.environ.get(var, "").strip()]
    if env_tokens and AISTUDIO_CLI_TOKEN_PATH.exists():
        print(f"[!]  已设置 {', '.join(env_tokens)}，会覆盖 SDK 缓存 token：{AISTUDIO_CLI_TOKEN_PATH}")
        print("     若验证失败，先 unset 旧环境变量，或显式传 --api-key。")

    # 网络连通性（只做 DNS，不发 API 请求）
    import socket
    try:
        socket.setdefaulttimeout(5)
        socket.getaddrinfo("train.aistudio-app.com", 443)
        print("[OK] 网络可访问 train.aistudio-app.com")
    except OSError:
        print("[X]  无法访问 train.aistudio-app.com，请检查网络或代理设置")
        ok = False

    print()
    if ok:
        print("环境自检通过，可以开始使用。")
    else:
        print("存在问题，请按上方提示修复后重试。")
        sys.exit(1)


# ------------------------------ status ----------------------- #

def cmd_status(args: argparse.Namespace) -> None:
    token = load_token(args)
    data = api("GET", f"/v1/train/jobs/{args.status}", token, args.base_url)
    _print_status(data)


def _print_status(data: dict) -> None:
    state = data.get("state", "未知")
    phase = data.get("currentPhase", "")
    error = data.get("errorMsg", "")
    tb_url = data.get("tensorboardUrl", "")
    log_url = data.get("logUrl", "")
    start = data.get("startTime", "")
    end = data.get("endTime", "")
    progress = data.get("trainingProgress") or {}
    output = data.get("modelOutputRepo")

    print(f"\n{chr(45) * 55}")
    print(f"  状态：        {_state_label(state, phase)}")
    if start:
        print(f"  开始时间：    {start}")
    if end:
        print(f"  结束时间：    {end}")

    if state == "running" and phase == "training" and progress:
        cur = progress.get("currentStep")
        tot = progress.get("totalSteps")
        elapsed = progress.get("elapsedTime", "")
        remaining = progress.get("remainingTime", "")
        loss = progress.get("trainLoss")

        if cur is not None and tot:
            pct = int(cur / tot * 100)
            bar = "#" * (pct // 5) + "-" * (20 - pct // 5)
            print(f"  进度：        [{bar}] {pct}%  ({cur}/{tot} 步)")
        if elapsed:
            print(f"  已耗时：      {elapsed}")
        if remaining:
            print(f"  预计剩余：    {remaining}")
        if loss is not None:
            print(f"  训练 Loss：   {loss:.4f}")

    if state == "succeeded" and output:
        repo = output.get("modelRepo", "")
        branch = output.get("branch", "")
        print(f"\n  模型仓库：    {repo}")
        print(f"  分支：        {branch}")

    if error:
        print(f"\n  错误信息：    {error}")

    if tb_url:
        if state == "running":
            print(f"\n  Tensorboard（实时 Loss 曲线）：")
        elif state in {"waiting_data", "pending"}:
            print(f"\n  Tensorboard（地址已生成，running 后再打开）：")
        else:
            print(f"\n  Tensorboard（训练已结束，实时数据流关闭，历史快照可能仍可访问）：")
        print(f"  {tb_url}")

    if log_url:
        print(f"\n  日志地址：    {log_url}")

    print(f"{chr(45) * 55}\n")

    if state == "waiting_data":
        print("提示：平台正在下载/挂载模型和数据集，通常需要 1-10 分钟。")
        if tb_url:
            print("Tensorboard 地址已生成，但要等进入 running 后再打开才有内容。")
        print("如果超过 10 分钟，按序核对 repo_id、--train-file、is_lfs、文件大小和下载回验。")
        print("若这些都正常且 system log 反复等待数据集下载，通常是平台内部的数据集挂载任务卡住，保留 jobId 便于排查。")
    elif state == "pending":
        print("提示：任务已进入队列，等待 GPU 调度，通常 1-5 分钟。")
    elif state == "running" and phase == "training":
        if tb_url:
            print(f"Tensorboard 可访问：{tb_url}")
        print("Loss 解读：持续下降 = 正常；每个 epoch 开始时突然上升 = 正常；")
        print("         一直不降或乱跳 = 数据质量有问题，先检查格式。")
    elif state == "succeeded":
        _print_test_guide(output or {})
    elif state == "failed":
        print("任务失败。常见原因：")
        print("  1. 数据格式错误（用 --check-data 检查）")
        print("  2. 超参数类型错误（值不能是字符串）")
        print("  3. 数据集为空（确认文件已上传）")


def _print_test_guide(output: dict) -> None:
    repo = output.get("modelRepo", "")
    separator = "=" * 55

    print(separator)
    print("  训练完成！以下是测试和使用模型的完整指南")
    print(separator)

    if repo:
        print(f"""
【第一步：API 调用测试模型（推荐先做）】

  训练完的模型通过 AiStudio API 调用，示例：

  curl -X POST https://aistudio.baidu.com/llm/lmapi/v1/chat/completions \\
    -H "Content-Type: application/json" \\
    -H "Authorization: token <your_token>" \\
    -d '{{
      "model": "{repo}",
      "messages": [{{"role": "user", "content": "你的测试问题"}}]
    }}'
""")

    print("""【第二步：验证微调效果的问题类型】

  推荐用以下几类问题测试，判断模型有没有真的学进去：

  ① 训练集内的问题（验证是否学到）
     -> 发一条 src 字段的原文，看回答是否接近 tgt
     -> 如果完全一样，可能过拟合；如果语义相近，效果良好

  ② 训练集外但同类型问题（验证是否泛化）
     -> 换个说法问同一类问题，看风格、术语是否统一
     -> 这是最重要的测试，体现微调是否有迁移效果

  ③ 无关问题（验证是否破坏基础能力）
     -> 问一些通用常识，确认模型没有遗忘基础能力
""")

    print("""【第三步：效果诊断标准】

  效果好的信号：
  [OK] 回答风格、用词与训练数据一致
  [OK] 对领域专有名词的处理明显改善
  [OK] 拒绝回答领域外问题（如果训练数据暗示这样做）

  效果不好的原因（按优先级排查）：
  1. 数据质量 — 答案是否准确、表述是否一致（最重要）
  2. 数据量 — 太少容易过拟合，建议至少 1,000 条
  3. 超参数 — 尝试调小学习率（1e-5）或增加 epochs
""")

    if repo:
        print(f"""【第四步：API 调用（用代码接入）】

  训练完的模型可以通过 AiStudio API 调用：
  模型仓库：{repo}
  参考文档：https://aistudio.baidu.com/doc/model-api
""")

        print("""【第五步：发布状态】

  训练产物已经上传到模型仓库。Skill 默认按公开发布处理；
  如果 AiStudio 网页端显示该模型仍是私密，请到模型库页面改为公开，
  并补充模型卡片、开源协议和可见性设置。
  API 调用成功代表当前账号/Token 下模型可用；是否已公开展示，
  仍需要以 AiStudio 模型库网页端为准。
""")

    print("如果效果不达预期，告诉我具体现象，我来帮你分析原因。")
    print(separator)


def _open_url(url: str) -> None:
    try:
        webbrowser.open(url)
        print(f"已在浏览器打开：{url}")
    except Exception:
        print(f"请手动打开：{url}")


def _state_label(state: str, phase: str) -> str:
    labels = {
        "waiting_data": "等待数据 (waiting_data) — 正在下载/挂载模型和数据集",
        "pending": "等待调度 (pending) — 等待 GPU 分配",
        "running": f"训练中 (running/{phase})" if phase else "运行中 (running)",
        "succeeded": "训练成功 (succeeded) [OK]",
        "failed": "训练失败 (failed) [FAIL]",
        "cancelled": "已取消 (cancelled)",
    }
    return labels.get(state, state)


# -------------------------------- poll ------------------------ #

def cmd_poll(args: argparse.Namespace) -> None:
    token = load_token(args)
    interval = args.interval or 30
    job_id = args.poll

    print(f"开始轮询任务 {job_id}，每 {interval} 秒刷新一次（Ctrl+C 停止）\n")

    terminal_states = {"succeeded", "failed", "cancelled"}
    tb_opened = False
    waiting_data_since: float | None = None
    waiting_system_checked = False

    try:
        while True:
            data = api("GET", f"/v1/train/jobs/{job_id}", token, args.base_url)
            state = data.get("state", "")
            phase = data.get("currentPhase", "")

            # 单行进度（先读，供 Tensorboard 判断用）
            progress = data.get("trainingProgress") or {}
            cur = progress.get("currentStep")
            tot = progress.get("totalSteps")
            remaining = progress.get("remainingTime", "")

            # Tensorboard：等首个 step 数据写入后再打开，避免打开空看板
            if not tb_opened and state == "running" and cur is not None and cur > 0:
                tb_url = data.get("tensorboardUrl", "")
                if tb_url:
                    print(f"\n  训练已产生数据（step={cur}），自动打开 Tensorboard…")
                    _open_url(tb_url)
                    print()
                    tb_opened = True

            ts = time.strftime("%H:%M:%S")
            if cur and tot:
                pct = int(cur / tot * 100)
                bar = "#" * (pct // 10) + "-" * (10 - pct // 10)
                suffix = f"  [{bar}] {pct}%  剩余 {remaining}" if remaining else f"  [{bar}] {pct}%"
            else:
                suffix = ""

            print(f"  [{ts}]  {_state_label(state, phase)}{suffix}")

            if state == "waiting_data":
                if waiting_data_since is None:
                    waiting_data_since = time.time()
                elapsed = time.time() - waiting_data_since
                if not waiting_system_checked and elapsed >= 600:
                    print("\nwaiting_data 已超过 10 分钟，主动查看 system log：")
                    _print_job_log(job_id, token, args.base_url, system=True, limit=80, soft=True)
                    print()
                    waiting_system_checked = True
            else:
                waiting_data_since = None

            if state in terminal_states:
                print()
                _print_status(data)
                if state in {"failed", "cancelled"}:
                    print("主动查看 system log：")
                    _print_job_log(job_id, token, args.base_url, system=True, limit=120, soft=True)
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n轮询已停止。")
        print(f"再次查询：python3 {Path(__file__).name} --status {job_id}")


# -------------------------------- logs ------------------------ #

def _fetch_job_log(
    job_id: str,
    token: str,
    base_url: str,
    system: bool = False,
    byte_limit: int = 409500,
    soft: bool = False,
) -> str:
    log_file = "system.log" if system else "master/output.log"
    url = base_url.rstrip("/") + f"/v1/train/jobs/{job_id}/{log_file}"
    safe_limit = max(1, int(byte_limit or 409500))
    # AI Studio 日志接口不支持 suffix range（bytes=-N），只支持 bytes=start-end。
    byte_range = f"bytes=0-{safe_limit - 1}"

    resp: requests.Response
    try:
        resp = requests.get(url, headers={**headers(token), "Range": byte_range}, timeout=30)
    except Exception as e:
        msg = f"获取{'system log' if system else 'stdout 日志'}失败：{e}"
        if soft:
            return f"[!] {msg}"
        die(msg)
        return ""

    if resp.status_code == 404:
        msg = "system log 不可访问或尚未生成" if system else "stdout 日志尚未生成（任务可能尚未进入 running，或 job_id 错误）"
        if soft:
            return f"[!] {msg}"
        die(msg)
    if resp.status_code == 401:
        if soft:
            return "[!] Token 认证失败"
        die("Token 认证失败")

    content = resp.content.decode("utf-8", errors="replace")

    # 检测是否是 JSON 错误响应（如 {"code":10002,"msg":"任务不存在"}）
    if content.strip().startswith("{"):
        try:
            err = json.loads(content)
            if isinstance(err, dict) and err.get("code", 0) != 0:
                msg = _format_api_error(err["code"], err.get("msg", content))
                if soft:
                    return f"[!] {msg}"
                die(msg)
        except json.JSONDecodeError:
            pass

    return content


def _print_log_tail(content: str, limit: int = 100) -> None:
    if not content.strip():
        print("日志为空（任务可能还在等待中）")
        return

    lines = content.splitlines()
    if len(lines) > limit:
        print(f"（共 {len(lines)} 行，显示最后 {limit} 行）\n")
        lines = lines[-limit:]

    print("\n".join(lines))


def _print_job_log(job_id: str, token: str, base_url: str, system: bool = False, limit: int = 100, soft: bool = False) -> None:
    content = _fetch_job_log(job_id, token, base_url, system=system, soft=soft)
    if content.startswith("[!] "):
        print(content)
        return
    _print_log_tail(content, limit=limit)


def cmd_logs(args: argparse.Namespace) -> None:
    token = load_token(args)
    _print_job_log(args.logs, token, args.base_url, system=args.system)


def cmd_diagnose(args: argparse.Namespace) -> None:
    token = load_token(args)
    job_id = args.diagnose
    data = api("GET", f"/v1/train/jobs/{job_id}", token, args.base_url)

    print("\n任务状态")
    _print_status(data)

    print("System log（主动排查平台下载/挂载/调度）：")
    _print_job_log(job_id, token, args.base_url, system=True, limit=120, soft=True)

    print("\nStdout 日志（训练 loss/脚本错误）：")
    _print_job_log(job_id, token, args.base_url, system=False, limit=80, soft=True)


# -------------------------- train summary -------------------- #

def _fetch_log_content(job_id: str, token: str, base_url: str) -> str:
    url = base_url.rstrip("/") + f"/v1/train/jobs/{job_id}/master/output.log"
    try:
        resp = requests.get(url, headers={**headers(token), "Range": "bytes=0-819200"}, timeout=30)
    except Exception as e:
        die(f"获取训练日志失败：{e}")
        return ""
    if resp.status_code == 404:
        die("日志文件不存在（任务可能还未开始，或 job_id 错误）")
    if resp.status_code == 401:
        die("Token 认证失败")
    content = resp.content.decode("utf-8", errors="replace")
    if content.strip().startswith("{"):
        try:
            err = json.loads(content)
            if isinstance(err, dict) and err.get("code", 0) != 0:
                die(_format_api_error(err["code"], err.get("msg", content)))
        except json.JSONDecodeError:
            pass
    return content


def _parse_metrics(lines: list[str]) -> list[dict]:
    metrics: list[dict] = []
    for line in lines:
        # LlamaFactory: {'loss': 1.23, 'learning_rate': 2e-05, 'epoch': 0.5, ...}
        if "'loss'" in line or '"loss"' in line:
            try:
                m = re.search(r"\{[^}]+\}", line)
                if m:
                    d = ast.literal_eval(m.group(0))
                    if isinstance(d, dict) and "loss" in d:
                        metrics.append(d)
                        continue
            except Exception:
                pass
        # PaddleFormers: "loss: 1.234" or "loss=1.234"
        m = re.search(r"\bloss[:\s=]+([0-9]+\.[0-9]+(?:e[+-]?[0-9]+)?)", line, re.IGNORECASE)
        if m:
            entry: dict = {"loss": float(m.group(1))}
            lr_m = re.search(r"\blr[:\s=]+([0-9]+\.[0-9]+e[+-]?[0-9]+|[0-9]+\.[0-9]+)", line, re.IGNORECASE)
            if lr_m:
                try:
                    entry["learning_rate"] = float(lr_m.group(1))
                except Exception:
                    pass
            step_m = re.search(r"\bstep[:\s]+(\d+)", line, re.IGNORECASE)
            if step_m:
                entry["step"] = int(step_m.group(1))
            metrics.append(entry)
    return metrics


def _print_ascii_loss_chart(losses: list[float], width: int = 50, height: int = 10) -> None:
    """在终端打印 ASCII loss 折线图。"""
    if len(losses) < 2:
        return
    # 降采样到 width 个点
    step = max(1, len(losses) // width)
    sampled = losses[::step][:width]
    mn, mx = min(sampled), max(sampled)
    if mx == mn:
        return  # 无变化，不画

    print(f"\n[loss 折线图]  {mn:.4f} (低) ~ {mx:.4f} (高)")
    print(f"  {'loss':^{width}}")

    for row in range(height, -1, -1):
        threshold = mn + (mx - mn) * row / height
        line = ""
        for v in sampled:
            line += "*" if v >= threshold - (mx - mn) / height / 2 else " "
        label = f"{threshold:.4f}" if row % (height // 2 or 1) == 0 else "      "
        print(f"  {label} |{line}")

    print(f"  {'':6} +{'-' * len(sampled)}")
    print(f"  {'':7}step 0{' ' * (len(sampled) - 12)}step {len(losses) - 1}")


def cmd_train_summary(args: argparse.Namespace) -> None:
    token = load_token(args)
    job_id = args.train_summary
    content = _fetch_log_content(job_id, token, args.base_url)
    if not content.strip():
        print("日志为空（任务可能还在等待中，或尚未产生训练日志）")
        return

    lines = content.splitlines()
    metrics = _parse_metrics(lines)

    print("-" * 50)
    print(f"训练汇报 - {job_id}")
    print("-" * 50)

    if not metrics:
        print("未能解析到 loss 指标，以下是日志末尾 20 行（供参考）：")
        print("\n".join(lines[-20:]))
        return

    losses = [m["loss"] for m in metrics]
    lrs = [m["learning_rate"] for m in metrics if m.get("learning_rate") is not None]

    first_loss, last_loss = losses[0], losses[-1]
    min_loss, max_loss = min(losses), max(losses)
    drop_pct = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0

    print(f"\n[loss 趋势]  共 {len(metrics)} 条记录")
    print(f"  起始：{first_loss:.4f}  ->  最终：{last_loss:.4f}")
    print(f"  最低：{min_loss:.4f}  最高：{max_loss:.4f}")
    if drop_pct > 10:
        verdict = "[OK] 明显下降，训练收敛正常"
    elif drop_pct > 2:
        verdict = "[OK] 轻微下降，可以尝试增加 epochs"
    elif drop_pct > -2:
        verdict = "[!!] 基本持平，建议调小学习率或增加 epochs"
    else:
        verdict = "[!!] loss 上升，训练不稳定，建议检查数据质量和学习率"
    if drop_pct >= 0:
        change_str = f"loss 下降 {drop_pct:.1f}%"
    else:
        change_str = f"loss 上升 {abs(drop_pct):.1f}%"
    print(f"  变化：{change_str}  {verdict}")

    if lrs:
        print(f"\n[学习率]")
        print(f"  起始：{lrs[0]:.2e}  最终：{lrs[-1]:.2e}")
        if lrs[-1] < lrs[0] * 0.5:
            print(f"  趋势：lr 已衰减（正常，调度器生效）")
        elif lrs[-1] > lrs[0]:
            print(f"  趋势：lr 先升（warmup 阶段）")

    # ASCII loss 折线图
    if len(losses) >= 2:
        _print_ascii_loss_chart(losses)

    error_lines = [l for l in lines if re.search(r"\b(error|exception|traceback)\b", l, re.IGNORECASE)]
    if error_lines:
        print(f"\n[!!] 发现 {len(error_lines)} 条错误/异常（最后 3 条）：")
        for l in error_lines[-3:]:
            print(f"  {l[:120]}")

    print()


# -------------------------- export artifacts ----------------- #

def cmd_export_artifacts(args: argparse.Namespace) -> None:
    token = load_token(args)
    job_id = args.export_artifacts
    out_dir = Path(args.out or f"training_artifacts/{job_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"导出训练证据：{job_id}")
    print(f"输出目录：{out_dir}")

    job_detail = api("GET", f"/v1/train/jobs/{job_id}", token, args.base_url)
    _write_text(out_dir / "job_detail.json", json.dumps(job_detail, ensure_ascii=False, indent=2))

    stdout_log = _fetch_job_log(
        job_id,
        token,
        args.base_url,
        system=False,
        byte_limit=args.log_bytes,
        soft=True,
    )
    system_log = _fetch_job_log(
        job_id,
        token,
        args.base_url,
        system=True,
        byte_limit=args.log_bytes,
        soft=True,
    )
    _write_text(out_dir / "master_output_raw.log", stdout_log)
    _write_text(out_dir / "system_raw.log", system_log)

    metrics = _parse_metrics(stdout_log.splitlines()) if not stdout_log.startswith("[!] ") else []
    summary = _build_training_summary_text(job_id, stdout_log, metrics)
    _write_text(out_dir / "train_summary.txt", summary)

    csv_path = out_dir / "loss_curve.csv"
    _write_metrics_csv(csv_path, metrics)

    chart_paths = _write_metric_charts(out_dir, metrics)

    print("\n导出完成：")
    for path in [
        out_dir / "job_detail.json",
        out_dir / "master_output_raw.log",
        out_dir / "system_raw.log",
        out_dir / "train_summary.txt",
        csv_path,
        *chart_paths,
    ]:
        if path.exists():
            print(f"  {path}")
    if not metrics:
        print("\n[!] 未解析到 loss 指标；已保留 raw log，请先检查任务是否进入 running 或日志格式是否变化。")
    elif not chart_paths:
        print("\n提示：未生成 PNG 曲线图。如需图表，请安装 matplotlib 后重试：pip install matplotlib")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_metrics_csv(path: Path, metrics: list[dict]) -> None:
    preferred = ["index", "step", "loss", "learning_rate", "epoch", "perplexity", "grad_norm"]
    extra = sorted({str(k) for m in metrics for k in m.keys()} - set(preferred))
    fieldnames = preferred + extra

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, metric in enumerate(metrics, start=1):
            row = {"index": idx}
            for key in fieldnames:
                if key == "index":
                    continue
                val = metric.get(key)
                if val is not None:
                    row[key] = val
            writer.writerow(row)


def _build_training_summary_text(job_id: str, content: str, metrics: list[dict]) -> str:
    lines = content.splitlines()
    output: list[str] = [
        "-" * 50,
        f"训练汇报 - {job_id}",
        "-" * 50,
        "",
    ]

    if not content.strip():
        output.append("日志为空（任务可能还在等待中，或尚未产生训练日志）")
        return "\n".join(output)

    if not metrics:
        output.append("未能解析到 loss 指标，以下是日志末尾 20 行（供参考）：")
        output.extend(lines[-20:])
        return "\n".join(output)

    losses = [float(m["loss"]) for m in metrics]
    lrs = [float(m["learning_rate"]) for m in metrics if m.get("learning_rate") is not None]
    first_loss, last_loss = losses[0], losses[-1]
    min_loss, max_loss = min(losses), max(losses)
    drop_pct = (first_loss - last_loss) / first_loss * 100 if first_loss > 0 else 0

    if drop_pct > 10:
        verdict = "[OK] 明显下降，训练收敛正常"
    elif drop_pct > 2:
        verdict = "[OK] 轻微下降，可以尝试增加 epochs"
    elif drop_pct > -2:
        verdict = "[!!] 基本持平，建议调小学习率或增加 epochs"
    else:
        verdict = "[!!] loss 上升，训练不稳定，建议检查数据质量和学习率"

    change_str = f"loss 下降 {drop_pct:.1f}%" if drop_pct >= 0 else f"loss 上升 {abs(drop_pct):.1f}%"
    output.extend([
        f"[loss 趋势]  共 {len(metrics)} 条记录",
        f"  起始：{first_loss:.4f}  ->  最终：{last_loss:.4f}",
        f"  最低：{min_loss:.4f}  最高：{max_loss:.4f}",
        f"  变化：{change_str}  {verdict}",
    ])

    if lrs:
        output.extend([
            "",
            "[学习率]",
            f"  起始：{lrs[0]:.2e}  最终：{lrs[-1]:.2e}",
        ])
        if lrs[-1] < lrs[0] * 0.5:
            output.append("  趋势：lr 已衰减（正常，调度器生效）")
        elif lrs[-1] > lrs[0]:
            output.append("  趋势：lr 先升（warmup 阶段）")

    error_lines = [l for l in lines if re.search(r"\b(error|exception|traceback)\b", l, re.IGNORECASE)]
    if error_lines:
        output.extend(["", f"[!!] 发现 {len(error_lines)} 条错误/异常（最后 3 条）："])
        output.extend(f"  {l[:120]}" for l in error_lines[-3:])

    return "\n".join(output)


def _write_metric_charts(out_dir: Path, metrics: list[dict]) -> list[Path]:
    if len(metrics) < 2:
        return []

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    xs = [int(m.get("step") or i) for i, m in enumerate(metrics, start=1)]
    losses = [float(m["loss"]) for m in metrics]
    lrs = [float(m["learning_rate"]) for m in metrics if m.get("learning_rate") is not None]
    lr_xs = [int(m.get("step") or i) for i, m in enumerate(metrics, start=1) if m.get("learning_rate") is not None]
    paths: list[Path] = []

    loss_path = out_dir / "loss_curve.png"
    plt.figure(figsize=(10, 4))
    plt.plot(xs, losses, linewidth=1.8)
    plt.title("Training Loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(loss_path, dpi=160)
    plt.close()
    paths.append(loss_path)

    if lrs:
        lr_path = out_dir / "learning_rate_curve.png"
        plt.figure(figsize=(10, 4))
        plt.plot(lr_xs, lrs, linewidth=1.8)
        plt.title("Learning Rate")
        plt.xlabel("step")
        plt.ylabel("learning_rate")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(lr_path, dpi=160)
        plt.close()
        paths.append(lr_path)

        combined_path = out_dir / "training_curves.png"
        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(xs, losses, color="#1f77b4", linewidth=1.8, label="loss")
        ax1.set_xlabel("step")
        ax1.set_ylabel("loss", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(lr_xs, lrs, color="#d62728", linewidth=1.4, label="learning_rate")
        ax2.set_ylabel("learning_rate", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        plt.title("Training Curves")
        fig.tight_layout()
        plt.savefig(combined_path, dpi=160)
        plt.close(fig)
        paths.append(combined_path)

    return paths


# ------------------------------- cancel ----------------------- #

def cmd_cancel(args: argparse.Namespace) -> None:
    token = load_token(args)
    job_id = args.cancel

    print(f"取消任务：{job_id}")
    api("POST", f"/v1/train/jobs/{job_id}/cancel", token, args.base_url)
    print("取消请求已发送。")
    print(f"\n确认状态：python3 {Path(__file__).name} --status {job_id}")


# --------------------------- open tensorboard ---------------- #

def cmd_open_tb(args: argparse.Namespace) -> None:
    token = load_token(args)
    job_id = args.open_tb
    data = api("GET", f"/v1/train/jobs/{job_id}", token, args.base_url)
    tb_url = data.get("tensorboardUrl", "")
    state = data.get("state", "")
    if state != "running":
        die(f"当前状态为 {state or 'unknown'}，Tensorboard 等任务进入 running 后再打开。")
    if not tb_url:
        die(f"该任务暂无 Tensorboard 地址（当前状态：{state}）。\n任务已进入 running 后可稍后再试。")
    _open_url(tb_url)


# ---------------------------- eval guide --------------------- #

def cmd_eval_guide(args: argparse.Namespace) -> None:
    path = Path(args.eval_guide)
    if not path.exists():
        die(f"文件不存在：{path}")

    records, _errors, _data_kind, _total_units = _load_data_records(path)
    samples = [obj for _idx, obj, _preview in records[:200]]

    if not samples:
        die("未能从文件中解析出有效样本。")

    def _unwrap(val: object) -> str:
        """ERNIE src/tgt 是 list，取第一个元素；否则直接转字符串。"""
        if isinstance(val, list):
            return str(val[0]) if val else ""
        return str(val) if val is not None else ""

    def _prompt_answer(sample: dict) -> tuple[str, str]:
        fmt = detect_format_obj(sample)
        if fmt == "ernie":
            return _unwrap(sample.get("src", "")), _unwrap(sample.get("tgt", ""))
        if fmt == "alpaca":
            prompt = str(sample.get("instruction", "") or "")
            if sample.get("input"):
                prompt += "\n" + str(sample.get("input"))
            answer = sample.get("output")
            if answer is None and isinstance(sample.get("chosen"), str):
                answer = sample.get("chosen")
            return prompt, str(answer or "")
        if fmt == "sharegpt":
            messages = sample.get("conversations")
            role_key = "from"
            content_key = "value"
            user_roles = {"human", "user"}
            assistant_roles = {"gpt", "assistant"}
            if not isinstance(messages, list):
                messages = sample.get("messages")
                role_key = "role"
                content_key = "content"
                user_roles = {"user"}
                assistant_roles = {"assistant"}
            prompt = ""
            answer = ""
            if isinstance(messages, list):
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    role = msg.get(role_key)
                    content = str(msg.get(content_key, "") or "")
                    if not prompt and role in user_roles:
                        prompt = content
                    if role in assistant_roles:
                        answer = content
            chosen = sample.get("chosen")
            if not answer and isinstance(chosen, dict):
                answer = str(chosen.get("value") or chosen.get("content") or "")
            return prompt, answer
        return "", ""

    # 选取最多 5 个有代表性的样本
    random.seed(42)
    picked = random.sample(samples, min(5, len(samples)))

    print(f"\n{chr(61) * 55}")
    print("  训练后效果测试建议（基于你的训练数据生成）")
    print(f"{chr(61) * 55}")
    print(f"\n  数据集：{path.name}，共 {len(samples)} 条样本\n")

    print("【测试问题 1：训练集内问题（验证是否学到）】")
    print("将以下问题发给微调后的模型，看回答是否接近期望输出：\n")
    for i, s in enumerate(picked[:3], 1):
        src, tgt = _prompt_answer(s)
        src = src[:120]
        tgt = tgt[:80]
        print(f"  问题 {i}：{src}")
        print(f"  期望回答（节选）：{tgt}...\n")

    print("\n【泛化测试：换个说法问同类问题】")
    print("把下面这些问题用自己的话改写后再问一遍，看风格是否保持一致：\n")
    for i, s in enumerate(picked[:2], 1):
        src, _tgt = _prompt_answer(s)
        src = src[:120]
        print(f"  问题 {i}：{src}\n")

    print("\n【效果判断标准】")
    print("  [OK] 回答风格与训练数据一致 -> 微调有效")
    print("  [OK] 专有名词处理更准确 -> 领域知识已注入")
    print("  [X]  回答完全复制训练样本 -> 可能过拟合，减少 epochs 或增加数据量")
    print("  [X]  回答与基底模型无区别 -> 可能欠拟合，增加 epochs 或检查数据质量")
    print("\n" + "=" * 55 + "\n")


# ----------------------------- main --------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AiStudio 无代码训练 CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 通用参数
    p.add_argument("--api-key", metavar="TOKEN", help="Access Token（也可用环境变量 AISTUDIO_ACCESS_TOKEN）")
    p.add_argument("--env-file", metavar="FILE", help="从 .env 文件读取 token")
    p.add_argument("--base-url", default=BASE_URL, metavar="URL", help=f"API 地址（默认 {BASE_URL}）")

    # 子命令
    p.add_argument("--verify-token", action="store_true", help="验证 Access Token 是否有效")
    p.add_argument("--verify-upload", action="store_true", help="验证数据集上传结果并打印回执")
    p.add_argument("--strict-lfs", action="store_true", help="配合 --verify-upload：is_lfs 不是 false 时直接失败")
    p.add_argument("--list-models", action="store_true", help="列出平台白名单中所有可用模型")
    p.add_argument("--list-datasets", action="store_true", help="列出内置推荐数据集（不用自己准备数据）")
    p.add_argument("--check-data", metavar="FILE", help="检查数据格式")
    p.add_argument("--suggest-params", metavar="FILE", help="根据数据集推荐超参数")
    p.add_argument("--model-type", choices=["ernie", "llama"], default="ernie", help="模型类型（配合 --suggest-params）")
    p.add_argument("--status", metavar="JOB_ID", help="查看任务状态")
    p.add_argument("--env-check", action="store_true", help="检查本地环境（Python 版本、依赖包）")
    p.add_argument("--poll", metavar="JOB_ID", help="持续轮询进度")
    p.add_argument("--interval", type=int, default=30, metavar="SEC", help="轮询间隔（秒，默认 30）")
    p.add_argument("--logs", metavar="JOB_ID", help="查看训练日志")
    p.add_argument("--system", action="store_true", help="查看系统日志（配合 --logs）")
    p.add_argument("--diagnose", metavar="JOB_ID", help="主动诊断任务状态、system log 和 stdout")
    p.add_argument("--cancel", metavar="JOB_ID", help="取消任务")
    p.add_argument("--open-tb", metavar="JOB_ID", help="任务 running 后打开 Tensorboard")
    p.add_argument("--train-summary", metavar="JOB_ID", help="训练完成后汇报 loss/lr 趋势")
    p.add_argument("--export-artifacts", metavar="JOB_ID", help="导出 job detail、raw logs、指标 CSV 和曲线图")
    p.add_argument("--out", metavar="DIR", help="输出目录（配合 --export-artifacts）")
    p.add_argument("--log-bytes", type=int, default=5_000_000, metavar="N", help="导出日志最大字节数（默认 5000000；AI Studio 日志接口使用 bytes=0-N）")

    # submit 参数
    p.add_argument("--submit", action="store_true", help="提交训练任务")
    p.add_argument("--base-model", metavar="MODEL", help="基底模型（如 PaddlePaddle/ERNIE-4.5-0.3B-PT）")
    p.add_argument("--train-type", default="SFT/Full", metavar="TYPE", help="训练类型（默认 SFT/Full；可选 SFT/LoRA）")
    p.add_argument("--train-data", metavar="REPO_ID", help="数据集仓库路径，必须是详情页真实 repo_id（如 gitlogin/my_dataset）")
    p.add_argument("--train-file", metavar="FILENAME", help="数据文件名（如 train.jsonl；强烈建议指定，省略时平台会自动选择首个 JSON/JSONL）")
    p.add_argument("--local-file", metavar="FILE", help="本地训练文件路径（配合 --verify-upload 对比大小）")
    p.add_argument("--params", metavar="JSON", help="超参数 JSON 字符串")
    p.add_argument("--name", metavar="NAME", help="任务名称（只允许字母、数字、下划线）")
    p.add_argument("--description", metavar="DESC", help="任务描述")
    p.add_argument("--output-repo", metavar="GITLOGIN/REPO", help="模型输出仓库（可选；命名空间必须可写）")
    p.add_argument("--max-run-time", type=int, metavar="HOURS", help="最长运行时间（小时，1-240）")

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.env_check:
        cmd_env_check()
    elif args.verify_token:
        cmd_verify_token(args)
    elif args.list_models:
        cmd_list_models(args)
    elif args.list_datasets:
        cmd_list_datasets(args)
    elif args.check_data:
        cmd_check_data(args)
    elif args.verify_upload:
        if not args.train_data:
            parser.error("--verify-upload 需要 --train-data")
        if not args.train_file:
            parser.error("--verify-upload 需要 --train-file")
        cmd_verify_upload(args)
    elif args.suggest_params:
        cmd_suggest_params(args)
    elif args.submit:
        if not args.base_model:
            parser.error("--submit 需要 --base-model")
        if not args.train_data:
            parser.error("--submit 需要 --train-data")
        cmd_submit(args)
    elif args.status:
        cmd_status(args)
    elif args.poll:
        cmd_poll(args)
    elif args.logs:
        cmd_logs(args)
    elif args.diagnose:
        cmd_diagnose(args)
    elif args.cancel:
        cmd_cancel(args)
    elif args.open_tb:
        cmd_open_tb(args)
    elif args.train_summary:
        cmd_train_summary(args)
    elif args.export_artifacts:
        cmd_export_artifacts(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
