# 星河社区 SDK 上传原则

## 核心原则

1. **优先使用仓库路径。** 上传目标必须是完整 `repo_id`，形如 `用户名/仓库名`；不要使用数字 `dataset_id`，旧接口不稳定且可能返回 500。
2. **一个数据集一个仓库。** 不同训练任务的数据不要混传到同一个仓库，避免后续训练选择文件、commit 和挂载排查混乱。
3. **token 只走环境变量。** 只从 `AISTUDIO_ACCESS_TOKEN` 读取 token，禁止硬编码到脚本、文档示例、命令行实参或提交记录里。不要用 `aistudio upload ... --token TOKEN`，部分 CLI 会把完整 argv 打印到终端日志。
4. **SDK 负责上传，脚本负责验收。** `aistudio-sdk upload_folder` 成功只代表文件提交成功；训练前仍必须用 `scripts/train.py --verify-upload` 确认文件名、大小和 `is_lfs` 状态。
5. **JSON/JSONL 训练文件推荐使用普通文件。** 新仓库默认 `.gitattributes` 可能把 `*.jsonl` / `*.json` 设为 LFS；上传前优先删对应规则。若已传成 `is_lfs:true`，不要直接判定任务必然失败：实测部分 LFS 文件也能被 AI Studio 成功挂载并完成训练。但 LFS 会降低下载回验和 `waiting_data` 排查可靠性，卡住时先删 LFS 文件、移除规则、再重传。
6. **不要只看 `is_lfs` 字段。** 仓库接口可能返回 `is_lfs:false`，但文件大小只有百余字节，正文实际是 Git LFS pointer。验收时要同时检查仓库大小、本地大小和 pointer 正文；需要严格验收时使用 `--verify-upload --strict-lfs`。
7. **普通 JSON/JSONL 大文件优先走 Git contents API fallback。** SDK/CLI 普通上传超过约 5MB 可能失败或回退 LFS；训练必须保留全量文件时，使用 `scripts/train.py --upload-plain-file "$LOCAL_FILE" --train-data "$REPO_ID" --path-in-repo train.jsonl --remove-json-lfs-rules` 覆盖为普通文件，再严格验收。
8. **路径异常优先使用根目录短文件名。** 如果提交后报 `trainDataFiles 指定的文件不存在`，即使 Git API 能看到原文件，也先把训练文件重新上传为根目录 `train.jsonl`，确认新 commit 后用同名 `--train-file train.jsonl` 重提。
9. **STS 报错不等于失败。** 出现 `'super' object has no attribute 'put_super_obejct_from_file'` 时，SDK 会回退到 HTTP LFS 上传；最终以 `201` 和 `Commit part 1 successful!` 为准。
10. **上传成功不等于训练挂载成功。** 如果训练日志已解析到 repo、commit 和 mount Job，但持续 `waiting_data`，通常是平台挂载任务卡住；保留 jobId、repo_id、commitId、mount Job、文件大小和 `is_lfs` 校验结果给平台排查。若 `is_lfs:true`，先修复为普通文件后重试。

## 最小示例

```python
import os
from aistudio_sdk.hub import upload_folder

upload_folder(
    repo_id="yourname/your-dataset-repo",
    folder_path="/abs/path/to/dataset/folder",
    path_in_repo="",
    repo_type="dataset",
    commit_message="upload dataset files",
    token=os.environ["AISTUDIO_ACCESS_TOKEN"],
    ignore_patterns=[".DS_Store", "*.tmp"],
    max_workers=4,
)
```

上传后继续执行：

```bash
python3 scripts/train.py --verify-upload \
  --train-data "$REPO_ID" \
  --train-file "$TRAIN_FILE" \
  --local-file "$LOCAL_FILE"
```
