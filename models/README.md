# 模型目录

两个 ONNX 模型**不入库**（共约 88MB），请按需放到本目录：

| 文件 | 大小 | 说明 |
|---|---|---|
| `nsfw-anime-medium-x1280.onnx` | 45.4 MB | 动漫向分割模型，**默认使用**，检测类别见程序「检测类别」 |
| `automatic_coding_original.onnx` | 42.7 MB | 原版模型，类别较少 |

获取方式（二选一）：

1. 从仓库 [Releases](https://github.com/duxingz/unified-image-tool/releases) 下载 `UnifiedImageTool.zip`，把 `_internal/models/` 里两个 `.onnx` 拷到本目录；
2. 或使用自己的兼容模型（需与 `app/core/detector.py` 的类别名称匹配，类别表由模型自带的 metadata 读取）。

程序启动后在「设置 → 检测模型」里会自动列出本目录下的 `.onnx` 文件。
