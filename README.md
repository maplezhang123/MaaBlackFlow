# MaaBlackFlow

MaaBlackFlow 是面向《明日方舟》集成战略“沉沦者的黑流树海”的独立开源智能路径规划器。当前包含 JSON 地图精确求解核心，以及从离线 PNG 提取可见地图节点候选的 OpenCV 基线。

> 非官方声明：本项目是非官方社区项目，目前未得到 MAA（MaaAssistantArknights / MaaFramework）或鹰角网络的认可、授权或背书。《明日方舟》及相关名称、内容的权利归其权利人所有。

## 当前能力

- 从 JSON 加载节点、道路、初始资源和配置驱动的加工品移动规则。
- 使用记忆化精确搜索，在确保抵达出口的前提下最大化收益。
- 自动清点本地截图的格式、尺寸、通道、大小、SHA-256、重复关系和文件名场景标签。
- 生成不改变原图的私人 contact sheet。
- 使用离线 OpenCV 圆形与局部纹理启发式，在正常、放大和缩小地图中提出可见节点候选。
- 输出原图分辨率下的节点编号、中心坐标、边界框、置信度、annotated PNG 和 JSON。

视觉功能目前只是离线节点检测基线：尚未接入 MaaFramework，尚不能完整识别节点类型或全部道路关系，也不能确定角色位置和出口。项目不控制游戏、不自动点击，并且没有 GUI。

## 安装

需要 Python 3.11 或更高版本。在 Windows PowerShell 中：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

求解核心不依赖 OpenCV；`maablackflow.vision` 使用 `opencv-python-headless` 和 Pillow。

## 路径规划 CLI

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m maablackflow.cli solve examples/maps/basic.json
```

成功到达出口时退出码为 0，无解为 1，输入错误为 2。

## 私人截图清点

仓库根目录的 `Screenshots/` 是本地私人数据目录，已被 Git 忽略，不会上传 GitHub。程序直接读取现有文件，不要求重命名或分类：

```powershell
.\.venv\Scripts\python.exe -m maablackflow.cli inspect-dataset Screenshots
```

默认在同样被忽略的 `data/outputs_private/run01/` 生成：

- `dataset_manifest.json`：图片元数据、哈希、重复关系、场景标签和是否适合节点检测；
- `contact_sheet.png`：按清单顺序编号的本地缩略图总览。

公开清单结构示例使用虚构数据：

```json
{
  "source_name": "Screenshots",
  "image_count": 1,
  "duplicate_count": 0,
  "resolutions": {"1280x720": 1},
  "images": [
    {
      "filename": "example_normal.png",
      "image_format": "PNG",
      "width": 1280,
      "height": 720,
      "channels": 3,
      "file_size": 123456,
      "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "duplicate_of": null,
      "scene_label": "地图正常状态",
      "suitable_for_node_detection": true
    }
  ]
}
```

## 离线节点检测

```powershell
.\.venv\Scripts\python.exe -m maablackflow.cli detect-nodes `
  "Screenshots\示例地图.png" `
  --output "data\outputs_private\run01"
```

成功后在输出目录查看：

- `示例地图.annotated.png`：保持原尺寸，绘制节点框、中心点和 `node_01` 编号；
- `示例地图.nodes.json`：原图尺寸及每个候选的中心、边界框和置信度。

输入必须是有效 PNG。无法读图或没有可信候选时命令返回非零退出码，不生成伪造结果。中文文件名、Windows 反斜杠、相对路径和绝对路径均受支持。

## 当前视觉限制

- 圆形高亮、文字图标和复杂弹窗可能产生误检；暗色、部分遮挡及屏幕边缘节点可能漏检。
- 不同缩放状态使用同一套比例化启发式，但精度尚未统一。
- 尚未识别完整节点类型、道路连接、当前位置、出口或地图拓扑。
- 真实截图、清单和标注结果只保存在被忽略的私人目录。

详细设计见 [架构文档](docs/architecture.md)，阶段计划见 [路线图](docs/roadmap.md)。

## 许可证

本项目暂定采用 GNU Affero General Public License v3.0，详见 [LICENSE](LICENSE)。