# MaaBlackFlow

MaaBlackFlow 是面向《明日方舟》集成战略“沉沦者的黑流树海”的独立开源智能路径规划器。当前包含 JSON 地图精确求解核心，以及从离线 PNG 提取可见地图节点候选的 OpenCV 基线。

> 非官方声明：本项目是非官方社区项目，目前未得到 MAA（MaaAssistantArknights / MaaFramework）或鹰角网络的认可、授权或背书。《明日方舟》及相关名称、内容的权利归其权利人所有。

## 当前能力

- 从 JSON 加载节点、道路、初始资源和配置驱动的加工品移动规则。
- 使用记忆化精确搜索，在确保抵达出口的前提下最大化收益。
- 自动清点本地截图的格式、尺寸、通道、大小、SHA-256、重复关系和文件名场景标签。
- 生成不改变原图的私人 contact sheet。
- 使用道路掩码、骨架关键点、全局周期网格拟合和按格点证据融合，在正常、放大和缩小地图中定位可见格点。
- 区分 `event_node`、`empty_waypoint`、唯一 `current_position` 与 `uncertain`，输出分项评分、证据来源、annotated PNG、JSON 和道路/网格调试图。
- 提供实验性的 MaaFramework v5.12.2 Pipeline v2 / Python Custom Recognition 接入骨架；Maa runtime 是可选依赖，当前只识别且 action 固定为 `DoNothing`。

视觉功能目前只是离线格点定位基线：已有可选 MaaFramework Custom Recognition 接口骨架，并已用真实 SDK/原生库验证注册和 Pipeline 加载，但尚未启动 Agent socket 或接入截图/控制能力；尚不能识别具体事件类型或完整道路关系，也不能可靠确定出口。`current_position` 同时保存人物标记的 `marker_center` 与吸附格点的 `grid_center`，最终节点 `center` 始终等于 `grid_center`。检测结果未经拓扑校验，禁止直接输入求解器。项目不控制游戏、不自动点击，并且没有 GUI。

## 安装

需要 Python 3.11 或更高版本。在 Windows PowerShell 中：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

求解核心不依赖 OpenCV；`maablackflow.vision` 使用 `opencv-python-headless` 和 Pillow。MaaFramework 不属于默认依赖；只有准备实验性 Agent runtime 时才安装官方可选项：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[maafw]"
```

这不会把 MaaFramework 二进制纳入 MaaBlackFlow wheel。本机没有 runtime 时，核心包、普通 CLI 和测试仍可使用。

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
- `示例地图.nodes.json`：原图尺寸、全局格距、网格行列、类别、`grid_center`、人物 `marker_center`、统一比例边界框、可靠性、分项评分和逐项证据；
- `示例地图.grid-debug.png`：道路掩码、骨架、淡色局部峰、最终周期网格轴，以及证据吸附关系。

输入必须是有效 PNG。无法读图、全局网格拟合失败或没有可信候选时命令返回非零退出码，不生成伪造结果。中文文件名、Windows 反斜杠、相对路径和绝对路径均受支持。

## 私人 Ground Truth 评估

人工真值保存在被 Git 忽略的 `data/ground_truth_private/`，检测器不会读取它。只有独立评估命令同时读取私人真值与检测 JSON：

```powershell
python -m maablackflow.cli evaluate-detection `
  --ground-truth data\ground_truth_private\run02 `
  --predictions data\outputs_private\run02
```

评估按预测格距动态设置容差并执行一对一匹配，输出 TP、FP、FN、precision、recall、F1、中心误差、人物 marker 误差、当前位置格点正确性和重复格点数量。`uncertain` 不允许匹配真值，仍会计入预测总数。
## MaaFramework 真实运行时兼容性

v0.3b 已使用 `MaaFw 5.12.2` 与 `MaaAgentBinary 1.0.1` 验证原生 Custom Recognition 注册、真实 `AnalyzeArg`/`AnalyzeResult` 返回契约，以及 `Resource.post_bundle()` 对 Pipeline v2 的加载。验证没有创建 Controller、连接 ADB、启动 Agent socket 或执行 action。

真实运行时测试位于 `tests/test_maafw_runtime.py`：安装可选 runtime 时执行，未安装时自动跳过，因此核心求解器和离线视觉仍不依赖 MaaFramework。
## MaaFramework 离线 adapter smoke

v0.3a 的 adapter 可在完全不启动 MaaFramework 的情况下复用当前检测器：

```powershell
.\.venv\Scripts\python.exe -m maablackflow.cli maa-adapter-smoke `
  "Screenshots\第三层起点.png" `
  --output "data\outputs_private\maa-smoke"
```

私人目录中生成的 `*.maa-detail.json` 包含网格、节点和人物 marker/grid 坐标，不包含输入绝对路径，且 `solver_ready` 恒为 `false`。Pipeline 资源位于 `maafw_project/resource/pipeline/blackflow.json`，只调用 `MaaBlackFlow.MapRecognize` 并执行 `DoNothing`。Agent 启动及接口版本见 [MaaFramework 集成说明](docs/maafw_integration.md)。

MaaBlackFlow 是独立社区项目，不隶属于 MaaAssistantArknights、MaaFramework 或鹰角网络，也未获得其认可、授权或背书。当前 Maa 接入是实验性的，不使用官方 Logo，不控制游戏。
## 当前视觉限制

- 道路纹理转折仍可能成为 `uncertain`；暗色、部分遮挡及屏幕边缘节点仍可能漏检。
- 不同缩放状态使用同一套比例化启发式，但精度尚未统一。
- 当前只区分格点视觉类别；尚未识别具体事件类型、完整道路连接、出口或可信地图拓扑。
- 被面板完全遮挡的节点不会外推恢复；输出不得直接输入求解器。
- 真实截图、清单和标注结果只保存在被忽略的私人目录。

详细设计见 [架构文档](docs/architecture.md)，阶段计划见 [路线图](docs/roadmap.md)。

## 许可证

本项目暂定采用 GNU Affero General Public License v3.0，详见 [LICENSE](LICENSE)。 可选 MaaFramework 运行时的第三方许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。