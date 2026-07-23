# MaaFramework 集成骨架

## 范围与版本

v0.3a 只建立 Pipeline v2 调用 Python Custom Recognition 的接口骨架。它不连接模拟器、不获取截图、不访问 controller、不执行点击或其他 action，也不把识别结果送入 solver。

接口于 2026-07-23 对照 MaaFramework v5.12.2、官方文档及官方仓库 `main` 提交 `25dff4c98eacadd6328685858f8981b52f3b56e2` 核对。当前 Python binding 的包名为 `MaaFw`、导入命名空间为 `maa`：

- `from maa.agent.agent_server import AgentServer`；
- `from maa.custom_recognition import CustomRecognition`；
- 使用 `AgentServer.register_custom_recognition(name, instance)` 注册并检查布尔返回值；官方装饰器内部也调用该函数，但不会向调用方暴露失败结果；
- `analyze(self, context, argv)` 接收 `CustomRecognition.AnalyzeArg`；
- `argv.image` 是 BGR、`uint8` 的 `numpy.ndarray`；
- 返回 `CustomRecognition.AnalyzeResult(box=..., detail=...)`，其中 `box` 为 `(x, y, width, height)` 或 `None`，`detail` 为可 JSON 序列化对象；
- Agent 生命周期为 `AgentServer.start_up(socket_id)`、`join()`、`shut_down()`。

早期文档中的简化 tuple 示例不足以描述当前 binding，因此实现以 v5.12.2 附近的 Python binding 源码、官方 AgentServer 测试和 ProjectInterface v2 schema 为准。

## 分层边界

```text
MaaFramework Pipeline v2
        |
        | BGR ndarray / AnalyzeResult
        v
integrations.maafw.agent       可选 SDK 导入、注册、生命周期
        |
integrations.maafw.adapter     框架无关调用与主要 box 选择
        |
integrations.maafw.serialization
        |
vision.NodeDetector            OpenCV 网格定位基线

solver                         完全独立，不消费上述输出
```

`maablackflow`、普通 CLI、`vision` 和 `solver` 都不会导入 Maa SDK。只有启动 Agent 时才惰性导入可选 runtime；缺失时明确报告 `MaaFramework optional runtime is not installed`。本项目不自动下载二进制、不修改 `PATH`，也不打包 Maa 原生库。

## Custom Recognition 约定

稳定注册名为 `MaaBlackFlow.MapRecognize`。成功时主要 `box` 优先选择 `current_position` 的实际格点框；未检测到人物时使用可信全局网格推导出的 `map_roi`。失败时返回 `box=None`，由 MaaFramework 视为未命中，同时把可预期错误写入结构化日志和失败 detail，不伪造成功。

`detail` schema 为 `maablackflow.maafw.detail.v1`，包含检测阶段、原图尺寸、网格拟合状态/格距、地图 ROI、节点、当前位置和 warnings。人物节点分别保存视觉 `marker_center` 与地图 `grid_center`；`center` 始终等于 `grid_center`。序列化只接收检测结果，不接收输入文件路径，并稳定排序。`solver_ready` 被强制固定为 `false`。

## Pipeline v2

资源在 `maafw_project/resource/pipeline/blackflow.json`。唯一节点使用当前 Pipeline v2 嵌套结构：

```json
{
  "recognition": {
    "type": "Custom",
    "param": {
      "custom_recognition": "MaaBlackFlow.MapRecognize",
      "custom_recognition_param": {
        "output_detail": true,
        "require_solver_ready": false,
        "recognition_mode": "grid_baseline"
      }
    }
  },
  "action": {"type": "DoNothing", "param": {}},
  "next": []
}
```

`maafw_project/interface.json` 使用 ProjectInterface 主结构版本 2，声明 Python Agent 与离线资源。 资源根不命名为 `maa/`，是为了避免在未安装 SDK 时被 Python 误识别为空的 `maa` namespace；ProjectInterface 只要求其内部相对路径一致，并不固定项目根目录名。`controller` 故意为空；它是接入骨架而非可操作游戏的成品 ProjectInterface。Pipeline 中不包含 `Click`、`Swipe`、`Shell`、`Command` 或 Custom Action。

## 离线 smoke

无需 MaaFramework：

```powershell
.\.venv\Scripts\python.exe -m maablackflow.cli maa-adapter-smoke `
  "Screenshots\第三层起点.png" `
  --output "data\outputs_private\maa-smoke"
```

命令直接用 Agent 相同的 adapter 与 serializer，检查识别成功、`solver_ready=false`、人物 marker/grid 分离、格点唯一及 detail 不含输入绝对路径，然后写入 `*.maa-detail.json`。输出目录及真实文件名均是 Git 忽略的私人数据。

## 可选 Agent 启动

安装并准备官方 Python/native runtime 后，由 ProjectInterface 提供 socket ID：

```powershell
.\.venv\Scripts\python.exe -m maablackflow.integrations.maafw.agent <socket_id>
```

`--help` 不加载 Maa runtime。缺少 socket 参数由参数解析器明确拒绝。

v0.3b 已在本机使用 `MaaFw 5.12.2`、`MaaAgentBinary 1.0.1` 和 `MaaAgentServer.dll` 完成以下真实验证：

- 原生 `AgentServer.register_custom_recognition()` 返回 `true`，稳定名称进入 SDK holder；
- 使用真实 `CustomRecognition.AnalyzeArg` 调用识别器，得到真实 `CustomRecognition.AnalyzeResult`；
- 私人截图返回人物格点 box、完整 detail，且 `solver_ready=false`；
- 标准 `Resource.post_bundle()` 成功加载 `maafw_project/resource`，解析出 Custom recognition、`DoNothing` 和空 `next`。

没有调用 `AgentServer.start_up()`：该函数用于与 AgentClient 提供的 socket identifier 建立通信，随后正常入口会进入阻塞式 `join()`。在没有配套 AgentClient 生命周期的独立 smoke 中启动它既不能增加识别契约覆盖，又可能留下等待中的本地 socket；因此本轮只验证安全的原生注册边界。没有创建 Controller、连接 ADB 或执行 action。

安装的 `MaaFw`/`MaaAgentBinary` Python 发行包不包含 MaaPiCli 或 ProjectInterface host/loader API，所以 `interface.json` 仍只能按官方 schema 验证；真正由通用 UI/ProjectInterface host 拉起子进程的链路尚未验证。Pipeline 资源本身已经由真实 MaaFramework Resource API 加载。

## 未来 TemplateMatch

`MaaTemplateEvidenceProvider` 目前只把外部已经产生的模板命中 `{template, box, score}` 转成 `CandidateEvidence`；其 `collect()` 默认不产生任何结果，不调用 Maa，也不伪造匹配。后续私人实验可由 MaaFramework TemplateMatch 或本地 OpenCV 模板提供真实命中，再沿现有网格吸附与同格融合流程处理。真实模板不得提交，且在道路拓扑、出口和人工校验完成前仍不能设置 `solver_ready=true`。
