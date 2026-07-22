# MaaBlackFlow

MaaBlackFlow 是面向《明日方舟》集成战略“沉沦者的黑流树海”的独立开源智能路径规划器。当前 v0.1 只包含与游戏画面、设备和 UI 解耦的精确路径规划核心。

> 非官方声明：本项目是非官方社区项目，目前未得到 MAA（MaaAssistantArknights / MaaFramework）或鹰角网络的认可、授权或背书。《明日方舟》及相关名称、内容的权利归其权利人所有。

## 当前能力

- 从 JSON 加载节点、道路、初始资源和配置驱动的加工品移动规则。
- 使用记忆化精确搜索，在确保抵达出口的前提下最大化收益。
- 按“收益、剩余行动力、加工品消耗、稳定顺序”选择唯一可复现的最优路线。
- 返回完整步骤、每步收益与消耗；无解时返回结构化原因。
- 提供 Python API、PowerShell 友好的 CLI、样例地图和 pytest 测试。

当前不包含 MaaFramework 接入、截图/OCR/模板识别、GUI、自动点击或任何 MAA 资源。

## 安装

需要 Python 3.11 或更高版本。在 Windows PowerShell 中：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

无需激活虚拟环境即可运行。若选择激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

## 运行

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m maablackflow.cli solve examples/maps/basic.json
.\.venv\Scripts\python.exe -m maablackflow.cli solve examples/maps/jump_required.json
```

CLI 会输出是否到达出口、最优路线、每步移动方式、总收益、剩余行动力、加工品使用量和无解原因。命令成功到达出口时退出码为 0，无解为 1，输入错误为 2。

## JSON 格式

顶层包含 `nodes`、`edges`、可选的 `movement_rules` 和 `game_state`。加工品完全由向量、行动力成本、剩余次数及能否越过未完成节点配置；可参考 [jump_required.json](examples/maps/jump_required.json)。

## 路线图

- v0.1：JSON 地图精确求解（当前阶段）
- v0.2：截图文件识别
- v0.3：MaaFramework 接入
- v0.4：独立可视化界面
- v1.0：动态重规划和 Windows 发布

详细设计见 [架构文档](docs/architecture.md)，阶段计划见 [路线图](docs/roadmap.md)。

## 许可证

本项目暂定采用 GNU Affero General Public License v3.0，详见 [LICENSE](LICENSE)。
