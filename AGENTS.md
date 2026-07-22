# MaaBlackFlow 协作约束

## 项目边界

MaaBlackFlow 是独立、非官方的社区项目。不得暗示获得 MAA、MaaFramework 或鹰角网络的认可、授权或背书。

当前 v0.1 仅开发纯 Python 路径规划核心。除非用户明确进入后续阶段，否则：

- 不接入 MaaFramework；
- 不复制 MAA 的代码、图片或其他资源；
- 不开发 GUI；
- 不实现截图识别、OCR 或模板匹配；
- 不上传 GitHub 或其他远端；
- 不执行自动点击或设备控制。

## 工程约束

- 支持 Python 3.11+、Windows PowerShell、src layout、pyproject.toml 和 pytest。
- 核心领域模型及求解器必须与 UI、识别和 MaaFramework 解耦。
- 加工品移动必须配置驱动，禁止在求解器中写死具体名称或效果。
- 求解器必须优先保证抵达出口，且不得修改输入对象。
- 路线比较依次为：收益更高、剩余行动力更多、加工品消耗更少、稳定顺序。
- 修改行为时同步测试、JSON 样例和架构文档。
- 禁止提交 `.venv`、缓存、构建产物、临时文件和 `data/raw_private`。
- 项目暂定使用 AGPL-3.0；引入依赖或素材前检查许可证兼容性。

## 验证

提交前至少运行全部 pytest，并实际运行一个成功和一个无解 CLI 样例。检查 `git diff` 与 `git status`，确保没有私有原始数据或生成物进入提交。
