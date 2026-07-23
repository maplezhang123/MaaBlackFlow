# MaaBlackFlow 协作约束

## 项目边界

MaaBlackFlow 是独立、非官方的社区项目。不得暗示获得 MAA、MaaFramework 或鹰角网络的认可、授权或背书。

当前完成 v0.1 路径规划核心，并正在开发 v0.2 离线截图识别。除非用户明确进入后续阶段，否则：

- 不接入 MaaFramework；
- 不复制 MAA 的代码、图片或其他资源；
- 不开发 GUI；
- 不上传 GitHub 或其他远端；
- 不执行自动点击、模拟器控制或设备控制；
- 不修改既有求解器行为来迎合视觉输出。
- 未经道路拓扑、出口和人工校验的视觉候选不得输入求解器。

## 私人数据

- 仓库根 `Screenshots/` 是只读的本地私人输入；不得移动、删除、修改或重命名其中的原图。
- `Screenshots/`、`data/raw_private/`、`data/outputs_private/`、`data/ground_truth_private/` 必须保持 Git 忽略。
- 不提交真实截图、真实文件名清单、标注图或其他私人派生产物。
- 数据清单只能在私人输出目录保存绝对/真实信息；公开文档只使用虚构示例。
- 处理截图前先核对 Git 忽略与索引状态；若私人图片已被跟踪或暂存，应停止并汇报。

## 工程约束

- 支持 Python 3.11+、Windows PowerShell、src layout、pyproject.toml 和 pytest。
- `solver` 不得依赖 OpenCV；视觉能力放在独立 `maablackflow.vision` 包。
- 核心领域模型及求解器必须与 UI、识别和 MaaFramework 解耦。
- 不得为单张真实截图硬编码节点坐标；算法必须配置化或使用通用视觉特征。
- 视觉测试只使用程序生成的合成图，不复制真实截图到 `tests/`。
- 加工品移动必须配置驱动，禁止在求解器中写死具体名称或效果。
- 求解器必须优先保证抵达出口，且不得修改输入对象。
- 路线比较依次为：收益更高、剩余行动力更多、加工品消耗更少、稳定顺序。
- 项目暂定使用 AGPL-3.0；引入依赖或素材前检查许可证兼容性。

## 验证

提交前运行全部 pytest、`git diff --check` 和 `git status`。明确检查 Git 索引中不存在 `.venv`、缓存、`Screenshots/`、`data/raw_private/` 或 `data/outputs_private/`。只提交源码、合成测试、文档、依赖配置和忽略规则。