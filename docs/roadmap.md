# 路线图

## v0.1：JSON 地图求解（已完成）

建立不可变领域模型、JSON 校验、配置驱动移动规则、记忆化精确求解、CLI、样例与自动测试。

## v0.2：截图文件识别（进行中）

当前已完成第一版离线原型：私人截图清点、SHA-256 重复判断、场景标签、contact sheet、OpenCV 节点候选检测，以及 annotated PNG/JSON 输出。

本阶段后续仍需提高暗节点和缩放状态的召回率、降低文字/高亮误检，并设计节点类型、道路连接、当前位置和出口识别。只处理用户提供的本地截图，不控制设备。

## v0.3a：MaaFramework Custom Recognition 骨架（已完成）

以可选依赖方式提供 Pipeline v2、Python AgentServer Custom Recognition、框架无关 adapter 和稳定 detail schema。当前 action 固定为 `DoNothing`，不接 controller、ADB、截图或点击，`solver_ready=false`。

## v0.3b：真实 MaaFramework 运行时兼容性（已完成）

使用官方 Python/native runtime 验证 Custom Recognition 注册、真实 AnalyzeArg/AnalyzeResult 契约和 Pipeline v2 资源加载；不启动 Agent socket，不接 controller、ADB、截图或动作。

## v0.3c-A：私有节点模板候选集（已完成）

从非 holdout 的无损截图裁取正常/缩小比例候选，完成风险标注、感知去重和人工检查包；只加载私人 Pipeline 草案，不执行 TemplateMatch。

## v0.3c-B：私人 TemplateMatch 评估（已完成可信实验）

人工批准少量模板后，使用 MaaFramework 5.12.2 的真实 `Resource` 与 `Tasker.post_recognition()` 在静态图片上执行 TemplateMatch，并在固定 holdout 上对比 OpenCV、模板单独及融合结果。模板命中必须经过 ROI/UI、动态格距吸附和按格去重；所有输出仍为私人数据，`solver_ready=false`。只有通过道路拓扑、出口与人工校验后，才可另行设计视觉结果到 solver 的安全门。

## v0.4：独立可视化界面

提供地图校正、候选路线解释、资源设置和模拟展示。UI 不承载求解规则。

## v1.0：动态重规划和 Windows 发布

根据每一步执行后的新观测更新状态并重新规划，完善异常恢复、打包、版本升级和 Windows 发布流程。任何自动控制能力都需要单独的安全设计与用户授权。
