# 路线图

## v0.1：JSON 地图求解（已完成）

建立不可变领域模型、JSON 校验、配置驱动移动规则、记忆化精确求解、CLI、样例与自动测试。

## v0.2：截图文件识别（进行中）

当前已完成第一版离线原型：私人截图清点、SHA-256 重复判断、场景标签、contact sheet、OpenCV 节点候选检测，以及 annotated PNG/JSON 输出。

本阶段后续仍需提高暗节点和缩放状态的召回率、降低文字/高亮误检，并设计节点类型、道路连接、当前位置和出口识别。只处理用户提供的本地截图，不控制设备。

## v0.3a：MaaFramework Custom Recognition 骨架（当前阶段）

以可选依赖方式提供 Pipeline v2、Python AgentServer Custom Recognition、框架无关 adapter 和稳定 detail schema。当前 action 固定为 `DoNothing`，不接 controller、ADB、截图或点击，`solver_ready=false`。

## v0.3b：私人 TemplateMatch 实验（待确认）

在不提交真实模板的前提下，把 MaaFramework TemplateMatch 命中作为新的 `CandidateEvidence`，使用私人 GT 评估增益。只有通过道路拓扑、出口与人工校验后，才可另行设计视觉结果到 solver 的安全门。

## v0.4：独立可视化界面

提供地图校正、候选路线解释、资源设置和模拟展示。UI 不承载求解规则。

## v1.0：动态重规划和 Windows 发布

根据每一步执行后的新观测更新状态并重新规划，完善异常恢复、打包、版本升级和 Windows 发布流程。任何自动控制能力都需要单独的安全设计与用户授权。