# 架构设计

## 分层边界

项目分成三个独立边界：`vision` 把离线图片转换为候选坐标，未来的识别/适配层把视觉结果转换为领域地图，`solver` 只消费结构化的 `MapState` 和 `GameState`。求解器不导入 OpenCV，也不了解截图、OCR、设备、点击坐标或 UI。

这种解耦允许视觉误差在进入规划前被校验和人工修正，同时保持 v0.1 的精确求解测试确定、快速且无设备副作用。未来 MaaFramework 只能作为外层适配器接入，不能进入规划核心。

## 路径规划领域模型

`Node` 描述节点坐标、类型、一次性收益、完成状态、可重复属性和出口属性；`Edge` 描述有向徒步连接及成本。`MovementRule` 用位移向量集合、行动力成本、剩余次数和越过未完成节点的许可描述加工品。

`MapState` 聚合节点、道路和移动规则，`GameState` 保存当前位置、行动力、已完成节点及加工品剩余次数。所有输入领域对象均为冻结 dataclass，求解过程只创建内部状态，不修改调用方对象。

## 路径规划 DP

记忆化搜索键为：

```text
(current_node, remaining_action_points, completed_mask, movement_uses)
```

徒步只沿配置边移动，不能跨过未完成的共线中间节点；加工品转移由规则向量驱动。只比较能到达出口的候选，依次最大化总收益、剩余行动力，再最小化加工品使用量，最后使用稳定路线签名打破平局。

设节点数为 `N`、初始行动力为 `A`，第 i 种加工品可用次数为 `U_i`，状态上界为：

```text
O(N × (A + 1) × 2^N × Π(U_i + 1))
```

## 私人数据清点

`vision.dataset` 递归读取指定本地目录中的常见图片格式。每项清单包含相对原始文件名、格式、宽高、通道数、文件大小、SHA-256、重复来源、场景标签及是否适合节点检测。清单不保存输入目录的绝对路径。

重复判断以完整文件 SHA-256 为准。场景标签仅根据原文件名关键词推测，不要求用户重命名。`contact_sheet.png` 使用原图缩略图和稳定序号生成，缩略图坐标不参与检测。

`Screenshots/`、`data/raw_private/` 和 `data/outputs_private/` 均由 Git 忽略。真实截图、真实文件名清单及标注结果不属于公开仓库内容。

## 两阶段离线格点检测

`vision.detector.NodeDetector` 将局部视觉观察和最终节点严格分开。Hough 圆、道路端点、交叉点、转折、暗中心及白色人物组件都只是 `CandidateEvidence`，不能直接成为 `DetectedNode`。

第一阶段位于 `vision.grid`：

1. 使用宽动态地图 ROI 保留靠近左右边缘的地图内容，只排除实际的左侧缩放控件矩形、顶部/底部操作带；右侧边缘密度检测展开的移动工具面板，面板后的内容不外推；
2. 从低饱和灰白像素提取横纵道路掩码，并用形态学骨架化获得中心线；
3. 提取道路端点、交叉点和转折作为局部证据，横纵投影峰仅作为周期拟合投票和调试信息；
4. 高可信圆心先在二维空间聚类，避免同一大图标的多个圆重复投票；
5. 对横纵锚点进行确定性的 RANSAC 式周期与相位投票，分别估计 `sx`、`sy` 和原点偏移。纵向数据不足时可使用横向周期作软约束，但不会复制局部投影峰作为多条轴；
6. 只绘制 `origin + n × spacing` 产生的最终轴。若锚点覆盖不足或无法形成稳定周期，返回 `grid_fit_failed`，不生成伪网格；
7. 在最终网格交点计算道路方向连接和暗中心证据。本阶段仍不生成完整道路边。

第二阶段将所有证据吸附到最近格点。吸附阈值和最终框尺寸按估计格距动态缩放；同一 `(grid_row, grid_col)` 的证据合并为唯一结果，所以同一物理格点最多有一个编号。跨格大圆只能作为该格点的一项证据，不能把原始大框带入最终输出。没有道路、圆或人物支持的纯笛卡尔交点不会被提升为节点。

人物定位先提取满足尺寸、纵向形状与面积约束的白色主体连通域，再要求它与一个已拟合格点关联。人物视觉位置保存为 `marker_center`/`marker_bbox`，地图节点的 `center` 和 `grid_center` 始终使用吸附后的格点坐标。放射线、白字或作战图标不能单独成为人物；人物证据在融合前即选成唯一候选。`event_node`、`empty_waypoint`、`current_position` 和 `uncertain` 都对应具体格点。完全遮挡节点不猜测；`false_positive` 在融合阈值处被拒绝。

算法不包含单图固定坐标，不使用在线 API、训练模型或游戏图片模板。

## 检测输出

`DetectedNode` 包含临时 ID、`grid_row`、`grid_col`、类别、统一的 `center`/`grid_center`、可选人物 `marker_center`/`marker_bbox`、统一比例边界框、总体置信度、可靠性、证据来源、分项评分和逐项 evidence 列表。`DetectionResult.analysis` 记录拟合状态、`sx`、`sy`、原点、最终轴、原始/吸附证据数量、唯一格点数量、合并数量、移动面板判断及“禁止直接进入求解器”的警告。

annotated PNG 只绘制融合后的唯一最终节点，并用不同颜色区分类别。独立的 `grid-debug.png` 以强线绘制最终周期网格，以淡线显示局部投影峰，同时显示原始证据到吸附格点的连线。所有坐标均映射到原图分辨率。

无有效图片、`grid_fit_failed` 或零可信格点时抛出结构化 `VisionError`，CLI 返回非零退出码且不生成伪造结果。


### 证据提供器边界

`EvidenceProvider` 只负责产生局部 `CandidateEvidence`，并声明是否依赖已拟合网格。当前 OpenCV 圆形提供器在网格拟合前运行，人物 marker 提供器在网格拟合后运行。未来可在相同边界加入私人 OpenCV 模板、MaaFramework TemplateMatch 或 OCR 适配器，而不修改全局网格与按格融合。v0.3a 只新增 MaaFramework 可选适配骨架；当前 provider 仍未调用 MaaFramework，也没有真实模板。

### 私人评估

`vision.evaluation` 是独立只读评估层，检测器不导入也不读取 Ground Truth。评估器按 `0.30 × min(sx, sy)` 默认容差进行最小代价最大一对一匹配；`uncertain` 不可消费真值匹配但仍计为预测，因此不会被暗中算作正确。总体指标按 TP、FP、FN 聚合，中心误差只对匹配对统计；人物 marker 和当前位置格点另行验证。私人 GT 位于被忽略的 `data/ground_truth_private/`。

## 当前限制与演进方向

全局周期拟合消除了同一行列附近堆积的重复轴，并显著压缩同格重复候选，但视觉类别仍不等于具体游戏节点类型。顶部 UI 附近的圆形装饰、道路纹理和极暗的孤立节点仍可能造成误检或漏检；两轴周期只是近似，不能替代人工真值或道路拓扑。

检测结果尚无完整道路边、出口和拓扑校验，严禁直接输入路径求解器。下一步应先建立少量本地人工真值并逐节点量化精确率/召回率，再决定继续优化传统视觉还是加入合法的 MaaFramework 模板匹配证据。
## MaaFramework 可选集成边界

`integrations.maafw.adapter` 是纯 Python/OpenCV 边界：接收 BGR `numpy.ndarray`，调用既有 `NodeDetector`，并把结果交给独立的稳定序列化层。detail 不接收文件路径，节点按检测器的稳定行列顺序输出，`solver_ready` 恒为 `false`。主要识别框优先使用当前位置的地图格点框，无当前位置时才使用拟合网格覆盖的 `map_roi`。

`integrations.maafw.agent` 是唯一允许导入 `maa` 的模块，而且仅在实际注册/启动时惰性导入。它只实现 Custom Recognition，不读取 `Context`、controller 或设备；失败返回空 box 并记录结构化错误。Pipeline v2 只有一个 Custom recognition 节点、`DoNothing` action 与空 `next`。因此 Maa runtime 缺失不会影响 `solver`、`vision` 或普通 CLI。

未来 `MaaTemplateEvidenceProvider` 只负责把外部 TemplateMatch 命中的 box/score 转换为 `CandidateEvidence`。它不能直接生成最终节点；命中仍必须经过全局网格吸附、同格融合和现有可靠性判定。v0.3a 不含真实模板，不执行模板匹配。
## 私有模板候选准备

`vision.templates` 是离线数据准备层，不是识别 provider。它在读取来源前按调用方提供的文件名集合排除固定 holdout，只接受无损 1280×720 PNG，并仅裁取现有检测器输出的可靠 `event_node`、`empty_waypoint` 与 `current_position`。放大比例在本轮明确跳过；正常和缩小比例根据网格间距/图高比例分组，裁片大小随格距缩放。

每个候选保存来源哈希标识、原图 bbox、视觉类别、置信度和人物/文字/面板/高亮/裁边风险。感知哈希只用于同类别、同缩放组内的近重复分组；原始候选不删除，推荐集从每组中按风险、置信度和稳定 ID 选择并限制人工检查规模。contact sheet 和摘要不包含绝对路径或真实来源文件名。

私有 Pipeline v2 草案使用 `resource/image` 相对模板路径、地图 ROI、`TemplateMatch` recognition、`DoNothing` action 和空 `next`。候选只在被忽略的 private resource preview 中暂存，不进入公开资源。加载 resource 只验证 JSON、模板路径和资源解析，不等于执行 TemplateMatch；本阶段不创建 Tasker、AgentClient、Controller 或 Context。