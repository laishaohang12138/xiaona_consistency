# 小娜一致性系统底层数学模型 v0.1

## 1. 文档地位

本文是项目底层数学语义的正式规范，定义系统测量什么、哪些量可以比较、
哪些量当前不可识别，以及测量结果能够进入哪一层工程流程。

本文不替代运行手册，也不改变现有审核排序。其规范优先级如下：

1. `01_project_charter.md` 定义项目目标和治理边界；
2. 本文定义底层数学模型和可识别性边界；
3. `38_identity_evidence_contract_vnext.md` 定义当前 Shadow 数据合同；
4. `32_same_truth_projection_uncertainty.md`、`35_consistency_confidence_matrix.md`
   等文档描述旧审核表面和运行用途。

旧产物中名为 `score`、`confidence` 或 `uncertainty` 的字段仍可用于兼容性审核
路由，但除非本文另有说明，它们不是概率、统计置信度或身份真值。

## 2. 项目目标与非目标

项目目标是为小娜候选批次建立可追溯、可复现、可逐步校准的一致性证据系统，
服务于强身份一致性 LoRA 数据准备和后续 checkpoint 比较。

项目只负责：

- 机器筛选；
- 原生一致性测量；
- 证据缺口暴露；
- 风险路由和人工复核排序；
- 可供外部决策流程消费的证据打包。

项目不负责：

- 最终图集构造或成员资格；
- 最终训练集准入；
- 自动改写小娜身份真相；
- 在项目优化完成前拟合参数、阈值、权重或协方差；
- 将 Winner Bank 变成真相源。

因此，本项目的数学对象是“测量证据”，不是“自动裁决”。

## 3. 状态词

本文使用四个实现状态：

| 状态 | 含义 |
| --- | --- |
| `IMPLEMENTED_GOVERNANCE` | 已实现的硬治理不变量，可阻断运行，但不产生一致性分数 |
| `IMPLEMENTED_SHADOW` | 已在代码中实现，可输出原生量，但无决策影响且未校准 |
| `LEGACY_REVIEW_HEURISTIC` | 当前审核链仍在使用的启发式分数，仅用于兼容和路由 |
| `DEFERRED_CALIBRATION` | 数学方向成立，但当前数据不足以识别，禁止进入生产决策 |

所有 vNext 数学记录必须保留：

```text
calibration_state = SHADOW_UNCALIBRATED
decision_influence = NONE
parameter_fitting_allowed = false
```

## 4. 符号系统

定义：

| 符号 | 含义 |
| --- | --- |
| `A_f` | 唯一脸部真相 `A-Core_01_0deg_MASTER.png` |
| `A_b` | 唯一身材真相 `Task-63987060-116-1.png` |
| `x` | 当前候选图像 |
| `k` | 测量轴，例如脸部身份、脸部形状、身体形状、身体拓扑 |
| `c` | 观测条件，包括视角、姿态、步态、相机、裁切、衣物、遮挡和光照 |
| `O_k(x,c)` | 轴 `k` 在条件 `c` 下的观测合同 |
| `M_k` | 轴 `k` 的测量器及其 provider contract |
| `r_k` | 轴 `k` 的原生残差，数值越小表示越一致 |
| `R(x)` | 多轴原生残差向量，不是综合分 |
| `Q_k` | 轴 `k` 的可靠性描述向量，不是概率 |
| `G` | 证据派生谱系 DAG |

底层输出定义为：

```math
\mathcal E(x)=\{O_k, M_k, r_k, Q_k, G_k\}_{k\in K}
```

当前禁止将其默认压缩成单个万能分数：

```math
S(x)=\sum_k w_k s_k
```

原因是不同轴位于不同几何空间、共享上游、缺失机制不同，而且目前没有独立
benchmark 支持统一尺度或权重。

## 5. 真相模型

### 5.1 两项唯一真相

```math
\mathcal T=\{A_f,A_b\}
```

- `A_f` 具有 `FACE_MASTER / ABSOLUTE_FROZEN` 权威；
- `A_b` 具有 `FULL_BODY_MASTER / ABSOLUTE_FROZEN` 权威；
- 两者均由路径、角色和 SHA-256 固定；
- 真相完整性失败必须先于 provider 和 GPU 初始化终止运行。

Winner Bank 的数学地位是：

```text
mutable_review_memory_only
truth_authority = NONE
```

它可以记录人工复核历史，但不能进入真相估计、阈值拟合或自动准入。

### 5.2 身体真相与步态轨道

`A_b` 是唯一权威来源，但身体真相不能解释为一张静态像素轮廓。相同身体在
姿态、步态和相机条件变化下会产生不同投影。概念上定义同真相观测轨道：

```math
\mathcal O_b(A_b;\mathcal C)
=\{\Phi(A_b,c):c\in\mathcal C\}
```

其中 `Phi` 是条件观测过程，`c` 包括 pose、gait、view 和 camera。

这一定义表达“绝对身材真相按步态解释”，但不产生新的真相锚：

- `A_b` 仍是唯一 authority source；
- 步态是条件变量，不是第三个 anchor；
- HMR2/SMPL 重建或渲染是 prior-dependent 派生观测，不是真相本体；
- 当前未冻结 `c` 的概率分布，因此不能声称计算了轨道期望。

## 6. 分层测量模型

系统按以下顺序工作：

```text
Frozen Truth Registry
        -> Axis Observation Eligibility
        -> Provider Comparability Contract
        -> Native Geometry Measurement
        -> Repeatability / Chain Diagnostics
        -> Evidence Lineage and Family Grouping
        -> Shadow Evidence Record
        -> Legacy Review Router (兼容层)
```

数学层不输出 PASS/FAIL。政策层不得修改数学层的原生残差。

## 7. 逐轴观测资格

每个轴独立声明观测资格：

```text
MEASURABLE
CONDITIONAL
PRIOR_DEPENDENT
UNOBSERVABLE
UNAVAILABLE
UNASSESSED
```

观测资格不表示一致或不一致，只表示当前图像是否有资格支持该测量。

当前脸部 vNext 范围：

| 视角 | `face_identity` | `face_shape` |
| --- | --- | --- |
| front | `MEASURABLE` | `MEASURABLE` |
| three-quarter | `CONDITIONAL` | `CONDITIONAL` |
| side | `PRIOR_DEPENDENT` | `PRIOR_DEPENDENT` |
| back | `UNOBSERVABLE` | `UNOBSERVABLE` |

当前身体 vNext 由 HMR2/SMPL 重建提供，因此只要 body core 工件可用，front、
three-quarter、side 和 back 都必须声明为 `PRIOR_DEPENDENT`；工件或有效分量不足
时输出 `UNAVAILABLE`。这不是说各视角信息量相同，而是避免把模型补全误写成直接
可测。缺失分量不得用剩余分量重新归一化为高可信结论。

观测记录还必须区分：

```text
CHAIN_VALID
CHAIN_INVALID
CHAIN_UNASSESSED
```

`CHAIN_VALID` 只说明所需工件可读取，不说明检测器稳定。稳定性必须由独立扰动
实验描述。

## 8. Provider 可比性合同

参考和候选必须由同一把尺测量。provider contract 至少冻结：

- provider 名称与版本；
- 模型 ID 与模型资产 SHA-256；
- 实现 SHA-256；
- CPU/CUDA execution backend；
- detector、alignment 和 preprocessing contract；
- embedding 维数或 landmark 数；
- landmark schema、点序和坐标约定；
- 原始字段来源和归一化规则。

身体 provider contract 还必须冻结 HMR2 checkpoint 与 SHA-256、SMPL body model
与 SHA-256、bbox/preprocessing contract、measurement schema 与顺序、shape/topology
维数、坐标约定及源字段。身体参考与候选的 CPU/CUDA backend 不同同样属于
`MISMATCH`。

两个哈希语义不同：

```text
observed_contract_sha256
  = 对当前已观测字段做指纹，即使合同不完整也存在

comparable_contract_sha256
  = 仅在全部关键字段已解析时存在
```

比较状态为：

```text
MATCH
PARTIAL_MATCH
MISMATCH
UNAVAILABLE
```

只有 `MATCH` 可以声称测量器可比。关键字段冲突必须 withholding residual；
`PARTIAL_MATCH` 不能被静默提升为可比。CPU 与 CUDA 是不同测量合同。

## 9. 脸部身份原生几何

状态：`IMPLEMENTED_SHADOW`

当前 identity vector 来自 InsightFace 运行时对齐链，不是从 3DDFA canonicalized
图像重新提取的 embedding。因此规范字段为 `runtime_face_embedding_raw`，旧字段
`canonical_identity_vector` 仅作为兼容 alias。

对参考和候选 embedding 显式 L2 归一化：

```math
\hat e=\frac{e}{\max(\lVert e\rVert_2,\epsilon)}
```

令 `a` 和 `x` 分别为归一化后的真相与候选 embedding，则二者位于单位超球面：

```math
a,x\in S^{d-1}
```

原生身份残差为球面角距离：

```math
r_{face-id}
=\arccos\left(\operatorname{clip}(a^\top x,-1,1)\right)
```

- 单位：`radian`；
- 方向：`lower_is_more_consistent`；
- 零范数、非有限值、维数不一致或 provider mismatch 时不输出残差；
- 当前不把角距离映射成 0 到 1 的“身份概率”。

旧链中的 `exp(-MAE)` 相似度属于 `LEGACY_REVIEW_HEURISTIC`，不是 vNext 原生
身份测量。

## 10. 脸部投影形状几何

状态：`IMPLEMENTED_SHADOW`

令对应的参考和候选 2D canonical landmarks 为：

```math
A=(a_1,\ldots,a_n),\qquad X=(x_1,\ldots,x_n)
```

基础可见性权重为 `m_i >= 0`。在每轮 IRLS 中，以 `w_i=m_i u_i` 计算加权
中心和加权 RMS 尺度，得到 `A_hat` 与 `X_hat`。随后求 proper rotation：

```math
R^*=\arg\min_{R\in SO(2)}
\sum_i w_i\lVert \hat x_iR-\hat a_i\rVert_2^2
```

反射不允许。点残差为：

```math
d_i=\lVert \hat x_iR^*-\hat a_i\rVert_2
```

Huber IRLS 权重更新为：

```math
u_i=\min\left(1,\frac{\delta}{\max(d_i,\epsilon)}\right)
```

Huber 损失为：

```math
H_\delta(d)=
\begin{cases}
\frac{1}{2}d^2,&d\le\delta\\
\delta(d-\frac{1}{2}\delta),&d>\delta
\end{cases}
```

最终 Shadow 残差为：

```math
r_{face-shape}
=\sqrt{
\frac{2\sum_i m_iH_\delta(d_i)}
{\max(\sum_i m_i,\epsilon)}}
}
```

当前固定工程参数：

```text
huber_delta = 0.05
max_iterations = 20
tolerance = 0.000000001
```

这些是预注册工程默认值，不是由候选批次拟合的阈值。

该量只表示 canonical 2D 投影形状兼容性，不是绝对 3D 脸部真相。低、中、高
y 分带、lateral band 和 center-axis band 是同一 landmark 集的诊断分解，不能
作为独立证据重复计票。

## 11. 身体形状、步态与拓扑

### 11.1 当前旧审核链

状态：`LEGACY_REVIEW_HEURISTIC`

当前 HMR2 身体链包含如下形式：

```math
s_\beta=\exp(-\operatorname{MAE}(\beta_A,\beta_X))
```

```math
s_{topology}=\exp(-3\operatorname{MAE}(t_A,t_X))
```

单个 measurement 使用固定尺度：

```math
s_l=\frac{1}{1+|q_l(X)-q_l(A_b)|/c_l}
```

并对 torso、shoulder-neck、waist-pelvis、leg-axis、lower-body-volume 和 gait
phase 做固定权重聚合。这些量当前仍可用于人工审核解释，但其中的指数尺度、
`c_l` 和聚合权重没有独立 benchmark 校准，不能被称为概率、标准化残差或
底层真值距离。

`body_pose_explained_delta_score` 同样是审核解释量。姿态变化不能因为该分数高
就自动把结构差异变成“正确”。

### 11.2 vNext body core 原生残差

状态：`IMPLEMENTED_SHADOW`

当前实现只选择五个正值、无量纲的 core ratio，并固定顺序：

```text
shoulder_width_to_torso
hip_width_to_torso
shoulder_to_hip_ratio
upper_to_lower_leg_ratio
foot_length_to_leg
```

对每个参考与候选均有效的分量，输出有符号对数比：

```math
r_l(x)=\log\frac{q_l(x)}{q_l(A_b)}
```

因此 `r_l=0` 表示该比例相同，正负号保留变化方向。记录输出
`residual_vector`、`component_residuals`、`used_components`、缺失/非法分量和 coverage，
但 `residual=null`，禁止将向量压成标量。至少三个有效分量才输出向量；这只表示
最低工件可用性，不是统计置信阈值。五个分量共享 HMR2/SMPL 上游，只构成一个
`body_shape_geometry` evidence family，不能当成五张独立票。

该残差仍是 HMR2/SMPL prior-dependent 的图像观测，不是从绝对 3D 身体真值直接量得。
姿态差向量、可见身体比例、衣物覆盖和 HMR2 coverage 分别写入 condition/reliability
描述，不进入残差，也不奖励一致性。provider 明确冲突时必须 withholding；合同不完整
时保留缺口，不能声称测量器可比。

### 11.3 vNext native 3D 身体拓扑残差

状态：`IMPLEMENTED_SHADOW`

HMR2 对参考与候选分别预测 shape beta 后，必须在同一个 neutral SMPL 模型上将
`global_orient` 和全部 `body_pose` 旋转替换为单位旋转，得到索引严格对应的 zero-pose
顶点矩阵：

```math
V_A,V_x\in\mathbb R^{6890\times3}
```

只去除每个网格的质心平移：

```math
\widetilde V=V-\mathbf 1\bar v^T,
\qquad
r_{topology}(x)=\operatorname{vec}(\widetilde V_x-\widetilde V_A)
\in\mathbb R^{20670}
```

输出保留候选减参考的有符号 xyz 坐标差，且 `residual=null`。实现禁止旋转拟合、尺度
拟合、Procrustes、pose fitting、向量范数、综合分和阈值分类。这样只消除无身份含义的
全局平移，同时保留由 SMPL shape 导出的尺度与局部结构差异。

provider contract 必须同时匹配 HMR2 checkpoint、实现哈希、execution backend、neutral
SMPL 资产哈希、preprocessing、zero-pose canonicalization、centroid-only alignment、完整
6890 顶点数量、20670 坐标维数和精确顶点索引语义。任一项缺失或冲突时 measurement
withholding，readiness 为 `BLOCKED`。

`body_topology_signature` 仍是 beta 与 body25 ratio 的 legacy 派生启发式，只服务旧审核
兼容层；即使参考与候选都存在该签名，也不能解除 native blocker。native topology 与
body core 共用 HMR2/SMPL 上游，二者同属 `body_shape_geometry` evidence family，不能作为
两张独立票。zero-pose 消除了显式 pose 参数进入 mesh 的路径，但图像、衣物、遮挡、视角
对 beta 回归的影响仍在，因此它仍是 prior-dependent compatibility evidence，不是绝对 3D
身体真相。

该轴已进入身体重复性协议 v0.2，但状态仍是 `PREREGISTERED_NOT_EXECUTED`。它复用与
body-core 完全相同的 baseline-plus-13 HMR2 工件，不增加重型推理次数。每个可用 trial
保留完整 20670 维坐标残差；汇总仅按固定 x、y、z 轴分别报告 signed quantiles
`0.05/0.25/0.50/0.75/0.95` 与 absolute quantiles
`0.50/0.90/0.95/0.99/1.00` 的跨 trial 描述。禁止坐标轴聚合、vertex norm、topology
score、稳定标签和阈值。协议预注册不等于已经观察到稳定性。

### 11.4 条件标准化的未来目标

状态：`DEFERRED_CALIBRATION`

理想身体 measurement `q_l` 应在条件 `c` 下比较：

```math
z_l(x,c)=
\frac{q_l(x)-\mu_l(A_b\mid c)}
{s_l(c)+\epsilon}
```

其中：

- `mu_l(A_b | c)` 是唯一身体真相在相同 view/pose/gait/camera 下的条件观测中心；
- `s_l(c)` 是同一测量链在该条件下的经验敏感性尺度；
- coverage 和 eligibility 决定该轴是否有资格输出，而不是作为身份残差分量。

当前只有一张身体真相，且没有冻结的姿态、相机、衣物与检测扰动分布，因此
`mu_l` 和 `s_l` 目前不可识别。当前禁止将人工常数包装为条件均值或统计方差。

该未来标准化量不得覆盖 11.2 的原始对数比。现阶段只记录 view、pose、gait、camera、
coverage、garment/occlusion condition、provider contract 和 `prior_dependent`，不计算
条件均值、条件尺度或姿态扣除量。不支持的轴必须 withholding。

### 11.5 身体反事实场景

未来可从 `A_b` 派生 HMR2/SMPL 模型，在目标条件下渲染，并将渲染图重新通过与
候选完全相同的 detection、crop、parsing、HMR2 和 measurement 链。这样可减少
“真相从干净网格直接量、候选从带衣物 2D 图量”的尺子不一致。

但在场景分布冻结前，只能定义有限场景集：

```math
\mathcal S_c=\{s_1,\ldots,s_n\}
```

输出 `scenario_median/min/max/spread`，不能输出数学期望。每条记录必须声明：

```text
truth_source = Task-63987060-116-1.png
prior_dependent = true
renderer_dependent = true
observation_chain_replayed = true/false
decision_influence = NONE
```

## 12. 衣物、姿态和可见性的数学地位

衣物、姿态、步态、光照、裁切与遮挡不是“小娜身份的附加残差轴”。它们是：

1. 观测条件；
2. 可见性和 eligibility 的来源；
3. detector/alignment 扰动来源；
4. 未来 nuisance 研究的候选变量。

轻微亮度、Gamma、JPEG、resize 和微小 crop 可进入受控重复性实验。严重模糊、
大面积遮挡、极端裁切、脸过小以及不可见背面结构不能作为可扣除 nuisance；它们
应降低可观测性或使对应轴不可测。

原则是：

```text
姿态变化可以增加测量不确定性或改变适用域，
但不能自动奖励一致性，也不能自动消除真实结构漂移。
```

## 13. 3D 拓扑语义

当前 3DDFA 和 HMR2/SMPL 都包含模型先验：

- 3DDFA landmarks 表达 canonical projection geometry；
- HMR2 beta、pose、legacy topology signature 和 zero-pose canonical mesh 表达 prior-dependent reconstruction；
- side/back 同真相投影不能创造新锚；
- 分区指标共享同一上游，不是独立 3D 证人。

因此当前允许的表述是：

```text
projection/topology compatibility evidence
prior-dependent same-truth support
```

当前禁止的表述是：

```text
absolute 3D face truth recovered
absolute 3D body truth recovered
independent topology votes
```

## 14. 重复性三分法

状态：`IMPLEMENTED_SHADOW`

同图重复执行不等于现实测量稳定。每个原生轴必须分开记录：

### 14.1 Numerical repeatability

固定输入、模型、backend 和代码重复执行。它只验证数值确定性。

### 14.2 Preprocessing repeatability

当前预注册：

- lossless PNG roundtrip；
- JPEG quality 95 roundtrip；
- resize 0.98 roundtrip；
- resize 1.02 roundtrip。

### 14.3 Admissible perturbation stability

当前预注册：

- crop x 负向和正向微移；
- crop y 负向和正向微移；
- Gamma 负向和正向微扰。

每个域只输出：

```text
trial_count
available_residual_count
min / median / max / spread
perturbation_family
detector_chain_diagnostics
```

不输出统一 repeatability score，不拟合 stable/unstable threshold。

脸部与身体使用独立的预注册协议 ID。身体协议固定执行 baseline 加 13 个串行试验，
同一次 HMR2 重建同时服务 body core 与 native topology。五个 body-core 对数比分量分别
保留 signed descriptor 与 absolute magnitude descriptor，分量之间不求均值、L2、权重
和总分。topology 保留逐 trial 原始坐标向量，并只在相同坐标轴与相同预注册分位数内做
描述统计。身体试验只描述 HMR2 测量链对输入扰动的敏感性，不回答候选是否属于小娜。
所有身体轴均不可测时不启动后续 13 次重型执行；trial 失败时停止并保留原子断点，真实
HMR2 执行之间使用预注册冷却时间。

多源运行先逐图汇总，再描述 source median、source maximum 和 source spread 的
跨源分布。相同 trial 可以跨源比较，但不能把一张图的 13 个相关扰动当成 13 个
独立受试样本。

## 15. 检测与对齐链诊断

每个扰动试验同时记录：

- normalized bbox IoU；
- bbox center displacement；
- bbox width/height relative delta；
- normalized InsightFace kps5 raw RMS；
- 去除平移、尺度和旋转后的 kps5 similarity-shape residual；
- 3DDFA canonical yaw/pitch/roll delta 和 L2 delta。

身体重复性链相应记录 body bbox IoU/中心/宽高变化、HMR2 coverage delta、fit-confidence
delta 和 provider-native pose-parameter RMS delta。这些量仍是同一重建链的诊断字段，
不是额外的身份票。

这些量用于区分：

```text
输入 framing 变化
-> detector bbox 变化
-> landmark/alignment 变化
-> embedding 或 shape residual 变化
```

它们和最终身份/形状残差共享同一测量链，不能作为额外独立投票。

## 16. 可靠性向量，而不是总方差

当前每个轴的可靠性表示为并列向量：

```math
Q_k=(
eligibility,
scope,
coverage,
chain\ state,
provider\ comparability,
repeatability\ descriptors,
prior\ dependence,
evidence\ gaps
)
```

当前禁止：

```math
\sigma^2_{total}
=\sigma^2_{measurement}
+\sigma^2_{observation}
+\sigma^2_{scope}
+\sigma^2_{prior}
```

因为 scope 是离散适用域，prior risk 可能是系统偏差，各项既不同尺度，也未证明
独立、零均值或具有方差语义。

同理，`coverage` 不是身份相似度。缺失证据不能通过剩余分数重新归一化成高可信
结论。当前也不能把工程惩罚项命名为 95% 置信上界。

## 17. 证据谱系与去重

状态：`IMPLEMENTED_SHADOW`

证据图 `G=(V,E)` 是有向无环图。当前持久化的直接派生边只有：

```text
OBSERVED_FROM
TRANSFORMED_FROM
DERIVED_FROM
```

共享上游关系通过图遍历推导，不作为派生边写入。图必须无环。

底层证据家族为：

```math
R(x)=
\begin{bmatrix}
r_{face-id}\\
r_{face-shape}\\
r_{body-shape}\\
r_{body-topology}
\end{bmatrix}
```

可见性属于每个轴的条件，不进入残差向量。

同一 provider 内的 partition、weakest part、mean、top drift 和局部诊断只能解释
一个 evidence family，不能因为字段数量多就获得更多票。

## 18. 聚合与当前审核兼容层

vNext Shadow 层当前固定：

```text
combined_repeatability_score = null
stable_unstable_classification = null
decision_influence = NONE
```

现有 `review_only_score_v2`、`review_only_confidence_v2`、同真相 projection score
和 consistency confidence matrix 继续作为 `LEGACY_REVIEW_HEURISTIC` 兼容运行。
它们的 PASS/WARN/FAIL 只表示审核优先级和证据路由，不表示训练准入或概率判断。

任何未来聚合必须满足：

1. 冻结独立 benchmark；
2. 冻结 provider contract；
3. 估计并验证 evidence family 间依赖；
4. 单轴逐步晋级；
5. 明确政策层与测量层隔离；
6. 完成外部人工审计。

在此之前，不启用 Mahalanobis、GLS、Bayes 或学习型 judge。

## 19. 局部 nuisance 切空间研究

状态：`DEFERRED_CALIBRATION`

对单位球面身份 embedding，可在脸部真相 `a` 的切空间使用对数映射：

```math
v=\log_a(x)
=\theta\frac{x-\cos(\theta)a}{\sin(\theta)}
```

对合法局部变换 `T_j` 的中心有限差分方向：

```math
u_j(h)=
\frac{
\log_a E(T_j(A_f,+h))-\log_a E(T_j(A_f,-h))
}{2h}
```

进入 nuisance basis 前必须独立验证：

- 不同步长的方向余弦稳定；
- 幅度比例稳定；
- 正负扰动局部对称；
- 未参与建基步长的线性重建误差；
- detector、bbox、landmark 和 alignment 未发生离散跳变。

若使用截断 SVD：

```math
B=U\Sigma V^\top,\qquad P_r=U_rU_r^\top
```

若使用原始基矩阵 ridge：

```math
P_\lambda=B(B^\top B+\lambda I)^{-1}B^\top
```

两条路线不能混写。秩 `r` 和 ridge `lambda` 都必须由独立预注册验证集确定，
当前禁止拟合。

还必须使用与候选批次独立的 hard-negative bank 做 identity leakage audit。投影
必须显著解释合法正探针，同时保持负样本安全边际。当前只能生成诊断报告，不能
生成 leakage gate。

## 20. 未来 checkpoint 判敛统计层

状态：`DEFERRED_CALIBRATION`

底层测量引擎只输出：

```math
r_{t,c,k},\qquad Q_{t,c,k}
```

其中 `t` 是 checkpoint，`c` 是 case，`k` 是轴。

固定 seed、prompt、workflow、encoder 和 reference 时，可做配对差：

```math
\Delta_{c,k}=r_{old,c,k}-r_{new,c,k}
```

正值表示新 checkpoint 的残差降低。Nano Banana 未暴露 seed 时不得伪造配对，
只能按 prompt family 比较独立样本分布。

冻结 benchmark 后，统计层应分别报告：

- 配对中位改善；
- 新 checkpoint 的 P90/P95 尾部残差；
- 严重失败率；
- 跨 seed IQR；
- 按 prompt family 分层的 bootstrap 区间；
- 多样性和 mode-collapse 诊断。

`CONVERGING`、`PLATEAU`、`REGRESSING`、`OVERFITTING`、`MODE_COLLAPSE` 和
`INCONCLUSIVE` 属于未来独立统计层，不能由当前候选批次或 Winner Bank 拟合。

## 21. 当前可识别性矩阵

| 数学对象 | 当前状态 | 当前允许的结论 |
| --- | --- | --- |
| 两项真相路径、角色、SHA-256 | `IMPLEMENTED_GOVERNANCE` | 完整性 PASS/FAIL |
| 脸部 embedding L2 归一化与角距离 | `IMPLEMENTED_SHADOW` | 原生弧度残差 |
| 2D weighted IRLS Procrustes | `IMPLEMENTED_SHADOW` | canonical 投影形状残差 |
| 逐轴 eligibility 和 chain state | `IMPLEMENTED_SHADOW` | 可测性与缺口 |
| provider comparability | `IMPLEMENTED_SHADOW` | MATCH/PARTIAL/MISMATCH/UNAVAILABLE |
| 重复性三域和跨源描述 | `IMPLEMENTED_SHADOW` | 描述统计，不是稳定标签 |
| Evidence Lineage DAG | `IMPLEMENTED_SHADOW` | 派生追踪和防重复计票 |
| 身体 core ratio 对数残差向量 | `IMPLEMENTED_SHADOW` | prior-dependent 原生向量，不是综合分 |
| 身体姿态/步态与衣物遮挡条件 | `IMPLEMENTED_SHADOW` | nuisance/reliability 描述，不参与投票 |
| 身体 topology provider contract | `IMPLEMENTED_SHADOW` | 完整 SMPL 顶点、模型与 canonicalization 可比性 |
| native 3D 身体拓扑残差 | `IMPLEMENTED_SHADOW` | 20670 维有符号坐标差，不是综合分 |
| native 3D 身体拓扑重复性 | `IMPLEMENTED_SHADOW` | v0.2 已预注册、尚未真实执行，不输出稳定结论 |
| 当前身体固定尺度相似度 | `LEGACY_REVIEW_HEURISTIC` | 审核解释和排序兼容 |
| 身体条件中心与条件尺度 | `DEFERRED_CALIBRATION` | 当前不可识别 |
| nuisance basis 和 leakage gate | `DEFERRED_CALIBRATION` | 只能设计控制实验 |
| 完整协方差、Mahalanobis、总方差 | `DEFERRED_CALIBRATION` | 禁止生产使用 |
| 统计置信区间和判敛概率 | `DEFERRED_CALIBRATION` | 禁止声称 |

## 22. Shadow 记录的规范形状

```json
{
  "truth": {
    "face": "A-Core_01_0deg_MASTER.png",
    "body": "Task-63987060-116-1.png",
    "integrity": "PASS"
  },
  "axis": "face_identity",
  "observation": {
    "eligibility": "MEASURABLE",
    "scope_state": "FRONT_SUPPORTED",
    "observation_chain_state": "CHAIN_VALID",
    "raw_observations": {}
  },
  "measurement": {
    "native_space": "unit_hypersphere",
    "measurement": "runtime_aligned_face_angular_distance",
    "residual": 0.0,
    "unit": "radian",
    "calibration_state": "SHADOW_UNCALIBRATED"
  },
  "provider_contracts": {
    "comparison_state": "MATCH"
  },
  "repeatability": {
    "numerical_repeatability": {},
    "preprocessing_repeatability": {},
    "admissible_perturbation_stability": {},
    "combined_repeatability_score": null
  },
  "lineage": {},
  "governance": {
    "decision_influence": "NONE",
    "may_affect_ranking": false,
    "may_affect_review_route": false,
    "may_affect_winner_bank": false,
    "may_modify_truth": false,
    "parameter_fitting_allowed": false
  }
}
```

示例中的 `residual` 只展示字段位置，不是阈值或项目实测结果。动态批次数据不得
写入本规范。

## 23. 晋级条件

任一 Shadow 数学轴获得决策影响前，必须全部满足：

1. 独立 benchmark 和标签冻结；
2. benchmark 与当前候选、Winner Bank、Top-K 完全隔离；
3. provider/model/preprocessing/backend 合同冻结；
4. numerical、preprocessing 和 admissible perturbation 重复性完成；
5. eligibility、coverage 与 withholding 行为通过回放；
6. 正探针和 hard negatives 泄漏审计通过；
7. 参数选择只使用预注册开发集，最终报告使用未见验证集；
8. 统计区间完成经验覆盖率验证；
9. 单轴晋级影响审查通过，不允许整套一次性接管；
10. 治理层显式修改 `decision_influence`，代码不得自行提升。

## 24. 代码映射

| 数学层 | 当前实现 |
| --- | --- |
| 真相完整性 | `core/qa_truth_integrity.py` |
| 身份/形状合同 | `core/qa_identity_evidence_contract.py` |
| Shadow 构造 | `core/qa_identity_evidence_shadow.py` |
| 身体观测与原生向量合同 | `core/qa_body_evidence_contract.py` |
| 身体 Shadow 构造 | `core/qa_body_evidence_shadow.py` |
| 身体重复性预注册协议 | `configs/body_repeatability_protocol.yaml` |
| 身体重复性测量适配器 | `core/qa_body_repeatability_adapter.py` |
| 身体重复性执行、core 逐分量与 topology 逐轴分位数汇总 | `core/qa_body_repeatability_runner.py` |
| 球面角残差 | `core/qa_identity_evidence_contract.py` |
| IRLS Procrustes | `core/qa_procrustes_shape.py::weighted_irls_procrustes` |
| Provider contract | `core/qa_provider_contract.py` |
| Evidence DAG | `core/qa_evidence_lineage.py` |
| 重复性描述 | `core/qa_repeatability_shadow.py` |
| 共享重复性执行骨架/脸部执行器 | `core/qa_identity_repeatability_runner.py` |
| 决策边界与外部复核路由 | `core/qa_governance.py` |
| 检测链适配 | `core/qa_face_repeatability_adapter.py` |
| 3DDFA 工件桥 | `core/qa_face_pose_canonical_3ddfa.py` |
| 当前身体旧审核链 | `core/qa_heavy_body_canonical.py` |
| 当前审核兼容层 | `core/qa_review_only_score.py`、`core/qa_consistency_confidence_matrix.py` |

## 25. 最终不变量

```text
1. 真相层只有 A_f 和 A_b。
2. 步态是 A_b 的条件观测语义，不是新锚。
3. 原生几何残差先于任何 0 到 1 映射。
4. 观测资格、可靠性和残差必须分离。
5. 同源派生指标不得重复计票。
6. 缺失证据不得被剩余高分补偿。
7. 当前 Shadow 不输出概率、置信区间、综合分或稳定标签。
8. 当前候选和 Winner Bank 不参与参数拟合。
9. 机器输出不决定图集或训练集准入。
10. 任何晋级都必须由冻结 benchmark 和显式治理变更触发。
11. 缺失、未知或旧版治理输入必须 fail-closed；兼容准入字段只能输出 `false`。
```

底层数学路线由此冻结为：

```text
异构原生几何残差
+ 逐轴观测资格
+ Provider 可比性
+ 重复性与检测链诊断
+ 证据谱系去重
+ 明确的不可识别性
```

而不是继续扩大“多个 0 到 1 分数的加权平均”。
