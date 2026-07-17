# 小娜一致性系统：非补偿式批次选择与多模型生成闭环实施规划 v0.1

## 0. 文档状态

```text
status = DRAFT_FOR_REVIEW
normative = false
intended_reviewers = Codex / GPT / project owner
current_project_stage = MEASUREMENT_QUALIFICATION
```

本文是实施规划，不修改下列现有规范的权威地位：

1. `docs/01_project_charter.md`：项目目标与治理边界；
2. `docs/39_underlying_mathematical_model.md`：底层数学语义与可识别性边界；
3. `docs/38_identity_evidence_contract_vnext.md`：当前 Shadow 身份证据合同；
4. `configs/project_stage.json`：当前阶段权限。

本文提出的所有新选择、排序、生成反馈和跨模型良率能力，在当前阶段必须保持：

```text
calibration_state = SHADOW_UNCALIBRATED
decision_influence = NONE
training_admission_participation = false
image_set_membership_participation = false
truth_authority = NONE
parameter_fitting_allowed = false
```

任何“暂定最佳”“Top Cluster”“Pareto 第一层”只表示批次内机器证据排序，不表示最终训练准入、最终图集成员或新真相锚。

---

## 1. 背景与问题定义

当前项目已经能够输出大量细粒度 JSON 证据，包括但不限于：

- 脸部身份 embedding 球面角残差；
- 脸部 canonical 投影形状残差；
- 身体五分量有符号对数比残差向量；
- 6890 个对应 SMPL 顶点形成的 20670 维有符号拓扑坐标差；
- provider 可比性；
- 逐轴 eligibility；
- pose / gait / clothing / occlusion / lighting 条件；
- 重复性与检测链诊断；
- 证据谱系 DAG；
- legacy review-only 排序与 confidence matrix。

当前缺少的不是更多底层指标，而是一个符合现有数学不变量的上层闭环：

1. 如何在不做简单加权平均的情况下，比较同一批次候选；
2. 如何避免“脸特别好”补偿“身材严重漂移”；
3. 如何避免 20670 个同源坐标被当成 20670 张独立选票；
4. 如何在只有一张脸锚和一张身体锚的情况下处理其他角度；
5. 如何在不引入人工疲劳、审美偏好和标准漂移的情况下自动筛选；
6. 如何让双 GPU、本地多底模、参考节点和姿态控制形成可复现的大规模生成闭环；
7. 如何统计不同底模、工作流和参数在不同任务槽位中的真实高纯度良率；
8. 如何保证候选数量无限增长时，绝对锚始终只有两项且永不被候选平均化。

目标不是制造一个表面精确的：

```text
小娜一致性 = 93.71%
```

目标是建立一个可审计、可拒绝裁决、不可补偿、不会移动真相的机器选择系统。

---

## 2. 不可破坏的系统不变量

### 2.1 唯一真相不变量

```text
A_f = A-Core_01_0deg_MASTER.png
A_b = Task-63987060-116-1.png
truth_anchor_count = 2
```

必须永久禁止：

- 候选图晋升为自动真相；
- Winner Bank 参与真相估计；
- 候选 embedding 求均值后替换脸锚；
- EMA prototype；
- accepted-image centroid；
- 用本轮冠军更新下一轮参考中心；
- 用派生侧脸、背面或重建网格创建新绝对锚；
- 用候选批次均值和方差定义“小娜正常范围”。

### 2.2 测量层与政策层隔离

底层测量层只产生：

```text
observation eligibility
provider comparability
native residuals
repeatability descriptors
evidence lineage
conditions and evidence gaps
```

上层选择器不得修改底层残差，只能读取并执行预注册政策。

### 2.3 非补偿不变量

任何关键轴的严重失败不得被其他轴的优秀表现抵消。

错误示例：

```text
face = excellent
body = catastrophic
weighted_average = acceptable
```

正确行为：身体关键轴触发硬阻断或成为最大短板，候选不得因脸部优秀而获胜。

### 2.4 同源证据去重不变量

同一个 provider 或同一个上游工件派生出的多个分区、均值、P95、weakest part 和 top drift，只能解释一个 evidence family，不能增加投票权。

当前建议的顶层 evidence families：

```text
face_identity
face_projection_geometry
body_shape_geometry
measurement_reliability
traceability_and_scope
```

其中 `body_core_shape` 与 `body_topology` 共享 HMR2/SMPL 上游，仍属于 `body_shape_geometry` 大家族，但可作为同家族内不同诊断块保留。

### 2.5 缺失不奖励不变量

缺失证据不得通过剩余轴重新归一化为高分。

```text
missing != good
unobservable != consistent
provider mismatch != zero residual
```

### 2.6 机器可弃权不变量

系统必须允许：

```text
NO_ELIGIBLE_CANDIDATE
TOP_CLUSTER
INDETERMINATE
INSUFFICIENT_EVIDENCE
MEASUREMENT_NOT_QUALIFIED
```

不得为了每批必须产出 Top 1 而伪造精确排名。

---

## 3. 总体目标架构

```text
多底模 / 多工作流生成器
          |
          v
Generation Manifest + 图片 SHA-256
          |
          v
Metadata-only Preflight
          |
          v
Visual Runtime + Native Evidence
          |
          v
Candidate Evidence Index
          |
          v
同槽位可比性与硬门禁
          |
          v
Evidence-family 非补偿压缩
          |
          v
Pareto 分层
          |
          v
Repeatability-normalized Minimax
          |
          v
可分辨性检查 / 自动弃权
          |
          +----------------------------+
          |                            |
          v                            v
Batch Selection Shadow          Generation Feedback Shadow
          |                            |
          v                            v
外部图集决策流程             多模型良率地图 / 下一批固定实验计划
```

当前仓库继续负责“筛选与证据”。最终图集成员和训练准入仍由外部流程拥有。

---

## 4. 核心术语与数据对象

### 4.1 Selection Slot

候选只能在相同或被明确声明为可比的任务槽位内竞争。

建议槽位键：

```text
selection_slot_id = hash(
  target_profile,
  intended_view_family,
  intended_view_center,
  crop_class,
  framing_class,
  pose_family,
  body_visibility_requirement,
  outfit_occlusion_class,
  prompt_family,
  resolution_class
)
```

初始槽位示例：

```text
front_face_closeup
front_halfbody_neutral
front_fullbody_neutral
three_quarter_left_halfbody
three_quarter_right_fullbody
side_left_profile
side_right_profile
back180_fullbody_neutral
back180_fullbody_subtle_gait
```

不得将正脸、严格侧脸、全身和半身混在一个排行榜。

### 4.2 Candidate Evidence Index

新增一个轻量索引，将一张图散落在多个 JSON 中的证据链接起来，但不复制全部高维向量。

建议输出：

```text
outputs/candidate_evidence_index.json
```

候选记录至少包含：

```json
{
  "candidate_id": "sha256-based-id",
  "image": "candidate.png",
  "image_sha256": "...",
  "selection_slot_id": "...",
  "generation_contract_sha256": "...",
  "measurement_contract_ids": {},
  "evidence_refs": {
    "qa_item": "qa_report.json#/items/...",
    "identity_shadow": "identity_evidence_shadow.json#/items/...",
    "body_shadow": "body_evidence_shadow.json#/items/...",
    "repeatability": [],
    "lineage": "..."
  },
  "truth_refs": {
    "face": "A_f",
    "body": "A_b"
  },
  "truth_authority": "NONE"
}
```

### 4.3 Selection Evidence Record

选择器使用统一结构读取不同原生空间，但不强制转换成统一总分：

```text
axis
family
native_residual / residual_vector
unit
direction
eligibility
scope
provider_comparison
chain_state
repeatability_state
prior_dependence
coverage
evidence_gaps
lineage_family_id
```

---

## 5. 三种选择模式

### 5.1 Mode A：Calibration-free Shadow Filtering

适用阶段：当前 `MEASUREMENT_QUALIFICATION`。

允许：

- 元数据与合同硬门禁；
- 同槽位比较；
- 同一个轴内按原生残差排序；
- Pareto 支配与 Pareto 层；
- 输出不可区分候选集；
- 输出每张图的最严重原生轴和证据缺口；
- 机器弃权。

禁止：

- 跨轴 minimax 标量比较；
- 通过候选批次拟合尺度；
- 综合权重；
- 生产准入；
- 唯一冠军自动入库。

输出建议：

```text
selection_mode = CALIBRATION_FREE_SHADOW
unique_winner_authorized = false
```

### 5.2 Mode B：Repeatability-normalized Shadow Selection

前置条件：每个参与决胜的轴完成预注册重复性协议，并通过 provider 合同冻结和执行验收。

对候选 `i`、轴 `k`，定义锚点自身测量扰动包络：

```math
B_k(c) = \text{preregistered upper envelope under condition } c
```

定义稳健尺度：

```math
s_k(c) = \text{preregistered robust spread descriptor}
```

建议使用预注册的 median、MAD、IQR 或固定分位差，具体形式必须在执行前冻结，不得根据当前候选调参。

标准化超额偏差：

```math
e_{ik}=\max\left(0,\frac{d_{ik}-B_k(c_i)}{s_k(c_i)+\epsilon}\right)
```

解释：

- `e = 0`：候选偏差尚未超过该测量链预注册扰动包络；
- `e > 0`：超出正常测量扰动；
- 数值越大，超额越明显；
- 这是测量超额，不是身份概率。

当前阶段可先输出该描述，但仍保持 `decision_influence=NONE`。

### 5.3 Mode C：Benchmark-calibrated External Decision

前置条件：

- 独立 benchmark 与标签冻结；
- benchmark 不包含当前候选、Winner Bank 或 Top-K；
- provider 合同冻结；
- 重复性完成；
- hard-negative 和 leakage audit 完成；
- `configs/project_stage.json` 显式开放对应权限；
- 外部图集决策流程实现并通过审计。

此阶段才允许研究阈值、经验覆盖率和更正式的判敛逻辑。本文不授权直接进入 Mode C。

---

## 6. 非补偿式选择数学

### 6.1 硬门禁

每张候选先得到：

```text
hard_gate_state = PASS | BLOCKED | NOT_COMPARABLE
```

建议初始 blocker：

```text
TRUTH_INTEGRITY_FAIL
INPUT_METADATA_INCOMPLETE
SELECTION_SLOT_UNRESOLVED
INTENDED_OBSERVED_LANE_MISMATCH
REQUIRED_FACE_NOT_DETECTED
REQUIRED_BODY_CROP_INCOMPLETE
PROVIDER_CONTRACT_MISMATCH
OBSERVATION_CHAIN_INVALID
REQUIRED_AXIS_UNAVAILABLE
REPEATABILITY_CONTRACT_MISMATCH
GPU_EXECUTION_CONTRACT_MISMATCH
NONFINITE_NATIVE_RESIDUAL
EVIDENCE_LINEAGE_INVALID
```

硬 blocker 不参与扣分，直接失去本槽位竞争资格。

### 6.2 Evidence-family 压缩

压缩目的不是生成总分，而是阻止高维同源字段泛滥。

#### Face identity

保留：

```text
angular_residual_radians
eligibility
provider_state
repeatability_excess_or_native_rank
```

#### Face projection geometry

保留：

```text
global_irls_procrustes_residual
worst_partition_residual
second_worst_partition_residual
visibility_coverage
```

分区只用于同家族诊断，不能作为额外投票。

#### Body core

五个有符号对数比分量不得求均值或 L2。

保留按绝对超额排序后的：

```text
worst_component
worst_component_excess
second_worst_component
second_worst_component_excess
component_coverage
signed_component_vector_ref
```

#### Body topology

20670 坐标不得逐坐标参与 Pareto。

需要新增固定 SMPL 顶点分区合同，例如：

```text
head_neck
shoulder_chest
arms
waist_abdomen
pelvis_hips
left_leg
right_leg
feet
```

每个区域只输出预注册描述：

```text
signed quantiles by x/y/z
absolute P50/P90/P95/P99
worst_axis
worst_region
coverage
```

选择层保留：

```text
worst_region_p95_excess
second_worst_region_p95_excess
worst_region_id
```

不得计算全网格平均、全局 vertex norm 或 topology score。

### 6.3 Pareto 支配

对同槽位候选 `i` 与 `j`，在可比且有效的核心维度集合 `K` 上：

```math
i \prec j
```

表示 `i` 在所有维度不劣于 `j`，并至少一维严格更优。

若 `i` 支配 `j`，则 `j` 不应继续争夺同一 Pareto 层。

必须满足：

- 只比较同槽位；
- 每个参与维度方向已统一为 `lower_is_better`；
- 缺失维度不得被当作零；
- 两候选有效维度集合不兼容时输出 `NOT_COMPARABLE`；
- 同一 evidence family 不得重复展开成多个票权维度。

输出：

```text
pareto_layer
is_nondominated
is_dominated_by
candidate_dominates
comparison_dimension_set
```

### 6.4 Lexicographic Minimax

在重复性尺度具备后，对第一 Pareto 层候选按非补偿排序键比较：

```math
K_i=(
F_i,
P_i,
M^{direct}_{i,1},
M^{direct}_{i,2},
M^{conditional}_{i,1},
M^{prior}_{i,1},
U_i,
C_i
)
```

其中：

- `F_i`：硬 blocker 数；
- `P_i`：Pareto 层；
- `M_direct_1`：直接可测轴中最大超额偏差；
- `M_direct_2`：第二大直接超额偏差；
- `M_conditional_1`：条件可测轴最大超额偏差；
- `M_prior_1`：先验依赖轴最大超额偏差；
- `U_i`：不确定性负担；
- `C_i`：证据覆盖缺口。

从左到右做字典序比较，不相加、不加权。

直接证据优先于条件证据，条件证据优先于先验依赖证据。

### 6.5 Top Cluster 与可分辨性

第一名与第二名只有在优势超过测量器自身可分辨边界时，才允许输出 `PROVISIONAL_WINNER`。

否则输出：

```text
TOP_CLUSTER
winner = null
```

建议每对候选输出：

```text
pairwise_decision = A_BETTER | B_BETTER | INDISTINGUISHABLE | NOT_COMPARABLE
separation_margin
repeatability_margin
separation_supported = true/false
```

不得使用固定小数差值假装统计显著性。

### 6.6 自动弃权

以下任一条件触发弃权：

- 无候选通过硬门禁；
- 第一 Pareto 层过大且重复性不足；
- Top 1 与 Top 2 不可分辨；
- 关键轴未完成重复性资格；
- 跨角度轴全部为先验依赖且 provider 之间意见冲突；
- 当前槽位没有足够直接或条件证据；
- 生成批次 metadata 无法追溯；
- 运行合同混入 CPU/CUDA 或不同 provider 版本。

---

## 7. 跨角度同真相可行域

### 7.1 原则

单张正脸锚不能绝对观察侧脸和后脑；单张正面身体锚不能绝对观察背部和前后厚度。

因此其他角度不得与“虚构的绝对侧脸锚”比较，而应回答：

```text
候选是否可以由固定绝对锚在合理的视角、姿态、相机和模型先验条件下解释？
```

### 7.2 Anchor-conditioned Feasible Envelope

对绝对锚 `A` 和目标条件 `c`，定义模型依赖可行域：

```math
\mathcal{Z}(A;c)=\{z:\operatorname{Obs}(z,c_0)\approx A\}
```

目标视角投影集合：

```math
\mathcal{P}_c(A)=\{\operatorname{Render}(z,c):z\in\mathcal{Z}(A;c)\}
```

这些派生对象必须携带：

```text
derived_from = A_f or A_b
truth_authority = NONE
prior_dependent = true
may_modify_truth = false
```

### 7.3 Shared-identity Counterfactual Fit

未来研究接口：

```math
L_{anchor}(x)=\min_{z\in\mathcal{Z}(A)} L(x,z,c)
```

允许候选自由更换身份结构时：

```math
L_{free}(x)=\min_z L(x,z,c)
```

定义：

```math
\Delta L(x)=L_{anchor}(x)-L_{free}(x)
```

解释：

- `Delta L` 小：保持锚点身份结构也能解释候选；
- `Delta L` 大：只有更换身份结构才容易解释候选；
- 不能因此声称恢复了绝对侧脸真相。

此能力当前必须标记 `DEFERRED_RESEARCH`，不得直接接入排序。

### 7.4 多 provider 分歧

侧面和背面应优先使用多个独立重建或表示链做分歧检测：

```text
provider_agreement = CONSISTENT | DISAGREEMENT | UNAVAILABLE
```

provider 分歧不取平均，直接增加不确定性或触发弃权。

### 7.5 Lane 结论权限

```text
front:
  direct measurement permitted

three_quarter:
  conditional compatibility permitted

side:
  prior-dependent compatibility / contradiction / indeterminate

back face:
  UNOBSERVABLE

back body:
  prior-dependent body compatibility only
```

---

## 8. 生成数据合同

### 8.1 Generation Manifest v1

本地 ComfyUI / FLUX.2 / 其他底模输出必须伴随逐图可复现元数据：

```json
{
  "schema_version": "generation_manifest_v1",
  "batch_id": "...",
  "candidate_id": "...",
  "image_relative_path": "...",
  "image_sha256": "...",
  "generator": {
    "model_family": "FLUX.2",
    "model_id": "...",
    "model_sha256": "...",
    "vae_id": "...",
    "text_encoder_ids": [],
    "lora_ids": [],
    "workflow_sha256": "...",
    "sampler": "...",
    "scheduler": "...",
    "steps": 0,
    "guidance": 0.0,
    "seed": 0,
    "width": 0,
    "height": 0
  },
  "identity_control": {
    "truth_anchor_refs": {
      "face": "A_f",
      "body": "A_b"
    },
    "reference_node_ids": [],
    "reference_strengths": {},
    "control_node_ids": [],
    "control_strengths": {},
    "pose_control_asset_sha256": "..."
  },
  "intent": {
    "prompt_id": "...",
    "prompt_family": "...",
    "intended_view": "...",
    "intended_view_center_deg": 0.0,
    "pose_family": "...",
    "crop_class": "...",
    "framing_class": "...",
    "outfit_occlusion_class": "..."
  },
  "execution": {
    "gpu_id": "...",
    "started_at_utc": "...",
    "finished_at_utc": "...",
    "generation_seconds": 0.0
  }
}
```

`anchor_source` 建议在新 schema 中替换或兼容映射为 `truth_anchor_refs`，避免误解为候选可成为锚。

### 8.2 生成合同哈希

生成合同标准化后计算：

```text
generation_contract_sha256
```

同一个批次实验必须冻结除目标变量之外的其他字段。

### 8.3 批次实验规则

每个实验批次最多主动改变一到两个变量，例如：

```text
reference_strength ladder
control_strength ladder
guidance ladder
model family comparison
pose controller comparison
```

禁止同时改动多个无法归因的变量后声称找到最优配方。

---

## 9. 多模型良率地图

### 9.1 目的

粮仓中的不同底模不要求争夺一个综合冠军，而应分别回答：

```text
哪个模型 / 工作流 / 参数区间，在哪个 selection slot 中更容易产生高纯度候选？
```

### 9.2 Yield Record

建议输出：

```text
outputs/model_yield_map_shadow.json
```

每个 `(model, workflow, slot, parameter_cell)` 记录：

```text
generated_count
metadata_ready_count
preflight_pass_count
visual_runtime_success_count
hard_gate_pass_count
pareto_layer1_count
top_cluster_count
provisional_winner_count
abstention_count
failure_reason_counts
worst_axis_counts
provider_unavailable_counts
mean_generation_seconds
mean_review_seconds
estimated_energy_kwh
artifact_bytes
```

### 9.3 高纯度良率

当前阶段不得把候选结果拟合为最终质量概率。

可报告纯描述比例：

```text
hard_gate_pass_share
pareto_front_share
top_cluster_share
provisional_winner_share
```

必须附：

```text
selection_mode
measurement_qualification_state
slot_id
sample_count
```

### 9.4 算力分配

当前阶段只允许预注册、确定性的阶梯实验和固定配额。

未来在 benchmark 和权限开放后，才研究：

- successive halving；
- conservative multi-armed bandit；
- Bayesian optimization；
- Optuna；
- 置信下界驱动的模型分配。

不得在当前候选数据上直接开启自适应参数拟合。

---

## 10. 双 GPU 运行拓扑

建议默认职责：

```text
GPU0:
  generation_primary

GPU1:
  evidence_runtime_primary
  generation_secondary_when_evidence_queue_empty
```

调度器必须：

- 尊重现有 GPU device-scoped lease；
- 不在同一 GPU 上同时启动冲突的重型 provider；
- 记录 WHEA 前后差值；
- 记录每个任务的模型、显存峰值、耗时和退出状态；
- 允许生成队列背压；
- 允许检测失败重试，但不得无限重试；
- 为生成、视觉预检、HMR2、重复性使用独立 workload class；
- 不允许 CPU 结果与 CUDA 结果混合为同一个测量合同。

建议未来外部队列状态：

```text
QUEUED
LEASE_WAIT
RUNNING
SUCCEEDED
FAILED_RETRYABLE
FAILED_TERMINAL
WITHHELD
```

---

## 11. 新增输出建议

### 11.1 `candidate_evidence_index.json`

用途：统一候选、原始证据、生成合同和真相引用。

### 11.2 `batch_selection_shadow.json`

建议结构：

```json
{
  "schema_version": "batch_selection_shadow_v0_1",
  "selection_mode": "CALIBRATION_FREE_SHADOW",
  "slot_id": "...",
  "truth_policy": {
    "face": "A_f",
    "body": "A_b",
    "truth_anchor_count": 2
  },
  "candidate_count": 0,
  "eligible_count": 0,
  "blocked_count": 0,
  "pareto_layers": [],
  "top_cluster": [],
  "provisional_winner": null,
  "abstention_reason": "MEASUREMENT_NOT_QUALIFIED",
  "decision_influence": "NONE"
}
```

### 11.3 `batch_pareto_explanation.json`

逐候选记录：

```text
dominated_by
candidate_dominates
comparison_axes
not_comparable_reasons
worst_family
second_worst_family
```

### 11.4 `generation_feedback_shadow.json`

面向外部生成控制器，输出：

```text
failure_reason_distribution
worst_axis_distribution
slot_shortage
parameter_cell_yield
next_fixed_experiment_recommendations
```

该文件不得自动修改真相或生成参数；当前阶段只生成建议。

### 11.5 `model_yield_map_shadow.json`

聚合多批次、多模型、多工作流的描述良率。

---

## 12. 实施阶段与任务拆解

# Phase 0：治理冻结与术语清理

目标：先防止未来实现无意中破坏绝对锚与项目边界。

### P0-T01：新增规划文档

- [ ] Codex 审阅本文；
- [ ] 明确哪些内容进入规范，哪些保留为研究路线；
- [ ] 不修改 `docs/39` 直到评审完成。

验收：

- 文档明确 `normative=false`；
- 不改变现有运行结果。

### P0-T02：冻结 truth reference schema

建议新增：

```text
configs/truth_reference_policy.json
```

内容锁死：

```text
truth_anchor_count = 2
allowed_face_truth_ref = A_f
allowed_body_truth_ref = A_b
candidate_truth_authority = NONE
winner_bank_truth_authority = NONE
derived_projection_truth_authority = NONE
```

测试：

- 候选文件路径填入 truth ref 时 fail；
- Winner Bank 条目填入 truth ref 时 fail；
- 未知 ref fail closed。

### P0-T03：重命名或兼容 `anchor_source`

- [ ] 新 schema 使用 `truth_anchor_refs`；
- [ ] 旧 `anchor_source` 只作兼容输入；
- [ ] 输出时显式标记 `legacy_field_alias`；
- [ ] 禁止候选路径充当 truth ref。

### P0-T04：项目阶段权限扩展预留

仅新增默认 false 的权限字段，不开放：

```text
noncompensatory_selector_shadow
repeatability_normalized_selector
external_dataset_decision
adaptive_generation_optimization
```

验收：旧配置缺失字段时全部 fail closed。

---

# Phase 1：Candidate Evidence Index

目标：解决“一张图产生一堆 JSON，无法统一追踪”的工程问题。

### P1-T01：定义 schema

新增：

```text
core/qa_candidate_evidence_index.py
configs/candidate_evidence_index.schema.json
```

### P1-T02：稳定候选 ID

建议：

```text
candidate_id = sha256(image_sha256 + canonical_generation_contract_sha256)
```

图片相同但生成合同不同必须能区分 provenance，同时保留 `image_sha256` 去重能力。

### P1-T03：证据引用解析

支持链接：

- `qa_report.json`；
- `identity_evidence_shadow.json`；
- `body_evidence_shadow.json`；
- identity/body repeatability manifests；
- generation manifest；
- evidence lineage DAG。

### P1-T04：孤儿和冲突检查

输出：

```text
ORPHAN_QA_ITEM
ORPHAN_IDENTITY_SHADOW
ORPHAN_BODY_SHADOW
DUPLICATE_IMAGE_HASH
CONFLICTING_GENERATION_CONTRACT
CANDIDATE_ID_COLLISION
```

### P1-T05：CLI workflow

新增：

```text
--workflow refresh_candidate_evidence_index
```

验收：

- 幂等；
- 原子写入；
- 不加载视觉模型；
- 不改变现有 QA 输出。

---

# Phase 2：Selection Slot 与可比性

目标：禁止不同任务、角度和裁切条件混排。

### P2-T01：定义 slot schema

新增：

```text
configs/selection_slots.yaml
core/qa_selection_slots.py
```

### P2-T02：slot resolver

输入 generation manifest 和 observed lane，输出：

```text
INTENT_RESOLVED
INTENT_OBSERVATION_MISMATCH
SLOT_UNRESOLVED
MIXED_SLOT_BATCH
```

### P2-T03：批次自动拆槽

新增 Shadow 输出：

```text
outputs/selection_slot_split_plan.json
```

只给计划，不自动移动原始图片，除非用户显式调用 materialize workflow。

### P2-T04：跨模型同槽位比较

允许不同生成模型竞争，但要求：

- 相同 slot；
- 相同测量 provider contract；
- 相同 backend；
- 相同原生测量 schema。

### P2-T05：测试

覆盖：

- front 与 side 不可比较；
- halfbody 与 fullbody 不可比较；
- 同 slot 的 FLUX.2 与 HunyuanImage 可比较；
- CPU/CUDA 测量合同不可混合；
- intended view 与 observed lane 冲突时 fail closed。

---

# Phase 3：Evidence-family Reducers

目标：将高维证据压缩成非补偿诊断块，而不是总分。

### P3-T01：Face identity reducer

新增：

```text
core/qa_selection_face_identity.py
```

输出原生角残差、资格、合同状态和重复性状态。

### P3-T02：Face geometry reducer

输出：

```text
global_residual
worst_partition
second_worst_partition
visibility_coverage
```

### P3-T03：Body core reducer

输出最严重和第二严重 log-ratio 分量，保留符号与原向量引用。

### P3-T04：SMPL 顶点分区合同

新增：

```text
configs/smpl_vertex_regions_v1.json
```

要求：

- 固定 body model SHA；
- 固定 6890 顶点索引；
- 每个顶点只能属于明确分区或明确 overlap policy；
- 分区配置本身有 SHA-256。

### P3-T05：Body topology reducer

按区域和 xyz 分别输出固定分位数，不计算全局 norm。

### P3-T06：Lineage 去重验证

确保 reducers 不把共享上游变成额外票权。

### P3-T07：测试

- 20670 坐标不直接进入 Pareto；
- 单一局部灾难不会被全局平均稀释；
- 五个 body core 分量不求均值；
- 分区字段不能被计为独立 evidence families。

---

# Phase 4：硬门禁与 Calibration-free Pareto

目标：当前阶段先实现不需要跨轴标定的保守筛选。

### P4-T01：Hard gate engine

新增：

```text
core/qa_selection_gates.py
configs/selection_gate_policy.yaml
```

### P4-T02：Pareto comparator

新增：

```text
core/qa_pareto_selection.py
```

要求：

- 只接受方向明确的维度；
- 缺失维度不补零；
- 不可比返回原因；
- 支持 Pareto layer peeling；
- 稳定排序只用于输出可复现，不代表语义优胜。

### P4-T03：Calibration-free 输出

新增：

```text
outputs/batch_pareto_explanation.json
outputs/batch_selection_shadow.json
```

当前阶段：

```text
unique_winner_authorized = false
```

### P4-T04：CLI workflow

```text
--workflow refresh_batch_selection_shadow
```

### P4-T05：测试

- 全面更差候选被支配；
- 各有优劣候选同处第一前沿；
- 缺失轴不自动占优；
- 不同 slot 不比较；
- blocker 候选不进 Pareto；
- 输出幂等。

---

# Phase 5：重复性执行与轴尺度资格

目标：给每根尺子建立自身扰动尺度，而不是从候选批次学习“小娜平均值”。

### P5-T01：执行 face repeatability 基准

至少覆盖：

- `A_f`；
- 预注册正探针；
- 独立 hard negatives；
- front 和 three-quarter 条件。

### P5-T02：执行 body repeatability v0.2

完成：

- `A_b` baseline + 13 trials；
- body core 五分量；
- native topology 20670 坐标；
- per-region topology descriptors；
- GPU lease 与 WHEA 前后记录。

### P5-T03：Repeatability envelope schema

新增：

```text
outputs/repeatability_envelopes_shadow.json
```

只记录预注册描述，不拟合生产阈值。

### P5-T04：轴 readiness contract

每轴输出：

```text
NOT_EXECUTED
INCOMPLETE
CONTRACT_MISMATCH
READY_FOR_SHADOW_NORMALIZATION
BLOCKED
```

### P5-T05：标准化超额实现

新增：

```text
core/qa_repeatability_excess.py
```

要求：

- envelope 来源必须是锚点和独立探针；
- 禁止从当前候选批次估计中心和尺度；
- 条件不匹配时 withholding；
- 不输出概率。

### P5-T06：测试

- 整批候选同时漂移时不能互相证明正常；
- 当前批次均值不能进入 envelope；
- 不同 backend envelope 不可混用；
- 协议 SHA 变化阻断复用。

---

# Phase 6：Lexicographic Minimax 与 Top Cluster

目标：在尺度资格具备后，自动选择“最严重短板最小”的候选，同时允许弃权。

### P6-T01：Minimax key builder

新增：

```text
core/qa_noncompensatory_selector.py
```

### P6-T02：直接 / 条件 / 先验证据分层

要求：

- direct 轴先比较；
- conditional 次之；
- prior-dependent 再次；
- uncertainty 最后；
- 不得跨层加权求和。

### P6-T03：第二、第三短板

最大短板相同或不可分时，比较第二严重、第三严重轴，仍使用字典序。

### P6-T04：Top Cluster

新增：

```text
core/qa_selection_separation.py
```

输出候选间可分辨性和弃权原因。

### P6-T05：Provisional winner

只有满足全部条件时输出：

```text
PROVISIONAL_WINNER
```

但仍：

```text
decision_influence = NONE
may_enter_training_bank = false
```

### P6-T06：测试

- 脸好不能补偿身体灾难；
- 最坏轴更小的候选胜出；
- 差距小于 repeatability margin 时输出 Top Cluster；
- 证据不足时弃权；
- 稳定 ID 不得被解释为质量 tie-break。

---

# Phase 7：跨角度可行域研究

目标：在不创建新锚的情况下提高 three-quarter / side / back 的兼容性判断。

### P7-T01：派生投影合同

新增：

```text
configs/derived_projection_contract.yaml
```

所有派生数据强制 `truth_authority=NONE`。

### P7-T02：Anchor perturbation cloud

对同一锚做预注册轻扰动和重建，描述模型测量云，不使用不同候选求平均。

### P7-T03：多 provider reconstruction comparison

至少定义两个可独立比较的 provider 接口，记录分歧。

### P7-T04：Anchor-conditioned fit prototype

先离线输出：

```text
anchor_conditioned_fit
free_fit
fit_gap
provider_agreement
```

不进入选择器。

### P7-T05：Lane-specific evidence contract

为 front / 3q / side / back 固定允许参与的轴和结论词汇。

### P7-T06：反证优先

优先实现 `CONTRADICTED` 检测，不急于声称 `ABSOLUTELY_CORRECT`。

### P7-T07：测试

- back face 必须 withholding；
- side 派生图不能成为新锚；
- provider 分歧触发不确定性；
- anchor-conditioned fit 不得修改锚点参数。

---

# Phase 8：Generation Manifest 与 ComfyUI Bridge

目标：让本地大规模生成可复现、可归因、可自动入检测队列。

### P8-T01：Generation manifest schema

新增：

```text
configs/generation_manifest.schema.json
core/qa_generation_manifest.py
```

### P8-T02：ComfyUI metadata adapter

读取：

- workflow JSON；
- seed；
- model path/hash；
- LoRA；
- sampler/scheduler；
- reference/control 节点；
- pose asset；
- prompt slot；
- 输出图片 SHA。

### P8-T03：批次目录约定

建议：

```text
input_generated/<batch_id>/images/
input_generated/<batch_id>/generation_manifest.json
input_generated/<batch_id>/input_manifest.json
```

### P8-T04：导入 workflow

新增：

```text
--workflow ingest_generation_batch
```

### P8-T05：可复现性检查

输出：

```text
GENERATION_CONTRACT_COMPLETE
SEED_MISSING
WORKFLOW_HASH_MISSING
MODEL_HASH_MISSING
CONTROL_ASSET_HASH_MISSING
```

### P8-T06：测试

- 同图同合同幂等；
- 同图不同合同 provenance 可区分；
- 缺 seed 的闭源批次可用 reason 字段，但本地批次 seed 必填；
- 未知模型资产 fail closed。

---

# Phase 9：多模型良率地图

目标：把“粮仓很多模型”转化为按任务槽位可测的生产能力地图。

### P9-T01：Yield aggregator

新增：

```text
core/qa_model_yield_map.py
```

### P9-T02：失败原因标准化

统一：

```text
face_identity_drift
face_geometry_drift
body_core_drift
body_topology_drift
lane_mismatch
crop_failure
lighting_risk
occlusion_risk
provider_failure
metadata_failure
measurement_unqualified
```

### P9-T03：参数网格单元

```text
parameter_cell_id = hash(model + workflow + slot + controlled variables)
```

### P9-T04：算力与能耗记录

先记录：

```text
generation_seconds
review_seconds
gpu_id
power_limit_snapshot
```

电量可后续按功耗遥测积分，不允许用未经验证的常数冒充精确能耗。

### P9-T05：输出

```text
outputs/model_yield_map_shadow.json
outputs/slot_model_leaderboard_shadow.json
```

leaderboard 只能按描述良率排序，并标记样本量和测量阶段。

### P9-T06：测试

- 小样本高比例不得隐藏样本量；
- 不同 slot 不混合；
- 不同测量合同不混合；
- 历史算法版本变化必须分段。

---

# Phase 10：无人值守生成反馈闭环

目标：人在日常批次中不做逐图筛选，机器自动生成、筛选、弃权和安排固定实验。

### P10-T01：外部队列接口

建议单独模块或外部服务，不直接塞入底层数学代码：

```text
generation_orchestrator/
```

### P10-T02：固定实验规划器

当前阶段仅支持：

- 固定 seed 列表；
- 固定参数阶梯；
- 固定每格样本量；
- 一次一到两个变量；
- 预注册停止条件。

### P10-T03：Evidence queue backpressure

生成速度不得无限超过检测速度，避免积压不可追溯资产。

### P10-T04：自动补批

只有在配置中预注册时，系统才可根据：

```text
slot_shortage
hard_gate_pass_shortage
no_eligible_candidate
```

创建下一批固定计划。

不得根据本批候选动态拟合最优参数。

### P10-T05：自动弃权策略

允许：

```text
整批丢弃并按原计划重喷
保留 Top Cluster
停止该参数单元
转移固定配额到另一个预注册模型
```

### P10-T06：双 GPU 任务调度

接入现有 GPU execution guard 和 WHEA 记录。

### P10-T07：测试

- GPU lease 冲突时等待而非并发冲撞；
- 检测队列拥堵时生成降速；
- 失败任务有限重试；
- 输出和 manifest 原子写入；
- 重启可恢复；
- 不因自动闭环修改绝对锚。

---

# Phase 11：独立 Benchmark 与晋级治理

目标：为未来决策影响建立科学资格，而不是让当前候选自证。

### P11-T01：Benchmark 隔离

目录、哈希和访问策略与当前候选、Winner Bank 隔离。

### P11-T02：标签协议

标签重点不是“更漂亮”，而是：

```text
identity contradiction
body contradiction
projection-compatible
indeterminate
measurement failure
```

### P11-T03：Hard negative bank

覆盖：

- 相似脸但非小娜；
- 相似身材但非小娜；
- 同脸不同年龄感；
- 同脸错误颅面结构；
- 同身体比例但局部拓扑不同；
- 不同角度的模型先验陷阱。

### P11-T04：经验覆盖率

任何未来区间或阈值必须检查实际覆盖率，不得只输出理论 95%。

### P11-T05：逐轴晋级

推荐顺序：

1. front face identity；
2. front face projection geometry；
3. front body core；
4. three-quarter face conditional；
5. body topology；
6. side/back prior-dependent compatibility。

禁止整套系统一次性获得决策权。

### P11-T06：外部决策层

最终图集成员选择应在独立模块中完成，并消费冻结版本的 evidence packet。

---

# Phase 12：CI、测试与可观测性

### P12-T01：CPU-only 单元测试 CI

GitHub Actions 至少执行：

- truth policy；
- schema validation；
- candidate ID；
- slot resolver；
- hard gates；
- Pareto；
- minimax；
- Top Cluster；
- lineage dedupe；
- JSON compatibility。

### P12-T02：GPU integration 手动工作流

不在公共 runner 强制下载重模型。使用自托管 runner 或本地显式 workflow。

### P12-T03：Golden fixtures

小型 synthetic fixtures 验证数学不变量：

- 全面支配；
- 互不支配；
- 局部灾难；
- 缺失不可奖励；
- provider mismatch；
- Top Cluster；
- back face withholding。

### P12-T04：Schema migration

所有新 schema 必须：

- 有版本；
- 有 migration 或明确拒绝；
- 未知版本 fail closed；
- 不覆盖历史 artifact。

### P12-T05：运行可观测性

记录：

```text
run_id
batch_id
slot_id
model_id
workflow_hash
measurement_contract_hash
selector_version
policy_hash
artifact_paths
```

---

## 13. 推荐代码与文件映射

```text
core/qa_candidate_evidence_index.py
core/qa_selection_slots.py
core/qa_selection_gates.py
core/qa_selection_face_identity.py
core/qa_selection_face_geometry.py
core/qa_selection_body_core.py
core/qa_selection_body_topology.py
core/qa_pareto_selection.py
core/qa_repeatability_excess.py
core/qa_noncompensatory_selector.py
core/qa_selection_separation.py
core/qa_generation_manifest.py
core/qa_model_yield_map.py

configs/truth_reference_policy.json
configs/selection_slots.yaml
configs/selection_gate_policy.yaml
configs/smpl_vertex_regions_v1.json
configs/generation_manifest.schema.json
configs/candidate_evidence_index.schema.json
configs/derived_projection_contract.yaml

outputs/candidate_evidence_index.json
outputs/batch_pareto_explanation.json
outputs/batch_selection_shadow.json
outputs/generation_feedback_shadow.json
outputs/model_yield_map_shadow.json
```

避免把这些逻辑继续堆入 `check_consistency.py`。CLI 只做路由和摘要打印。

---

## 14. 首个可实施最小切片

建议 Codex 首先只实现以下内容，不碰重型模型：

### Slice 1

1. `truth_reference_policy`；
2. `candidate_evidence_index`；
3. `selection_slot`；
4. hard gates；
5. calibration-free Pareto；
6. `batch_selection_shadow.json`；
7. 单元测试；
8. CLI workflow。

明确不实现：

- repeatability normalization；
- minimax 唯一冠军；
- cross-angle fit；
- generation scheduler；
- model optimization；
- admission。

### Slice 1 验收标准

- 不改变现有 QA 排名和 review packet；
- 新流程可读取已有 outputs；
- 同槽位全面更差候选能被解释性淘汰；
- 互有优劣候选保留在第一前沿；
- 缺失和 provider mismatch 不会获得优势；
- 输出 `unique_winner_authorized=false`；
- 所有新记录 `decision_influence=NONE`；
- 测试可在无 GPU 环境执行；
- 现有测试全部通过。

---

## 15. 风险登记

### R1：Pareto 前沿爆炸

原因：维度过多或同源字段展开。

缓解：先做 evidence-family reducer，只保留有限、预注册诊断维度。

### R2：批次相对归一化掩盖整批漂移

缓解：尺度只能来自锚点自身重复性和独立探针，不使用当前候选均值。

### R3：Minimax 被噪声轴劫持

缓解：未完成重复性资格的轴不得进入跨轴 minimax，只能作为缺口或原生诊断。

### R4：侧脸模型先验被当成真相

缓解：所有 side/back 结论保留 `prior_dependent`，派生数据永远零权威，多 provider 分歧触发弃权。

### R5：生成越多，平均脸污染越严重

缓解：任何候选、Winner Bank、Top Cluster、yield map 都不得更新锚点或原型。

### R6：自动闭环变成参数过拟合

缓解：当前阶段只允许固定阶梯和预注册配额，不允许基于当前候选动态拟合。

### R7：大量 JSON 失控

缓解：使用 evidence index 和引用，不复制高维向量；设置 artifact retention policy。

### R8：双 GPU 稳定性

缓解：复用 device-scoped lease、WHEA watermark、运行前后差分和原子恢复。

### R9：算法稳定地犯错

缓解：独立 hard negatives、多 provider 分歧、自动弃权、未来 benchmark 审计。

---

## 16. Codex 重点审查问题

请重点检查：

1. Pareto 输入维度是否仍存在同源重复计票；
2. Calibration-free 模式是否无意产生唯一冠军；
3. repeatability envelope 是否可能被当前候选污染；
4. minimax 排序是否严格非补偿；
5. direct / conditional / prior-dependent 的字典序是否合理；
6. Top Cluster 的可分辨性接口是否需要更严格定义；
7. SMPL 分区能否在不生成全局 topology score 的情况下稳定实现；
8. candidate ID 与 generation contract 哈希设计是否会造成错误去重；
9. `anchor_source` 到 `truth_anchor_refs` 的兼容迁移是否安全；
10. 新模块是否应该全部留在当前仓库，还是生成调度器拆成独立项目；
11. 当前 project-stage 权限是否需要新增但默认关闭的字段；
12. 哪些任务可以先用纯 synthetic fixtures 完成；
13. 哪些跨角度研究依赖尚未准备好的外部模型；
14. 如何避免 `check_consistency.py` 继续膨胀；
15. 是否存在更简单但同样满足非补偿和不移动真相的方案。

---

## 17. 最终目标状态

系统最终应能在没有逐图人工挑选的情况下输出：

```text
本批次：1000 张
同槽位可比较：842 张
硬门禁通过：173 张
Pareto 第一层：11 张
重复性尺度合格：8 张
Top Cluster：2 张
唯一可分辨冠军：无
机器结论：保留 Top Cluster 或整批重喷
```

或：

```text
唯一暂定最佳：candidate_00482.png
原因：
- 无硬阻断
- 位于第一 Pareto 层
- 最大直接测量超额为本批次最小
- 第二严重短板同样优于其他候选
- 与第二名的差距超过预注册重复性边界
- 没有关键 provider 分歧
- 仍为 Shadow 证据，不自动进入训练集
```

同时，系统能够跨批次回答：

```text
FLUX.2 在 front_fullbody_neutral 槽位高纯度良率更高；
HunyuanImage 在 three_quarter_left 槽位更稳定；
某参考节点强度区间会显著增加 body_core_drift；
某姿态控制器降低 lane mismatch，但提高 face_geometry_drift；
```

这些结论用于优化生成工厂，而不是修改“小娜是谁”。

最终原则：

```text
锚点永远不动；
候选可以无限生成；
测量器可以不断变准；
生成配方可以不断优化；
机器可以拒绝裁决；
任何候选都不能通过投票重写绝对真相。
```
