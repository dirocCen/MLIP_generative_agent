# 高导热材料的闭环搜索

MLIP × 结构枚举 × agent：一个会从自己算过的结果里学习、每一步都可审查的高 κ 材料搜索框架。

> 这是 `loop` 分支。`main` 分支记录的是之前的单向漏斗（MatterGen / AIRSS / CrystalFormer → 弛豫 → E_hull → L0 → L1），完整过程见 [docs/search_log.md](docs/search_log.md)。

## 为什么改成闭环

`main` 上的漏斗跑通了，但每一批都和上一批无关：

- **没有回流**：下游的 L0 / L1 结果不影响下一批生成什么、先算什么。三种原理完全不同的生成器反复收敛到同一批结构（AlSiCN sg 160 被找到三次），L1 κ 在 195–240 W/mK 封顶。
- **门槛和目标冲突**：高 κ 的致密 sp³ 相相对层状相本来就是亚稳的。按 MP 凸包，金刚石在凸包以上 0.138 eV/atom，c-BN 0.077，BAs 0.080，B–C–N 三元最低 0.173。「E_hull ≤ 0.1 且 MP 里没有」留下的基本只有四面体二元化合物的有序合金，而它们的 κ 低于端元。
- **奖励信号是平的**：命中定义为超过 c-BN，命中数为 0，候选之间的差距和模型误差相当，没有可以学习的信号。

闭环把问题换成：**在固定的候选池上，一个从已算结果里学习的选样策略，能比固定规则和随机更快地找到高 κ 结构吗？** 这个问题无论有没有找到新材料都有可报告的结果。

## 循环

```
候选池 ──► 代理模型挑一批 ──► 弛豫 → L0（FC2 描述符）→ L1（3ph κ）
   ▲                                          │
   │                                          ▼
   └── 新的稳定结构做元素替换 ◄── 重训代理模型（L0 描述符 → L1 κ）
```

1. **候选池**：确定性生成的小原胞结构（见下）。
2. **每一轮**：代理模型按预测的 κ 和不确定度挑一批结构，跑 L0、L1。
3. **结果回流两处**：
   - 重训代理模型，替代和 L1 排序相关性很弱的 Slack 估计（Kendall τ = 0.36）；
   - 新的 L0 稳定结构作为原型，做同族元素替换，加进下一轮的候选。
4. **目标是连续量**：κ 和 E_hull 一起报告，E_hull 不再是硬门槛。

**先全算，再回放**：先把整个池子的 L0 和一大批 L1 算完，然后在 `replay` 模式下让各策略多次运行（不同随机种子），比较三种策略：
- 代理模型（主动学习）；
- 按 Slack 排序的固定规则；
- 随机选择。

这样曲线有误差棒，也不依赖集群状态。

**要展示的三条曲线**：
- **重发现**：池里藏着 c-BN、BAs、BP、SiC、金刚石和阳性对照 R3m-BNC₂，看每种策略第几轮找到它们；
- **代理模型逐轮变准**：与 L1 排序的相关性随轮数的变化；
- **单位成本**：每找到一个 L0 稳定结构、一个高 κ 结构花多少 GPU·h。

## 当前进展（2026-11-11）

**候选池 v1 已建好**（`scripts/build_pool.py`，确定性生成，没有随机结构），限定在 16 个化学体系、原胞 ≤ 8 原子：

| 来源 | 数量 | 内容 |
|---|---|---|
| `enum` | 447 | 闪锌矿、纤锌矿、岩盐超胞上的有序占位，按固定氧化态电荷平衡 |
| `bcn` | 570 | 同样的超胞上 B、C、N 占所有位点，n(B) = n(N) |
| `subst` | 135 | 项目里 L0 稳定的结构和 MP 结构做同族元素替换 |
| `known` | 57 | MP 结构（E_hull ≤ 0.3），作为锚点 |

- 占位先按母格超胞的空间群约化，再用近邻距离签名和 StructureMatcher 去重，共 1209 个（集合 `loop.pool.v1`）。
- B–C–N 三元原来占 605 个，按化学式轮流、每个化学式内对称性从高到低保留到 200 个（`scripts/trim_pool.py`），否则回放曲线会被一个体系主导。**最终 804 个**（集合 `loop.pool.v1.bcn200`）。
- 池里有 157 个 κ_SRME 化学式的结构，**只作为隐藏的重发现目标**：代理模型学的是循环自己算出的 L1 值，这些结构不进任何 MLIP 训练或调参数据。

**实测单价**（DPA-4.1-Max-OAM，结果库里已有的作业）：

| 计算 | 平均 GPU·h | 说明 |
|---|---|---|
| 对称约束精弛豫 | 约 0.005 | CrystalFormer 批次的单价 |
| L0 FC2 描述符 | 约 0.01 | 轻元素超胞要用 80 GB 显卡 |
| L1 3ph Wigner-RTA | 0.69（中位 0.49） | 25 个结构 |
| L2 LBTE | 中位 0.87 | 只给最后的前几名 |

**下一步**：
1. 全池弛豫和 L0（约 10–30 GPU·h）；
2. 按 L0 分层抽样 120–150 个上 L1（约 100 GPU·h）；
3. 代理模型和循环驱动，回放比较三种策略；
4. 前几名上 L2，和 S0 的 c-BN、BAs 阶梯结果放在一起。

## 之前的结论（`main` 分支，详见 search_log）

- **S0 校准**：c-BN、w-AlN、3C-SiC、BAs 上，三个委员会模型相对 phononDB 的偏差都在 ±5.5 % 以内，SRME 0.028–0.040；移植的 κ 协议和 Matbench Discovery 排行榜上同一模型的结果相差不超过 2.4 %。

  ![κ_SRME subset](docs/figures/s0_ksrme.png)

- **L1 → L2**：LBTE 让 c-BN 的 κ 提高 12–13 %，BAs 提高 89–90 %。只算 3ph 时 BAs 被高估是已知现象，需要四声子散射修正（Feng, Lindsay & Ruan, [PRB 96, 161201 (2017)](https://doi.org/10.1103/PhysRevB.96.161201)）。

  ![L1 → L2](docs/figures/s0_ladder.png)

- **生成器对比**：MatterGen、AIRSS、CrystalFormer CSP 在 E_hull ≤ 0.1 上的命中率都是百分之几，而且收敛到同一批结构。L1 前 9 名都在 195–240 W/mK，DPA 和 TECE 一致（Kendall τ = 0.85）。
- **MLIP 会让生成的结构坍缩**：原子靠得太近时 DPA 的排斥力不够，E_hull 可以低到 −210 eV/atom。现在有三道防线：生成时的最短间距、E_hull 下限、弛豫后的最短间距检查。
- **Slack 只能当门槛**：它对微小的对称性破缺很敏感，和 L1 的排序相关性弱（τ = 0.36）。这是闭环里用代理模型替代它的原因。

## 保真度阶梯

| 级 | 内容 | 工具状态 |
|---|---|---|
| L0 | FC2 描述符：虚频、声学–光学 gap、加权相空间、群速度、Grüneisen、Slack 估计 | 已有 |
| L1 | 3ph Wigner-RTA κ（κ_SRME 协议），力常数存入缓存 | 已有 |
| L2 | LBTE 直接解，读 L1 的力常数缓存 | 已有 |
| L3 | MLIP 四阶力常数（随机位移 + symfc）→ FourPhonon，3+4ph κ | 已有，闭环里暂不用 |
| L4 | DFT：**人批准后才提交** | 已有，闭环里暂不用 |

主力 MLIP 是 DPA-4.1-Max-OAM；TECE-OAM-RRA 和 PET-OAM-XL 作为委员会，只在复核时使用。

## 设计原则

- **分工**：工具负责「怎么算」，默认参数都写在工具里，每个都注明出处；agent 负责「算什么、信不信」。
- **记录**：每次调用写一条 JSON 记录（`runs/<run_id>/records/`），包括参数及来源、决定由谁做出（`agent` / `human`）、输入和输出。闭环的记录在 `runs/loop-v1/`。
- **人工关口**：提交 VASP 和开始微调之前，agent 必须停下来；审批只能由人在终端里完成。
- **同一口径**：MLIP 的 κ 移植自 κ_SRME 官方实现（位移 0.03 Å、MS-SMM19、noNAC、phono3py 3.30.0），结果能和 Matbench Discovery 排行榜直接比较。
- **防泄漏**：κ_SRME 的 103 个测试条目对应 70 个化学式，这些化学式的所有多形体及其扰动后代都不能进入训练、回放或调参数据。

## 四个 MCP server

| server | 作用 | 主要工具 |
|---|---|---|
| `mga-structures` | 结构导入、过滤、去重、新颖性和泄漏检查 | `import_structures` `filter_structures` `deduplicate` `check_novelty` `check_benchmark_leakage` |
| `mga-mlip` | MatterGen 生成；MLIP 弛豫、声子和 κ；阶梯 L0 / L2 / L3；收敛检查 | `submit_mlip_relax` `submit_mlip_fc2_descriptors` `submit_mlip_phonon_kappa` `submit_mlip_kappa_lbte` `submit_mlip_fc4_fourphonon` `submit_mlip_convergence` |
| `mga-dft` | VASP 输入预览、成本估计、审批单、提交 | `preview_vasp_inputs` `estimate_vasp_cost` `request_vasp_submission` `submit_vasp` |
| `mga-analysis` | 作业状态、取回结果、E_hull、κ_SRME、模型分歧；预算账本和升级策略基线 | `job_status` `fetch_results` `compute_ehull` `model_disagreement` `budget_ledger` `escalation_baseline` `query_records` |

运行模式用环境变量 `MGA_MODE` 切换：

| 模式 | 提交类工具的行为 |
|---|---|
| `dry_run`（默认） | 构建作业、校验参数、写记录，**不提交** |
| `live` | 通过 jobflow-remote 提交到集群，需要先配好 MongoDB 和 jobflow-remote，见 [plan §7](docs/mcp_servers_plan.md) |
| `replay` | 只返回缓存的结果，用来做可复现的策略比较；VASP 不能回放 |

`live` 模式的本地 runner（jobflow-remote 1.0.1）：

```bash
cp configs/jobflow_remote/mga.yaml.template ~/.jfremote/mga.yaml   # 填 TODO；pre_run 里不能写 # 注释
export PATH=$HOME/.venvs/mga/bin:$PATH   # runner 要找到 supervisord
jf project check --errors
jf admin reset        # 只在全新的空库上做一次：会清空所有作业数据
jf -p mga runner start
```

MongoDB 的数据目录要放在 ext4 上（不要放 /mnt 下的 Windows 盘）。笔记本休眠后 runner 会报 `No route to host`，`jf -p mga runner restart` 即可。

## 安装和测试

```bash
python -m venv ~/.venvs/mga && source ~/.venvs/mga/bin/activate
pip install -e ".[local,test]"   # 本地：MCP server、作业提交、测试
pip install -e ".[worker]"       # 每个集群 worker：phono3py==3.30.0、phonopy、symfc
python -m pytest
```

phono3py 必须用 **3.30.0**：3.31 起删掉了 κ_SRME 用的 MS-SMM19 求解器。

重建候选池（结构文件不进 git，可以从脚本重新生成）：

```bash
python scripts/build_pool.py                       # → data/structures/pool/loop_pool_v1.extxyz
MGA_RUN_ID=loop-v1 python scripts/trim_pool.py     # 导入后：B–C–N 三元保留 200 个
```

## 和 Claude Code 一起用

1. 在项目根目录启动 `claude`，四个 server 由 `.mcp.json` 加载。`.mcp.json` 里写的是本机的绝对路径，换机器时要改。
2. 默认是 `dry_run`；要切到 live，用 `MGA_MODE=live claude` 启动。
3. VASP 的审批流程：agent 调用 `request_vasp_submission` 生成审批单后停下；人在另一个终端里用 `mga approvals list / show / approve <id>` 审批（要重新输入审批单 ID 并写一句理由）；批准后 agent 用 `submit_vasp(approval_id)` 提交，每张审批单只能用一次。

## 目录

```
src/mga/
  tools/        普通 Python 函数（所有逻辑都在这里）
  mcp_servers/  四个 MCP server（薄 adapter）
  workflows/    κ 协议、MLIP / MatterGen 作业、VASP flow
  analysis/     κ_SRME、E_hull、升级策略基线
  specs.py      各计算的参数 schema：默认值、范围、出处
scripts/        候选池构建（build_pool.py、trim_pool.py）、集群安装脚本、README 作图
tests/          pytest
configs/        集群信息、模型列表、jobflow-remote 模板
data/sets/      命名的结构集合（结构本身和结果库不进 git）
docs/           搜索记录、MCP 规划、文献调研
runs/           每次工具调用的 JSON 记录
DESIGN.md       总体设计与时间线
```

## 文档

- [docs/search_log.md](docs/search_log.md)：从 S0 到闭环每一步的详细记录（数字、step id、放弃的原因）
- [DESIGN.md](DESIGN.md)：问题、流程、参数来源、时间线
- [docs/mcp_servers_plan.md](docs/mcp_servers_plan.md)：工具清单、默认参数及出处、测试覆盖
- [docs/lit_review.md](docs/lit_review.md)：高 κ 判据、已有工作

## 来源

- **参考数据**：`data/reference/kappa_srme/` 中的 103 个结构和 noNAC κ 来自 [phononDB](https://github.com/atztogo/phonondb)（A. Togo），经由 [k_SRME](https://github.com/MPA2suite/k_SRME)（GPL-3.0）取得，具体版本见 `SOURCE.txt`。这些数据只用于测试，沿用 k_SRME 的 GPL-3.0。
- **许可证**：代码以 MIT 发布，见 [LICENSE](LICENSE)。
- **κ 协议**：`src/mga/workflows/kappa_protocol.py` 移植自 [matbench-discovery](https://github.com/janosh/matbench-discovery)（MIT，commit `2b7bff9`），其实现又改编自 k_SRME，见 B. Póta, P. Ahlawat, G. Csányi, M. Simoncelli, [arXiv:2408.00755](https://arxiv.org/abs/2408.00755)。
