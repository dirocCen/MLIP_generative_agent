---
title: MLIP × 生成模型 × Agent —— 高导热材料的可审查闭环工作流
created: 2026-10-20
status: Phase A（MCP server、记录、审批，10-20）+ DFT FC3 flow（phono3py_fd，11-03）已完成；11-03 按调研报告重定科学问题（§0、§8）；未开始集群计算
tags: [project, MLIP, MatterGen, agent, thermal-conductivity, workflow]
---

# MLIP × 生成模型 × Agent：高导热材料的可审查闭环工作流

> 目的：一个公开项目（GitHub + 项目页 + 短文）：MLIP 委员会、生成模型和 agent 组成的可审查高 κ 材料筛选工作流。
> **硬约束**：集群账号预计**能用到 11 月底**（11-03 用户确认）→ 重计算（MatterGen、MLIP、VASP、微调）**11-27 前跑完**，之后只做本地 / 回放。
> **新 session 先读 §11**（协作约定）和 §8（已定 / 未定事项）。

---

## 0. 一句话 & 核心问题

**一句话**：一个由 agent 驱动、每一步都可审查的工作流：MatterGen 按化学先验生成多元轻元素晶体 → MLIP 委员会算声子和 κ → **agent 在固定计算预算下决定每个候选升到哪一级保真度**（描述符 → 3ph RTA → LBTE → 4ph → DFT），识别计算伪影 → 回答「3ph 筛出的高 κ 排序，在更高保真度下还剩多少」。

**科学问题（11-03 定，见 §1.6）**
> 生成的多元轻元素结构中，3ph-RTA 筛出来的高 κ 候选，加上迭代解（LBTE）和四声子散射之后还剩多少？能否找到计入 4ph 后仍超过 c-BN（天然丰度约 850 W/mK）、甚至接近 BAs（约 1300）的新结构？
- 目标定在 c-BN / BAs，**不是金刚石**：调研报告已经说明超过金刚石基本无望
- 这个问题无论找没找到新材料都有可报告的结果：排序在每一级保真度之间的变化（Spearman、top-k 留存率），以及 agent 用多少预算得到这些结论

**Agent 问题**：同样的计算预算下，agent 的升级决策和固定规则（按委员会均值取 top-k）、随机选择相比，命中率高多少、浪费多少、能不能抓住伪影（§3.2、§6.3）

**三个组件的分工（缺一个问题就不成立）**
- **生成模型**：提供已知数据库以外的多元结构，排序稳健性要在这些结构上检验
- **MLIP**：让 3ph / LBTE / 4ph 对几十到上百个候选都算得起；委员会分歧是升级决策的输入
- **Agent**：在多保真度阶梯上做预算分配和伪影判断，每一步写记录、可审查

**附带实验（原目标 B，11-03 降级）**：用 DFT 位移超胞的力微调 MACE-MPA-0，报告 κ_SRME 前后变化。把 0.41 的模型调好也追不上直接可用的 0.09 级模型，所以只作为「这份 DFT 数据还能做什么」的附带结果，不作主要结论（§5）。

**工程主张（项目的差异化）**：agent 决定「做什么、什么时候做」，工具决定「怎么做」。参数有来源、每一步有记录、出错可定位，人可以审查和接管。

---

## 1. 关键事实（10-20 查证）

### 1.1 κ_SRME 是什么（Matbench Discovery 的热导率指标）

来源：Póta, Ahlawat, Csányi, Simoncelli, *Thermal conductivity predictions with foundation atomistic models*, [arXiv:2408.00755](https://arxiv.org/abs/2408.00755)

- **参考数据**：phononDB-PBE（Togo）里的 **103 个二元成分–原型条目**，对应 **70 个不同化学式、34 种元素**，结构只有**岩盐、闪锌矿、纤锌矿**三类，**泛函是 PBE**。已逐行核对[官方 PBE 目录](https://github.com/atztogo/phonondb/blob/main/mdr/phono3py_103compounds_fd_PBE/README.md)，[本地排除表](docs/phonondb_pbe_103_manifest.csv)含原始数据链接
  - ⚠️ 闪锌矿 / 纤锌矿正是高 κ 材料的主场（BAs、BP、SiC、AlN、GaN……）。**新增微调、回放及验证数据保守排除这 70 个化学式的所有多形体及扰动后代**；基准 κ 标签不用于选超参数或 checkpoint。预训练阶段是否包含相关母相须另披露
  - **C（金刚石）、Si、BSb 不在这 103 条中**；SiC 收录的是 3C 与 2H，不是 4H/6H
- **输运理论**：Wigner 输运方程（κ = κ_P 粒子部分 + κ_C 相干部分），**在 RTA（弛豫时间近似）下求解**，phono3py 计算，300 K。其中高 κ 简单晶体的 κ_C 很小，基本等于三声子 RTA；不能把这点外推到整个集合的低 κ 材料
- **公式**：
  - SRE（总 κ 的对称相对误差）= 2|κ_ML − κ_DFT| / (κ_ML + κ_DFT)
  - **SRME**（按模态的对称相对平均误差）= 2/(N_q·V) · Σ_{q,s} |𝒦_ML(q,s) − 𝒦_DFT(q,s)| / (κ_ML + κ_DFT)
  - 先比较每个声子模态的 κ 贡献再求和，所以「模态误差互相抵消、总 κ 碰巧对上」的情况会被抓出来（SRME 大、SRE 小 = 误差抵消）
  - 范围 0（完美）到 2（最差）；**排行榜上的数字是 103 个材料的平均值**；出现虚频的材料记 SRME = 2
- **弛豫流程**：先加对称性约束弛豫（FrechetCellFilter + FIRE），再去掉约束弛豫；对称性变了就用第一步的结构；fmax = 1e-4 eV/Å，最多 300 步；spglib 默认精度
  - **10-20 读官方代码（matbench-discovery `2b7bff9`）后的更正**：当前实现**默认单步弛豫**（`FixSymmetry` + 无剪切的 `FrechetCellFilter` + FIRE，两步是可选模式）；**位移 0.01 Å**（11-04 改为 0.03 Å：委员会三个模型在排行榜上跑 κ 时的 yml 都写 0.03）；`is_isotope=True`；输运用 `MS-SMM19`，要求 **phono3py 3.30.0**（3.31 起删掉了这个求解器）；参考数据是 **noNAC**。我们照官方当前实现，逐条对照见 [docs/mcp_servers_plan.md](docs/mcp_servers_plan.md) §2.2
- **对我们的意义**：这套流程直接当作我们 MLIP κ 计算的**默认参数来源**（§4.3）；它的 103 个化合物就是微调前后的**公开测试集**

### 1.2 排行榜上 κ_SRME 靠前的模型（2026-11 快照：matbench-discovery main `c70947f`，11-01；共 62 个模型有 κ_SRME）

| 模型 | κ_SRME | 框架 / 来源 | 训练数据 / 备注 |
|---|---:|---|---|
| Prophet-OAME-MBD | 0.0648 | Kairos Materials（初创），架构未细查 | MPtrj+OMat24+sAlex+**ELEMENTA（非公开）**；代码 MIT、权重 CC-BY-4.0，但 openness 标为 OSCD |
| DPA-4.1-Max-OAM-PFT | 0.0806 | DeepModeling | 训练用了 MDR 声子频率，**可能和测试集相关**，不进委员会，引用时注明 |
| **DPA-4.1-Max-OAM** | 0.0903 | DeepModeling（AISI / 深势），SO(2) 等变卷积 | OAM |
| **TECE-OAM-RRA-1.0** | 0.0927 | TACE 系列（arXiv 2509.14961 / 2607.10664），笛卡尔张量形式的 ACE 类模型；HF `xvzemin/tace-foundations`，CC-BY | OAM |
| EquFlashV2 | 0.0941 | Samsung，等变 GNN（ICML 2025） | OAM |
| EquiformerV3+DeNS | 0.1178 | MIT（Smidt 组），SE(3) 等变图注意力 Transformer | OAM |
| **PET-OAM-XL** | 0.1192 | EPFL（Ceriotti 组），Point Edge Transformer，**结构上不强制旋转等变**（靠数据增强），7.3 亿参数 | OAM |
| GRACE-3L-OAM-L | 0.1211 | ICAMS（Drautz 组），Graph ACE，和 MACE 同属 ACE 一脉 | OAM |
| NequIP-OAM-XL / **L** | 0.1252 / 0.1657 | Harvard（Kozinsky 组），**就是你用过的 NequIP**（新版框架） | OAM |
| TACE-OAM-L | 0.1260 | 同 TECE | OAM |
| **MACE-MPA-0** | 0.4123 | Cambridge（Csányi 组） | MPtrj+sAlex |
| DPA-3.1-MPtrj | 0.6503 | DeepModeling | MPtrj；已被 DPA-4 取代 |
| NEP89 | 不在榜上 | Chalmers / 樊哲勇等 | 多数据集混合 + D3（见 §1.4） |

加粗的是 §8.1 委员会的候选和旧参照。10-20 选的委员会（MACE-MPA-0 + DPA-3.1-MPtrj + NequIP-OAM-L）已经落后一个数量级，11-03 换掉。

**最重要的规律：κ 精度的提升主要来自数据，不是架构。** 同一个架构，从 MPtrj 换成 OAM（加入 OMat24 的大量非平衡构型：rattle、AIMD）后：

| 架构 | MPtrj → OAM 后的 κ_SRME |
|---|---|
| NequIP-L | 0.466 → 0.166 |
| GRACE-2L | 0.525 → 0.169 |
| eSEN-30M | 0.340 → 0.170 |
| EquiformerV3 | 0.275 → 0.118 |
| Allegro-L | 0.504 → 0.319 |

另外 **PFT（Phonon Fine-Tuning，[arXiv:2601.07742](https://arxiv.org/abs/2601.07742)）**：只用声子数据微调，Nequix 从 0.446 降到 0.307。→ **这两条是 §5「补什么数据能提升 κ」的直接证据**，也是本项目必须引用、必须比较的已有工作。

### 1.3 DPA-3 是什么，和 MACE 有什么不同

来源：[arXiv:2506.01686](https://arxiv.org/abs/2506.01686)、[DeePMD-kit 文档](https://docs.deepmodeling.com/projects/deepmd/en/latest/model/dpa3.html)

| | MACE | DPA-3 |
|---|---|---|
| 怎么表示角度信息 | **等变特征**：节点特征带 L > 0 的球谐分量，消息之间用张量积耦合 | **不变特征 + 线图**：只用标量（距离、角度），把角度变成图的显式元素 |
| 图结构 | 原子图（原子 = 节点，近邻对 = 边） | **线图序列 LiGS**：G1 原子图 → G2 以「键」为节点、「键角」为边 → G3 二面角……消息在原子 / 键 / 角上同时传递（代码里叫 RepFlow） |
| 多体项从哪来 | ACE 基组：每层直接构造到 4 体的消息，所以 2 层就够 | 层数多（十几层），多体信息靠线图一层层累积 |
| 计算特点 | 张量积开销大，精度和数据效率高 | 没有张量积，单步运算便宜；思路接近 ALIGNN / DimeNet |
| 预训练方式 | 单一数据集（MP / MPA / OMAT 各一个模型） | **多任务预训练**：OpenLAM 包含很多不同 DFT 设置的数据集，**每个数据集对应一个输出头**；用的时候选和自己 DFT 设置一致的那个头 |
| 生态 | ASE、LAMMPS；欧洲学术圈常用 | DeePMD-kit、DP-GEN、LAMMPS、ABACUS；**国内工业界（深势、AISI）主力** |

- 后续版本 **DPA-4** 改成了 SO(2) 等变卷积，κ_SRME 0.211，比 DPA-3（0.469）好很多。DPA-4 的通讯作者就是张铎（见 progress.md）→ 用 DPA-4 而不是 DPA-3，联系他时更有话可讲（**待定**，取决于 DPA-4 是否易用）

### 1.4 泛函一致性（问题 3 的答案）

- **原则对**：验证和微调用的 DFT，要和基础模型的训练数据一致（泛函、PAW、+U 约定）。否则测出来的「误差」里混着泛函差异，微调时模型还得额外学一个泛函偏移
- 但「不一致就肯定不准」说过头了：不一致时问题出在**误差没法归因**，预测值本身不一定离实验更远（PBE 和 PBEsol 算出的 κ 本来就可能差 10–30%）
- **各模型的参考**：
  - MPtrj、sAlex、OMat24：都是 PBE / PBE+U，MP 兼容设置（OMat24 的 PAW 版本和 MP 略有不同）
  - κ_SRME 参考（phononDB）：PBE
  - DPA-3 / DPA-4：多头模型，选 MPtrj 头就是 PBE
  - **NEP89：多个数据集混合，并且加了 D3 色散** → 和其他模型的参考不一样
- **决定**：**DFT 用 PBE + MP 兼容的 PAW 和输入集**。我们只做轻元素体系（B、C、N、O、Al、Si、Be……），**不涉及 +U**，所以 PBE / PBE+U 的歧义自然消失
  - 注意：MP 的默认输入集是为弛豫设计的，**收敛参数对力常数来说太松**。泛函和 PAW 保持一致，但收紧数值参数（EDIFF ≤ 1e-8、ENCUT ≥ 520 eV、更密的 k 点、PREC = Accurate）。这只降低数值噪声，不改变「参考」
- **由此确定的模型组合**（10-20 定，11-03 换成 OAM 模型，见 §8.1）：把模型分歧当作不确定性的前提是各模型的参考相同。OAM 系列都训练在 MPtrj + OMat24 + sAlex 上（PBE / PBE+U）。NEP89 带 D3，**不进分歧委员会**，单独作为「速度模型」做对照

### 1.5 工具调研

- **matcalc**（[GitHub](https://github.com/materialsvirtuallab/matcalc)，Ong 组）：
  - 把 MLIP 计算封装成 PropCalc：RelaxCalc、PhononCalc、**Phonon3Calc（三阶力常数 + κ）**、EnergeticsCalc（形成能）、ElasticityCalc、QHACalc、ChainedCalc（串联）
  - 支持 MACE、DeePMD、matgl、GRACE、PET、FAIRChem、SevenNet 等；**NEP 没有原生支持**，但它接受任意 ASE calculator，calorine 的 NEP calculator 可以直接传进去
  - **没有 MCP server**，也没有和 jobflow / atomate2 的集成
  - → **MLIP 那一侧几乎可以全部交给它**（弛豫 → 声子 → κ），我们只写薄薄一层
  - **10-20 更正：κ 不用 matcalc**。`Phono3Calc` 会先以 fmax = 0.1 再弛豫一遍，而且不输出按模态的 Wigner κ，算不了 SRME。改为移植 κ_SRME 官方函数（MIT 许可，代码里注明出处），保证我们的 MLIP κ 和排行榜同一口径（plan §2.3）
- **atomate2**（MP 的工作流库，基于 jobflow）：
  - **VASP 输入集来自 MP**（MPRelaxSet / MPStaticSet / MatPES 静态集）→ 正好保证和训练数据泛函一致（§1.4）
  - forcefield 模块支持的 MLIP：MACE（MP-0 / MPA-0 / MP-0b3）、**NEP**、**DeepMD**、NequIP、Allegro、SevenNet、MatPES、FAIRChem、MatterSim、UPET 等
  - 现成流程：relax、phonons（phonopy）、elastic、eos、gruneisen、qha、hiphive、pheasy；**没有现成的 phono3py κ 流程**（MLIP 的 κ 用移植的 κ_SRME 协议，见上面 matcalc 一条的更正）
  - jobflow 的 Flow 可以导出 mermaid 图 → 现成的流程可视化
  - **jobflow-remote**：从本地通过 SSH 把作业提交到远程集群的 Slurm，结果回传到数据库 → 正好符合「小脚本在本地跑、重计算在计算节点跑、不在登录节点算」的要求
- **AiiDA**：完整的溯源图（每个计算和数据都是图上的节点）、错误处理器、可视化最强；但需要 PostgreSQL + RabbitMQ + daemon，学习曲线陡；phono3py 集成弱
  - → **10 天内上 AiiDA 风险太大**。选 atomate2 + jobflow(-remote)，可视化用 jobflow 的 mermaid 图加上我们自己的审查页面（§6）。AiiDA 写进「以后可扩展」
- **MatterGen 的条件模型**：chemical_system、space_group、dft_band_gap、ml_bulk_modulus（ML 预测的体模量）、chemical_system + energy_above_hull 联合。**没有 κ 条件**

### 1.6 四声子、多保真度和 DFT 力常数（11-03 查证）

**四声子的分量**
- **R3m-BNC₂**（Wu, Fan, Hu, PRB 103, L041203 (2021)，arXiv:2104.07833）：3+4ph 下 a/b 轴 2100、c 轴 1554 W/mK；只算 3ph 是 2300，即 4ph 降低约 9%。同文 Pmm2 相 1242/958/846，P4̄m2 相 723/723/620；还算了 BPC₂、BAsC₂
- **Ravichandran & Broido**（PRX 10, 021063 (2020)）：17 种闪锌矿里，只算 3ph 会 "drastically overestimate" 很多材料的 κ；含 B/C/N 的化合物 4ph 散射弱一个数量级
- → 4ph 修正的幅度因体系而异（BNC₂ 约 −9%，BAs 降一半以上），取决于 3ph 有没有被相空间压制。所以「哪些候选值得花 4ph 的钱」本身就是要判断的决策
- **Li, Lee, Wolverton, Xia**（npj Comput. Mater. 12 (2026)，doi:10.1038/s41524-025-01920-y）：773 个化合物，CSLD + SCPH + FourPhonon，含 3ph、4ph 和非对角项，562 个完整收敛。数据在 Zenodo（doi:10.5281/zenodo.15937696，`results.csv` + 约 191 MB 的 IFC 压缩包，CC-BY-4.0）。**泛函和 SCPH 设置还没核实**
- **uMLIP 的四阶力常数 / 4ph κ 没有系统 benchmark**（调研没找到）。最接近的：Anam…Hu（arXiv:2509.03401）只评测 2/3 阶，结论是力的精度不能预示 κ 精度；JAP 140, 081101 (2026) 的教程里 BAs 用定制 SOAP-GAP 做 4ph → §3.3 的 4ph benchmark

**软件**
- **phono3py 不支持 4ph 散射**（`phono3py/phonon4` 只做 fc4 和一阶频移，标为 experimental）。4ph 要用 **FourPhonon**（ShengBTE 扩展）
- FourPhonon 成本：BAs 在 16³ 网格、RTA 下，4ph 约 340 CPU·h，3ph 约 15 CPU·h（JAP 140, 081101）。加速版：最大似然**采样**（Guo et al., npj Comput. Mater. 10, 31 (2024)，提速 3–4 个数量级，已并入 FourPhonon）；**FourPhonon_GPU**（arXiv:2510.00518，Cell Rep. Phys. Sci. 2026；总耗时快 >10 倍，只支持 RTA）
- **symfc 能拟合四阶力常数**（自带 O4 基组和 O2O3O4 联合求解器，11-03 在本地 symfc 里确认）→ MLIP 的 FC4 可以走和 FC3 一样的「随机位移 + symfc」路线，再转成 FourPhonon 的输入格式

**DFT 三阶力常数：随机位移 + symfc（11-03 定，实现见 `workflows/phono3py_fd.py`）**
- 系统位移（每对原子单独位移）的超胞数随对称性降低急剧增长；随机位移让所有原子同时动，一个快照给 3N 个力方程，所需快照数只和独立的 FC3 参数个数有关。用 symfc 数出的下限：

  | 结构 | FC3 超胞原子数 | 系统位移（不截断） | symfc 下限（截断 5 Å / 4 Å） |
  |---|---|---|---|
  | zb-SiC | 64 | 222 | 3 / 2 |
  | w-AlN | 72 | 1254 | 11 / 3 |
  | 生成的 Pm 8 原子结构 | 96 | **27696** | **166 / 51** |

- 实际快照数取下限 × 4（phono3py 的 `auto` 规则）；高对称的校准体系 FC2 两个系统位移就够，逐阶自动选便宜的那种
- 随机位移幅度 0.03 Å（phononDB-LTC 的值）。11-04 起 MLIP κ 也用 0.03 Å（委员会模型的排行榜设置），两侧一致。DFT 本来就不能用更小的位移：有数值噪声，太小的位移力信号淹在噪声里
- 收敛诊断写进每个结果：两个 symfc 截断（4 / 5 Å）的 κ 差、快照数学习曲线（一半快照 vs 全部）
- 收敛判据没有官方数值。Phonon Olympics（JAP 138, 135108 (2025)）：不同专家、不同代码算同一批材料，κ 散布最多约 15%，主要来自 FC3 设置 → MLIP 和 DFT 差在这个量级以内时，不过度解读；项目自定「κ 变化 < 5%」作内部收敛判据

---

## 2. 计算资源与放置原则

| 机器 | 用途 | 规则 |
|---|---|---|
| **本地（WSL）** | 小脚本：pymatgen 去重、MP API 查询、分析、画图、agent / MCP server、审查页面、MongoDB + jobflow-remote runner | 不跑重计算 |
| **datalab（GPU 集群）** | MatterGen 生成、MLIP 批量弛豫和声子 / κ、MACE 微调 | **只在计算节点跑**（Slurm），不在登录节点（slurm-head-3）算 |
| **dirona（CPU 集群）** | VASP | **只在计算节点跑** |

### 2.1 datalab（用户名 `yujie.cen`，登录节点 slurm-head-3；10-20 查看）

| 分区 | 每节点 GPU | 每节点 CPU / 内存 | 用途建议 |
|---|---|---|---|
| **GPU-l40s**（默认分区） | 8× L40S | 128 核 / 1.5 TB | MatterGen 生成、MLIP 推理（弛豫、声子、κ） |
| GPU-a100 | 4–8× A100 | 128+ 核 / 1–2 TB | **MACE 微调**（显存大）；avm-a100-d-10 在维护 |
| GPU-a100s | 4–8× A100s | 68–256 核 / 1.2–2 TB | 同上，A100 排队时的备选 |
| GPU-a40 | 4–8× A40 | 40–96 核 | MLIP 推理的备选 |
| GPU-rtx6000 | 8× RTX6000 | 192 核 | 推理备选 |
| CPU-amd | — | 128 核 / 1.5 TB | phono3py 的 κ 积分（κ 求解本身是 CPU 计算）；VASP 的备选 |

- **所有分区的单作业时间上限都是 7 天**
- 大部分节点是 mix 状态（有空闲 GPU），单卡作业应该能很快排上
- 每个 MLIP 一个 conda 环境（mace-torch / deepmd-kit / nequip 的 torch 版本可能冲突），先在 GPU 节点上用交互作业（`srun --pty`）测试安装

### 2.2 dirona（用户名 `yjcen`；10-20 查看）

| 分区 | 节点 | 每节点核数 | 时间上限 |
|---|---|---|---|
| **normal**（默认） | c005–c012，每台分 Bb / Stb 两个节点 | c005–c009：56；c010–c012：80 | 无限制 |
| perform | c013 / c014 | 256 / 128 | 无限制 |

- 10-20 时 normal 分区基本满载，**perform 的 c014（128 核）空闲**
- **VASP：`vasp/6.4.2_intel`**（`module load vasp/6.4.2_intel; source /opt/intel/oneapi/setvars.sh --force; mpirun -np $SLURM_NTASKS vasp_std`，和之前 MatterGen 弛豫作业同一版本）
- **PAW：PBE_54 = `/home/yjcen/vasp/potpaw/potpaw_PBE/`**（10-20 确认）；系统目录 `/opt/ohpc/pub/dft/vasp.6.4.2/paw/PBE` 是 PBE_64。pymatgen 要求 `POT_GGA_PAW_PBE_54/` 这个目录名，需建符号链接（plan §8）。存储：home 已用 52/200 GB，`/scratch` 不可用。⚠️ 注意 MPtrj / MPA 的训练数据来自 MP 的旧版 GGA 计算，用的是**旧版 PBE POTCAR**，不是 PBE_54（MatPES 用 PBE_64）。对 B、C、N、O、Al、Si 这些轻元素，不同版本对力和声子的影响很小，但绝对能量会有偏移。所以：
  - **POTCAR 的变体选择和 MP 一致**（例如 Be 用 Be_sv），版本用 PBE_54，在报告里写明
  - **E_hull 不直接拿我们的 DFT 能量和 MP 的凸包比**：竞争相（元素单质、已知二元相）用同一套设置自己算一遍，再构建凸包
  - S6 开始前，拿 1–2 个 MP 结构对比一下我们设置下的能量 / 力 / 声子和 MP 数据的差异，写进记录
- **并行策略**：我们的原胞 ≤ 8 原子，**每个作业用少量核（8–16 核），很多作业并行跑**，不要一个作业占整台节点
- **提交大批量之前先做测试，定下参数**：选 1–2 个代表性结构，扫描核数（8 / 16 / 28）和 NCORE / KPAR，选出效率最高的组合；同时做 ENCUT、k 点、超胞大小的收敛测试（这一步的结果就是 §4.2 第 2 层「收敛检查」的第一批数据）

### 2.3 共同规则

- 作业提交方式：jobflow-remote（本地 runner → SSH → Slurm），数据库是本地 MongoDB（§8.2）
- ⚠️ 账号预计 11 月底失效 → 所有原始输出**每天同步回本地**；回放模式需要的缓存在 11-27 前齐全
- dirona 上的 jobflow-remote worker 也要装 `mga[worker]`（phono3py 3.30.0、phonopy 3.5、symfc ≥ 1.7）：DFT FC3 的位移生成和力常数拟合作业跑在 dirona 上
- ⚠️ jobflow-remote 的 runner 跑在本地：电脑休眠时，集群上的作业照常运行，但流程不会推进到下一步（不会自动提交后续作业、不会取回结果）。长时间离开时，确认 runner 在运行或接受这一延迟

---

## 3. 流程与每一步用的软件（11-03 改为多保真度阶梯）

> 括号里是预计的数量级，最后由成本估计工具和 S0 校准。

### 3.1 前段：生成和初筛（不变）

| 步骤 | 做什么 | 软件 | 在哪跑 |
|---|---|---|---|
| **S0 校准** | 先用表内 c-BN、w-AlN、3C-SiC 检查公开 PBE/RTA 流程，再扩至 5–10 条；Si、金刚石另作额外校准，不计入 103 条。区分文献实验 κ 与同口径 DFT κ；不以测试 κ 标签调参。**同时在 c-BN、BAs、R3m-BNC₂ 上把整条阶梯（L0–L3）跑一遍，定出 §0 的比较基准**（阈值用同一阶梯算出的值，不直接拿实验值比） | 移植的 κ_SRME 协议（ASE + phono3py 3.30.0）+ FourPhonon | datalab |
| **S1 生成** | MatterGen 按 §4.1 的体系生成，原胞 ≤ 8 原子；先每体系 16 个，再按有效率扩配额 | MatterGen | datalab |
| **S2 去重 / 新颖性** | 对称性去重；和 MP / Alexandria 比较，标记「已知 / 新」；**排除 κ_SRME 的 70 个化学式**；R3m-BNC₂ 复现检查（§4.1） | pymatgen StructureMatcher、mp-api | 本地 |
| **S3 MLIP 弛豫 + E_hull** | 按 κ_SRME 的对称性弛豫流程（默认单步，§1.1）；对照 MP 凸包算 E_hull（MP2020 修正），保留 < 0.1 eV/atom（~200） | 移植的 κ_SRME 弛豫（ASE），pymatgen 凸包 | datalab（E_hull 本地） |

### 3.2 后段：多保真度阶梯 + agent 升级

S3 之后的候选进入阶梯。**每升一级，成本涨一个数量级左右，保真度也更高**；agent 在总预算内决定每个候选升到哪一级。

| 级 | 做什么 | 成本（相对） | 输出 / 用途 |
|---|---|---|---|
| **L0 FC2 描述符** | 委员会各算 FC2：虚频检查、a-o gap / 声学带宽、声学支聚拢度、光学带宽、3ph 加权相空间（phono3py `--wgp`）、群速度 | 很低 | 去掉动力学不稳定的；描述符给 agent 判断「3ph 可能被压制」的依据 |
| **L1 委员会 3ph RTA** | κ_SRME 协议（Wigner-RTA，300 K），委员会每个模型各算一次；超胞和 q 网格在提交前固定，各模型相同 | 低 | κ 和按模态的模型间 SRME（分歧） |
| **L2 LBTE** | phono3py 直接解（`--lbte`），只给 L1 排名靠前或分歧大的；RTA 会低估高 κ 材料（Normal 过程被当作热阻），金刚石类差别大 | 中（内存随 (q 点数 × 支数)² 增长，先小网格估） | κ_LBTE / κ_RTA 比值 |
| **L3 MLIP 4ph** | 随机位移 + symfc 拟合 FC4 → FourPhonon（采样版或 GPU 版）算 3+4ph κ。4ph 修正幅度因体系而异（§1.6） | 高 | κ_3+4ph；4ph 修正比例 |
| **L4 DFT** | **L4a**：PBE 弛豫 + 静态 + 带隙（非金属判据，§4.1），便宜；**L4b**：DFT FC2 / FC3（`phono3py_fd`：随机位移 + symfc，§1.6）→ DFT 3ph κ；可再接 MLIP FC4 做混合 3+4ph。**提交前人工审批** | 很高 | 最高保真度；同时检验 MLIP 在这个候选上的模态误差 |

**Agent 做什么**
- 读前一级的结果、模型分歧和描述符，决定谁升级、谁停下、谁要先做检查
- **识别伪影**：近 Γ 点的小虚频（声学求和规则没满足 vs 真不稳定）、弛豫中对称性漂移、平带、几千 W/mK 的离谱 κ（Ohnishi 2026 踩过这类坑）、超胞没收敛、只有一个模型给出高 κ。识别到之后调用检查工具（换超胞、换 symprec、换模型），不能直接改参数（§4.2）
- 每一步写记录：为什么升级、预算花在哪、哪个结论来自哪一级

**怎么评估（和固定规则、随机选择比）**
- **预算账本**：每一级有统一的成本单位（GPU·h / CPU·h 折算），由成本估计工具给出；agent、固定规则、随机选择用同一个总预算
- **固定规则**：每一级按委员会均值取 top-k 往上升（k 按预算定）；**随机选择**：每一级随机挑同样多个
- **公平比较的做法**：MLIP 的 L0–L2 对整个候选池全算，L3 算到比任何策略都多的一个子集 → 三种策略在**回放模式**下用同一份缓存比较，便宜、可重复。L4 DFT 只能对实际选中的候选算，反事实没法回放，这一点如实写
- **「命中」的定义**：以能拿到的最高一级为准（L3，部分有 L4 复核），κ 超过 S0 定出的 c-BN 基准即为命中
- **指标**：命中数 / 召回、花掉的预算、top-k 在各级之间的留存率（Spearman）、抓到的伪影数（候选池里放几个已知的伪影案例作为对照）
- agent 不如固定规则也如实报告（§11.4）

### 3.3 独立 benchmark：uMLIP 的四阶力常数（11-03 加）

- **问题**：没有人系统评测过通用 MLIP 算 FC4 / 4ph 的准确性（§1.6）。L3 的可信度直接依赖它
- **数据**：Li/Lee/Wolverton/Xia 的 773 个化合物 DFT IFC（Zenodo，CC-BY-4.0）。不需要新的 DFT；**只作 benchmark，不进训练**，不违反防泄漏规则
- **做法**：先挑一个小子集（轻元素、高 κ、结构简单的 20–40 个），委员会模型按同样的超胞和截断拟合 FC2/3/4：
  1. 直接比力常数（按近邻壳层分组的误差）
  2. 用 FourPhonon 比 4ph 散射率和 κ_3+4ph
- **先核实**：他们的泛函（是不是 PBE）和 SCPH 设置。SCPH 会改 FC2，比较时要用同样的 FC2 处理方式，否则误差没法归因

### 3.4 附带实验：微调（原目标 B，11-03 降级）

- L4b 的位移超胞力（随机位移构型，局部环境比单原子位移更多样）作为微调数据，微调 MACE-MPA-0（多头微调，回放预训练数据防遗忘）
- 评估：κ_SRME 103 条分组报告（元素在 B/C/N/Al/Si 里的 6 条 vs 其余 97 条）+ 留出的生成候选
- 不作主要结论：0.41 的模型调好了也追不上 0.09 级的现成模型。能说的是「多少 DFT 数据换来多少改进」。普通位移力微调**不能称作 PFT**（PFT 直接监督 Hessian，§4.1）

### 3.5 和原流程比改了什么

1. 「RTA 下算热导，三阶力常数先不算」不成立：**RTA 的声子寿命就来自三阶力常数**，没有它就算不了 κ。可以省的是 **DFT 的三阶力常数**；MLIP 的三阶力常数很便宜，照算
2. 10-20 版把 LBTE 和四声子写成「局限性」；11-03 起它们是阶梯的 L2、L3，**排序在这两级里怎么变正是科学问题本身**
3. κ_SRME 的参考和模型都用 RTA，所以**比较模型时**（L1 委员会、附带实验）口径一致；只有报告绝对 κ 和排序稳健性时才用到 L2、L3
4. DFT FC3 改用随机位移 + symfc：低对称生成结构的系统位移数在几千到几万，不可行（§1.6）

---

## 4. 参数从哪来、怎么控制（问题 2）

### 4.1 S1 生成：文献调研完成（10-20）
- 调研正文：[docs/lit_review.md](docs/lit_review.md)（[四页 PDF](docs/lit_review.pdf)）；[文献表与核查记录](docs/lit_review_references.md)；任务书：`tasks/codex_lit_review_brief.md`
- **建议体系与原始生成配额**：B–C–N 640、Al–C–N–Si 512、C–N–Si 384、Al–B–N 256、C–N 128、B–N 128，总计 **2048**。先每组 16 个检查有效率；全部按标准化原胞 ≤8 原子；本轮不硬锁空间群
- **11-03 调整**：
  - **加 B–C–P 小组（128）**作机理组：和 B–C–N 比，区别在质量比和 a-o gap（Wu/Fan/Hu 同文算过 BPC₂）。1:1 BP 在 κ_SRME 排除表里，生成到要剔除
  - **R3m-BNC₂ 复现检查**（阳性对照）：B–C–N 的生成结果里有没有 R3m-BNC₂（StructureMatcher 比对文献结构）；没生成出来也把它放进候选池，看阶梯和 agent 能不能把它排到前面。这对应调研报告「基础」档的要求：能复现已知体系
  - **带隙判据放宽为「非金属」**（PBE 带隙 > 0，L4a 检查）：排除金属是因为电声散射超出本项目范围；带隙大小只作为报告的属性，不作筛选条件
- B–N 主要作重发现/流程对照，1:1 BN 不进入微调；Be–O 只用来复现流程（BeO 在测试表里，不能用来选参数）；Al–N–O 的常见大胞缺陷结构与本轮预算不匹配，暂不分配主线配额。MP/Alexandria 各体系全量条目数与库外比例尚未核实，不能预先许诺新颖性
- **条件限制**：官方没有现成的“chemical_system + space_group + ml_bulk_modulus”联合权重；用化学体系 checkpoint，其他属性后筛。官方另有 `chemical_system_energy_above_hull` 联合 checkpoint，试运行时和单独的 `chemical_system` 比较有效率（见文献表末尾的复核意见）。高体模量组仅作可选探索对照：B 不约束剪切响应与声子寿命，不是 κ 的充分条件
- **必须比较的直接先例**：Guo et al. 2025（[DOI](https://doi.org/10.1038/s41524-025-01592-8)）已做 CDVAE + Allegro/QBC + 高 κ 碳材料闭环；Cheng et al. CrysVCD（[DOI](https://doi.org/10.1038/s43588-026-01037-2)）已做价态先验 + 生成 + 高 κ/带隙。差异化须用相同 DFT 预算下的选样效率、性质链诊断和可审查 agent 决策来验证
- **PFT 的含义**：直接监督 Hessian；普通位移力微调不能称作 PFT。原建议「比较分歧选样与随机/固定规则选样」在 11-03 后改为比较 **agent 升级决策 vs 固定规则 vs 随机**（§3.2），微调只检查 κ 改进及遗忘

### 4.2 参数分三层
1. **默认值有出处**（写死在工具里，每个都注明来源）：
   - VASP：atomate2 的 MP 兼容输入集 + 声子用的更严收敛参数（§1.4）
   - MLIP 弛豫和 κ：沿用 κ_SRME 官方当前实现（默认单步对称性弛豫、fmax = 1e-4、位移 0.03 Å（11-04 起）、300 K、`is_isotope`、MS-SMM19、phono3py 3.30.0、noNAC；§1.1）。生成结构没有 phononDB 的超胞 / q 网格预设，用长度规则（fc3 ≥ 8 Å、fc2 ≥ 14 Å、q 长度 40 Å），标为启发式、待收敛检查
   - MatterGen：官方推荐的 guidance factor 和 batch 设置
2. **收敛检查是一个单独的工具**：超胞大小、q 网格、位移幅度。agent 只能**调用检查**，不能凭感觉改参数
3. **agent 只能在允许范围内覆盖参数**：每个参数都有 schema（类型 + 范围）；每次覆盖都要写理由，记进日志，在审查页面上高亮

### 4.3 MCP 工具 vs. 让 agent 读手册后直接调用软件

| | Agent 直接调用（读手册 → 写脚本 → bash） | MCP 工具（agent 调用封装好的函数） |
|---|---|---|
| 灵活性 | 高，什么都能做 | 只能做工具暴露出来的事 |
| 可重复性 | 低：每次写的脚本、选的参数都可能不同 | 高：同样的调用，同样的方法和默认参数 |
| 物理正确性 | 容易悄悄出错（忘了对称化、超胞太小、单位错） | 正确做法写在工具里，由人验证过一次 |
| 可审查 | 要读 agent 临时写的代码 | 看调用记录即可：哪个工具、什么参数、为什么 |
| 能否评估 | 难：动作空间无限，没法和基线比 | 容易：动作空间固定，可以和固定规则比 |
| token 成本 | 高（每次都要读手册、写代码） | 低 |

- **结论**：不是二选一。**agent 负责「做什么、什么时候做、要不要信」，工具负责「怎么做」**。科研场景里可靠性比灵活性重要，所以核心计算走 MCP；探索性的分析（画图、临时检查）允许 agent 直接写代码，但不能进入正式结果
- MCP 工具是薄的一层：**内部调用 atomate2 / pymatgen / phono3py，κ 协议移植自官方 κ_SRME 代码**，不重复造轮子（matcalc 不用于 κ，见 §1.5）
- **实现与工具清单**：[docs/mcp_servers_plan.md](docs/mcp_servers_plan.md)（四个 server：mga-structures / mga-mlip / mga-dft / mga-analysis；三种模式 dry_run / live / replay；VASP 审批单只能由人在终端批准）

---

## 5. 修正闭环的逻辑（问题 3：「bug」）

**你看出的问题是对的**：如果最终目的是让 MACE 算 κ 更准，那「生成高 κ 结构 → 微调」这条逻辑有漏洞。生成的「条件」决定的是化学空间，而真正提升 κ 精度的是**补哪一类构型的数据**。

**证据（§1.2）**：
- 同一个架构加入 OMat24 的非平衡构型后，κ_SRME 降低 2–3 倍
- PFT 只用声子数据微调就有明显改进
- 已知现象：基础模型普遍存在势能面系统性软化（我们 B5 benchmark 里 TensorNet 整体软约 25%），根源是训练数据以弛豫轨迹为主，缺少平衡位置附近的曲率信息

→ **对 κ 有用的数据 = 平衡位置附近的位移和扰动构型的精确力**（有限位移超胞、小幅 rattle、低温 MD 快照），而不只是「更多高 κ 的结构」。本项目只用**有限位移超胞**（10-20 决定）：同一份 DFT 计算既给力常数，又给微调数据。11-03 起 DFT FC3 主要用随机位移（所有原子同时动，§1.6），这类构型的局部环境比单原子位移多样，作为微调数据更好。

**修正后的闭环：主线和附带实验分开处理（11-03 更新：目标 B 降为附带实验）**

| | 主线：多保真度发现（§3.2） | 附带实验：微调（§3.4） |
|---|---|---|
| 生成条件的作用 | 把搜索集中在可能高 κ 的化学空间（文献调研 + 11-03 调整） | 决定微调数据**覆盖哪片化学空间**（轻元素共价体系） |
| 选结构的标准 | agent 按阶梯结果、分歧和描述符决定升级；最终要 κ 高 + 稳定 + 非金属 | 不单独选：直接用主线 L4b 算出来的位移超胞 |
| 要算的 DFT | L4a（弛豫、带隙）给前十几个；L4b（DFT FC3）给少数 | 不额外加 DFT |
| 怎么评估 | 各级之间的排序留存率；同预算下 agent vs 固定规则 vs 随机 | 微调前后的 κ_SRME（分组）+ 留出的生成候选 |

- **为什么降级**：委员会换成 0.09 级的 OAM 模型后，「把 0.41 的 MACE-MPA-0 调好」追不上直接可用的模型，说服力弱。数据效率（多少 DFT 换多少改进）仍然可以作为附带结果报告
- **10-20 版的「分歧选样 vs 随机选样」对照（§8 原第 9 条）随之取消**：DFT 预算全部用于主线的 L4，不再分给随机对照组；附带实验的结论只能写成「用这批数据微调的效果」，不能写成「分歧选样比随机好」
- 外层还可以再加一层：如果某个化学体系的分歧特别大，下一轮 MatterGen 就多生成这个体系 → 生成条件 = 「模型最不确定的化学体系」。本轮时间内不做，写进展望
- **防泄漏**：新增微调、回放及验证数据按排除表的 70 个化学式保守剔除所有多形体及扰动后代（覆盖全部 103 个测试条目）；基准 κ 标签不用于调参，预训练重叠另披露。§3.3 的 Li/Xia 数据只作 benchmark，不进训练
- **预期的诚实结论**：在轻元素体系上改进明显；在其他化学体系（碱金属卤化物等）上可能不变甚至变差 → 按组报告，这本身就是关于「专精 vs 遗忘」的结果

---

## 6. 可视化与可审查性（GitHub 成果的重点）

**最终成果** = 流程代码 + 必要软件的 MCP server + 可视化的 agent 计算流程 + 数据

### 6.1 这个想法成立吗？
**成立，而且这是项目最有差异化的部分**。现在的 agent × 科学计算演示大多只展示「agent 能跑起来」，很少回答「结果凭什么可信」。「agent 跑的每一步，人都能看懂、复查、接管」直接回应这个问题，也接上你 research statement 里「workflow 和 agent 合作而不牺牲可靠性」那句话。

### 6.2 风险和建议
1. **范围膨胀是最大风险**：MCP + 工作流 + 可视化 + 科学结果，4 周都做是做不完的
   - → **科学结果（各级保真度下的 κ 排序 + agent 对比基线的结果）是主角**，可视化做到「够用、能审查」即可；微调是附带实验，时间不够就砍
   - → 4ph benchmark（§3.3）先做小子集，FourPhonon 装不顺就推迟，不能拖住主线
   - → **绝对不要自己写工作流引擎**：用 jobflow / atomate2 管理计算，可视化只做一层「读记录 → 出页面」
2. **数据结构先定**：每一步都写一条统一的记录（JSON），可视化和 agent 都读这份记录
   ```
   step_id, parent_ids, tool, params(含来源: default/agent_override+理由),
   code_versions, inputs(结构 id), outputs(文件/数值), status, error, wall_time, host
   ```
3. **审查页面（最小可用版本）**：一个静态 HTML
   - 流程 DAG（mermaid，由记录生成）
   - 每个候选结构一行：每一步的状态和关键数值（E_hull、最低声子频率、κ、各模型分歧）
   - 高亮：失败的步骤、agent 覆盖过的参数、模型分歧超过阈值的地方
   - 点进去能看到参数、日志、出错信息
4. **人工审查关口**（agent 必须停下来等人确认）：提交 VASP 之前（费钱）；开始微调之前（数据集定稿）

### 6.3 Agent 的形式：直接用 Claude Code（问题 2）
- **Claude Code 本身就是建立在 Claude API 上的 agent 框架**：工具调用循环、读写文件、bash、MCP 客户端、权限控制、子 agent、hooks 都是现成的。直接用 Claude API 的话，这些都要自己写
- → **同意先用 Claude Code**：开发阶段交互式使用；正式评估用它的**非交互模式**（`claude -p` + `--mcp-config` + 只允许调用我们的 MCP 工具），可以脚本化地重复跑 N 次任务
- MCP 是开放协议，Codex 等其他客户端也能接 → 成果不绑定某一家，换成开源模型也能用
- **Agent 评估**（证明项目够格；11-03 起按 §3.2 的多保真度升级来评）：
  - 任务：在固定预算下决定每个候选升到哪一级、哪些结果是伪影，最后给出高 κ 候选清单
  - 指标：命中数 / 召回、花掉的预算、各级排序留存率、抓到的伪影数、步数、失败停在哪一步
  - **基线**：固定规则（每级按委员会均值取 top-k）和随机选择，同一个总预算。agent 不如固定规则也要如实报告
  - **回放模式**：工具直接返回缓存的计算结果 → 评估可复现、便宜、不依赖集群（11 月底集群账号失效后也能做）。前提是 MLIP 各级对整个候选池的结果在 11-27 前缓存齐全（§3.2）

---

## 7. 时间线（11-03 重排：集群用到 11 月底，重计算 11-27 前跑完）

| 日期 | 计算（集群） | 本地 / 人 |
|---|---|---|
| **10-20** | — | 文献调研；定化学体系；确认 SSH、Slurm、VASP / PAW；MCP server、记录、审批 + 51 个测试（✅） |
| **11-03** | — | DFT FC3 flow（`phono3py_fd`，随机位移 + symfc）+ 58 个测试；按调研报告重定科学问题（✅） |
| **11-04 → 11-06** | datalab：委员会候选逐个装环境、在 GPU 节点上测（§8 第 1 条）；phono3py 3.30.0、symfc、MatterGen；编译 FourPhonon。dirona：worker 装 `mga[worker]` | MongoDB + jobflow-remote 配好，`live` 模式用 1 个结构跑通 |
| **11-06 → 11-09** | S0 校准（c-BN、w-AlN、3C-SiC）；c-BN、BAs、R3m-BNC₂ 跑 L0–L2 定基准。VASP 参数测试（核数、ENCUT、k 点）；w-AlN 跑一次 `phono3py_fd` 对照 phononDB | S1 试生成（每体系 16 个）→ 确认配额 |
| **11-09 → 11-13** | S1 生成 → S3 弛豫 + E_hull；全池 L0–L2 | S2 去重和新颖性；准备伪影对照案例 |
| **11-13 → 11-18** | L3（MLIP FC4 + FourPhonon）；4ph benchmark 小子集（§3.3） | agent 升级决策（交互）→ **人工确认 L4 名单** |
| **11-18 → 11-25** | **L4 VASP 在 dirona 跑**（11-20 前全部提交）；DFT 超胞抽查（§8 第 8 条） | 整理回放缓存；agent vs 基线（回放模式） |
| **11-25 → 11-27** | 附带实验：微调 + κ_SRME（时间不够就砍） | **所有原始数据拷回本地** |
| **11-28 → 12 月** | 不依赖集群 | 审查页面、回放评估重复跑、GitHub、网站项目页、短文 |

**兜底方案**
- FourPhonon 装不顺：去掉 L3，科学问题收窄为「RTA → LBTE → DFT 下排序的稳健性」，4ph 写进局限
- VASP 跑不完：L4b 只做 3 个候选 + w-AlN 校准，其余只做 L4a
- 微调来不及：直接砍，不影响主线

---

## 8. 待定事项

### 已定（10-20）

1. ~~模型组合（MACE-MPA-0 + DPA-3.1-MPtrj + NequIP-OAM-L）~~ → **11-03 更换**，见第 4 条
   - 保留的结论：**NEP89 单独做速度模型对照**（带 D3，参考不同，不进委员会；在 matcalc 里通过 calorine 的 ASE calculator 调用）；多头预训练模型要记录完整 checkpoint 与输出头，不能自动套用排行榜分数
2. **jobflow-remote 的数据库**：在本地项目目录下起一个 MongoDB（例如 `db/mongo/`，数据目录不进 git）
   - jobflow-remote 的设计里，**只有本地 runner 连数据库**，集群上的作业不需要访问数据库，所以放在本地可行
   - ⚠️ 风险：项目在 `/mnt/d`（Windows 磁盘，WSL 通过 9p 访问），MongoDB 的存储引擎在这种文件系统上可能很慢，或者出现文件锁问题。**先在项目目录试；如果启动报错或明显卡顿，就把数据目录移到 WSL 的 ext4（`~/` 下），项目目录里只放定期的 `mongodump` 备份**
   - 安装：conda-forge 的 `mongodb` 包，或者 Docker（如果 WSL 里装了）
3. **集群信息**：见 §2.1（datalab）和 §2.2（dirona）；VASP `vasp/6.4.2_intel`，PBE_54 在 `/home/yjcen/vasp/potpaw/potpaw_PBE/`；NAC 先不做；DFT 力常数用有限位移，数据量按 DFT 能力控制

### 已定（11-03）

4. **委员会**：从 **DPA-4.1-Max-OAM（0.090）、TECE-OAM-RRA-1.0（0.093）、PET-OAM-XL（0.119）** 里选 3 个，以 datalab 上的安装测试为准；装不上的用 **NequIP-OAM-L（0.166）** 补。**MACE-MPA-0（0.412）**留作旧模型参照和附带实验的微调对象，不进委员会
   - 不进委员会：DPA-4.1-Max-OAM-PFT（训练用了 MDR 声子频率，可能和测试集相关）；Prophet-OAME-MBD（0.065，权重开放但训练集含非公开 ELEMENTA，可作委员会外参照）
   - 新委员会精度接近，分歧不再被最差的模型主导。但它们**都训练在 OAM 上，误差可能相关**，分歧会低估真实误差 → 这正是 L4 DFT 抽查要回答的
5. **科学问题和 agent 角色**：§0 的排序稳健性问题；agent 在多保真度阶梯上做预算分配和伪影判断，和固定规则、随机选择比（§3.2）
6. **目标 B 降为附带实验**（§3.4、§5）；原第 9 条「DFT 预算分给随机对照组」随之取消
7. **化学体系**：加 B–C–P 小组；R3m-BNC₂ 复现检查；带隙判据放宽为非金属（§4.1）
8. **DFT 力常数和超胞收敛**
   - FC3：随机位移 + symfc，逐阶自动选随机 / 系统位移，快照数在请求时算好、写进审批单（`phono3py_fd`，✅ 11-03；plan §9.2）
   - 超胞收敛：没有现成的包能自动收敛，也没有官方数值判据（§1.6）。做法是：
     - MLIP 端对所有候选扫 FC2 / FC3 超胞和 q 网格（`submit_mlip_convergence`，几乎不花钱），待补一个频率判据（两档超胞的 FC2 频率差）
     - DFT 端只抽 1–2 个结构（w-AlN + 一个低对称生成结构），用两档超胞算 FC2，确认 MLIP 选出的超胞在 DFT 下也够大
     - FC3 的截断和快照数收敛已写进每个 `phono3py_fd` 结果
   - 取代 10-20 的「VASP 超胞由收敛测试定（10 vs 14 Å）」

### 已定（11-09，S1-main）

16. **E_hull 闸门在 L0 之前**：MLIP 弛豫后 E_hull ≤ 0.1 eV/atom 的结构才进 L0
17. **单质不进 L1 排序**：选 L1 候选集前用 `filter_structures(min_elements=2)` 去掉只含一种元素的结构（decided_by=human）。目标是新化合物；b2 里 7 个单质（As、P、N₂）在 L0 全有虚频
18. **PET 退出 L1**（11-10）：s1-trial 里 PET-OAM-XL 的 L1 在 4 个结构上跑满 24 h 仍未完成，不补救；PET 只留在 L0 委员会。S1-main 的 L1 由 DPA 和 TECE 计算，去重后的 12 个化合物全部上 L1
19. **生成方式**（11-10）：MatterGen 只按化学体系生成时命中率低（化合物约 2 %），加 E_hull 条件也没有改善。接下来同时试两条路，都先固定化学计量：AIRSS + DPA 随机结构搜索，以及 CrystalFormer 的 CSP 模式

### 还没定

9. 文献建议的体系与配额已给出（§4.1）；试运行后按有效率确认最终配额，并完成 MP/Alexandria 的实际新颖性查询
10. datalab 的家目录 / scratch 配额（dirona 已查：home 52/200 GB，`/scratch` 不可用 → VASP 输出放 home，只保留小文件，`LWAVE`/`LCHARG` 关闭）
11. Codex 里「gpt6.1sol」的准确模型 ID（`codex exec -m <id>` 报错时查可用模型列表）
12. **4ph 的具体做法**：FourPhonon 用采样版还是 GPU 版；FC4 的截断和快照数；symfc 的 FC4 怎么转成 FourPhonon 的输入格式；要不要 SCPH。FourPhonon 编译后用 BAs 试一次再定
13. **Li/Xia 数据的泛函和 SCPH 设置**（§3.3），定 benchmark 子集前核实
14. **预算账本**：工具已实现（11-04，`budget_ledger` / `escalation_baseline`，单位 GPU·h，k 由预算反推）。还要定的是 agent 评估的**总预算数值**，等 S0 和 L0–L2 有了实测成本再定
15. **κ 粗网格噪声**：玩具体系 3×3×3 q 网格上，1e-8 THz 量级的频率噪声能让 κ_P 变 3.5%（简并模态在四面体积分里的问题）；q 长度 20–30 时降到 0.15%。11-04 又碰到一次：phono3py yaml 把质量舍入到 ~1e-7，EMT 铝 5×5×5 网格上 κ 就差 6%，12×12×12 时 0.03%。L1 → L2 的力常数缓存因此存精确的晶胞和质量。按模态比较 DFT vs MLIP 时要用够密的网格，S0 时在真实体系上确认

---

## 9. 目录结构

```
MLIP_generative_agent/
├── DESIGN.md                 ← 本文件
├── .mcp.json                 ← 四个 MCP server（MGA_ACTOR=agent，MGA_MODE 默认 dry_run）
├── .claude/settings.json     ← 禁止 agent 批准审批单（只在从本目录启动 Claude Code 时生效）
├── pyproject.toml            ← 包 `mga`；extras: local / worker（phono3py==3.30.0、symfc）/ test
├── tasks/                    ← 交给 Codex / 子 agent 的任务书
├── docs/                     ← 文献调研、mcp_servers_plan.md、短文草稿
├── src/mga/
│   ├── config / records / params / specs / registry / approvals / results / models / jobs / cli
│   ├── tools/                ← 业务函数（MCP 和脚本共用）
│   ├── workflows/            ← 计算节点上跑的 job（MLIP κ 协议、MatterGen、VASP flow、DFT FC3 `phono3py_fd`）
│   ├── analysis/             ← κ_SRME / 模型分歧、E_hull
│   ├── mcp_servers/          ← 四个 server，每个几十行
│   └── review/               ← 记录 → 审查页面（11 月）
├── tests/                    ← pytest（58 个，plan §10）
├── configs/                  ← models.yaml、cluster.yaml、jobflow_remote/ 模板
├── data/                     ← 结构注册表、命名集合、结果缓存、参考数据（大文件不进 git）
├── approvals/                ← VASP / 微调审批单
├── db/                       ← 本地 MongoDB 数据目录 + mongodump 备份（不进 git）
└── runs/                     ← 每次运行的记录（JSON）
```

---

## 11. 给新 session 的注意事项（来自之前对话的记录，10-20 整理）

### 11.1 协作约定
- **始终用中文回复**；可以质疑方向：目标不清就先问，路径不优就直说
- **先设计、后计算**：大批量提交前，先用 1–2 个结构测试参数；VASP 提交前、微调前都要停下来等用户确认（§6.2）
- **计算放置**：小 Python 脚本在本地；MatterGen / MLIP 批量 / VASP / 微调只在计算节点跑，不在登录节点算（§2）
- 文献调研交给 Codex，任务书在 `tasks/codex_lit_review_brief.md`，用户指定模型「gpt6.1sol」+ xhigh reasoning（本地 `~/.codex/config.toml` 默认是 gpt-6-astra / medium，需要在命令里覆盖）
- 项目文件都放在本目录（Obsidian 库里，用 Markdown 写笔记）

### 11.2–11.3 个人背景
- 个人背景和项目动机在 `DESIGN.private.md`（不进 git，只在本地）。新 session 两份一起读

### 11.4 产出的诚实性
- **每个结论都要有证据**（数字、文件、提交记录）
- **只写真正做了的**：用户之前明确不愿意把 benchmark 说成「agent 跑的」。本项目里哪些步骤是 agent 决定的、哪些是人决定的、哪些是固定脚本，**必须在记录里区分清楚**，报告时如实说
- agent 不如固定规则的基线、微调在某些化学体系上变差、VASP 没跑完：这些都**如实报告**，不美化
- κ_SRME 的 103 个测试化合物**不能进训练数据**（§5）

### 11.5 不要做的
- 不要在登录节点跑计算
- 不要在没有用户确认的情况下大批量提交 VASP

---

## 10. 参考

- Póta et al., Thermal conductivity predictions with foundation atomistic models, [arXiv:2408.00755](https://arxiv.org/abs/2408.00755)
- [Matbench Discovery 排行榜](https://matbench-discovery.materialsproject.org/)
- PFT: Phonon fine-tuning for MLIPs, [arXiv:2601.07742](https://arxiv.org/abs/2601.07742)
- 高 κ 化学/结构判据、生成筛选先例与完整文献表：[文献调研](docs/lit_review.md)、[文献表](docs/lit_review_references.md)
- DPA-3, [arXiv:2506.01686](https://arxiv.org/abs/2506.01686)；DPA-4, [arXiv:2606.02419](https://arxiv.org/abs/2606.02419)
- NEP89, [arXiv:2504.21286](https://arxiv.org/abs/2504.21286)
- GRACE, [arXiv:2508.17936](https://arxiv.org/abs/2508.17936)
- EquiformerV3, [arXiv:2604.09130](https://arxiv.org/abs/2604.09130)
- NequIP（新版框架）, [arXiv:2504.16068](https://arxiv.org/abs/2504.16068)
- [matcalc](https://github.com/materialsvirtuallab/matcalc)；[atomate2](https://github.com/materialsproject/atomate2)；[AiiDA](https://aiida.net/)
- [MatterGen](https://github.com/microsoft/mattergen)
- 四声子：Wu, Fan, Hu, PRB 103, L041203 (2021)；Ravichandran & Broido, PRX 10, 021063 (2020)；Li, Lee, Wolverton, Xia, npj Comput. Mater. 12 (2026)（Zenodo doi:10.5281/zenodo.15937696）；FourPhonon 教程 JAP 140, 081101 (2026)；采样版 Guo et al., npj Comput. Mater. 10, 31 (2024)；FourPhonon_GPU arXiv:2510.00518
- 力常数：[symfc](https://github.com/symfc/symfc)；Phonon Olympics, JAP 138, 135108 (2025)
- 自己的前期工作：[基础模型声子与热输运 benchmark](https://diroccen.github.io/projects/mlip-foundation-benchmark/)
