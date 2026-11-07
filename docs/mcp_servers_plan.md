---
title: MCP server 规划与实现说明
created: 2026-10-20
status: Phase A 已实现；DFT FC3 flow `phono3py_fd` 11-03 加入（§9.2）；58 个测试通过（§10）；dry_run / replay 可用；live 需先配好 MongoDB + jobflow-remote，尚未在真实集群上跑过
---

# MCP server 规划

> 上级文档：[DESIGN.md](../DESIGN.md) §4.2（参数三层）、§4.3（MCP vs 直接调用）、§6（记录与审查）
> 代码：`src/mga/`；测试：`tests/`；客户端配置：`.mcp.json`

## 1. 定位与时间线修正

- **MCP 只是薄 adapter**。所有逻辑写在普通 Python 函数里（`mga.tools.*`），MCP server 只做参数转发、错误包装。同一套函数也能被脚本和 CLI 直接调用。
- **时间线修正**：DESIGN §7 原来把「MCP server 定稿」排在 11-01 之后。但 10-22 起的计算如果不经过这些函数，就没有统一记录，回放模式也无从谈起。所以**核心函数现在写好**，10-22 起的所有计算都通过它们执行并写记录；11 月只做定稿（审查页面、回放评估）。
- **谁做的决定要分清（§11.4）**：每条记录带 `actor`（谁调用的：`agent` / `human` / `script`，取自环境变量 `MGA_ACTOR`，`.mcp.json` 里设为 `agent`，CLI 设为 `human`）；所有提交类工具另有必填参数 `decided_by`（`agent` / `human`：这批结构和参数是谁选的）和 `rationale`（理由）。

## 2. 10-20 查证的事实（影响默认参数）

### 2.1 dirona：VASP 与 PAW

| 项目 | 结果 |
|---|---|
| VASP 模块 | `vasp/6.4.2_intel`（默认）、6.4.2 / 6.4.2_amd / 6.4.2_libxc、6.5.1 / 6.5.1_intel / 6.5.1_amd |
| 选用 | **`vasp/6.4.2_intel`**：用户之前的 MatterGen 弛豫作业就是用它跑的（`~/vasp/mattergen/space_group/asejob.sh`）。整个项目不混用编译版本 |
| 启动方式 | `module purge; module load vasp/6.4.2_intel; source /opt/intel/oneapi/setvars.sh --force; mpirun -np $SLURM_NTASKS vasp_std` |
| **PBE_54** | **`/home/yjcen/vasp/potpaw/potpaw_PBE/`**（330 项，`<sym>/POTCAR`，最后更新 2015-09-04）。pymatgen 对 B、C、N、O、Al、Si、Be_sv、Li_sv、Mg_pv、Ga_d 的识别结果是 PBE_54/PBE_52，W_sv 只匹配 PBE_54 |
| 系统 PAW | `/opt/ohpc/pub/dft/vasp.6.4.2/paw/PBE` = PBE_64（带 SHA256 头）。上述轻元素的数据与 PBE_54 相同 |
| 与 MP 旧版 "PBE" | 上述轻元素**都没有**被识别为 MP 旧版 PBE 集 → 绝对能量与 MP 有偏移，支持 DESIGN §2.2「竞争相自己算」 |
| pymatgen 目录 | 需要 `PMG_VASP_PSP_DIR/POT_GGA_PAW_PBE_54/<sym>/POTCAR`，dirona 上要建一个指向 `potpaw_PBE` 的符号链接（见 §8） |
| 存储 | home 已用 52/200 GB；`/scratch` 不可用（root 所有，在根分区） |

### 2.2 κ_SRME 官方实现（matbench-discovery `2b7bff9`，2026-10-19）与 DESIGN 原写法的差异

直接读了 `matbench_discovery/phonons/{pipeline,thermal_conductivity}.py`、`adapters/standard.py`、`metrics/phonons.py`：

| 参数 | DESIGN 原写法 | 官方当前实现 | 处理 |
|---|---|---|---|
| 位移幅度 | 0.03 Å | `KappaSettings.displacement_distance` 默认 0.01 Å；但 DPA-4.1、TECE、PET-OAM-XL、NequIP-OAM-L 的 `models/*/*.yml` 都写 **0.03 Å** | 11-04 定为 **0.03**：和委员会模型的排行榜结果同口径，也和 DFT 侧（phononDB-LTC）一致。比较 MACE-MPA-0 时覆盖成 0.01 |
| 弛豫 | 两步（先约束后放开） | **默认单步**：`FixSymmetry(symprec=0.01)` + `FrechetCellFilter(mask=无剪切)` + FIRE，fmax 1e-4，300 步；两步是可选模式 | 默认单步，`two-stage` 作为允许的覆盖值 |
| 对称性判定 | spglib 默认 | symprec 1e-5（moyopy） | 用 spglib，symprec 1e-5（记录里写明） |
| 同位素散射 | 未写 | **`is_isotope=True`**，`boundary_mfp=1e6` μm | 照搬 |
| 输运方程 | Wigner | `transport_type="MS-SMM19"`，**锁定 phono3py==3.30.0**（≥3.31 删掉了这个求解器） | **所有 MLIP worker 环境锁定 phono3py 3.30.0** |
| 虚频判据 | 「出现虚频」 | Γ 点光学支 < 0、Γ 点声学支 < −0.01 THz、其他 q 点任意 < 0 | 照搬 |
| 不算 κ 的情况 | — | 有虚频，或弛豫后空间群变了 → SRME 记 2 | 照搬 |
| NAC | — | 参考数据是 **noNAC** | 不做 NAC（用户 10-20 决定，§9.1） |

phononDB 的超胞 / q 网格（从 103 个参考结构统计）：岩盐 / 闪锌矿用惯用胞 fc3 2×2×2、fc2 4×4×4，纤锌矿 fc3 3×3×2、fc2 5×5×3；q 网格 19³ / 19×19×15。对轻元素条目，fc3 超胞最小宽度 6.6–9.1 Å，fc2 11–18 Å，q 网格长度 40–53 Å。**生成结构没有这些预设**，我们用长度规则（fc3 ≥ 8 Å、fc2 ≥ 14 Å、q 长度 40 Å），标为「按 phononDB 选择校准的启发式，待收敛检查」。

### 2.3 工具链里会悄悄出错的默认值

1. **atomate2 的 `PhononMaker` 默认是 PBEsol**（atomate2 0.1.5 实测，NaCl / BeO：所有子 Maker 都是 `GGA=PS`；POTCAR 是 PBE_54，但 Be 用 `Be` 而不是 MP 的 `Be_sv`；ENCUT 弛豫与位移 700、static 与 Born 680）→ 与 DESIGN §1.4 的 PBE 一致性冲突。所有子 Maker 都换成 MPRelaxSet / MPStaticSet（`src/mga/workflows/vasp_flows.py`）。`tests/test_vasp_flows.py` 固定了这些上游默认值：atomate2 改默认值时测试会失败，提醒重新检查。
2. **pymatgen 的 `MPRelaxSet` / `MPStaticSet` 用的是旧版 `PBE` POTCAR 集**（实测 `potcar_functional = "PBE"`）→ 必须显式传 `user_potcar_functional="PBE_54"`。POTCAR 变体（`Be_sv`、`Na_pv` 等）仍按 MP 的选择。
3. **matcalc 的 `Phono3Calc` 默认先以 fmax=0.1 再弛豫一遍**，而且不输出 SRME 需要的按模态 Wigner κ → **κ 计算不用 matcalc**，改为移植官方 κ_SRME 函数（MIT 许可，代码里注明出处）。这是对 DESIGN §1.5「MLIP 侧交给 matcalc」的偏离。
4. **phono3py 4.x 的 `SMM19` 不再给出 `mode_kappa_P_RTA` / `mode_kappa_C`**，算不了官方 SRME。
5. **ENCUT 用 520 eV，不用 atomate2 位移计算默认的 700 eV**：520 是 MP 的值（MPtrj / MPA 的训练数据就是这个设置），并且 EDIFF 1e-8、PREC Accurate、LREAL False 已经收紧了力的数值噪声。ENCUT 是否够，由 S6 前的收敛测试（520–800 eV）回答；不够时作为有理由的覆盖值提高，对所有结构统一。
6. **非零 MAGMOM 会让 atomate2 的 `generate_phonon_displacements` 报错**（`primitive_matrix="auto"` 时不接受磁矩）。MP 输入集默认 `ISPIN=2` 并给初始磁矩，所以声子 flow 的**弛豫、static、位移计算统一用 `ISPIN=1`**（参数 `ispin`，可覆盖），同一 flow 内不混用自旋处理。`preview_vasp_inputs` 对奇数电子数和含过渡金属的结构给出警告。
7. **atomate2 的超胞大小在运行时决定**（`get_supercell_size` 是一个 `@job`，作用在 VASP 弛豫后的结构上）。所以 `estimate_vasp_cost` 和 preview 里的超胞 / 位移数是按输入结构算的估计值；`job_status` 只能看到提交时已知的作业，位移计算作业是运行时动态加进来的。
8. **MLIP 的超胞和 q 网格在提交前固定**：按输入结构用长度规则（或 phononDB 参考设置）定好 fc2 / fc3 超胞和 q 网格，写进作业参数和记录；worker 上弛豫后晶格稍有变化也不重新选。否则同一结构在不同模型下可能得到不同的 q 网格，按模态 SRME 就没法比较。

## 3. 架构

```
src/mga/
├── config.py        路径、运行模式（MGA_MODE = dry_run | live | replay，默认 dry_run）、actor
├── records.py       StepRecord：每次工具调用一条 JSON，runs/<run_id>/records/<step_id>.json
├── params.py        参数解析：默认值 + 覆盖（必须写理由）+ 每个参数的来源
├── specs.py         所有参数 schema 集中在这里，默认值逐条注明出处
├── registry.py      结构注册表：ID = 约化化学式 + 规范化结构的 sha1 前 10 位；命名集合
├── approvals.py     人工审批单（VASP、微调）：pending → approved → consumed，带 plan_hash
├── results.py       结果缓存：data/results/<kind>/<model>/<sid>__<paramhash>.json（回放模式只读这里）
├── models.py        configs/models.yaml 的读取与校验
├── jobs.py          jobflow-remote 的薄封装（提交、查状态、取输出）
├── cli.py           mga approvals list/show/approve/reject；mga records list
├── tools/           业务函数（MCP server 和脚本共用）
├── workflows/       在计算节点上跑的 jobflow job（MLIP 批处理、MatterGen、VASP flow）
├── analysis/        κ 指标（SRE / SRME / 模型间分歧）、E_hull
└── mcp_servers/     四个 server，每个只有几十行
```

### 3.1 三种运行模式

| 模式 | 提交类工具的行为 | 用途 |
|---|---|---|
| `dry_run`（默认） | 构建 jobflow Flow、校验参数、写记录（status = `dry_run`），**不提交** | 开发、给人看计划 |
| `live` | 通过 jobflow-remote 提交到 datalab / dirona | 正式计算 |
| `replay` | 不提交，直接从 `data/results/` 返回缓存结果；没有缓存就报错（`ResultMissing`）。`fetch_results` 只读缓存。**VASP 不能回放**：`submit_vasp` 只在 live 下运行，其他模式下拒绝且不消耗审批单 | agent 评估（11 月集群失效后也能跑） |

### 3.2 记录（DESIGN §6.2 的字段 + 补充）

`step_id, run_id, parent_ids, tool, actor, decided_by, rationale, mode, params{name: {value, source, ref, reason}}, code_versions, inputs, outputs, status, error, wall_time_s, host, created_at`。worker 端的软件版本和主机名由作业本身返回，`fetch_results` 时补进结果文件。

### 3.3 参数三层（DESIGN §4.2）的实现

1. **默认值有出处**：`specs.py` 里每个字段都有 `source`；没有可靠出处的写 `heuristic; to validate`，不装作有出处。
2. **收敛检查是单独的工具**：`submit_mlip_convergence`（超胞、q 网格、位移）、VASP 收敛测试走审批。
3. **覆盖必须在 schema 范围内并写理由**：pydantic `extra="forbid"` 拒绝未知参数，`ge/le` 限制范围；有覆盖但没写 `override_reason` 直接报错。记录里每个参数标注 `default` / `override`，审查页面据此高亮。

### 3.4 结构不进 LLM 上下文

2048 个结构的 ID 塞进上下文既贵又容易抄错。工具之间传**命名集合**（`data/sets/<name>.json`，不可变，带父集合和生成方式），agent 只看摘要和少量样例。

## 4. 四个 server 的工具清单

### mga-structures（本地，轻）

| 工具 | 作用 |
|---|---|
| `import_structures` | 从文件（cif/extxyz/MatterGen 输出 zip）导入，注册，生成命名集合 |
| `structure_summary` | 集合的成分 / 原子数 / 空间群分布 + 少量样例 |
| `filter_structures` | 按原子数、元素、空间群、键长等过滤，生成新集合 |
| `deduplicate` | StructureMatcher 去重（先按约化化学式分组） |
| `check_novelty` | 与 MP 比较（mp-api，按化学体系缓存到 `data/reference/mp/`），标注已知 / 新 |
| `check_benchmark_leakage` | **保守排除 κ_SRME 70 个化学式的所有多形体**（DESIGN §1.1），以及以它们为父结构的后代 |

### mga-mlip（提交到 datalab）

| 工具 | 作用 |
|---|---|
| `list_models` | 模型、checkpoint、能量参考、是否属于分歧委员会 |
| `describe_parameters` | 某类计算的参数 schema、默认值与出处 |
| `submit_mattergen` | MatterGen 生成（化学体系条件） |
| `submit_mlip_relax` | κ_SRME 协议的弛豫，分块批处理，返回能量 / 结构 / 对称性变化 |
| `submit_mlip_phonon_kappa` | FC2 → 虚频判据 → FC3 → Wigner-RTA κ（300 K） |
| `submit_mlip_convergence` | 超胞 / q 网格 / 位移幅度扫描 |

微调（11-03 起是附带实验，DESIGN §3.4）到时再加 `request_finetune` / `submit_finetune`，同样走审批。多保真度阶梯需要的新工具见 §9.3。

### mga-dft（提交到 dirona，有人工关口）

| 工具 | 作用 |
|---|---|
| `preview_vasp_inputs` | 生成 INCAR / KPOINTS / POSCAR / POTCAR.spec，列出与 MP 默认值的差异；`phono3py_fd` 另给出超胞设置和位移计划 |
| `estimate_vasp_cost` | 计算数、原子数、k 点数的统计；核时只给粗估（系数待 dirona 测试校准）。计算类型：`relax_static`、`phonon_fd`（FC2）、`phono3py_fd`（FC2 + FC3 + κ，§9.2）、`convergence` |
| `request_vasp_submission` | 冻结计划（plan_hash），生成审批单，**agent 必须在这里停下** |
| `submit_vasp` | 只接受已批准、未用过、hash 未变的审批单 |

**MCP 不提供批准工具**。批准只能由人在自己的终端里运行 `mga approvals approve <id>`；CLI 要求交互式 TTY 并要求手动输入确认，项目 `.claude/settings.json` 另外禁止 agent 通过 Bash 调用它。

### mga-analysis（本地）

| 工具 | 作用 |
|---|---|
| `job_status` | 查询 jobflow-remote 作业状态 |
| `fetch_results` | 分两步：**ingest**（把完成的作业输出写进 `data/results/`，已有的不覆盖）→ **derive**（生成下游命名集合，例如 `<set>.relaxed.<model>.<hash6>`、`mattergen.<key>`）。VASP 的能量 / 力 / 应力另存为 `*.frames.extxyz`，就是 S7 的微调数据 |
| `compute_ehull` | MP2020 兼容性修正 + MP 凸包（只对 `energy_reference` 为 `mp_pbe` 的模型）。**不接受 VASP 结果**：我们用 PBE_54，MP 凸包用旧版 PBE，要比就得自己算竞争相（DESIGN §2.2） |
| `model_disagreement` | 模型间 κ 的 SRE、按模态 SRME、声子频率差。q 点不一致时**标记 `mode_comparable=False`**（SRME 不算，SRE 仍报告），不报错；两次提交的参数不同时列在 `param_differences` 里 |
| `kappa_srme` | 与 phononDB 参考比较（移植官方实现）；只对参考集里的结构有效 |
| `kappa_convergence` | 读 `submit_mlip_convergence` 的结果，给出每个结构各取值的 κ 和最后一步的相对变化 |
| `query_records` | 按 run / 工具 / 结构查记录 |

## 5. 默认参数一览

| 计算 | 参数 | 默认 | 出处 |
|---|---|---|---|
| MLIP 弛豫 | 优化器 / 晶胞滤波 | FIRE / FrechetCellFilter | matbench-discovery `KappaSettings` |
| | fmax / 最大步数 | 1e-4 eV/Å / 300 | 同上 |
| | 模式 / FixSymmetry symprec | single-stage / 0.01 | 同上 |
| | 空间群判定 symprec | 1e-5 | 同上 |
| MLIP 声子 + κ | 位移 | 0.03 Å | 委员会模型的排行榜设置（§2.2） |
| | 温度 | 300 K | 同上 |
| | 输运 | MS-SMM19（Wigner-RTA），is_isotope，boundary_mfp 1e6 μm | `thermal_conductivity.calculate_conductivity` |
| | fc3 / fc2 超胞最小宽度 | 8 Å / 14 Å | 启发式，按 phononDB 对轻元素的选择校准 |
| | q 网格长度 | 40 Å | 同上 |
| | 精度 | float64 | MLIP 声子的通常做法（k_SRME 用 float64 跑 MACE）；待核 |
| MatterGen | guidance factor | 2.0 | MatterGen README 的条件生成示例；待在 datalab 核对 |
| | batch size | 64 | MatterGen 默认；待核 |
| VASP | 输入集 | MPRelaxSet / MPStaticSet + `PBE_54` | DESIGN §1.4 / §2.2 |
| | ENCUT / EDIFF / PREC / LREAL | 520 / 1e-8 / Accurate / False | MP + DESIGN §1.4 收紧（不用 atomate2 默认的 700，见 §2.3 第 5 条） |
| | ISPIN | 1（声子 flow 的所有步骤） | 启发式，见 §2.3 第 6 条 |
| | ISMEAR / SIGMA | 0 / 0.01 | 绝缘体声子的常用设置；启发式 |
| | k 点 | reciprocal_density 100（静态 / 位移） | MPStaticSet 默认 |
| | 位移 / 超胞最小长度 | 0.01 Å / 10 Å | phonopy 默认 / 按成本定的启发式，按 §9.2 第 4 点抽查 |
| | NAC | 不做 | 与 κ_SRME noNAC 参考一致 |
| DFT FC3（`phono3py_fd`） | 位移方式 | auto（逐阶取系统 / 随机较便宜的） | §9.2 |
| | 位移幅度 | 0.03 Å | phononDB-LTC |
| | symfc 截断 / 快照系数 | 4、5 Å / FC3 × 4、FC2 × 2 | phono3py `auto` 规则 |
| | 学习曲线 / 随机种子 | 一半快照 / 1103 | 项目自定 |
| | 超胞 / q 网格 | fc3 ≥ 8 Å、fc2 ≥ 14 Å、q 长度 40 Å | 与 MLIP κ 相同（`length_rule_settings`） |
| | symprec / 输运 | 1e-5 / MS-SMM19 | κ_SRME |

## 6. 数据流

```
MatterGen 输出 → import_structures → filter → deduplicate → check_benchmark_leakage → check_novelty
   → submit_mlip_relax(每个模型) → fetch_results → compute_ehull → filter
   → submit_mlip_phonon_kappa(每个模型) → fetch_results → model_disagreement
   → [人工确认名单] → preview_vasp_inputs → request_vasp_submission → [人批准] → submit_vasp
   → fetch_results（VASP frames.extxyz → 微调数据，附带实验）
```

每个提交工具都按 (结构 ID, 模型, 参数 hash) 去查缓存，已有结果就不重复提交（除非显式 `force`）。

## 7. jobflow-remote 部署（live 模式前要做）

- worker：`datalab_mace`、`datalab_dpa`、`datalab_tace`、`datalab_upet`、`datalab_nequip`、`datalab_nep`、`datalab_mattergen`、`dirona_vasp`（模板：`configs/jobflow_remote/mga.yaml.template`）
- 每个 worker 的 conda 环境都要装 `mga[worker]`（`pip install -e` 或 rsync 后安装）：**phono3py==3.30.0**、phonopy 3.5.x、symfc ≥ 1.7。**dirona_vasp 也要装**，DFT FC3 的位移生成和拟合在那里跑
- 大数组（按模态 κ、频率）通过 jobflow 的附加存储（GridFS）保存，不塞进主文档
- runner 跑在本地：电脑休眠时流程不推进（DESIGN §2.3）

## 8. 需要用户做的事

1. **装 MongoDB**（conda-forge `mongodb`）。先在 `db/mongo/` 试；9p 文件系统出问题就移到 `~/` 的 ext4
2. `cp configs/jobflow_remote/mga.yaml.template ~/.jfremote/mga.yaml`，填好后 `jf project check --errors`
3. dirona：`mkdir -p ~/pmg_psp && ln -s ~/vasp/potpaw/potpaw_PBE ~/pmg_psp/POT_GGA_PAW_PBE_54`，`~/.pmgrc.yaml` 写 `PMG_VASP_PSP_DIR: /home/yjcen/pmg_psp`
4. 设 `MP_API_KEY`（本地）
5. datalab 上逐个核实（11-03 换委员会后）：DPA-4.1-Max-OAM 的 checkpoint 与是否多头、TECE-OAM-RRA-1.0 的 ASE calculator 接口（`tace` loader 还没接）、PET-OAM-XL 在 upet 里的模型名和版本、NequIP-OAM-L 编译模型的加载方式（备选）、NEP89 的 D3 是否内置、MatterGen 的 CLI 参数。核实后改 `configs/models.yaml` 里的 `TODO`
6. 计算节点上 `lscpu`，确认 intel 编译的 VASP 在 AMD 节点上的效率（S6 前的核数 / NCORE 测试会顺带回答）
7. **从项目目录启动 Claude Code**（`cd research_projects/MLIP_generative_agent && claude`）：`.mcp.json`（四个 server）和 `.claude/settings.json`（禁止 agent 调 `mga approvals approve/reject`、禁止编辑 `approvals/`）只在这个目录下生效。从 Obsidian 根目录启动时两者都不加载

## 9. 用户决定（10-20 定，11-03 更新）

1. **NAC / Born 电荷：先不做**。与 κ_SRME 的 noNAC 参考一致；`born_charges` 固定为 `False`（schema 里只允许这一个值）。以后要画含 N、O 极性材料的色散图或报告绝对 κ 时再加 DFPT，届时作为新的计算类型走审批
2. **DFT 力常数用有限位移**（不用 rattle + hiphive / pheasy 拟合）。位移超胞的 DFT 力同时就是微调数据（DESIGN §3.4）。**数据量按 DFT 能力控制**：`estimate_vasp_cost` 给出每个结构的位移数 / 计算数和成本估计，人在批准审批单时决定做多少个结构；单张审批单最多 200 个结构
3. **VASP 超胞**：10-20 的「10 Å vs 14 Å 收敛测试」在 11-03 改为 §9.2 第 4 点的做法

### 9.2 DFT FC3：`phono3py_fd`（11-03 实现）

**流程**（`src/mga/workflows/phono3py_fd.py`）：

```
relax（MP 兼容双步弛豫，PBE_54）
  → phono3py_displacements（dirona worker：生成 FC3 / FC2 位移超胞，按 VASP 的原子顺序预排序）
  → run_fc3_displacements / run_fc2_displacements（atomate2 run_phonon_displacements → N 个 static）
  → phono3py_collect（回收力 → 拟合 FC2 / FC3 → 虚频检查 → Wigner-RTA κ，写 fc2/fc3.hdf5）
```

1. **逐阶选位移方式**（`displacement_mode="auto"`）：
   - 系统位移：每对原子单独位移，超胞数随对称性降低急剧增长
   - 随机位移 + symfc 拟合：快照数 = symfc 的理论下限（独立参数数 / 3N，取最大截断）× 系数（FC3 × 4、FC2 × 2，phono3py `auto` 的规则）
   - 每一阶取较便宜的那个，审批单里两种数都列出来

   | 结构 | FC3 超胞 | FC3：系统 vs 随机 | FC2 超胞 | FC2：系统 vs 随机 |
   |---|---|---|---|---|
   | zb-SiC | 2×2×2 惯用胞（64） | 222 vs **12** | 4×4×4（512） | **2** vs 2 |
   | w-AlN | 3×3×2（72） | 1254 vs **44** | 6×6×3（432） | 6 vs **2** |
   | 生成的 Pm 8 原子结构 | 96 | 27696 vs 约 660 | — | — |

2. **计划在请求时算好、运行时强制执行**：`request_vasp_submission` 时就算出位移计划（2–9 s / 结构），写进冻结的 plan；worker 端重新生成后核对数目，对不上就报错要求重新审批。随机位移用固定种子（`random_seed`，FC2 用 seed + 1），可复现
3. **VASP 会重排原子**：pymatgen 的输入集 `sort_structure=True` 按电负性排序。位移超胞在生成时就先排好序，回收时用置换把力映射回 phonopy 的顺序；测试里用 EMT 代替 VASP，排序后映射的力常数和原始顺序的结果在 1e-10 内一致
4. **收敛诊断写进每个结果**：
   - 两个 symfc 截断（默认 4 / 5 Å）各拟合一次，报告 κ 相对差；系统位移另加一次不截断的完整解
   - 快照学习曲线：一半快照 vs 全部快照的 κ 相对差（随机位移模式）
   - 超胞大小：MLIP 端对所有候选扫（`submit_mlip_convergence`，待补 FC2 频率判据）；DFT 端只抽 1–2 个结构（w-AlN + 一个低对称生成结构）用两档 FC2 超胞确认
   - 没有官方的数值判据。项目自定 κ 变化 < 5%；不同代码、不同专家之间本身就有约 15% 的散布（Phonon Olympics, JAP 138, 135108 (2025)）
5. **和 MLIP κ 同网格**：超胞和 q 网格都由 `length_rule_settings` 给出，DFT 和 MLIP 的结果可以按模态直接比较。位移幅度也相同：都是 0.03 Å（DFT 照 phononDB-LTC，MLIP 照委员会模型的排行榜设置，11-04 起）
6. **粗网格上 κ 的数值噪声**：EMT Cu₃Au 在 3×3×3 q 网格上，约 5e-8 THz 的频率差就让 κ_P 变 3.5%（推测是简并模态在四面体积分里的问题）；q 长度 20–30 时降到约 0.15%。按模态比较时要用够密的网格；测试里因此比力常数，不比 κ
7. **部署**：位移生成和拟合作业跑在 dirona worker 上，那里的环境要装 `mga[worker]`（phono3py 3.30.0、phonopy 3.5.x、symfc ≥ 1.7）；本地 `mga[local]` 也要有 phono3py + symfc（请求时算计划）

### 9.3 11-03 方向调整带来的新工具（11-04 实现）

DESIGN §3.2 的多保真度阶梯。计算在 `workflows/ladder.py`，作业在 `workflows/mlip_jobs.py`（`mlip_ladder_chunk`），回放评估是纯函数 `analysis/escalation.py`：

| 工具 | 作用 | 级 |
|---|---|---|
| `submit_mlip_fc2_descriptors` | 弛豫 + FC2：虚频、a-o gap / 声学带宽、声学支聚拢度、群速度、加权相空间（phono3py `--wgp`）、Grüneisen 和 Slack 估计（默认排序分数 `kappa_slack_W_mK`） | L0 |
| `submit_mlip_phonon_kappa`（改） | 照旧算 Wigner-RTA；另把紧凑 fc2 / fc3 存到 `<fc_cache>/<model>/<sid>/<κ 参数 hash>/` | L1 |
| `submit_mlip_kappa_lbte` | 读 L1 缓存直接解 LBTE，不再调用 MLIP；先估碰撞矩阵内存，超了就缩网格或跳过（跳过时 L1 的估计继续往上走）；输出 κ_LBTE / κ_RTA | L2 |
| `submit_mlip_fc4_fourphonon` | 随机位移 + symfc 拟合 FC2 / FC3 / FC4，留出集上按阶报力 RMSE；写 ShengBTE 格式的力常数（读回校验）和 3ph / 3+4ph 两份 CONTROL；配置了 FourPhonon 命令就跑，输出 4ph 修正比 | L3 |
| `budget_ledger` | 按级、按提交汇总 GPU·h（每个结构的 wall time，单 GPU 作业），可给总预算看剩余 | 评估 |
| `escalation_baseline` | 在同一份缓存上回放：固定规则（每级按上一级估计取 top-k）、随机选择（默认 200 次）、可选 agent 的实际选择；同一个硬预算；k 取「固定规则刚好花完预算」的最大保留比例；附相邻两级之间的 Spearman | 评估 |

实现上的几点：
- **缓存路径**：worker 端的根目录来自 `configs/cluster.yaml` 的 `fc_cache` / `ladder_work`，环境变量 `MGA_FC_CACHE` / `MGA_LADDER_WORK` 优先。
- **缓存要存精确的晶胞和质量**：phono3py yaml 里的质量和晶格是舍入过的（~1e-7）。在 5×5×5 网格上，单凭这一点就能让 EMT 铝的 κ 差 6%（13.00 vs 12.19）。所以 `mga_fc_meta.json` 存原值，L2 用它们重建 `Phono3py`；重建后三种网格上都和 L1 逐位相同。
- **各级的分数**：L0 委员会均值（只用来排序）；L1 κ 均值；L2 = L1 × LBTE/RTA；L3 =（L2，否则 L1）× 4ph 比。「真值」取缓存里最高一级。稳定性用 L0 的委员会多数票。
- **成本里没算的**：L4（dirona CPU）、排队时间、每个作业加载模型的时间。

L4 用已有的 `phono3py_fd`。4ph benchmark（DESIGN §3.3）需要读 Li/Xia 的 Zenodo IFC，格式核实后再定接口。

## 10. 测试（10-20 起，11-03 更新）

`~/.venvs/mga/bin/python -m pytest`：**65 个通过**，约 4 分钟（本地 WSL，Python 3.12，phono3py 3.30.0；11-04）。

| 文件 | 测什么 |
|---|---|
| `test_core.py`（20） | 记录在成功 / 出错时都落盘；参数覆盖必须在范围内并写理由；`result_hash` 忽略只影响执行的参数；结构 ID 对平移和原子顺序不变；注册表和集合不可改、能发现篡改；审批单生命周期、批准后改 plan 被拒；CLI 在非 TTY 或 `MGA_ACTOR=agent` 时拒绝批准 |
| `test_kappa.py`（7） | 参考表 103 行；用参考值当预测时 SRME = 0；虚频 / 对称性改变 / q 网格不符被 censor 而不是崩溃；**EMT 铝的完整协议**（弛豫 → FC2 → FC3 → Wigner κ，确认 phono3py 3.30.0 的 MS-SMM19 输出键）；固定好的 settings 原样使用 |
| `test_tools.py`（12） | 导入 → 过滤 → 去重 → 泄漏检查（包括以 NaCl 为父结构的 CsCl 后代）；103 个参考结构带 phononDB settings；dry_run 什么也不提交；**用假集群（进程内 `run_locally` + 内存 JobStore）跑通 live 的 弛豫 → 取回 → 派生集合 → κ → 缓存命中 → replay**；两个「模型」在同一 settings 下 SRD = 0；MatterGen 取回；VASP 预览的 INCAR、成本估计；**VASP 审批单在批准前不能提交，dry_run 下不被消耗，用过一次后不能再用** |
| `test_vasp_flows.py`（6） | 三个 flow 里**每一个 VASP 步骤**（包括嵌套 flow 和作为参数传进去的位移 Maker）都是 PBE_54、ENCUT 520、没有 `GGA`、声子 flow 统一 `ISPIN=1`；固定 atomate2 的上游默认值 |
| `test_phono3py_fd.py`（6） | 位移计划逐阶选便宜的方式（SiC：FC3 随机 12 个，FC2 系统 2 个）；同种子可复现、数目和计划不符时报错；VASP 排序后的超胞和 POSCAR 原子顺序一致、置换可逆；flow 里所有 VASP 步骤是 PBE_54 + ISPIN 1，审批的计划随 flow 传下去；**用 EMT 代替 VASP 跑完 力 → FC → κ**（auto 和 systematic 两种），排序映射后的 fc2/fc3.hdf5 与原始顺序一致 |
| `test_ladder.py`（7） | Slack 公式的标度；碰撞矩阵内存估计；合成阶梯表上的分数链、固定规则 / 随机选择守预算、`ks_for_budget`、Spearman；**EMT 铝 L0 描述符 + L1 缓存 → L2 读回，κ_RTA 与 L1 相差 < 1e-6**；**EMT 铝 L3：FC4 拟合（留出集 RMSE 按阶下降）、ShengBTE 文件读回误差 < 1e-8、R 都是弛豫后原胞的格矢**；三个新工具 + 账本 + 基线用假集群端到端跑通 |
| `test_servers.py`（7） | 四个 server 通过真实 MCP 协议（进程内 `mcp.Client`）列工具和调用；没有任何批准类工具；错误带类型和消息返回给 agent；通过 MCP 发 VASP 请求后停在 pending |

另外：全部 34 个源文件按 Python 3.10 语法解析通过（集群 worker 环境的 Python 版本未定）；worker 端模块的顶层依赖只有 numpy / ase / jobflow / pymatgen。

**没有测过**：MongoDB、jobflow-remote 与真实集群、VASP 实际运行、任何真实 MLIP（MACE / DPA / NequIP / NEP）和 MatterGen、MP API。这些要在 §8 的准备做完后，用 1–2 个结构逐个验证。
