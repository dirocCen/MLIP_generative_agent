"""All parameter schemas, with the source of every default (DESIGN §4.2 layer 1).

Conventions
-----------
- ``source`` names the document/code a default comes from. Defaults without a
  reliable source say ``heuristic`` and how they will be validated; nothing here
  pretends to have a citation it does not have.
- Fields that cannot be overridden are typed as a single-value ``Literal``.
- ``affects_result=False`` marks execution knobs excluded from the cache hash.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from mga.params import P, ParamSpec

# --- sources -----------------------------------------------------------------
KSRME = (
    "matbench-discovery KappaSettings, protocol phonondb-v1 "
    "(commit 2b7bff9, 2026-10-19; Pota et al. arXiv:2408.00755)"
)
KSRME_COND = "matbench-discovery thermal_conductivity.calculate_conductivity (commit 2b7bff9)"
# The committee's own leaderboard kappa runs override the protocol's 0.01 A displacement
KSRME_COMMITTEE = (
    "matbench-discovery models/*/*.yml kappa settings of DPA-4.1-Max-OAM, TECE-OAM-RRA-1.0, PET-OAM-XL, "
    "NequIP-OAM-L (displacement 0.03 A; commit c70947f); also the phononDB-LTC DFT value used at L4b. "
    "MACE-MPA-0's leaderboard run used 0.01 A (override for a like-for-like kappa_SRME of MACE)"
)
KSRME_IMAG = "matbench-discovery phonons.IMAGINARY_FREQ_THRESHOLD (commit 2b7bff9)"
HEUR_PHDB = (
    "heuristic: calibrated on phononDB-PBE choices for light-element entries "
    "(docs/mcp_servers_plan.md §2.2); validate with submit_mlip_convergence"
)
MP_RELAX = "pymatgen MPRelaxSet (via atomate2 MPGGARelaxMaker)"
MP_STATIC = "pymatgen MPStaticSet (via atomate2 MPGGAStaticMaker)"
DESIGN_14 = "DESIGN §1.4: MP-compatible PBE with tightened numerics for force constants"
EXEC = "execution only (does not change results)"

Temperature = Annotated[float, Field(gt=0, le=3000)]


# --- MLIP ----------------------------------------------------------------------
class MLIPRelaxParams(ParamSpec):
    """Relaxation before phonons, following the kappa_SRME protocol."""

    optimizer: Literal["FIRE", "FIRE2", "BFGS", "LBFGS"] = P("FIRE", KSRME)
    cell_filter: Literal["FrechetCellFilter", "ExpCellFilter", "none"] = P(
        "FrechetCellFilter", KSRME
    )
    fmax: float = P(1e-4, KSRME, gt=0, le=0.1, description="eV/Angstrom")
    max_steps: int = P(300, KSRME, ge=1, le=5000)
    relaxation_mode: Literal["single-stage", "two-stage"] = P(
        "single-stage",
        KSRME + "; two-stage = constrained then unconstrained, keep stage 1 if the space group changes",
    )
    enforce_symmetry: bool = P(True, KSRME + " (FixSymmetry, no-tilt cell mask)")
    relax_symprec: float = P(0.01, KSRME + " (FixSymmetry symprec)", gt=0, le=0.2)
    symprec: float = P(1e-5, KSRME + " (space-group detection; spglib here)", gt=0, le=0.1)
    dtype: Literal["model_default", "float64", "float32"] = P(
        "model_default",
        "configs/models.yaml per model, resolved at submission (decided 11-04): float64 by default "
        "(k_SRME runs MACE in float64); PET-OAM-XL float32 (42-rotation symmetrizer 45 s -> 4.7 s per "
        "64-atom force call on L40S, max force change 1.1e-5 eV/A; checked on c-BN in S0)",
    )
    chunk_size: int = P(64, EXEC, affects_result=False, ge=1, le=2048)


class MLIPPhononKappaParams(ParamSpec):
    """FC2 -> stability gate -> FC3 -> Wigner-RTA kappa, kappa_SRME protocol."""

    displacement: float = P(0.03, KSRME_COMMITTEE, ge=0.001, le=0.1, description="Angstrom")
    is_plusminus: Literal["auto", "true", "false"] = P("auto", KSRME)
    symprec: float = P(1e-5, KSRME, gt=0, le=0.1)
    temperatures: tuple[Temperature, ...] = P((300.0,), KSRME, min_length=1, max_length=20)
    use_reference_settings: bool = P(
        True,
        "kappa_SRME: phononDB entries carry their own fc2/fc3 supercells and q-mesh; "
        "used whenever a structure has them",
    )
    fc3_min_width: float = P(8.0, HEUR_PHDB, ge=4.0, le=25.0, description="Angstrom")
    fc2_min_width: float = P(14.0, HEUR_PHDB, ge=6.0, le=40.0, description="Angstrom")
    q_mesh_length: float = P(40.0, HEUR_PHDB, ge=10.0, le=150.0, description="Angstrom")
    is_isotope: bool = P(True, KSRME_COND)
    boundary_mfp_um: float = P(1e6, KSRME_COND, gt=0)
    transport: Literal["MS-SMM19"] = P(
        "MS-SMM19", KSRME_COND + "; requires phono3py==3.30.0 on the worker"
    )
    imaginary_threshold_thz: float = P(-0.01, KSRME_IMAG, le=0, ge=-1.0)
    kappa_if_symmetry_broken: bool = P(False, KSRME + " (conductivity_broken_symm)")
    dtype: Literal["model_default", "float64", "float32"] = P("model_default", "see MLIPRelaxParams.dtype")
    chunk_size: int = P(4, EXEC, affects_result=False, ge=1, le=256)


ConvergenceTarget = Literal["fc3_min_width", "fc2_min_width", "q_mesh_length", "displacement"]


# --- multi-fidelity ladder (plan §9.3; workflows/ladder.py) ---------------------------
LADDER = "heuristic (ladder rung, plan §9.3): a ranking/escalation signal, validated against L1/L4"


class MLIPFc2DescriptorParams(ParamSpec):
    """L0: FC2-only descriptors. Supercells, q-mesh, displacement and the imaginary-mode gate
    come from MLIPPhononKappaParams, so L0 and L1 see the same harmonic phonons."""

    temperature: Temperature = P(300.0, LADDER)
    weighted_phase_space: bool = P(
        True, "phono3py JointDos with temperature (weighted JDOS = 3-phonon phase space with occupations)"
    )
    gruneisen: bool = P(True, LADDER + "; mode Grueneisen by +-volume_strain (phonopy PhonopyGruneisen), feeds the Slack estimate")
    volume_strain: float = P(0.01, "heuristic: +-1 % volume, internal coordinates relaxed at fixed cell", gt=0.001, le=0.05)


class MLIPLbteParams(ParamSpec):
    """L2: Wigner kappa with the direct LBTE solution (phono3py, MS-SMM19) on the L1 force constants."""

    temperatures: tuple[Temperature, ...] = P((300.0,), KSRME, min_length=1, max_length=5)
    max_collision_matrix_gb: float = P(
        32.0, "execution limit: the q-mesh is reduced until the collision matrix fits; the reduction is recorded",
        gt=0, le=1000,
    )
    min_mesh_scale: float = P(0.5, "heuristic: below half the L1 mesh the LBTE/RTA ratio is not trusted", gt=0.1, le=1.0)
    pinv_cutoff: float = P(1e-8, "phono3py default (--pinv-cutoff)", gt=0, le=1e-3)


class MLIPFc4Params(ParamSpec):
    """L3: FC2/FC3/FC4 from random displacements + symfc, ShengBTE/FourPhonon 3ph vs 3+4ph."""

    supercell_min_width: float = P(8.0, HEUR_PHDB + " (same as fc3_min_width; diagonal in the primitive basis)", ge=5.0, le=20.0)
    displacement: float = P(
        0.03, "heuristic: random-displacement amplitude commonly used with symfc / pypolymlp fits; "
        "large enough for the quartic term to show in the forces", ge=0.005, le=0.1, description="Angstrom")
    fc3_cutoff: float = P(5.0, "heuristic: symfc cutoff for FC3; validate by cutoff convergence on a top candidate", ge=2.0, le=12.0)
    fc4_shells: int = P(
        2, "heuristic: symfc FC4 cutoff midway between the n-th and (n+1)-th neighbour shell (replaced a fixed 3.0 A "
        "on 11-07: c-BN's 3rd shell sits at 2.997 A, so a fixed radius flips with each model's lattice constant)",
        ge=1, le=4)
    fc4_shell_gap: float = P(0.1, "heuristic: distances closer than this x the nearest-neighbour distance are one shell",
                             gt=0, le=0.5)
    snapshot_factor: float = P(3.0, "heuristic: snapshots x 3N forces >= factor x number of symfc basis functions", ge=1.0, le=20.0)
    min_snapshots: int = P(20, "heuristic", ge=2, le=5000)
    random_seed: int = P(0, "reproducibility")
    max_fc4_gb: float = P(
        64.0, EXEC + "; refuse when symfc's FC4 compression matrix (row pointer over n_prim N^3 81 elements) would "
        "exceed it; the job needs >= 2x this in host memory (GpuResources.mem)", affects_result=False, gt=0, le=500)
    temperature: Temperature = P(300.0, LADDER)
    q_mesh: tuple[int, int, int] | None = P(None, "None: the L1 q-mesh; FourPhonon 4ph scattering cost grows steeply with it")
    scalebroad: float = P(1.0, "ShengBTE manual default", gt=0, le=10)
    num_sample_4ph: int = P(
        100000, "FourPhonon manual / Example-Si-sampling: 4ph phase space and RTA rates from this many sampled "
        "processes per mode (Guo et al., npj Comput. Mater. 2023); <= 0 = exact, which is impractical: the "
        "exact Example-Si (16^3) did not finish in 12 h on 128 cores (job 960801)", ge=-1, le=10_000_000)
    is_isotope: bool = P(True, KSRME_COND)
    iterative: bool = P(False, "RTA for both 3ph and 3+4ph, so the ratio compares like with like")
    fourphonon_cmd: str | None = P(
        None, EXEC + "; e.g. 'srun /path/ShengBTE' (FourPhonon build); default $MGA_FOURPHONON; unset = write inputs only",
        affects_result=False)
    fourphonon_timeout_s: int = P(86400, EXEC, affects_result=False, ge=60)
    chunk_size: int = P(1, EXEC + "; structures per job (an L3 structure can take hours)", affects_result=False, ge=1, le=50)


# --- structure matching ------------------------------------------------------------
PMG_SM = "pymatgen StructureMatcher default"


class StructureMatcherParams(ParamSpec):
    """Used by deduplicate and check_novelty."""

    ltol: float = P(0.2, PMG_SM, gt=0, le=1.0)
    stol: float = P(0.3, PMG_SM, gt=0, le=1.0)
    angle_tol: float = P(5.0, PMG_SM, gt=0, le=30.0, description="degrees")
    primitive_cell: bool = P(True, PMG_SM)
    scale: bool = P(True, PMG_SM + " (volume-normalised, so MLIP- and DFT-relaxed cells still match)")
    attempt_supercell: bool = P(False, PMG_SM)


# --- MatterGen -------------------------------------------------------------------
class MatterGenParams(ParamSpec):
    pretrained_name: Literal["chemical_system", "mattergen_base"] = P(
        "chemical_system",
        "DESIGN §4.1: chemical-system checkpoint; other properties are post-filters",
    )
    diffusion_guidance_factor: float = P(
        2.0, "MatterGen README conditional-generation example; TODO verify on datalab", ge=0, le=10
    )
    batch_size: int = P(64, "MatterGen default; TODO verify on datalab", ge=1, le=512)
    record_trajectories: Literal[False] = P(False, "storage: trajectories are not needed")


# --- VASP --------------------------------------------------------------------------
class VaspCommonParams(ParamSpec):
    potcar_functional: Literal["PBE_54"] = P(
        "PBE_54", "dirona ~/vasp/potpaw/potpaw_PBE (docs/mcp_servers_plan.md §2.1); DESIGN §2.2"
    )
    encut: float = P(520.0, MP_RELAX + "; " + DESIGN_14, ge=520.0, le=1000.0)
    ediff: float = P(1e-8, DESIGN_14, gt=0, le=1e-5)
    prec: Literal["Accurate"] = P("Accurate", MP_RELAX)
    lreal: Literal[False] = P(False, DESIGN_14 + " (real-space projection off for forces)")
    ismear: Literal[0, -5] = P(0, "heuristic: Gaussian smearing for insulators with displaced cells")
    sigma: float = P(0.01, "heuristic: small smearing for insulators", gt=0, le=0.2)
    reciprocal_density: int = P(
        100, MP_STATIC + " default; used for all calcs for consistency", ge=50, le=1000
    )
    algo: Literal["Normal", "Fast", "All"] = P(
        "Normal", "atomate2 PhononDisplacementMaker (ALGO Normal)"
    )


class VaspRelaxStaticParams(VaspCommonParams):
    ediffg: float = P(
        -1e-3, "heuristic: tight force criterion before phonons (eV/Angstrom)", ge=-0.05, le=-1e-4
    )
    double_relax: bool = P(True, "atomate2 DoubleRelaxMaker (MP practice: relax twice)")


class VaspPhononParams(VaspRelaxStaticParams):
    displacement: float = P(
        0.01, "phonopy / atomate2 PhononMaker default", ge=0.005, le=0.05, description="Angstrom"
    )
    supercell_min_length: float = P(
        10.0, "starting point; fixed by a 10 vs 14 Angstrom convergence test before mass submission (plan §9.3)", ge=8, le=25
    )
    symprec: float = P(1e-3, "atomate2 PhononMaker default", gt=0, le=0.1)
    ispin: Literal[1, 2] = P(
        1,
        "heuristic: closed-shell light-element insulators; applied to relax, static and displacements. "
        "Also avoids atomate2 generate_phonon_displacements refusing non-zero magmoms "
        "with primitive_matrix='auto'. preview flags odd-electron cells",
    )
    born_charges: Literal[False] = P(
        False, "not done (user decision 10-20): matches the noNAC kappa_SRME reference (plan §9.1)"
    )


PH3_AUTO = "phono3py 3.30.0 generate_displacements(number_of_snapshots='auto')"


class VaspPhono3pyParams(VaspRelaxStaticParams):
    """DFT FC2 + FC3 by finite displacements, kappa with the kappa_SRME settings
    (workflows/phono3py_fd.py, plan §9.2)."""

    ispin: Literal[1, 2] = P(1, "heuristic: closed-shell light-element insulators (see VaspPhononParams.ispin)")
    symprec: float = P(1e-5, KSRME + "; same as the MLIP kappa tool", gt=0, le=0.1)
    displacement_mode: Literal["auto", "systematic", "random"] = P(
        "auto",
        "heuristic: per order, the cheaper of systematic (traditional solver) and random + symfc; "
        "the request shows both counts",
    )
    displacement: float = P(
        0.03, "phono3py default distance (systematic and random); phononDB-LTC random sets use 0.03",
        ge=0.01, le=0.06, description="Angstrom",
    )
    fit_cutoffs: tuple[Annotated[float, Field(ge=2.0, le=15.0)], ...] = P(
        (4.0, 5.0),
        "heuristic: symfc FC3 pair cutoffs; snapshots are sized for the largest, kappa at the smallest is "
        "the cutoff-convergence check (plan §9.2 basis-size table)",
        min_length=1, max_length=4,
    )
    fc3_snapshot_factor: float = P(4.0, PH3_AUTO + ": symfc minimum x 4 for FC3", ge=1.5, le=20.0)
    fc2_snapshot_factor: float = P(2.0, "phono3py 3.30.0 generate_fc2_displacements('auto'): symfc minimum x 2", ge=1.0, le=20.0)
    learning_curve_fraction: float = P(
        0.5, "heuristic: kappa from this fraction of the FC3 snapshots is the snapshot-count check", gt=0.1, lt=1.0
    )
    random_seed: int = P(1103, "fixed so the approved displacement set is reproducible", ge=0)
    fc3_min_width: float = P(8.0, HEUR_PHDB + "; same as MLIPPhononKappaParams", ge=4.0, le=25.0, description="Angstrom")
    fc2_min_width: float = P(14.0, HEUR_PHDB + "; same as MLIPPhononKappaParams", ge=6.0, le=40.0, description="Angstrom")
    fc2_supercell_shape: Literal["symmetric", "any"] = P(
        "symmetric", "decided 11-07: the MLIP path takes the smallest FC2 cell of any shape (GPU memory), but random FC2 "
        "displacements at the symfc minimum x 2 gave imaginary modes in a symmetry-broken cell (EMT Cu3Au); "
        "DFT keeps the point group")
    q_mesh_length: float = P(40.0, HEUR_PHDB + "; same as MLIPPhononKappaParams", ge=10.0, le=150.0, description="Angstrom")
    temperatures: tuple[Temperature, ...] = P((300.0,), KSRME, min_length=1, max_length=20)
    is_isotope: bool = P(True, KSRME_COND)
    boundary_mfp_um: float = P(1e6, KSRME_COND, gt=0)
    transport: Literal["MS-SMM19"] = P("MS-SMM19", KSRME_COND + "; requires phono3py==3.30.0 on the worker")
    imaginary_threshold_thz: float = P(-0.01, KSRME_IMAG, le=0, ge=-1.0)
    born_charges: Literal[False] = P(False, "not done (user decision 10-20): noNAC, as kappa_SRME")


class VaspConvergenceParams(VaspCommonParams):
    encut_values: tuple[Annotated[float, Field(ge=400, le=1200)], ...] = P(
        (520.0, 600.0, 700.0, 800.0), "convergence scan (DESIGN §2.2)", min_length=2, max_length=8
    )
    reciprocal_density_values: tuple[Annotated[int, Field(ge=20, le=2000)], ...] = P(
        (64, 100, 200, 400), "convergence scan (DESIGN §2.2)", min_length=2, max_length=8
    )


VASP_CALC_SPECS: dict[str, type[VaspCommonParams]] = {
    "relax_static": VaspRelaxStaticParams,
    "phonon_fd": VaspPhononParams,
    "phono3py_fd": VaspPhono3pyParams,
    "convergence": VaspConvergenceParams,
}


# --- resources (execution only) ----------------------------------------------------
class VaspResources(ParamSpec):
    partition: Literal["normal", "perform"] = P("normal", "dirona partitions, DESIGN §2.2", affects_result=False)
    ntasks: int = P(16, "DESIGN §2.2: 8-16 cores per job; calibrate in the core-count test", affects_result=False, ge=4, le=128)
    ncore: int = P(4, "heuristic; calibrate in the core-count test", affects_result=False, ge=1, le=64)
    kpar: int = P(1, "heuristic; calibrate in the core-count test", affects_result=False, ge=1, le=16)

    @model_validator(mode="after")
    def _divisible(self) -> VaspResources:
        if self.ntasks % (self.ncore * self.kpar):
            raise ValueError(f"ntasks={self.ntasks} must be divisible by ncore*kpar={self.ncore * self.kpar}")
        return self


class GpuResources(ParamSpec):
    partition: Literal["GPU-l40s", "GPU-a100", "GPU-a100s", "GPU-a40", "GPU-rtx6000"] = P(
        "GPU-l40s", "datalab default partition, DESIGN §2.1", affects_result=False
    )
    gpus: int = P(1, EXEC, affects_result=False, ge=1, le=8)
    cpus_per_task: int = P(8, EXEC, affects_result=False, ge=1, le=64)
    time: str = P("1-00:00:00", "datalab limit is 7 days, DESIGN §2.1", affects_result=False, pattern=r"^\d+-\d{2}:\d{2}:\d{2}$")
    nodelist: str | None = P(
        None, EXEC + "; one node, e.g. 'a-a100-o-1' (A100-SXM4-80GB): the GPU partitions mix 40 and 80 GB cards "
        "and 512-atom supercells do not fit 48 GB (configs/models.yaml)", affects_result=False, pattern=r"^[A-Za-z0-9.-]+$")
    exclude_nodes: str | None = P(None, EXEC + "; comma-separated node names", affects_result=False,
                                  pattern=r"^[A-Za-z0-9.,-]+$")
    mem: str | None = P(None, EXEC + "; host memory per job, Slurm --mem (e.g. '200G'); unset = the partition "
                        "default per CPU. L3 needs >= 2x the FC4 estimate (MLIPFc4Params.max_fc4_gb)",
                        affects_result=False, pattern=r"^\d+[MG]$")
