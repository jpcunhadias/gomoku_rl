from cli.utils import get_config_and_override
from train.self_play import run_selfplay_pipeline
from utils.paths import cycle_paths, save_config
from utils.seeding import set_global_seed


def main() -> None:
    """Run a single self-play pipeline for a specific cycle."""
    cfg, _, args = get_config_and_override(
        description="Run a self-play pipeline for a specific cycle."
    )

    print(f"[Self-Play] Starting run for cycle {args.cycle}")

    set_global_seed(cfg.seed)

    # Save the resolved config
    paths = cycle_paths(args.cycle)
    save_config(cfg, paths["config"])

    print(
        "[SP config]",
        "uniform_root_p=",
        cfg.sp_uniform_root_p,
        "block_opening_repeats=",
        cfg.sp_block_opening_repeats,
        "opening_memory_size=",
        cfg.opening_memory_size,
        "restart_cap_fraction=",
        cfg.restart_cap_fraction,
        "dirichlet_epsilon_root=",
        cfg.dirichlet_epsilon_root,
        "tau_early_plies=",
        getattr(cfg, "tau_early_plies", None),
    )

    run_selfplay_pipeline(
        config=cfg,
        load_checkpoint=True,  # Always load the latest policy
    )

    print(f"[Self-Play] Cycle {args.cycle} finished.")


if __name__ == "__main__":
    main()
