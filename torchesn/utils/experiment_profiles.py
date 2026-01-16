from __future__ import annotations

from types import SimpleNamespace

KS_KEYS = [
    "L",
    "Q",
    "dt",
    "mu",
    "lam",
    "burn_in",
    "total_steps",
]
RC_KEYS = [
    "g",
    "l",
    "hidden_size",
    "spectral_radius",
    "sigma",
    "kappa",
    "density",
    "leaking_rate",
    "readout_training",
    "lambda_reg",
    "readout_features",
]
EVAL_KEYS = [
    "T_train",
    "T_eval",
    "K",
    "tau",
    "epsilon",
    "n_trials",
]
CONFIG_KEYS = KS_KEYS + RC_KEYS + EVAL_KEYS

PAPER_COMMON = {
    "hidden_size": 5000,
    "T_train": 70000,
    "spectral_radius": 0.6,
    "sigma": 1.0,
    "kappa": 3,
    "readout_features": "linear_and_square",
    "readout_training": "cholesky",
    "K": 30,
    "tau": 1000,
    "epsilon": 10,
    "n_trials": 10,
}

FIGURE_BASE_DEFAULTS = {
    "L": 22.0,
    "Q": 64,
    "dt": 0.25,
    "mu": 0.0,
    "lam": 100.0,
    "burn_in": 0,
    "g": 1,
    "l": 0,
    "hidden_size": 500,
    "spectral_radius": 0.6,
    "sigma": 1.0,
    "kappa": 3,
    "density": 1.0,
    "leaking_rate": 1.0,
    "readout_training": "cholesky",
    "lambda_reg": 1e-4,
    "readout_features": "linear_and_square",
    "T_train": 1000,
    "T_eval": 500,
    "epsilon": 10,
    "tau": 100,
    "K": 2,
    "n_trials": 1,
}

EVAL_BASE_DEFAULTS = {
    "L": 200.0,
    "Q": 512,
    "dt": 0.25,
    "mu": 0.01,
    "lam": 100.0,
    "burn_in": 1000,
    "T_train": 70000,
    "T_eval": None,
    "K": 30,
    "tau": 1000,
    "epsilon": 10,
    "n_trials": 10,
    "g": 64,
    "l": 6,
    "hidden_size": 5000,
    "spectral_radius": 0.6,
    "sigma": 1.0,
    "kappa": 3,
    "density": 1.0,
    "leaking_rate": 1.0,
    "lambda_reg": 1e-4,
    "readout_training": "cholesky",
    "readout_features": "linear_and_square",
}

PAPER_PROFILES = {
    "fig2_single": {
        "L": 22.0,
        "Q": 64,
        "dt": 0.25,
        "mu": 0.0,
        "lam": 100.0,
        "g": 1,
        "l": 0,
    },
    "fig4_parallel": {
        "L": 200.0,
        "Q": 512,
        "dt": 0.25,
        "mu": 0.01,
        "lam": 100.0,
        "g": 64,
        "l": 6,
    },
    "fig5_scaling": {
        "dt": 0.25,
        "mu": 0.01,
        "lam": 100.0,
        "fig5_a_L_values": [200, 400, 800, 1600],
        "fig5_a_g_values": [64, 128, 256, 512],
        "fig5_b_L": 200.0,
        "fig5_b_Q": 512,
        "fig5_b_g_values": [1, 2, 4, 8, 16, 32, 64],
    },
    "fig6_shared_weights": {
        "L": 200.0,
        "Q": 512,
        "dt": 0.25,
        "lam": 100.0,
        "g": 64,
        "l": 6,
        "mu_values": [0.0, 0.01],
    },
    "eval": {
        "L": 200.0,
        "Q": 512,
        "dt": 0.25,
        "mu": 0.01,
        "lam": 100.0,
        "g": 64,
        "l": 6,
        "burn_in": 1000,
    },
}

QUICK_PROFILES = {
    "fig2_single": {
        "Q": 64,
        "g": 1,
        "l": 0,
        "hidden_size": 150,
        "T_train": 400,
        "T_eval": 200,
        "K": 2,
        "tau": 60,
        "epsilon": 10,
        "n_trials": 1,
        "lambda_reg_min": 1e-2,
    },
    "fig4_parallel": {
        "Q": 128,
        "g": 8,
        "hidden_size": 150,
        "T_train": 400,
        "T_eval": 200,
        "K": 2,
        "tau": 60,
        "epsilon": 10,
        "n_trials": 1,
        "lambda_reg_min": 1e-2,
    },
    "fig5_scaling": {
        "fig5_a_L_values": [100, 200],
        "fig5_a_g_values": [8, 16],
        "fig5_b_Q": 64,
        "fig5_b_g_values": [1, 2, 4, 8],
        "hidden_size": 150,
        "T_train": 400,
        "T_eval": 200,
        "K": 2,
        "tau": 60,
        "epsilon": 10,
        "n_trials": 1,
        "lambda_reg_min": 1e-2,
    },
    "fig6_shared_weights": {
        "Q": 128,
        "g": 8,
        "hidden_size": 150,
        "T_train": 400,
        "T_eval": 200,
        "K": 2,
        "tau": 60,
        "epsilon": 10,
        "n_trials": 1,
        "lambda_reg_min": 1e-2,
    },
    "eval": {
        "L": 22.0,
        "Q": 64,
        "dt": 0.25,
        "mu": 0.0,
        "lam": 100.0,
        "burn_in": 200,
        "g": 8,
        "l": 2,
        "hidden_size": 200,
        "spectral_radius": 0.6,
        "T_train": 400,
        "K": 2,
        "tau": 50,
        "epsilon": 10,
        "n_trials": 1,
        "lambda_reg_min": 1e-2,
    },
}

PAPER_QUICK_PROFILES = {
    "fig2_single": QUICK_PROFILES["fig2_single"],
    "fig4_parallel": QUICK_PROFILES["fig4_parallel"],
    "fig5_scaling": QUICK_PROFILES["fig5_scaling"],
    "fig6_shared_weights": QUICK_PROFILES["fig6_shared_weights"],
    "eval": {
        "Q": 128,
        "g": 8,
        "hidden_size": 200,
        "T_train": 400,
        "K": 2,
        "tau": 50,
        "epsilon": 10,
        "n_trials": 1,
        "lambda_reg_min": 1e-2,
    },
}


def _apply_defaults(
    cfg: dict,
    defaults: dict,
    user_provided: set[str],
    skip_keys: set[str] | None = None,
) -> None:
    skip_keys = skip_keys or set()
    for key, value in defaults.items():
        if key in skip_keys:
            continue
        if key not in user_provided and cfg.get(key) is None:
            cfg[key] = value


def _resolve_value(cfg: dict, key: str, fallback) -> None:
    if cfg.get(key) is None:
        cfg[key] = fallback


def resolve_experiment_config(args, mode: str) -> dict:
    if mode not in PAPER_PROFILES:
        raise ValueError(f"Unknown mode '{mode}'.")

    user_provided = {key for key in CONFIG_KEYS if getattr(args, key, None) is not None}
    cfg = {key: getattr(args, key, None) for key in CONFIG_KEYS}

    if args.paper_defaults:
        _apply_defaults(cfg, PAPER_COMMON, user_provided)
        _apply_defaults(cfg, PAPER_PROFILES[mode], user_provided)
        skip_keys = {"density", "T_eval"}
        if mode == "eval":
            _apply_defaults(cfg, EVAL_BASE_DEFAULTS, user_provided, skip_keys=skip_keys)
        else:
            _apply_defaults(cfg, FIGURE_BASE_DEFAULTS, user_provided, skip_keys=skip_keys)
    else:
        if mode == "eval":
            _apply_defaults(cfg, EVAL_BASE_DEFAULTS, user_provided)
        else:
            _apply_defaults(cfg, FIGURE_BASE_DEFAULTS, user_provided)

    if mode == "fig6_shared_weights":
        mu_values = getattr(args, "mu_values", None)
        if mu_values is None:
            mu_values = cfg.get("mu_values")
        if mu_values is None:
            mu_values = [cfg["mu"]] if cfg.get("mu") is not None else [0.0]
        cfg["mu_values"] = mu_values
        if cfg.get("mu") is None:
            cfg["mu"] = mu_values[0]

    if mode == "fig5_scaling":
        fig5_profile = PAPER_PROFILES["fig5_scaling"]
        if cfg.get("fig5_a_L_values") is None:
            cfg["fig5_a_L_values"] = fig5_profile["fig5_a_L_values"]
        if cfg.get("fig5_a_g_values") is None:
            cfg["fig5_a_g_values"] = fig5_profile["fig5_a_g_values"]
        if cfg.get("fig5_b_L") is None:
            cfg["fig5_b_L"] = fig5_profile["fig5_b_L"]
        if cfg.get("fig5_b_Q") is None:
            cfg["fig5_b_Q"] = fig5_profile["fig5_b_Q"]
        if cfg.get("fig5_b_g_values") is None:
            cfg["fig5_b_g_values"] = fig5_profile["fig5_b_g_values"]

        if getattr(args, "L", None) is not None:
            cfg["fig5_a_L_values"] = [args.L]
            cfg["fig5_b_L"] = args.L
        if getattr(args, "g", None) is not None:
            cfg["fig5_a_g_values"] = [args.g]
            cfg["fig5_b_g_values"] = [args.g]
        if getattr(args, "Q", None) is not None:
            cfg["fig5_b_Q"] = args.Q

    if args.quick:
        quick_profile = (
            PAPER_QUICK_PROFILES[mode] if args.paper_defaults else QUICK_PROFILES[mode]
        )
        for key, value in quick_profile.items():
            if key.endswith("_min"):
                continue
            if key.startswith("fig5_"):
                if mode == "fig5_scaling" and {"L", "g", "Q"}.isdisjoint(user_provided):
                    cfg[key] = value
            elif key not in user_provided:
                cfg[key] = value

        min_lambda = quick_profile.get("lambda_reg_min")
        if min_lambda is not None:
            if cfg.get("lambda_reg") is None:
                cfg["lambda_reg"] = min_lambda
            else:
                cfg["lambda_reg"] = max(cfg["lambda_reg"], min_lambda)

        if mode == "fig2_single" and "g" not in user_provided:
            cfg["g"] = 1
        if mode == "fig2_single" and "l" not in user_provided:
            cfg["l"] = 0

        if mode != "fig5_scaling" and cfg["Q"] % cfg["g"] != 0:
            if "Q" not in user_provided:
                cfg["Q"] = cfg["g"] * (cfg["Q"] // cfg["g"])
            else:
                raise ValueError("Q must be divisible by g.")

    _resolve_value(cfg, "burn_in", 0)
    _resolve_value(cfg, "leaking_rate", 1.0)
    _resolve_value(cfg, "readout_training", "cholesky")
    _resolve_value(cfg, "lambda_reg", 1e-4)
    _resolve_value(cfg, "readout_features", "linear_and_square")

    if mode != "fig5_scaling" and cfg["Q"] % cfg["g"] != 0:
        raise ValueError("Q must be divisible by g.")

    q_value = None if mode == "fig5_scaling" else cfg["Q"] // cfg["g"]

    required_eval = cfg["K"] * cfg["tau"] + cfg["epsilon"] + 1
    if cfg.get("T_eval") is None:
        cfg["T_eval"] = required_eval
    if cfg["T_eval"] < required_eval:
        raise ValueError("T_eval is shorter than required_eval.")

    if cfg.get("total_steps") is None:
        cfg["total_steps"] = cfg["burn_in"] + cfg["T_train"] + cfg["T_eval"]
    if cfg["total_steps"] < cfg["burn_in"] + cfg["T_train"] + cfg["T_eval"]:
        raise ValueError("total_steps is shorter than burn_in + T_train + T_eval.")

    density = cfg.get("density")
    effective_density = density
    if effective_density is None:
        effective_density = min(1.0, cfg["kappa"] / float(cfg["hidden_size"]))

    result = {
        "mode": mode,
        "paper_defaults": bool(args.paper_defaults),
        "quick": bool(args.quick),
        "ks": {
            "L": cfg["L"],
            "Q": cfg["Q"],
            "dt": cfg["dt"],
            "mu": cfg["mu"],
            "lam": cfg["lam"],
            "burn_in": cfg["burn_in"],
            "total_steps": cfg["total_steps"],
        },
        "rc": {
            "g": cfg["g"],
            "q": q_value,
            "l": cfg["l"],
            "hidden_size": cfg["hidden_size"],
            "spectral_radius": cfg["spectral_radius"],
            "sigma": cfg["sigma"],
            "kappa": cfg["kappa"],
            "density": density,
            "effective_density": effective_density,
            "leaking_rate": cfg["leaking_rate"],
            "lambda_reg": cfg["lambda_reg"],
            "readout_training": cfg["readout_training"],
            "readout_features": cfg["readout_features"],
        },
        "eval": {
            "T_train": cfg["T_train"],
            "T_eval": cfg["T_eval"],
            "K": cfg["K"],
            "tau": cfg["tau"],
            "epsilon": cfg["epsilon"],
            "n_trials": cfg["n_trials"],
        },
        "derived": {
            "required_eval": required_eval,
        },
    }

    if mode == "fig6_shared_weights":
        result["fig6"] = {"mu_values": cfg["mu_values"]}
    if mode == "fig5_scaling":
        result["fig5"] = {
            "a": {
                "L_values": cfg["fig5_a_L_values"],
                "g_values": cfg["fig5_a_g_values"],
            },
            "b": {
                "L": cfg["fig5_b_L"],
                "Q": cfg["fig5_b_Q"],
                "g_values": cfg["fig5_b_g_values"],
            },
        }

    return result


def config_to_namespace(cfg: dict) -> SimpleNamespace:
    fig5 = cfg.get("fig5", {})
    fig6 = cfg.get("fig6", {})
    return SimpleNamespace(
        mode=cfg["mode"],
        paper_defaults=cfg["paper_defaults"],
        quick=cfg["quick"],
        L=cfg["ks"]["L"],
        Q=cfg["ks"]["Q"],
        dt=cfg["ks"]["dt"],
        mu=cfg["ks"]["mu"],
        lam=cfg["ks"]["lam"],
        burn_in=cfg["ks"]["burn_in"],
        total_steps=cfg["ks"]["total_steps"],
        g=cfg["rc"]["g"],
        q=cfg["rc"]["q"],
        l=cfg["rc"]["l"],
        hidden_size=cfg["rc"]["hidden_size"],
        spectral_radius=cfg["rc"]["spectral_radius"],
        sigma=cfg["rc"]["sigma"],
        kappa=cfg["rc"]["kappa"],
        density=cfg["rc"]["effective_density"],
        leaking_rate=cfg["rc"]["leaking_rate"],
        lambda_reg=cfg["rc"]["lambda_reg"],
        readout_training=cfg["rc"]["readout_training"],
        readout_features=cfg["rc"]["readout_features"],
        T_train=cfg["eval"]["T_train"],
        T_eval=cfg["eval"]["T_eval"],
        K=cfg["eval"]["K"],
        tau=cfg["eval"]["tau"],
        epsilon=cfg["eval"]["epsilon"],
        n_trials=cfg["eval"]["n_trials"],
        fig5_a_L_values=fig5.get("a", {}).get("L_values"),
        fig5_a_g_values=fig5.get("a", {}).get("g_values"),
        fig5_b_L=fig5.get("b", {}).get("L"),
        fig5_b_Q=fig5.get("b", {}).get("Q"),
        fig5_b_g_values=fig5.get("b", {}).get("g_values"),
        mu_values=fig6.get("mu_values"),
    )
