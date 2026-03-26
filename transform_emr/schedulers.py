"""
schedulers.py
=============

Unified auxiliary loss weighting scheduler for Phase-1 (embedding) and Phase-2 (transformer) training.

LambdaScheduleController:
  - Reads auxiliary schedule from training config (which auxiliaries, when they start, ramp durations).
  - Implements frozen-fraction calibration: lambda_max = fraction_cap * main_loss / aux_loss (once, then fixed).
  - Applies optional linear ramp from 0 to lambda_max (ramp_epochs=1 means no ramp, immediate full power).
  - Supports optional plateau-based curriculum for Phase-2 stage transitions.
  - Single interface for both phases; both pass their phase-specific config.
"""


def linear_schedule(epoch: int, start_epoch: int, end_epoch: int, max_val: float) -> float:
    """
    Linearly ramp from 0 to `max_val` over [start_epoch, end_epoch].
    If end_epoch <= start_epoch, immediately returns max_val at start_epoch (no ramp).
    """
    if epoch < start_epoch:
        return 0.0
    if end_epoch <= start_epoch:  # ramp_epochs <= 1: immediate activation
        return max_val
    progress = min(max((epoch - start_epoch) / float(end_epoch - start_epoch), 0.0), 1.0)
    return max_val * progress


class LambdaScheduleController:
    """
    Unified scheduler for auxiliary loss weighting.

    Handles both Phase-1 (embedding) and Phase-2 (transformer) training with:
    - Frozen-fraction calibration: compute lambda_max once, keep it fixed.
    - Linear ramp (ramp_epochs=1 means immediate, no actual ramping).
    - Optional plateau-based curriculum for Phase-2 stage gating.

    Expected usage:
      - Once per epoch (after validation): call update(epoch, vl_main, aux_loss_dict)
      - Per batch: call get_lambdas(epoch) to retrieve current lambda values
      - Optional logging: call status_line(epoch)
    """

    def __init__(self, training_settings: dict, start_epoch: int = 0):
        """
        Initialize scheduler by reading auxiliary schedule from training config.

        Parameters
        ----------
        training_settings : dict
            Training config. Expected keys for Phase-1:
              - phase1_aux_fraction_caps: dict of {name: fraction}
              - phase1_dynamic_schedule: dict with mlm_ramp_epochs, dt_ramp_epochs
              - aux_max_fraction_default: fallback fraction cap
            For Phase-2:
              - phase2_aux_fraction_caps: dict of {name: fraction}
              - phase2_dynamic_schedule: dict with ce/dt/penalty/outcome ramp_epochs and plateau settings
              - aux_max_fraction_default: fallback fraction cap

        start_epoch : int
            Current training epoch (for checkpoint resume).
        """
        self.training_settings = training_settings
        self.start_epoch = int(start_epoch)
        self.default_fraction = float(training_settings.get("aux_max_fraction_default", 0.2))
        self.min_aux_loss = 1e-8
        self.max_lambda_clamp = 100.0

        # Internal state: {name: {start_epoch, ramp_epochs, fraction, lambda_max, ...}}
        self._auxiliaries = {}

        # Determine phase based on available config keys
        if "phase2_auxiliary_schedule" in training_settings or "phase2_aux_fraction_caps" in training_settings:
            self._init_phase2()
        else:
            self._init_phase1()

    def _init_phase1(self):
        """Initialize Phase-1 (embedding) configuration."""
        caps = self.training_settings.get("phase1_aux_fraction_caps", {})
        sched = self.training_settings.get("phase1_dynamic_schedule", {})

        # Phase-1 has MLM and DT, both starting at epoch 0 with ramp=1 (immediate)
        self._register_aux(
            name="mlm",
            start_epoch=self.start_epoch,
            ramp_epochs=max(1, int(sched.get("mlm_ramp_epochs", 1))),
            fraction=caps.get("mlm"),
        )
        self._register_aux(
            name="dt",
            start_epoch=self.start_epoch,
            ramp_epochs=max(1, int(sched.get("dt_ramp_epochs", 1))),
            fraction=caps.get("dt"),
        )

        self.dynamic_enabled = False
        self.warmup_complete_epoch = None

    def _init_phase2(self):
        """Initialize Phase-2 (transformer) configuration with optional plateau curriculum."""
        caps = self.training_settings.get("phase2_aux_fraction_caps", {})
        dyn_cfg = self.training_settings.get("phase2_dynamic_schedule", {})

        # CE and DT start immediately at epoch 0
        self._register_aux(
            name="ce",
            start_epoch=self.start_epoch,
            ramp_epochs=max(1, int(dyn_cfg.get("ce_ramp_epochs", 5))),
            fraction=caps.get("ce"),
        )
        self._register_aux(
            name="dt",
            start_epoch=self.start_epoch,
            ramp_epochs=max(1, int(dyn_cfg.get("dt_ramp_epochs", 5))),
            fraction=caps.get("dt"),
        )

        # Check if dynamic curriculum is enabled
        self.dynamic_enabled = bool(dyn_cfg.get("enabled", False))

        if self.dynamic_enabled:
            # Penalty and outcome are unlocked dynamically based on plateau detection
            self._register_aux(
                name="penalty",
                start_epoch=None,  # Unlocked later
                ramp_epochs=max(1, int(dyn_cfg.get("penalty_ramp_epochs", 5))),
                fraction=caps.get("penalty"),
            )
            self._register_aux(
                name="outcome",
                start_epoch=None,  # Unlocked later
                ramp_epochs=max(1, int(dyn_cfg.get("outcome_ramp_epochs", 5))),
                fraction=caps.get("outcome"),
            )

            # Plateau tracking state
            self.plateau_min_delta = float(dyn_cfg.get("plateau_min_delta", 1e-4))
            self.base_plateau_patience = max(1, int(dyn_cfg.get("base_plateau_patience", 3)))
            self.penalty_plateau_patience = max(1, int(dyn_cfg.get("penalty_plateau_patience", 3)))
            self.min_base_epochs = max(1, int(dyn_cfg.get("min_base_epochs", 5)))
            self.min_penalty_epochs = max(1, int(dyn_cfg.get("min_penalty_epochs", 5)))

            self.base_best = float("inf")
            self.base_bad_epochs = 0
            self.pen_best = float("inf")
            self.pen_bad_epochs = 0
            self.warmup_complete_epoch = None
        else:
            # Static schedule: penalty and outcome start at fixed epochs
            warmup_epochs = int(self.training_settings.get("warmup_epochs", 100))
            foundational_epochs = int(self.training_settings.get("foundational_epochs", 50))
            secondary_epochs = int((foundational_epochs + 1) // 2)

            self._register_aux(
                name="penalty",
                start_epoch=secondary_epochs,
                ramp_epochs=foundational_epochs,
                fraction=caps.get("penalty"),
            )
            self._register_aux(
                name="outcome",
                start_epoch=foundational_epochs,
                ramp_epochs=warmup_epochs - foundational_epochs,
                fraction=caps.get("outcome"),
            )
            self.warmup_complete_epoch = warmup_epochs

    def _register_aux(self, name: str, start_epoch: int | None, ramp_epochs: int, fraction: float | None):
        """Register an auxiliary component."""
        self._auxiliaries[name] = {
            "name": name,
            "start_epoch": start_epoch,
            "ramp_epochs": max(1, int(ramp_epochs)),
            "fraction": float(self.default_fraction if fraction is None else fraction),
            "lambda_max": None,  # Calibrated on first call
            "anchor_main_loss": None,
            "anchor_aux_loss": None,
        }

    def update(self, epoch: int, vl_main: float, **aux_losses):
        """
        Calibrate auxiliaries and check for plateau-based stage transitions.

        Parameters
        ----------
        epoch : int
            Current epoch.
        vl_main : float
            Validation main loss (BCE for both phases).
        **aux_losses : float
            Named keyword args for each auxiliary loss (e.g., vl_mlm_raw, vl_dt_raw, vl_ce_raw, ...).

        Returns
        -------
        list[str]
            Messages for logging (calibration events, stage transitions).
        """
        messages = []

        # Calibrate each auxiliary once
        for name, spec in self._auxiliaries.items():
            if name not in aux_losses:
                continue
            if spec["lambda_max"] is not None:
                continue  # Already calibrated

            vl_aux = float(aux_losses[name])
            if vl_aux > self.min_aux_loss:
                # Compute lambda_max = fraction * main_loss / aux_loss
                spec["lambda_max"] = (spec["fraction"] * vl_main) / max(vl_aux, self.min_aux_loss)
                spec["lambda_max"] = min(spec["lambda_max"], self.max_lambda_clamp)
                spec["anchor_main_loss"] = vl_main
                spec["anchor_aux_loss"] = vl_aux

                # For dynamic auxiliaries, unlock them now
                if spec["start_epoch"] is None:
                    spec["start_epoch"] = epoch
                    messages.append(f"[Scheduler]: {name} auxiliary unlocked at epoch {epoch} with λ_max={spec['lambda_max']:.4f}")
                else:
                    messages.append(f"[Scheduler]: {name} auxiliary calibrated at epoch {epoch} with λ_max={spec['lambda_max']:.4f}")

        # Handle plateau-based stage gating for Phase-2 dynamic curriculum
        if self.dynamic_enabled:
            messages.extend(self._update_dynamic_stages(epoch, vl_main, aux_losses))

        return messages

    def _update_dynamic_stages(self, epoch: int, vl_main: float, aux_losses: dict) -> list:
        """Check and advance dynamic stages in Phase-2 curriculum."""
        messages = []
        vl_ce_raw = float(aux_losses.get("vl_ce_raw", 0.0))
        vl_dt_raw = float(aux_losses.get("vl_dt_raw", 0.0))
        vl_pen_raw = float(aux_losses.get("vl_pen_raw", 0.0))

        # Stage 1: Check if base objectives (CE + DT) have plateaued
        base_metric = vl_main + vl_ce_raw + vl_dt_raw
        if self._auxiliaries["penalty"]["start_epoch"] is None:
            self.base_best, self.base_bad_epochs, base_plateau = self._check_plateau(
                base_metric, self.base_best, self.base_bad_epochs, self.plateau_min_delta, self.base_plateau_patience
            )
            trained_base_epochs = epoch - self.start_epoch + 1
            if trained_base_epochs >= self.min_base_epochs and base_plateau:
                # Unlock penalty stage
                self._auxiliaries["penalty"]["start_epoch"] = epoch + 1
                messages.append(f"[Scheduler][Dynamic]: Penalty stage unlocked at epoch {self._auxiliaries['penalty']['start_epoch']}")

        # Stage 2: Check if penalty stage has plateaued
        if self._auxiliaries["penalty"]["start_epoch"] is not None and self._auxiliaries["outcome"]["start_epoch"] is None:
            pen_metric = vl_main + vl_ce_raw + vl_dt_raw + vl_pen_raw
            self.pen_best, self.pen_bad_epochs, pen_plateau = self._check_plateau(
                pen_metric, self.pen_best, self.pen_bad_epochs, self.plateau_min_delta, self.penalty_plateau_patience
            )
            trained_pen_epochs = epoch - self._auxiliaries["penalty"]["start_epoch"] + 1
            if trained_pen_epochs >= self.min_penalty_epochs and pen_plateau:
                # Unlock outcome stage
                self._auxiliaries["outcome"]["start_epoch"] = epoch + 1
                outcome_ramp = self._auxiliaries["outcome"]["ramp_epochs"]
                self.warmup_complete_epoch = epoch + 1 + outcome_ramp
                messages.append(f"[Scheduler][Dynamic]: Outcome stage unlocked at epoch {epoch + 1}; warmup completes at epoch {self.warmup_complete_epoch}")

        return messages

    @staticmethod
    def _check_plateau(metric_val: float, best_val: float, bad_epochs: int, min_delta: float, patience: int) -> tuple:
        """Check if metric has plateaued (no improvement for patience epochs)."""
        if metric_val < (best_val - min_delta):
            return metric_val, 0, False
        bad_epochs += 1
        return best_val, bad_epochs, bad_epochs >= patience

    def get_lambdas(self, epoch: int) -> dict:
        """
        Return current lambda values for all auxiliaries.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        dict
            {aux_name: lambda_value} for all registered auxiliaries.
        """
        lambdas = {}
        for name, spec in self._auxiliaries.items():
            if spec["start_epoch"] is None or spec["lambda_max"] is None:
                # Not yet unlocked or calibrated
                lambdas[name] = 0.0
            else:
                # Compute ramped lambda
                start = spec["start_epoch"]
                ramp = spec["ramp_epochs"]
                # If ramp <= 1, no actual ramp (end = start triggers immediate condition)
                end = start if ramp <= 1 else start + ramp
                lambdas[name] = linear_schedule(epoch, start, end, spec["lambda_max"])
        return lambdas

    def current_warmup_end_epoch(self) -> int | float:
        """
        Return the epoch after which early-stopping can begin counting.

        Returns
        -------
        int | float
            - Static Phase-2: configured warmup_epochs
            - Dynamic Phase-2: float('inf') until outcome unlocked, then concrete epoch
            - Phase-1: None (no warmup gating)
        """
        if self.warmup_complete_epoch is None:
            return float("inf") if self.dynamic_enabled else None
        return self.warmup_complete_epoch

    def status_line(self, epoch: int) -> str:
        """
        Build a human-readable status line for logging.

        Parameters
        ----------
        epoch : int
            Current epoch.

        Returns
        -------
        str
            Formatted string with current lambdas for all auxiliaries.
        """
        parts = []
        lambdas = self.get_lambdas(epoch)
        for name in sorted(self._auxiliaries.keys()):
            spec = self._auxiliaries[name]
            lam = lambdas[name]
            if spec["lambda_max"] is None:
                parts.append(f"{name}:λ={lam:.4f}(pending)")
            else:
                parts.append(f"{name}:λ={lam:.4f}/λ_max={spec['lambda_max']:.4f}")
        return f"[Scheduler]: {' | '.join(parts)}"


