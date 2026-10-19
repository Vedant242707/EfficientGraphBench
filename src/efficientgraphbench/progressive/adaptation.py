"""Validation-only transition metrics. All signed changes mean after minus before."""


def transition_metrics(
    previous: float | None,
    pre_old: float | None,
    pre_new: float | None,
    post_old: float | None,
    post_new: float | None,
) -> dict:
    def delta(after, before):
        return after - before if after is not None and before is not None else None

    old_delta = delta(post_old, previous)
    expansion = delta(pre_old, previous)
    return {
        "previous_stage_score": previous,
        "old_node_score_pre_training": pre_old,
        "new_node_score_pre_training": pre_new,
        "old_node_score_post_training": post_old,
        "new_node_score_post_training": post_new,
        "old_node_performance_delta": old_delta,
        "post_expansion_delta": expansion,
        "old_node_forgetting": max(0.0, -old_delta) if old_delta is not None else None,
        "expansion_degradation": max(0.0, -expansion) if expansion is not None else None,
        "old_node_adaptation_delta": delta(post_old, pre_old),
        "new_node_adaptation_gain": delta(post_new, pre_new),
        "old_new_gap": abs(post_old - post_new)
        if post_old is not None and post_new is not None
        else None,
    }


def recovery_status(
    previous: float | None, pre_old: float | None, target: float | None, recovered: bool
) -> str:
    if previous is None or pre_old is None or target is None:
        return "not_applicable"
    if pre_old >= target:
        return "recovery_not_required"
    return "recovered" if recovered else "not_recovered"


class RecoveryTracker:
    """First target hit and endpoint satisfaction are distinct observations."""

    def __init__(self, previous, before, threshold):
        self.previous, self.before = previous, before
        self.target = threshold * previous if previous is not None else None
        self.recovered = before is not None and self.target is not None and before >= self.target
        self.epoch = 0 if self.recovered else None
        self.seconds = 0.0 if self.recovered else None

    def observe(self, epoch, score, seconds):
        if not self.recovered and self.target is not None and score >= self.target:
            self.recovered, self.epoch, self.seconds = True, epoch, seconds

    def result(self, final):
        return {
            "recovery_status": recovery_status(
                self.previous, self.before, self.target, self.recovered
            ),
            "first_recovery_epoch": self.epoch,
            "first_recovery_seconds": self.seconds,
            "recovery_epochs": self.epoch,
            "recovery_seconds": self.seconds,
            "recovery_target": self.target,
            "final_target_satisfied": final >= self.target
            if final is not None and self.target is not None
            else None,
        }
