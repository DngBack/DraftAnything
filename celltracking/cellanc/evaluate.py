"""Evaluator: reads labels/, scores predictions per window (doc §9, §12)."""

import pandas as pd


def score_window(pred: dict[int, int], pred_sibs: set, tg: pd.DataFrame,
                 sib: pd.DataFrame) -> dict:
    """Ancestry accuracy on resolved in-FOI targets; unresolved are counted, never scored.

    Descendant-count MAE is over every anchor that has >=1 resolved descendant or >=1
    predicted descendant among resolved targets, so hard anchors stay in the denominator.
    """
    foi = tg[tg.in_foi]
    ok = foi[foi.target_status == "valid_anchor"]
    p = ok.target_detection_id.map(pred)
    correct = (p == ok.ancestor_detection_id).fillna(False).astype(bool)
    gt_cnt = ok.ancestor_detection_id.value_counts()
    pr_cnt = p.value_counts()
    anchors = gt_cnt.index.union(pr_cnt.index)
    mae = (gt_cnt.reindex(anchors, fill_value=0) - pr_cnt.reindex(anchors, fill_value=0)).abs()
    gt_pairs = {tuple(sorted(x)) for x in zip(sib.det_1, sib.det_2)}
    tp = len(gt_pairs & pred_sibs)
    return {
        "n_targets": len(tg), "n_scored": len(ok), "n_unresolved": int((foi.target_status != "valid_anchor").sum()),
        "n_outside_foi": int((~tg.in_foi).sum()),
        "n_correct": int(correct.sum()),
        **{f"n_scored_d{d}": int((ok.generation_depth.clip(upper=3) == d).sum()) for d in range(4)},
        **{f"n_correct_d{d}": int(correct[ok.generation_depth.clip(upper=3) == d].sum())
           for d in range(4)},
        "n_hidden_div": int((ok.hidden_intermediates > 0).sum()),
        "n_correct_hidden_div": int(correct[(ok.hidden_intermediates > 0).fillna(False)].sum()),
        "count_abs_err_sum": float(mae.sum()), "count_n_anchors": len(anchors),
        "sib_gt": len(gt_pairs), "sib_pred": len(pred_sibs), "sib_tp": tp,
    }
