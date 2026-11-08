# Analysis Plan

This summary describes the intended analyses. Freeze the specific data, model and analysis identities for each new run before test access.

## Closed-loop comparisons

The primary endpoint is normalized area under the tracking-error response curve across the fixed packet-drop grid from 0 to 0.8. Normalize mean tracking RMSE by the corresponding seed's nominal value and integrate with the trapezoidal rule. The estimand uses the full 500-step horizon; score unobserved post-failure steps using the registered 1.0 m penalty. Report failure alongside tracking error.

The primary paired contrasts are LTC versus fixed-time-constant CT-RNN, dense CfC versus LTC, and NCP-CfC versus degree-matched random-sparse CfC. The first two are operational architecture comparisons rather than single-factor interventions. A tracking-error improvement must meet the registered failure-rate noninferiority condition. Report recovery, latency and solver activity as distinct endpoints.

Resample paired training seeds and then paired test trajectories within each seed for hierarchical bootstrap intervals. Keep perturbation levels paired within each trajectory. Exact seed-level sign-flip comparisons and Holm corrections use their prespecified families. Five paired seeds permit a minimum two-sided exact p value of 0.0625, so emphasize effect estimates and uncertainty rather than significance claims.

## Offline comparisons

Use macro-F1 for UCI HAR and AUROC for PhysioNet 2012 as the primary predictive metrics. Report absolute performance and degradation from the zero-added-missingness condition. Keep optimization seeds as the outer replication unit; resample matched subjects or patients within each seed. Retain the same fixed checkpoint across corruption levels and report independent and block missingness separately.
