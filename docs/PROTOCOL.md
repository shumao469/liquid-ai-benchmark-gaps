# Experimental Protocol

This public design summary accompanies the code. It is not the original runtime authorization or a dataset or checkpoint release. Independently freeze the configuration, software environment, splits and analysis plan before any new confirmatory execution.

## Simulated closed loop

The confirmatory plant is Gymnasium Reacher-v5. The target follows forward kinematics of a seeded moving joint reference. Every learned model estimates the same four joint states from corrupted proprioception, masks, ages and time increments, the reference, and previous applied torque. The estimate drives the same joint-space PD controller and changes the next simulator state. The point-mass smoke test and the small Reacher-v5 pilot are development checks only.

Use disjoint training, validation and test trajectory seeds. The design specifies 2,000, 200 and 200 episodes respectively, with 500 steps per episode. Fit normalization on training only. Full-state PD demonstrations provide the supervised state target; they do not turn the observer into an action policy.

Compare LSTM, ODE-RNN, fixed-time-constant CT-RNN, LTC, dense CfC, NCP-CfC and degree-matched random-sparse CfC under a common effective trainable-parameter budget. The structured and random-sparse comparison uses the same cell and readout with exactly matched unsigned degrees. Use full-state PD, zero-order-hold PD and Kalman-plus-PD as nonlearned references.

Choose AdamW hyperparameters by validation on the dedicated tuning seed 17 from the common four-candidate grid. Train each registered model with seeds 101, 202, 303, 404 and 505 for 100 epochs of 500 optimizer steps. Validation determines the checkpoint, not the training budget. Keep model weights fixed during test execution. Use common test trajectories and perturbation schedules across methods.

## Offline sequence tasks

UCI HAR and PhysioNet 2012 use fixed subject- or patient-disjoint splits. Independently acquire and verify the official data files. Use the same input channels and missingness draws across models. Select the checkpoint on validation at a fixed training budget, then reuse it unchanged across the missingness sweep. Evaluate independent missingness and contiguous blocks separately. The PhysioNet representation uses a fixed hourly grid; the experiment does not test native within-hour event times.
