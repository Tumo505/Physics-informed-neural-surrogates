# Protocol registered 2026-10-16, before modern model test evaluation

Track A pilot: native PyTorch message passing, width 32, two steps, SiLU, mean incoming aggregation. Input training-normalized coordinates + all documented-as-opaque local channels; broadcast all global channels and first 32 supplied shape coefficients. Edge relative position and length. Output displacement in cm via training-fitted component target normalization. Loss: mean squared vector error on physical displacement divided by sum of training target component variances; Adam 0.001. Seeds 11,22,33,44,55. CPU float32, two threads, deterministic algorithms. No physics loss.

Pilot random seed 2027 chooses 64 of 2250 author training cases, independently of outcomes. Ten passes (640 optimizer steps per seed), validation on first 16 author validation cases every epoch; best validation component MSE checkpoint. Remaining author validation IDs 16:150 reserved for calibration (134), inaccessible to selection. Pilot evaluation first 64 author test cases, chosen by index before evaluation. This is a small training pilot, not a full reference reproduction or final test. No hyperparameter tuning from test scores. Remaining test cases reserved for later main experiment.

Fit modern normalizers on 64 pilot training cases only. No released pretrained representations or statistics used in modern model. Model source inputs do not contain target displacement. Feature semantics beyond geometry remain opaque pending paper review. No material/load shift claims. Geometry-shift evaluation deferred; no post-hoc threshold.

Five-member ensemble: sample variance ddof=1; raw variance is an indicator. Calibrate constant-radius simultaneous whole-geometry L2 balls with score=max node error, corrected order statistic at 90/95%. Calibration and test must have distinct geometries; exact duplication audit required. Exchangeability is assumed, not proven. Report binomial exact coverage intervals, rank association, and selective risk. Do not tune intervals on test.

Reference baseline: first fixed-geometry full released test set, then varying-geometry first 16 test cases as numerical reproduction pilot. Fixed-geometry data not used for geometry coverage. Report compatibility adapter vs author forward agreement. B0 released normalization is part of pretrained artifact, not modern fit.

Track B: blocked until trusted solver assembly can be run and all ten mechanics gates pass. Verified synthetic unit tests may be implemented but are not cardiac validation. Physics title is a proposed study title only.
