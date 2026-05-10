# Deployment Considerations

What moving this pipeline from research prototype to clinical tool would require - grounded in challenges encountered during development.

## Human-in-the-Loop QC

Tumor aggressiveness prediction is safety-critical, so no stage should run unsupervised. Segmentation review needs a lightweight GUI where a technician can relabel connected components, remove small-volume noise, or keep only the largest object per class without writing code. Registration failures should be detected automatically via cross-correlation scoring and flagged, rather than silently propagated downstream.

## Handling Distribution Shift

The primary failure mode: when a new patient's tissue differs in morphology or intensity, segmentation degrades sharply. However, the recovery loop is fast. A technician visually rejects bad slices, the remaining good predictions become pseudo-labels for retraining. This was validated empirically during dataset iteration, reducing days of manual annotation to ~30 min of curation + overnight compute. Cross-protocol shifts (different staining or microscope) remain an honest limitation requiring re-annotation or domain adaptation.

## Why Not C++?

Processing a full 3D stack takes a few hours on a GPU workstation. The bottleneck is sample preparation and imaging (days), not computation. A C++ port would cost hundreds of engineering hours and introduce new bugs in a safety-critical context, while saving time nobody is waiting for.
