## Analysis

All numbers below come from the tables above; nothing is from external sources.

**1. Test is clearly worse than validation.** Test macro F1 is 0.713–0.802, against 0.790–0.898
on validation. Part of this gap is optimism: the checkpoint was chosen on validation. The rest
is variation between pigs. Each split has only 2 pigs, and **8 of the 20 classes occur in only
one test pig (4 images)**, so per-class test F1 for those classes is a small-sample estimate.

**2. The two worst test classes fail because of a pig, not because the class is hard.**
- **Kidney** (test F1 0.09–0.44) occurs only in P096. On the validation pigs, kidney F1 was
  0.75–0.83 for every model. Most errors go to *liver* (0.31–0.94 of kidney pixels, depending
  on the model) and *heart*. The training-set mean spectra of kidney and liver are similar
  (see the figure).
- **Major vein** (test F1 0.05–0.48) occurs only in P095. Validation F1 was 0.81–0.97. It is
  mostly predicted as *bone* or *cartilage*.

We did not investigate the cause further, for example perfusion or the imaging conditions of
these recordings. With a single test pig per class, these results cannot separate "hard class"
from "unusual pig".

**3. 16 bands cost little once spatial context is used.**
- **Patch CNN:** −0.016 macro F1 (0.802 → 0.786). The two sd bands (±0.008 and ±0.011) do not
  overlap.
- **U-Net:** no measurable difference (0.761 vs 0.762).
- **Pixel MLP:** −0.038 (0.751 → 0.713). It has no spatial context, so it relies entirely on
  spectral detail.

The 16-band means track the 100-band class means closely (figure), which is consistent with
this. Per class, the 16-band patch CNN loses most on *muscle* (0.78 → 0.57) and *omentum*
(0.96 → 0.86), and gains on *kidney*, *major vein* and *pancreas*. All of these move by
amounts comparable to their seed sd.

**4. Model ranking on test: patch CNN > U-Net ≈ pixel MLP (full spectrum).** U-Net and patch CNN
were tied on validation. On test the U-Net is worse and less stable (full spectrum ±0.024).
Its best epoch was 38–40 of 40 in every run, so it is probably undertrained. We did not run a
longer schedule, so this is a hypothesis, not a result.

**5. Image-level accuracy is similar for all models (0.86–0.89).** A majority vote over each
annotated polygon removes most pixel-level noise. Even the per-pixel MLP identifies the organ
in 0.888 ± 0.005 of test images (full spectrum). The pixel-level differences between models
come mostly from boundary pixels and the hard classes above.

**Limitations of this evaluation:** 2 test pigs and 128 images; porcine organs imaged ex situ
or in an open-abdomen setting, not real surgical scenes; one organ labelled per image, so
background and organ boundaries are never scored; no hyperparameter search (one configuration
per model); simulated 16-band sensor (spectral binning only, no mosaic pattern or sensor
noise).
