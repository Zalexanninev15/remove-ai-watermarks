# Temporal video inpainting evaluation

Development-only study run on 2026-09-27. It asks whether a permissively
licensed temporal model improves visible video-mark removal enough to justify
an optional backend. The answer on this corpus is no: VACE and ROSE were less
temporally stable than the best shipped per-frame backend on every clip, VACE
left an obvious block on Veo, and ROSE reproduced the Veo wordmark strongly
enough for the production detector to accept all 17 output frames.

No provider oracle or private input was used. All source clips are tracked,
publication-cleared fixtures. Generated videos, masks, reports, and contact
sheets remain outside the repository.

## Models, licenses, and runtime requirements

| Model | Evaluated code and weights | License evidence | Official inference requirements |
| --- | --- | --- | --- |
| VACE | [`ali-vilab/VACE`](https://github.com/ali-vilab/VACE) at `48eb44f`, [`Wan-AI/Wan2.1-VACE-1.3B-diffusers`](https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers) at `ec4d2cb` | Code and 1.3B weights are Apache-2.0. | The upstream repository tested Python 3.10.13, CUDA 12.4, and PyTorch 2.5.1. Its 1.3B profile targets about 81 x 480 x 832. The official [input contract](https://github.com/ali-vilab/VACE/blob/main/UserGuide.md) requires missing pixels to be gray and a separate white-generate/black-retain mask. |
| ROSE | [`Kunbyte-AI/ROSE`](https://github.com/Kunbyte-AI/ROSE) at `6be41c5`, [`Kunbyte/ROSE`](https://huggingface.co/Kunbyte/ROSE) at `8a5e57c`, with [`alibaba-pai/Wan2.1-Fun-1.3B-InP`](https://huggingface.co/alibaba-pai/Wan2.1-Fun-1.3B-InP) at `9bcbbcf` | The top-level repository license, tuned transformer, and base model declare Apache-2.0. The separate `hugging_face/` demo vendors/imports ProPainter code and was excluded; a product package would have to keep that non-commercial path out. | The upstream recipe uses Python 3.12, CUDA 12.4, PyTorch 2.6.0, and torchvision 0.21.0. It requires `16n+1` frames, defaults to 480 x 720 and 50 steps, and combines the tuned 1.3B transformer with the Wan2.1-Fun base model. |

MiniMax-Remover and ProPainter were not evaluated because their weights or
code are non-commercial. ROSE's core `inference.py` imports only the `rose/`
Wan2.1-Fun path used here; the ProPainter-derived interactive demo is a
separate license boundary, not evidence that the complete checkout is safe to
redistribute as one Apache-only dependency.

## Design

The production `scan_video_marks` and `stabilize_localizations` paths selected
the first continuous 17-frame accepted run for each mark. The production
`_mask_for_region` then built every per-frame mask, including the mark-specific
box or silhouette policy and padding. Seventeen frames satisfy ROSE's `16n+1`
constraint and keep the cloud run small.

| Case | Tracked source | Frames | Source to working size | Mask pixels per frame |
| --- | --- | ---: | --- | ---: |
| Sora moving | `data/fixtures/visible/sora/provider-original.mp4` (`7dd8a9d4`) | 150-166 | 704 x 1280 to 448 x 832 | 4,508 |
| Veo fixed | `data/fixtures/visible/veo/provider-original.mp4` (`e778194b`) | 0-16 | 1280 x 720 to 832 x 464 | 209 |
| Kling fixed | `data/fixtures/visible/kling/provider-original.mp4` (`f3e8d6a7`) | 0-16 | 1280 x 720 to 832 x 464 | 2,968 |

The aspect-preserving working sizes are multiples of 16 and stay near the
1.3B models' 480p operating point. `cv2`, MI-GAN, and LaMa ran through the
library's own `fill` and `stabilize_filled_frame` calls on those same decoded
frames and masks. VACE used 30 steps and the required RGB-128 hole. ROSE used
its official unmasked-video-plus-mask conditioning at 50 steps. Seed 0 and an
empty removal prompt were fixed for both models.

This is a deliberately small decision study, not a population estimate: one
seed, one 17-frame window from each of three real exports, and no ground truth
for the pixels hidden by the marks. It spans one moving mark and two fixed
marks, but not every registered provider. Hailuo was screened and omitted
because its real fixture did not produce an accepted visual run without
provenance, so it could not satisfy the experiment's requirement to derive the
mask from production detection.

Only generated pixels inside the production mask were composited into the
source. This protects the rest of the image even though the raw VACE output
changed outside-mask pixels by 1.10-2.51 levels of RGB MAE and ROSE by
2.72-3.22. Every delivered arm was then encoded as H.264 CRF 14 at 16 fps and
decoded again for measurement.

The primary temporal metric is mean absolute frame-to-frame RGB difference
inside the union of adjacent masks. `excess` subtracts the same reading from
the marked source. Lower is smoother, but it is not ground-truth recovery: a
blurred patch can score well. Outside-mask MAE is measured after delivery, so
its small non-zero value includes H.264 error even though the compositor kept
the source pixels. The last two columns run the library's own detector on the
delivered working-resolution video.

## Results

| Case | Method | Mask temporal MAD | Excess vs source | Outside-mask MAE | Detector frames / 17, max score | Run seconds |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| Sora | OpenCV | 1.023 | -1.081 | 0.2613 | 0, 0.336 | 0.125 |
| Sora | MI-GAN | 1.072 | -1.032 | 0.2559 | 0, 0.336 | 12.865 |
| Sora | LaMa | 1.481 | -0.623 | 0.2577 | 0, 0.336 | 97.539 |
| Sora | VACE | 1.619 | -0.485 | 0.2520 | 0, 0.336 | 12.226 |
| Sora | ROSE | 1.581 | -0.523 | 0.2655 | 0, 0.336 | 14.051 |
| Veo | OpenCV | 0.150 | +0.097 | 0.0047 | 0, 0.449 | 0.105 |
| Veo | MI-GAN | 0.040 | -0.013 | 0.0007 | 0, 0.434 | 11.180 |
| Veo | LaMa | 0.049 | -0.004 | 0.0039 | 0, 0.434 | 98.762 |
| Veo | VACE | 1.338 | +1.286 | 0.0132 | 0, 0.491 | 11.301 |
| Veo | ROSE | 0.330 | +0.277 | 0.0043 | **17, 0.663** | 13.000 |
| Kling | OpenCV | 0.433 | +0.102 | 0.2653 | 0, 0.000 | 0.296 |
| Kling | MI-GAN | 0.234 | -0.097 | 0.2687 | 0, 0.000 | 30.799 |
| Kling | LaMa | 0.386 | +0.055 | 0.2756 | 0, 0.000 | 102.011 |
| Kling | VACE | 0.737 | +0.406 | 0.2740 | 0, 0.000 | 12.709 |
| Kling | ROSE | 0.840 | +0.510 | 0.2554 | 0, 0.000 | 13.045 |

OpenCV was narrowly lowest on Sora; MI-GAN was the best learned backend there
and the lowest-scoring method on Veo and Kling. Relative to MI-GAN, VACE was
1.51x, 33.65x, and 3.15x higher on Sora, Veo, and Kling; ROSE was 1.47x, 8.29x,
and 3.60x higher. The resized Sora and Veo source controls themselves
fell below the temporal acceptance rule (maximum scores 0.627 and 0.522), so a
zero accepted-frame count cannot prove removal for those two cases. It remains
useful as a mutation check: ROSE raised the resized Veo from 0 accepted frames
to 17. The resized Kling source remained accepted on 17/17 frames at 0.675,
and every output cleared that detector.

CPU runtimes were remeasured serially after other experiments stopped, with
88.96 percent host CPU idle immediately before the run. They exclude model
setup (MI-GAN 1.5-4.1 seconds; LaMa 3.4-13.1 seconds). GPU rows are clean
pipeline-call time on one H100, excluding container start and model loading.
VACE peaked at 22.60-22.87 GiB allocated and ROSE at 19.71-19.83 GiB.

At Modal's [2026-09-27 H100 list rate](https://modal.com/pricing) of
$0.001097 per second, the measured VACE calls cost about $0.0398 total and the
ROSE calls $0.0440, or $0.0837 for all six 17-frame calls. That is an
inference-only floor. Cold starts, weight loading, warm-idle time, CPU, memory,
and storage make the billed experiment higher.

## Frame review

- **Sora:** OpenCV produced a moving gray smear. MI-GAN, LaMa, correctly
  conditioned VACE, and ROSE all left a small bright sliver near the mascot's
  former edge; the temporal models did not reconstruct the deck seam better
  enough to offset their higher temporal MAD.
- **Veo:** MI-GAN and LaMa blended the tiny translucent label into the flat
  background. VACE replaced it with a conspicuous gray rectangular block.
  ROSE visibly reproduced the `Veo` text on frames 0, 8, and 16, matching the
  detector's 17/17 result.
- **Kling:** every method removed the wordmark, but the temporal models left a
  darker rectangular water patch. Their frame-to-frame MAD was 3.15-3.60x the
  MI-GAN result, so the patch moved in intensity rather than providing the
  expected temporal advantage.

An initial VACE control mistakenly passed the original marked pixels under the
mask. It visibly reconstructed `Sora` and raised the detector score to 0.427.
The official VACE preprocessing code replaces the hole with gray first; the
control is retained in the external report as `vace`, marked
`excluded_from_verdict`, while every tabled VACE row is the corrected
`vace-gray` arm. This is direct evidence that a future VACE integration must
pin the model's conditioning contract, not merely its mask polarity.

## Verdict and reproducibility

Do not add VACE or ROSE as an optional visible-video backend. Neither model
beat the practical CPU MI-GAN tier on temporal stability, both introduced
visible artifacts on simple backgrounds, and ROSE failed the basic Veo
removal case. Their roughly 20-23 GiB H100 footprint, multi-model downloads,
cold-start cost, and ROSE demo-license boundary add deployment complexity
without a measured quality gain. Revisit only with a temporally trained model
that can beat MI-GAN on a wider real corpus while preserving tiny-mask
conditioning. This one-seed result rejects an optional backend now; it does not
claim that no prompt or seed can improve either model. Any revisit should
predefine that tuning budget and still beat the unchanged shipped controls.

The external harness SHA-256 is
`e9bb2232bac2b719ae72d62e395699949609c3cdfdcd46e38f4d681ed3edee29`.
The retained JSON report SHA-256 is
`76da749eb0410cebb402914082f128ab339f943cc2596f05ae8c86d92cfedb41`.
The run is bound to library commit `742a0dc`. The report records the complete
model revisions, source and output hashes, masks, per-method metrics, and the
invalid VACE control; full-frame and enlarged mask contact sheets were reviewed
from the retained artifacts.
