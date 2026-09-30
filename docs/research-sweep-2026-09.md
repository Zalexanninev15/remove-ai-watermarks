# Research sweep, September 2026

A survey of every area the library works in, run on 2026-09-22 and 2026-09-23 to
find what changed since the topic pages were written and what earlier research
missed. Nine areas were covered: Google SynthID, other vendors' invisible
watermarks, the watermark attack literature, visible AI labels, metadata and
provenance standards, video and audio, AI image classifiers, law and regulation,
and fill or regeneration models with competing tools. Claims were checked
against primary sources (specifications, vendor documentation, legal texts,
arXiv pages); live captures from the maintainer's accounts settled several
questions directly.

This page is an index. Each finding lives in its canonical page.

## Where the findings went

| Area | Change | Page |
| --- | --- | --- |
| Law and regulation | Research purpose, no-liability notice, jurisdiction table (China, EU, India, US, South Korea), platform labeling behavior | [legal-and-safety.md](legal-and-safety.md) |
| SynthID adoption | 100B+ items, ElevenLabs and OpenAI audio (with OpenAI's verification API limits), Apple (announced), Vertex C2PA, Nano Banana 2 Lite and Gemini Omni Flash, optional visible mark, portal status | [synthid.md](synthid.md), [provider-oracles.md](provider-oracles.md) |
| Other vendors | Microsoft Foundry provenance for Azure-hosted Flux and MAI, Amazon Nova Reel, audio adopters | [watermarking-landscape.md](watermarking-landscape.md) |
| C2PA 2.3 and 2.4 | `c2pa.ai-disclosure` and ingredient `digitalSourceType` now read; parser claim corrected | [watermarking-landscape.md](watermarking-landscape.md) |
| Containers | GIF metadata stripped without re-encoding; TIFF and DNG refused; C2PA after large WAV and AVI data now scanned | [known-limitations.md](known-limitations.md) |
| TC260 in audio | WAV, MP3, OGG, Opus and FLAC label placements from the 2025-08 audio guide | [supported-signals.md](supported-signals.md) |
| Video labels | Sora discontinued; opening-only labels no longer filled over the whole clip; any video mark vetoed by another vendor's C2PA AI claim, Kling also by a non-Kling TC260 producer; TC260 video producers named | [supported-signals.md](supported-signals.md), [known-limitations.md](known-limitations.md) |
| Apple | Image Playground and Photos Clean Up credits attributed | [supported-signals.md](supported-signals.md) |
| New visible marks | `vidu` video mark and `wan` image mark, each calibrated on one real export | [supported-signals.md](supported-signals.md), [known-limitations.md](known-limitations.md) |
| Google Photos | AI edits reported as "Google Photos (AI edit)" rather than Gemini, with SynthID present: Google's checker found it in 4 of 4 Photos AI edits (Ask, eraser and two others) | [synthid.md](synthid.md), [supported-signals.md](supported-signals.md) |
| YouTube | Upload test: C2PA-driven "Made with AI" label, public streams without C2PA, Studio downloads re-signed by YouTube; re-encoded non-Google video no longer read as Google AI with SynthID | [supported-signals.md](supported-signals.md), [legal-and-safety.md](legal-and-safety.md), [module-internals.md](module-internals.md) |
| Audio | Generated audio from OpenAI, ElevenLabs and Microsoft passes through the video path unchanged | [known-limitations.md](known-limitations.md) |
| Attack literature | Fidelity band on open encoders now 30-45 dB; WmForger and WMCopier status | [synthid-removal-research.md](synthid-removal-research.md), [watermark-forgery-study.md](watermark-forgery-study.md) |
| Models | FLUX.2 klein has a strength-controlled pipeline in diffusers | [chroma1-engine-research.md](chroma1-engine-research.md) |
| Dependencies | TrustMark extra on every supported Python (`trustmark>=0.9.2`) | [installation.md](installation.md) |
| Live captures | Gemini Omni, Grok, Vidu, Wan, Apple, Google Photos, Nano Banana 2 Lite and Gemini Omni Flash through the API; Runway, Higgsfield and Dreamina | [watermarking-landscape.md](watermarking-landscape.md#live-captures-2026-09-23-and-2026-09-24) |

## Open items

- **Google Photos Reimagine manifests.** Magic Editor, and so Reimagine, is not
  offered in Google Photos on iOS, so no Reimagine file is captured. The
  question it was meant to answer is settled without one: Google's checker found
  SynthID in all four Photos AI edits tested on 2026-09-25, none of which
  records SynthID in its manifest.
- **Vidu and Wan marks on more samples.** Both are registered from one real
  export each (`vidu` video, `wan` image); their gates are provisional until a
  wider cohort is measured, and a Wan video label is not yet captured. Public
  Vidu feed videos (2026-09-26) carry the TC260 label naming ShengShu, and one
  names Tongyi Yunqi as producer with Vidu as propagator, but none shows the
  visible `Vidu` mark; Wan's public demo videos show no mark either. Both marks
  appear only on account downloads.
- **TC260 practice guides.** The audio placements from TC260-PG-202510A are
  now read (the guide PDFs load in a browser from
  `tc260.org.cn/tc260/sjzn/list.shtml`, not to scripted requests). The text-file
  guide (TC260-PG-20258A) covers documents this image and video tool does not
  process. The security-protection guide (TC260-PG-202511A) records a
  `SecurityData` JSON object, a digital signature over the label and optionally
  the content, in `ReservedCode1` and `ReservedCode2`. MiniMax is the one
  producer in the local corpus that writes a guide-shaped object (checked
  2026-09-29): four MiniMax videos served by Higgsfield and Runway, two of them
  tracked (`higgsfield-hailuo-2-3.mp4`, `higgsfield-minimax-h3.mp4`), carry
  `SecurityData` with two SM3withSM2 (`1.2.156.10197.1.501`) signatures and one
  public key, `KeyValue` `00a0b3b0...47e4fd` in all four. The label signature
  verifies on all four against the guide's appendix A.1 message, the ASCII string
  `"Label":"1","ContentProducer":"MiniMax","ProduceID":"<id>"`, with the default
  SM2 user ID `1234567812345678` and the key's x coordinate lifted to the even-y
  point; changing one `ProduceID` digit fails it. MiniMax's own spellings depart
  from the guide: `TBSData.Type` is `LabelMataData` (the guide's `Md`) and
  `Binding` with `BType` `0` (its `Bnd`); the second signature verifies over the
  compact JSON of the `Bindings` entry, also on all four. That entry's SM3
  content hash matched neither the whole file nor the file without its label,
  and it names no `CntSel`, so the content binding is unverified. Other producers fill the same
  fields with opaque vendor strings (Qwen a base64 digest, Vidu 32 hex digits,
  Wan and HappyHorse `L-`/`K-` prefixed values), not `SecurityData`. The key sits
  in the file it signs, so a check proves authenticity only against a pinned
  MiniMax key. `identify` now verifies it with a built-in SM3/SM2 and a pinned
  MiniMax key ([module internals](module-internals.md)).
- **Experiments.** Chroma restoration after regeneration (arXiv:2508.21072) was
  measured on 2026-09-27 and is not shipped: transplanted source chroma drew one
  explicit "inconclusive" and two unclear Gemini answers where the untouched
  outputs were all clean, and ghosts redrawn objects
  ([removal research](synthid-removal-research.md#color-restoration-after-regeneration-2026-09-27)).
  FLUX.2 klein 4B was measured the same day: stock inpainting copies the
  source through its reference tokens and keeps OpenAI SynthID, and without
  them it clears both vendors only at lower face and CJK-text fidelity than
  `qwen-zimage`
  ([Chroma1 research](chroma1-engine-research.md#flux2-klein-as-a-global-stage-2026-09-27)).
  Apache-2.0 temporal video inpainting was also measured: VACE and ROSE were
  less stable than MI-GAN on all three Sora, Veo, and Kling clips, VACE left a
  gray block on Veo, and ROSE retained the Veo wordmark on 17/17 frames. They
  are not shipped
  ([video inpainting research](video-inpainting-research.md)).
  Resemble Perth and Meta PixelSeal were added as exact-revision, MIT-licensed
  local benchmark oracles. The 2026-09-28 study found Perth on all three speech
  positives and confirmed that visible-video cleaning copies its marked AAC
  packets unchanged; PixelSeal survived JPEG q90 on both cleared image
  carriers, while its CUDA-only removal profiles could not run on this host
  ([benchmark study](benchmark-kernel.md#perth-and-pixelseal-local-oracle-study)).
  Re-watermarking (arXiv:2605.16796) was reproduced locally on 2026-09-27:
  same-encoder message B replaced A on all 12/12 valid TrustMark and 12/12
  VideoSeal image controls, both VideoSeal video clips, and all 6/6 valid
  controls in each DWT-DCT direction. Median image clean-to-output PSNR was
  39.73-44.15 dB; the DWT-DCT controls worked on only half the carriers.
  This is an open-encoder result, not a SynthID result
  ([removal research](synthid-removal-research.md#re-watermarking-on-open-encoders-2026-09-27)).
  A one-carrier production probe subsequently kept OpenAI SynthID `detected`
  on both its metadata-free control and 43.04 dB VideoSeal-overlaid image;
  a visible-clean, metadata-free Google control and its 42.86 dB VideoSeal
  overlay also remained `detected` under direct SynthID-tool checks. The Google
  control and candidate positives came from different approved account slots,
  so this remains a one-carrier coexistence result
  ([removal research](synthid-removal-research.md#production-synthid-overmark-probe-2026-09-27-local)).
  Z-Image-Turbo was measured on 2026-09-29. It is not a global replacement:
  at its OpenAI margin point it improves both no-face typography carriers but
  loses face identity, and the same face/text split is stronger in the Google
  fidelity probe. The Google no-face outputs at 0.40 also visibly redraw scene
  geometry and object detail, despite better OCR and aggregate metrics, so a
  face detector alone cannot make a safe complement. The last Google boundary
  is unresolved because the available oracle checks were quota-blocked or
  could not accept the prepared upload.
  The mflux arm works on an M5 with 32 GB and is the only measured native Apple
  Silicon image-regeneration path, since every shipped image profile is
  CUDA-only. It took 301-380 seconds per image. This proves M5 feasibility but
  is not a usable fallback at the measured quality
  ([Chroma1 research](chroma1-engine-research.md#z-image-turbo-as-a-global-stage-2026-09-29)).
  The experiment is closed with a decision not to add a global or mflux
  profile; retain its harness and local evidence for future comparison only.
  The existing source-conditioned Z-Image face-repair stage remains part of
  `qwen-zimage`, `sdxl-zimage`, and `chroma-zimage`.
  Still open: linear probes on newer frozen backbones for `classify` and RAVEN
  view-synthesis removal
  (arXiv:2601.08832, no code released as of 2026-09-24).
  MiniMax-Remover (CC-BY-NC-4.0 weights) and ProPainter (S-Lab License,
  non-commercial) are excluded on license.
- **New C2PA signers without a vendor row.** The
  [C2PA conformance list](https://github.com/c2pa-org/conformance-public/blob/main/conforming-products/conforming-products-list.json)
  names generator certificates `Amazon Bedrock`, `Getty Images
  AI-Generated Image` and `AI-Modified Image` (and the iStock pair), `Jasper`,
  `RefaceAI`, and vivo `vivo Albums` and `JOVI Albums`. It also lists `Google
  Photos` for both Android and iOS, the signer the Photos rule keys on. A
  real sample per vendor is still needed to confirm the manifest shape before
  adding a row. Runway is done: its own models sign as `RUNWAY AI, INC.`
  (measured 2026-09-24) and the row covers the conformance-list spelling too.
  Amazon is done: Nova Canvas and Nova Reel outputs published in
  `aws-samples/amazon-nova-samples` sign as `Amazon Web Services, Inc.` with
  claim generator `Amazon Bedrock` (2026-09-26), and had been reported as Canva
  because the byte fallback matched `Canva` inside `Nova Canvas`. A search on
  2026-09-26 found no public Getty, iStock, Jasper, RefaceAI or vivo Albums file
  carrying C2PA: their marketing and model-page images are unsigned. A vivo
  X300 camera capture carries a vivo device manifest (`digitalCapture`) that
  the bundled reader cannot open (`unknown algorithm`).
- **Chinese phone galleries.** Huawei's help pages say the Xiaoyi photo-edit AI
  watermark can be switched off; Honor, OPPO and Xiaomi label behavior is not
  documented in any primary source found. vivo's Album setting is already
  covered.
- **Claims not supported by a primary source.** Kling embedding SynthID
  (repeated by remover blogs; Kuaishou is not in Google's adopter list),
  Hailuo pixel watermarks, and platforms stripping C2PA on upload. None is used
  by the library.
- **Google Photos small edits (deferred).** The "SynthID present" rule for
  Photos AI edits rests on four checked edits, none of them tiny. Still to
  measure on the phone app: a tiny eraser edit, a large one, a small "Help me
  edit" change, a restyle, and a crop-only control, each checked in Gemini. iPhone
  Mirroring does not show the Photos editing toolbar, and the web editor has no
  generative tool, so this needs edits made on the device.
- **Captures blocked by region or device.** Doubao, Jimeng, Samsung. Higgsfield,
  Runway and Dreamina were captured on paid plans on 2026-09-24.

## Decisions

- **Dreamina `AI` badge: not registered (2026-09-26).** Dreamina stamps a small
  boxed `AI` badge top-left on images and video, and calls it its visible AI
  watermark. It stays unregistered for three reasons. A generic boxed `AI` is the
  shape of the disclosure icon the EU Code of Practice defines, which this
  project does not template, so a template would remove any such label. The same
  generic shape would match other platforms' badges and plain `AI` text and
  misattribute them to Dreamina. And it adds nothing to attribution: Dreamina
  exports are identified by their C2PA (`ByteDance Media Transcode Service`
  signer with a `Dreamina/7.5.0` ingredient), and Dreamina's own "Remove watermarks" setting
  removes the badge. Revisit only if the badge becomes a distinctive brand mark.
