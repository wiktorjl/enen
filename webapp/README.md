# Glyph — Neural Digit Studio

A standalone, dependency-free browser app for exploring the digit neural
network in this repository. It includes the original UCI optical-digits data
and the repository's trained model inside `assets/`.

## Run it

From this directory, start any static file server:

```sh
python3 -m http.server 8080
```

Then open <http://localhost:8080>.

A local server is required because browsers block JavaScript from loading the
bundled model and dataset directly from a `file://` page. There are no package
installs, build commands, external fonts, or network services.

## What is included

- **Train:** creates a fresh configurable `64 → hidden → hidden → 10` network,
  performs stochastic backpropagation on all 3,823 training samples, and checks
  accuracy against the 1,797 held-out samples after each epoch. Training can be
  configured from 4 through 200 epochs and includes mild translation
  augmentation plus locally generated mouse-drawn and typographic variations
  for freehand robustness.
- **Recognize:** accepts pointer, mouse, or touch drawing. Each stroke is
  centered and reduced to the network's 8×8 input before live inference. A few
  sub-pixel alignments are averaged through the same active network so small
  placement differences do not dominate the answer.
- **Visualization:** renders every input, hidden, and output activation in a
  compact canvas, plus a small selection of the strongest active connections.
- **Pretrained start:** parses the native `ENEN` C model format directly in the
  browser so recognition works immediately.

All learning and inference happen locally in the browser.
