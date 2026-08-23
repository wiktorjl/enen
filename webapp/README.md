# Browser digit recognizer

This is a minimal interface around the repository's C classifier. Emscripten
compiles `src/nn.c`, `src/dataset.c`, and `src/web_api.c` into `enen.wasm`.
JavaScript loads data, normalizes canvas pixels, and updates the interface; it
does not implement forward propagation, backpropagation, or model weights.

## Build

From the repository root, with Emscripten available:

```sh
make web
```

This writes `webapp/enen.js`, `webapp/enen.wasm`, and a browser copy of the
canonical `src/nn.c`. All three files are required by the complete interface.

## Run

Serve the directory over HTTP:

```sh
python3 -m http.server 8080 --directory webapp
```

Open <http://localhost:8080>. Browsers will not load the Wasm module and data
from a `file://` URL. If an older direct-file tab is already open, close it and
use the HTTP address above.

## Training and recognition

The Train tab exposes four C architectures, 10–100 epochs, and three learning
rates. The tested default is `64 → 128 → 64 → 10`, learning rate `0.05`,
and 25 epochs. Every recipe trains on:

- all 3,823 official UCI training rows, loaded and normalized by C; and
- 800 deterministic handwritten/typographic variations passed into C after
  going through the same crop, center, resize, and 8×8 reduction as the live
  drawing canvas.

Live strokes are retained on a padded in-memory canvas so brush pixels near the
visible edge are not discarded. Before every prediction, the complete ink
bounds are deterministically resized and centered; drawing position therefore
does not become a model-input offset.

After training, C evaluates the untouched 1,797-row UCI test split once. The
page also evaluates 160 separately generated alternate-path and typographic
variations that were not used for training. Recognition remains locked unless
the overall checks and every digit class pass their thresholds. Mouse, pen, and
touch input then call robust C inference on the exact model instance that was
just trained. Draw and Eraser tools update both the visible surface and its
padded in-memory copy, so erased strokes are also removed from the centered
8×8 input used by C.

The narrower network monitor remains visible to the right of both tabs. While a
stroke changes in Recognize, the browser periodically asks C for one direct
forward pass of the centered 8×8 input and animates the resulting input, hidden,
and output activations. The probability card still uses the robust C prediction,
which averages a few nearby
centered variants; the caption calls out that distinction explicitly.

While training, the setup form is replaced by a dense view of every network
node and connection. A small C training batch updates the real model, and the
browser copies its current example, activations, and weights for one frame about
every half second. JavaScript only draws those snapshots; it never computes a
forward pass or a weight update. Pause stops visual updates without stopping C
training, while **Finish without animation** completes every selected epoch and
quality check using larger batches.

After a run passes every quality check, the browser asks C to serialize that
exact network and stores the bytes in IndexedDB. On later visits it loads and
rechecks the last validated model before enabling recognition. Storage is local
to the current browser profile and site origin—there is no account or backend
model store, and another browser, profile, host, or port gets its own copy.

The **View nn.c** button opens a source dialog containing only the complete
canonical neural-network implementation, with local syntax highlighting and a
raw-source link. The build
keeps `webapp/assets/nn.c` byte-identical to `src/nn.c` and `make web-check`
enforces that relationship.

Run `make check` from the repository root for the native core and browser-API
quality gates, and `make web-check` to rebuild and validate the deployable web
artifacts. Run `make browser-check` with Chrome or Chromium installed to exercise
the complete page from startup through training and recognition.
