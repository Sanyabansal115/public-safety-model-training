# Deliverable 5 — Deployment Guide

How to run the deployed model, what each part does, and what the interface is
showing you.

---

## 1. What Deliverable 5 actually is

Deliverables 1–4 end with five trained models sitting in memory inside
`public_safety_ml.py`. The moment that script exits, they are gone. Deliverable 5
closes that gap: it takes the winning model, freezes it to disk, and puts an HTTP
service in front of it so something other than the training script can ask it for
predictions.

Three things had to be solved to get there, and they are worth understanding
because they are the parts a grader is most likely to ask about.

**The model alone was not servable.** The training script fits the
`ColumnTransformer` in section 7 and then fits every model on the already-encoded
matrix. Pickling the neural network on its own would produce something that only
accepts a ~200-column encoded array — useless to an endpoint receiving JSON. Both
fitted objects are wrapped into a single `Pipeline` that accepts raw feature rows.
Neither is refitted, so the deployed model is exactly the evaluated one.

**Rare-category grouping could not be recomputed.** Feature engineering folds
categories below 1% frequency into `"Other"` using frequencies from the whole
training set. A single API request has no frequency distribution, so recomputing
would group nothing — a category that trained as `"Other"` would arrive raw and be
silently encoded as all-zeros. That is a wrong answer with no error message. The
surviving category lists are saved at training time and replayed on each request.

**The 0.5 cut-off was wrong for this problem.** See §7.

Everything the API needs travels in **one pickle** — the pipeline, the threshold,
the feature order, the category maps, the form schema, and the metrics — so the
served model and the metadata describing it cannot drift apart.

---

## 2. Before you start

You need the Anaconda interpreter, not the system one:

```bash
/opt/anaconda3/bin/python
```

The default `python3` on this machine has no `scikit-learn` and no `flask` and
will fail on import.

You also need three generated files in the project folder. They are already
there:

| File | What it is |
|---|---|
| `model_bundle.pkl` | The pickled pipeline + all metadata |
| `test_samples.csv` | 25 held-out records the model never trained on |
| `model_metadata.json` | Human-readable copy of everything except the pipeline |

**You do not need to retrain to run the API.** Only re-run
`public_safety_ml.py` (~7 minutes) if you change feature selection or the model —
and if you do, re-run `make_postman_collection.py` afterwards so the saved request
bodies still match.

---

## 3. Running it

### Terminal 1 — start the API

```bash
cd ~/Desktop/Programming/Projects/public-safety-model-training
/opt/anaconda3/bin/python app.py
```

Wait for these four lines. If you don't see them, the bundle failed to load:

```
[startup] Loaded Neural Network (trained 2026-08-16T16:02:52, sklearn 1.8.0)
[startup] Decision threshold: 0.0160
[startup] Expecting 30 features
[startup] 25 held-out test records available at /samples
 * Running on http://127.0.0.1:5000
```

Leave this running. It serves until you press Ctrl-C.

### Terminal 2 — run the Python client

```bash
cd ~/Desktop/Programming/Projects/public-safety-model-training
/opt/anaconda3/bin/python client.py
```

### Browser — the Jinja client

Open <http://127.0.0.1:5000> while the API is running.

### Postman

File → Import → `postman_collection.json`, then use the Collection Runner. All 8
requests carry test assertions, so you get a pass/fail summary rather than raw
responses.

---

## 4. What the browser page shows you

The page is one screen, top to bottom. Here is every region and what it means.

### Header and badges

The title bar states the actual prediction task: whether a person involved in a
KSI collision is likely to be **Killed** rather than seriously injured. The
wording matters — each row in the dataset is one *person*, not one crash, so one
request scores one involved person.

Five badges read straight out of the pickle:

- **Model** — `Neural Network`, the algorithm actually being served
- **Threshold** — `0.0160`, the decision cut-off (see §7)
- **Features** — `30`, how many inputs the model consumes
- **scikit-learn** — the version that trained it, which must match the version
  running the API
- **Trained** — the timestamp of the training run that produced this bundle

If you retrain, these badges change on their own. They are not hardcoded.

### Result panel — appears only after you submit

This is the part to walk a grader through:

- **A large verdict**: "Predicted: Fatal" in red, or "Predicted: Non-Fatal" in
  green.
- **A sentence explaining the decision** — the estimated probability of a fatal
  outcome, and explicitly whether that sits above or below the deployed
  threshold. The classification is not sklearn's default `.predict()`; it is the
  probability compared against the tuned cut-off.
- **A probability meter.** The filled bar is P(fatal). The thin vertical mark is
  the decision threshold. Because the threshold is 0.016, that mark sits very
  close to the left edge — which is a visual explanation of why the model flags
  records that look low-probability at first glance.
- **A "Notes on this input" box**, when relevant. This is where the API is honest
  about what it did to your request: fields you left blank and what it
  substituted, values outside the training range, categories it never saw, and
  any `TIME` → `HOUR` conversion.

### "Load a held-out test record"

A dropdown of 25 records from the test split, each labelled with its **recorded
outcome**. Picking one fills in all 30 fields below and shows the true answer in
a badge.

This is the piece that satisfies the project requirement to test the client with
data not used in training. The grouped train/test split kept whole crashes on one
side, so these rows are genuinely unseen. Select one, submit, and compare the
prediction against the recorded outcome.

### The form — all 30 features in 5 groups

Grouping is presentation only; every field is submitted either way.

| Group | Fields | What it covers |
|---|---|---|
| **Location** | 7 | Latitude, longitude, district, division, neighbourhood, accident location type, road class |
| **Time & environment** | 4 | Hour of day, lighting, visibility, road surface |
| **Collision details** | 4 | Impact type, traffic control, initial direction, vehicle type |
| **Person involved** | 2 | Involvement type and age band — the person being scored |
| **Contributing factors** | 13 | Yes/No flags: pedestrian, cyclist, speeding, alcohol, aggressive driving, red light, and so on |

Field-level details:

- Each label carries the **justification** for keeping that feature, taken from
  the feature-selection dictionary in the training script.
- Numeric fields show their **training range**. You can go outside it, but the
  response will warn you the prediction is an extrapolation.
- Categorical fields are dropdowns built from the **fitted encoder's own
  categories**, so the form can only offer values the model was trained on.
- Binary flags are explicit **Yes/No dropdowns rather than checkboxes**. An
  unchecked checkbox sends no value at all, which the server would fill with the
  training default instead of "No" — silently changing the record you submitted.

### "Deployed model" panel

The honest scorecard, read from the bundle: the algorithm, the threshold and how
it was chosen, and test-set recall, precision, F1/F2, accuracy, and ROC-AUC. It
closes with the scope disclaimer — this is a screening aid built on historical
reports, not a verdict on any individual incident.

---

## 5. What the Python client proves

`client.py` runs six stages, in this order:

1. **Health check** — the API is up, and reports which model and threshold are loaded.
2. **Fetch held-out records** — pulls the 25 unseen test rows.
3. **Single prediction** — one record, full JSON response shown.
4. **Batch prediction** — all 25 scored at once, printed as a predicted-vs-actual
   table with a confusion breakdown (TP / FN / FP / TN) and the sample's accuracy,
   recall, and precision.
5. **Partial input** — sends only five fields to show the API fills the rest with
   training defaults and discloses every substitution.
6. **Error handling** — four bad requests that must be rejected cleanly.

### Reading the output correctly

**The 400 and 404 responses in stage 6 are the pass condition**, not failures.
They also show up in the Flask log in Terminal 1. If those had returned 200, that
would be the bug.

**Sample accuracy is lower than test accuracy, and that is expected.** The 25-row
sample is deliberately enriched to 40% Fatal (10 of 25) versus the real ~14% base
rate, so it over-weights the class the model is worst at. The honest numbers are
the full-test-set figures in `/health`.

**`MISS` rows are model errors, not software errors.** Catching about half the
fatalities matches the measured 53.7% recall.

---

## 6. API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Browser form |
| `POST` | `/predict-form` | Form submission, re-renders the page with the result |
| `POST` | `/predict` | JSON in, JSON out — the main analytics endpoint |
| `POST` | `/predict/batch` | Scores a list of records in one request |
| `GET` | `/schema` | Every accepted field, with valid options and ranges |
| `GET` | `/samples` | Held-out records with their recorded outcomes |
| `GET` | `/health` | Liveness, loaded model, threshold, test metrics |

Example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
     -H "Content-Type: application/json" \
     -d '{"INVTYPE": "Pedestrian", "SPEEDING": 1, "ALCOHOL": 0, "LIGHT": "Dark"}'
```

The response gives `prediction`, `label`, `probability_fatal`,
`probability_non_fatal`, the `threshold` used, the `model` name, and any
`warnings`.

Three input conveniences, each mirroring a training-time transformation:

- **Partial bodies work.** Missing fields get the training default, and every
  substitution is listed in `warnings`.
- **`TIME` is accepted instead of `HOUR`** (`1430` → hour 14), matching the
  `TIME // 100` feature engineering.
- **Flags accept `1`/`0`, `"Yes"`/`"No"`, or `true`/`false`.**

Invalid input returns HTTP 400 with a readable explanation, never a stack trace
and never a silent guess.

---

## 7. Decisions you should be able to defend

### Why the neural network

It had both the highest test F1 (0.340) and the highest ROC-AUC (0.736). The
ROC-AUC is the stronger argument: it is threshold-independent, so the
best-ranking model stays the right pick even after the cut-off is retuned.

### Why the threshold is 0.0160 and not 0.5

The default 0.5 treats both error types as equally costly. The whole framing of
this project is that a **false negative — a fatal collision predicted non-fatal —
is the expensive error**.

The obvious fix, maximising F-beta with beta=2 to weight recall, is a poor
criterion here. Precision on the Fatal class is floored near the ~15% class
prevalence while recall can be pushed toward 1.0, so F2 rewards ever-lower
cut-offs. Its optimum falls below 0.01 and scores 0.505 against **0.474 for a
classifier that simply labels everything Fatal** — a margin narrow enough that
most of the score comes from indiscriminate flagging rather than discrimination.

**Youden's J** (`TPR − FPR`) is used instead. It cannot degenerate: J = 0 for both
all-positive and all-negative predictions, so its maximum is necessarily an
interior, genuinely discriminating point. It is also the standard cut-point
criterion for the ROC curve already plotted in Deliverable 4. J is maximised over
**every distinct predicted probability** via `roc_curve` (742 candidates), not a
fixed grid — a grid can only report an optimum at one of its own points, so a
maximum near the edge is indistinguishable from one the grid is too coarse to
resolve.

Critically, the threshold is tuned on a **validation split carved out of the
training data, never on the test set**. Choosing a cut-off that maximises a score
on the test set and then reporting that score would be leakage.

### What the retuning bought

| Threshold | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.50 (default) | 82.4% | 35.4% | 32.7% | 0.340 |
| **0.0160 (deployed)** | 76.2% | 30.0% | **53.7%** | **0.385** |

**104 more fatalities caught** on the test set (missed fatalities fall 333 → 229),
paid for with 325 more false alarms (296 → 621).

**Be ready for the accuracy question.** Accuracy drops 6 points, and if your
report leads with accuracy this looks like a regression. It is a deliberate trade:
for a tool that flags records for human review, a false alarm costs a review while
a missed fatality costs the outcome the model exists to surface. Note that F1
improves too, so the recall gain is not simply bought at a proportional loss
elsewhere.

### One demo answer to prepare

Sending a pedestrian, at night, with alcohol involved can return
P(fatal) = 0.0000, which looks wrong. The reason is in the warnings directly
below it: 25 of the 30 fields fell back to training defaults, and those defaults
describe a typical non-fatal collision, so they outweigh the few fields you set.
Correct behaviour for a partial request — but worth being able to explain.

---

## 8. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'sklearn'` | Wrong interpreter. Use `/opt/anaconda3/bin/python`. |
| `FileNotFoundError: Model bundle not found` | `model_bundle.pkl` is missing. Run `public_safety_ml.py` to regenerate it. |
| `Address already in use` | An old API instance is still running: `pkill -f "python app.py"`. |
| `WARNING: model_bundle.pkl was created with scikit-learn X` | Version mismatch — a pickle is not portable across versions. Install the pinned version from `requirements.txt`, or retrain. |
| `GET /favicon.ico 404` in the log | Just the browser asking for a tab icon. Harmless. |
| `WARNING: This is a development server` | Normal. The project specifies localhost deployment. |
| Client says nothing is listening | Start `app.py` first, in a separate terminal. |

---

## 9. File map

| File | Role |
|---|---|
| `public_safety_ml.py` | Training pipeline; its Deliverable 5 section exports the bundle |
| `app.py` | Flask API and form layout |
| `model_service.py` | Bundle loading, request → DataFrame preparation, prediction |
| `templates/index.html` | The browser client |
| `client.py` | Python test client |
| `make_postman_collection.py` | Regenerates `postman_collection.json` from the bundle |
| `requirements.txt` | Pinned dependencies (scikit-learn pinned exactly) |
| `8_threshold_selection.png` | Threshold-selection figure for the report |

The Postman collection is generated from the bundle rather than hand-written, so
the saved request bodies can never drift out of sync with the model's actual
input contract.
