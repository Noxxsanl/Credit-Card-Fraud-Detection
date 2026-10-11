"""TC-12 — đối chiếu máy khách – máy chủ (UI-D1, T-54).

Giao diện tính TP/FP/FN trong trình duyệt (docs/07 §2), còn báo cáo, notebook và
``GET /threshold/preview`` tính bằng ``src/threshold.py``. Nếu hai bên lệch nhau, con số
trên màn hình demo sẽ không khớp con số trong báo cáo — và điều đó rất khó phát hiện bằng
mắt (docs/08 §2.2).

Có hai bản giao diện, mỗi bản một tệp UI-D1, và cả hai đều được đối chiếu:

- ``web/threshold.js`` — bản HTML + Alpine.js (giai đoạn 8), dạng UMD;
- ``frontend/src/lib/threshold.mjs`` — bản Next.js, dạng ES module.

Ca kiểm thử chạy **chính các tệp đó** bằng Node, không chép lại logic sang Python. Máy
không có Node thì tự bỏ qua.

Dữ liệu đưa sang Node bằng JSON: ``json.dumps`` ghi số thực bằng ``repr`` (biểu diễn ngắn
nhất khôi phục đúng số), ``JSON.parse`` đọc lại đúng số thực đó — giống hệt đường đi của
``GET /metrics?section=test_scores`` tới trình duyệt.
"""

import json
import re
import shutil
import subprocess

import numpy as np
import pytest

from src.config import METRICS_PATH, PROJECT_ROOT, THRESHOLD_PATH
from src.threshold import confusion_counts, metrics_at_threshold

NODE = shutil.which("node")
WEB_DIR = PROJECT_ROOT / "web"
FRONTEND_SRC = PROJECT_ROOT / "frontend" / "src"

IMPLEMENTATIONS = {
    "web": WEB_DIR / "threshold.js",
    "frontend": FRONTEND_SRC / "lib" / "threshold.mjs",
}

pytestmark = pytest.mark.skipif(NODE is None, reason="Cần Node.js để chạy mã JavaScript của giao diện (TC-12)")

# import() nạp được cả ES module lẫn tệp UMD/CommonJS (khi đó hàm nằm ở default)
_RUNNER = r"""
import { pathToFileURL } from "node:url";
const mod = await import(pathToFileURL(process.argv[1]).href);
const T = mod.default ?? mod;
let raw = "";
process.stdin.setEncoding("utf8");
for await (const chunk of process.stdin) raw += chunk;
const d = JSON.parse(raw);
const prepared = T.prepare(d.y_true, d.y_score);
const out = d.thresholds.map((t) => T.metricsAt(prepared, t, d.options));
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(params=sorted(IMPLEMENTATIONS))
def impl(request):
    path = IMPLEMENTATIONS[request.param]
    if not path.exists():
        pytest.skip(f"Không có {path.relative_to(PROJECT_ROOT)}")
    return path


def run_js(impl, y_true, y_score, thresholds, *, cost_fn, cost_fp, sample_fraction, days) -> list[dict]:
    payload = {
        "y_true": [int(v) for v in y_true],
        "y_score": [float(v) for v in y_score],
        "thresholds": [float(t) for t in thresholds],
        "options": {"costFn": cost_fn, "costFp": cost_fp, "sampleFraction": sample_fraction, "days": days},
    }
    done = subprocess.run([NODE, "--input-type=module", "-e", _RUNNER, str(impl)], input=json.dumps(payload),
                          capture_output=True, text=True, encoding="utf-8", timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def assert_parity(impl, y_true, y_score, thresholds, **options) -> list[dict]:
    """Từng ngưỡng: đếm khớp tuyệt đối, và mọi chỉ số khớp tới từng bit."""
    from_js = run_js(impl, y_true, y_score, thresholds, **options)
    assert len(from_js) == len(thresholds)
    for tau, js in zip(thresholds, from_js):
        tp, fp, fn, tn = confusion_counts(y_true, y_score, tau)
        assert (js["tp"], js["fp"], js["fn"], js["tn"]) == (tp, fp, fn, tn), f"τ = {tau!r}"

        py = metrics_at_threshold(y_true, y_score, tau, **options).as_dict()
        assert set(js) == set(py)
        for key, value in py.items():
            assert js[key] == value, f"τ = {tau!r}, {key}: JS {js[key]!r} ≠ Python {value!r}"
    return from_js


@pytest.fixture(scope="module")
def exported():
    """Điểm tập kiểm thử đúng như API phục vụ cho trình duyệt (metrics.json → test_scores)."""
    if not (METRICS_PATH.exists() and THRESHOLD_PATH.exists()):
        pytest.skip("Chưa có hiện vật — chạy notebooks/08_export_artifacts.ipynb")
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    threshold = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    return metrics, threshold


def sample_thresholds(y_true, y_score, threshold: dict) -> list[float]:
    """Khoảng 20 ngưỡng mẫu, dồn vào những chỗ dễ lệch nhất: đúng bằng một điểm có thật."""
    scores = np.asarray(y_score, dtype="float64")
    labels = np.asarray(y_true)
    positives = np.unique(scores[labels == 1])
    tau_star = float(threshold["default_threshold"])
    first_alert = float(scores[scores >= tau_star].min())

    picks = [
        *threshold["alternatives"].values(),                        # mọi phương án của threshold.json
        *positives[[0, 10, 40, 80, -1]],                            # 5 điểm gian lận có thật (hòa tại biên)
        first_alert,                                                # điểm nhỏ nhất còn bị cảnh báo ở τ*
        np.nextafter(first_alert, 0.0),                             # nhỏ hơn đúng một ulp
        np.nextafter(first_alert, 1.0),                             # lớn hơn đúng một ulp
        scores.min(), scores[labels == 0].max(), 1e-12, 1.0,        # hai đầu; điểm cao nhất của lớp
                                                                    # hợp lệ (điểm cao nhất cả tập là gian lận)
        *np.quantile(scores, [0.5, 0.99, 0.999]),                   # giữa và đuôi phân bố
    ]
    return [float(t) for t in picks]


def test_tc12_parity_on_exported_test_scores(impl, exported):
    metrics, threshold = exported
    y_true = metrics["test_scores"]["y_true"]
    y_score = metrics["test_scores"]["y_score"]
    thresholds = sample_thresholds(y_true, y_score, threshold)
    assert len(set(thresholds)) == len(thresholds) >= 20          # đều khác nhau — không ca nào thừa

    results = assert_parity(
        impl, y_true, y_score, thresholds,
        cost_fn=threshold["cost_false_negative"], cost_fp=threshold["cost_false_positive"],
        sample_fraction=metrics["dataset"]["test_fraction"], days=metrics["dataset"]["days"],
    )

    # Gắn với con số của báo cáo: tại τ* trình duyệt ra đúng ma trận nhầm lẫn trong metrics.json
    at_default = results[thresholds.index(threshold["default_threshold"])]
    expected = metrics["confusion_at_default"]
    assert {k: at_default[k] for k in ("tp", "fp", "fn", "tn")} == {k: expected[k] for k in ("tp", "fp", "fn", "tn")}


def test_tc12_parity_with_ties_at_the_threshold(impl):
    """Điểm làm tròn 3 chữ số → hàng trăm điểm trùng nhau; ngưỡng đặt đúng bằng các điểm đó.

    Đây là chỗ ``>`` và ``>=`` cho kết quả khác nhau. Chạy cả khi chưa có hiện vật.
    """
    rng = np.random.default_rng(42)
    y_true = (rng.random(5_000) < 0.02).astype(int)
    y_score = np.round(np.where(y_true == 1, rng.beta(5, 2, 5_000), rng.beta(1, 30, 5_000)), 3)
    values = np.unique(y_score)
    thresholds = [float(t) for t in values[np.linspace(0, values.size - 1, 17).astype(int)]] + [0.0, 0.5, 1.0]

    assert_parity(impl, y_true, y_score, thresholds, cost_fn=122.21, cost_fp=5.0, sample_fraction=0.2, days=2.0)


def test_tc12_parity_with_float32_scores(impl):
    """Điểm của XGBoost là float32 nâng lên float64 — đúng kiểu dữ liệu thật trong metrics.json."""
    rng = np.random.default_rng(7)
    y_true = (rng.random(3_000) < 0.05).astype(int)
    y_score = rng.random(3_000).astype("float32").astype("float64")
    thresholds = [float(t) for t in np.sort(y_score)[::150]]

    assert_parity(impl, y_true, y_score, thresholds, cost_fn=200.0, cost_fp=3.0, sample_fraction=0.25, days=1.0)


def test_empty_classes_do_not_divide_by_zero(impl):
    """Không có mẫu dương, hoặc không có cảnh báo: precision/recall/F1 về 0 như phía Python."""
    y_true = [0, 0, 0, 0]
    y_score = [0.1, 0.2, 0.3, 0.4]
    assert_parity(impl, y_true, y_score, [0.05, 0.25, 0.9], cost_fn=10.0, cost_fp=1.0, sample_fraction=1.0, days=1.0)


# Không có phép so sánh điểm – ngưỡng nào tự viết lại ngoài module UI-D1
_HANDMADE_COMPARISON = re.compile(r"score\s*[<>]=?\s*(?:this\.)?(?:tau|threshold)\b")


def test_web_uses_this_module():
    """TC-12 chỉ có nghĩa khi giao diện thật sự tính bằng đúng hàm vừa kiểm."""
    index = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    app = (WEB_DIR / "app.js").read_text(encoding="utf-8")

    assert re.search(r'<script src="threshold\.js"', index)
    assert "FraudThreshold.prepare(" in app
    assert "FraudThreshold.metricsAt(" in app
    assert not _HANDMADE_COMPARISON.search(app)


def test_frontend_uses_this_module():
    if not FRONTEND_SRC.exists():
        pytest.skip("Chưa có frontend/")
    sources = {p: p.read_text(encoding="utf-8") for p in FRONTEND_SRC.rglob("*.ts*")}
    importers = [p for p, text in sources.items() if re.search(r'from\s+"[^"]*/threshold\.mjs"', text)]

    assert importers, "không tệp nào của frontend nạp lib/threshold.mjs"
    assert any("prepare(" in sources[p] for p in importers)
    assert any("metricsAt(" in sources[p] for p in importers)
    for path, text in sources.items():
        assert not _HANDMADE_COMPARISON.search(text), path


# --------------------------------------------------------------------------
# Điểm chọn ngưỡng qua payload nhị phân (docs/11 M3)
#
# Từ review 2026-10-10, màn hình Ngưỡng tính trên điểm out-of-fold do GET /threshold/oof-scores
# gửi dưới dạng nhị phân, không trên tập kiểm thử. Ca dưới chạy CHÍNH hàm decodeSortedScores của
# hai giao diện trên payload do src/artifacts.py dựng, rồi so từng chỉ số với Python.
# --------------------------------------------------------------------------

_DECODE_RUNNER = r"""
import { pathToFileURL } from "node:url";
import { readFileSync } from "node:fs";
const mod = await import(pathToFileURL(process.argv[1]).href);
const T = mod.default ?? mod;
const bytes = readFileSync(process.argv[2]);
const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
const d = JSON.parse(readFileSync(process.argv[3], "utf8"));
const decoded = T.decodeSortedScores(buffer);
const options = { costFn: d.costFn, costFp: d.costFp, sampleFraction: decoded.sampleFraction, days: decoded.days };
const out = d.thresholds.map((t) => T.metricsAt(decoded.prepared, t, options));
const meta = { sampleFraction: decoded.sampleFraction, days: decoded.days, n: decoded.prepared.n, nPos: decoded.prepared.nPos };
process.stdout.write(JSON.stringify({ meta, out }));
"""


def run_js_decode(impl, payload: bytes, thresholds, *, cost_fn, cost_fp, tmp_path) -> dict:
    blob = tmp_path / "scores.bin"
    blob.write_bytes(payload)
    params = tmp_path / "params.json"
    params.write_bytes(json.dumps({"thresholds": [float(t) for t in thresholds],
                                   "costFn": cost_fn, "costFp": cost_fp}).encode("utf-8"))
    done = subprocess.run([NODE, "--input-type=module", "-e", _DECODE_RUNNER, str(impl), str(blob), str(params)],
                          capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def assert_decoded_parity(impl, y_true, y_score, thresholds, *, sample_fraction, days, cost_fn, cost_fp, tmp_path):
    from src.artifacts import sorted_scores_payload

    payload = sorted_scores_payload(y_true, y_score, sample_fraction=sample_fraction, days=days)
    result = run_js_decode(impl, payload, thresholds, cost_fn=cost_fn, cost_fp=cost_fp, tmp_path=tmp_path)
    assert result["meta"] == {"sampleFraction": sample_fraction, "days": days,
                              "n": len(y_true), "nPos": int(np.sum(y_true))}
    for tau, js in zip(thresholds, result["out"]):
        py = metrics_at_threshold(y_true, y_score, tau, cost_fn=cost_fn, cost_fp=cost_fp,
                                  sample_fraction=sample_fraction, days=days).as_dict()
        assert set(js) == set(py)
        for key, value in py.items():
            assert js[key] == value, f"τ = {tau!r}, {key}: JS {js[key]!r} ≠ Python {value!r}"


def test_tc12_parity_on_exported_selection_scores(impl, exported, tmp_path):
    """Đúng đường đi thật: oof_scores.npz → payload nhị phân → trình duyệt → chỉ số."""
    from src.artifacts import read_oof_scores
    from src.config import OOF_SCORES_PATH

    if not OOF_SCORES_PATH.exists():
        pytest.skip("Chưa có models/oof_scores.npz")
    _, threshold = exported
    oof = read_oof_scores(OOF_SCORES_PATH)
    thresholds = sample_thresholds(oof["y_true"], oof["y_score"], threshold)
    assert_decoded_parity(impl, oof["y_true"], oof["y_score"], thresholds,
                          sample_fraction=oof["train_fraction"], days=oof["days"],
                          cost_fn=threshold["cost_false_negative"], cost_fp=threshold["cost_false_positive"],
                          tmp_path=tmp_path)


@pytest.mark.parametrize("dtype", ("float32", "float64"))
def test_tc12_parity_of_the_binary_payload_on_synthetic_scores(impl, dtype, tmp_path):
    """float32 đi đường 4 byte; điểm không biểu diễn đúng bằng float32 phải đi đường 8 byte."""
    rng = np.random.default_rng(13)
    y_true = (rng.random(4_000) < 0.03).astype(int)
    y_score = rng.random(4_000)
    if dtype == "float32":
        y_score = y_score.astype("float32").astype("float64")
    thresholds = [float(t) for t in np.sort(y_score)[::173]] + [0.0, 0.5, 1.0]
    assert_decoded_parity(impl, y_true, y_score, thresholds, sample_fraction=0.8, days=1.6,
                          cost_fn=115.94, cost_fp=5.0, tmp_path=tmp_path)


def _threshold_screen_block(app: str) -> str:
    start = app.index('Alpine.data("thresholdScreen"')
    end = app.index("Alpine.data(", start + 1)
    return app[start:end]


def test_web_threshold_screen_uses_selection_scores_not_the_test_set():
    """Kéo thanh trượt là đang CHỌN ngưỡng — mọi con số phải trên điểm chọn ngưỡng (docs/11 M3)."""
    app = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "FraudThreshold.decodeSortedScores(" in app and "/threshold/oof-scores" in app
    block = _threshold_screen_block(app)
    assert "Selection.metrics(" in block
    assert "Scores.metrics(" not in block and "Scores.perDay(" not in block


def test_frontend_threshold_screen_uses_selection_scores_not_the_test_set():
    if not FRONTEND_SRC.exists():
        pytest.skip("Chưa có frontend/")
    scores_lib = (FRONTEND_SRC / "lib" / "scores.ts").read_text(encoding="utf-8")
    context = (FRONTEND_SRC / "context" / "AppContext.tsx").read_text(encoding="utf-8")
    screen = (FRONTEND_SRC / "components" / "screens" / "ThresholdScreen.tsx").read_text(encoding="utf-8")
    assert "decodeSortedScores(" in scores_lib
    assert '"/threshold/oof-scores"' in context
    # Màn hình không lấy bộ điểm tập kiểm thử ra để tính
    assert not re.search(r"const \{[^}]*\bscores\b[^}]*\} = app", screen)
    assert re.search(r"const \{[^}]*\bselection\b[^}]*\} = app", screen)
