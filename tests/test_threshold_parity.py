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
    """20 ngưỡng mẫu, dồn vào những chỗ dễ lệch nhất: đúng bằng một điểm có thật."""
    scores = np.asarray(y_score, dtype="float64")
    labels = np.asarray(y_true)
    positives = np.unique(scores[labels == 1])
    tau_star = float(threshold["default_threshold"])
    first_alert = float(scores[scores >= tau_star].min())

    picks = [
        *threshold["alternatives"].values(),                        # 5 phương án của threshold.json
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
    assert len(set(thresholds)) == 20

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
