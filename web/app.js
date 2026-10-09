/*
 * Fraud Review Console — logic phía trình duyệt (docs/07).
 *
 * Alpine.js giữ trạng thái và dựng giao diện; không có công cụ đóng gói, không node_modules
 * (docs/03 §6.2). Bảy biến trạng thái của 07 §9 nằm trong Alpine.store("app") và trong từng
 * thành phần màn hình. Riêng mảng điểm của tập kiểm thử (56.746 phần tử) và các đối tượng
 * Chart.js/EventSource được giữ NGOÀI trạng thái phản ứng của Alpine: bọc chúng trong Proxy
 * làm mỗi lần đọc chậm đi và làm hỏng Chart.js.
 *
 * Mọi chỉ số theo ngưỡng tính bằng FraudThreshold (web/threshold.js) — cùng hàm mà TC-12
 * đối chiếu với src/threshold.py. Tệp này không tự so điểm với ngưỡng ở bất kỳ đâu.
 */
(function () {
  "use strict";

  const CONFIG = window.FRAUD_CONSOLE_CONFIG || {};
  const API_BASE = (CONFIG.apiBase || "http://localhost:8000/api/v1").replace(/\/$/, "");

  // --------------------------------------------------------------------------
  // Định dạng số kiểu Việt Nam — dấu phẩy thập phân, dấu chấm hàng nghìn (UI-D3)
  // --------------------------------------------------------------------------

  const LOCALE = "vi-VN";
  const formats = new Map();
  const numberFormat = (options) => {
    const key = JSON.stringify(options);
    if (!formats.has(key)) formats.set(key, new Intl.NumberFormat(LOCALE, options));
    return formats.get(key);
  };
  const missing = (v) => v == null || !Number.isFinite(v);

  const fmt = {
    int(v) {
      return missing(v) ? "—" : numberFormat({ maximumFractionDigits: 0 }).format(v);
    },
    num(v, digits = 2) {
      return missing(v) ? "—" : numberFormat({ minimumFractionDigits: digits, maximumFractionDigits: digits }).format(v);
    },
    /** Số có dấu, dùng dấu trừ thật (−) cho dễ đọc: +2,31 / −0,45 */
    signed(v, digits = 2) {
      if (missing(v)) return "—";
      const body = fmt.num(Math.abs(v), digits);
      return v > 0 ? `+${body}` : v < 0 ? `−${body}` : body;
    },
    pct(v, digits = 1) {
      return missing(v) ? "—" : `${fmt.num(v * 100, digits)}%`;
    },
    /** Điểm rủi ro dạng phần trăm; không làm tròn 0,99999 thành "100%" hay 0,00001 thành "0%". */
    score(v) {
      if (missing(v)) return "—";
      if (v > 0 && v < 0.0005) return "< 0,1%";
      if (v < 1 && v >= 0.9995) return "> 99,9%";
      return fmt.pct(v, 1);
    },
    /**
     * Ngưỡng: 3 chữ số thập phân khi ≥ 0,1; nhỏ hơn thì 4 chữ số có nghĩa (0,02317).
     * Sát 1 thì thêm chữ số cho tới khi thấy được nó nhỏ hơn 1: 0,99986 hiện "0,9999", không phải "1,000".
     */
    tau(v) {
      if (missing(v)) return "—";
      if (v < 0.1) return numberFormat({ maximumSignificantDigits: 4 }).format(v);
      let digits = 3;
      while (digits < 8 && v < 1 && Number(v.toFixed(digits)) >= 1) digits++;
      return fmt.num(v, digits);
    },
    /** Số thực nguyên độ chính xác, chỉ đổi dấu thập phân */
    raw(v) {
      return missing(v) ? "—" : String(v).replace(".", ",");
    },
    tauTick(v) {
      return numberFormat({ maximumSignificantDigits: 1 }).format(v);
    },
    eur(v, digits = 2) {
      return missing(v) ? "—" : `${fmt.num(v, digits)} EUR`;
    },
    /** Giây trong ngày → HH:MM:SS */
    clock(seconds) {
      if (missing(seconds)) return "—";
      const s = Math.max(0, Math.floor(seconds)) % 86400;
      const pad = (n) => String(n).padStart(2, "0");
      return `${pad(Math.floor(s / 3600))}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}`;
    },
    /** Cột Time (giây kể từ giao dịch đầu tiên) → "ngày 1 · 02:14:33" */
    timeOffset(seconds) {
      if (missing(seconds)) return "—";
      return `ngày ${Math.floor(seconds / 86400) + 1} · ${fmt.clock(seconds)}`;
    },
    hour(h) {
      return missing(h) ? "—" : `${String(h).padStart(2, "0")}h`;
    },
    date(iso) {
      return iso ? new Date(iso).toLocaleDateString(LOCALE) : "—";
    },
    datetime(iso) {
      return iso ? new Date(iso).toLocaleString(LOCALE, { dateStyle: "short", timeStyle: "medium" }) : "—";
    },
    feature(name) {
      return FEATURE_NOTES[name] ? `${name} (${FEATURE_NOTES[name]})` : name;
    },
    featureValue(name, value) {
      return name === "Amount" ? `${fmt.num(value, 2)} EUR` : fmt.num(value, 2);
    },
  };

  const FEATURE_NOTES = { Amount: "số tiền", hour_sin: "giờ trong ngày", hour_cos: "giờ trong ngày" };

  const LABELS = {
    decision: { allow: "Cho qua", review: "Cần thẩm định", block: "Đề xuất chặn" },
    review: { pending: "Chờ xử lý", confirmed_fraud: "Xác nhận gian lận", false_alarm: "Báo động sai" },
    source: { upload: "Tải CSV", sample: "Mẫu", replay: "Phát lại", manual: "Nhập tay" },
    model: {
      logistic_regression: "Hồi quy logistic",
      decision_tree: "Cây quyết định",
      random_forest: "Rừng ngẫu nhiên",
      xgboost: "XGBoost",
    },
    strategy: {
      none: "Không xử lý",
      class_weight: "Trọng số lớp",
      smote: "SMOTE",
      smote_tomek: "SMOTE + Tomek",
      undersample: "Giảm mẫu lớp đa số",
    },
    split: {
      random_stratified: "Ngẫu nhiên phân tầng 80/20 (dùng trong báo cáo)",
      random_downsized_day1: "Ngẫu nhiên, tập huấn luyện thu nhỏ bằng ngày 1",
      temporal: "Theo thời gian: học ngày 1, kiểm ngày 2",
    },
    category: {
      fraud_easy: "Gian lận dễ",
      fraud_hard: "Gian lận khó",
      legit_easy: "Hợp lệ dễ",
      legit_hard: "Hợp lệ khó",
    },
  };

  /** Ngưỡng gợi ý sẵn của UI-03 — khóa của GET /threshold → alternatives, theo thứ tự 07 §5. */
  const ALTERNATIVES = [
    { key: "default_naive", label: "Mặc định 0,5", hint: "Không có cơ sở cho dữ liệu mất cân bằng 1:578" },
    { key: "min_expected_cost", label: "Cực tiểu chi phí kỳ vọng (τ*)", hint: "Chọn trên out-of-fold với chi phí của threshold.json" },
    { key: "max_f1", label: "F1 lớn nhất", hint: "Dùng khi không ước lượng được chi phí" },
    { key: "recall_at_least_90", label: "Ràng buộc recall ≥ 90%", hint: "Ngưỡng lớn nhất còn giữ recall 90% trên out-of-fold" },
    { key: "budget_200_alerts", label: "Ngân sách 200 lượt thẩm định/ngày", hint: "Ngưỡng nhỏ nhất còn nằm trong ngân sách trên out-of-fold" },
  ];

  /** Màu theo mức điểm tuyệt đối (07 §3): đỏ ≥ 90%, cam 60–90%, vàng < 60% — luôn kèm hình dạng. */
  function riskLevel(score) {
    if (score >= 0.9) return { key: "critical", shape: "▲", text: "rất cao (≥ 90%)" };
    if (score >= 0.6) return { key: "serious", shape: "◆", text: "cao (60–90%)" };
    return { key: "warning", shape: "●", text: "dưới 60%" };
  }

  const TITLES = {
    queue: "Hàng đợi thẩm định",
    threshold: "Cấu hình ngưỡng quyết định",
    performance: "Hiệu năng mô hình",
    replay: "Chế độ phát lại",
  };

  function routeFromHash() {
    const name = location.hash.replace(/^#\/?/, "").split("?")[0];
    return name in TITLES ? name : "queue";
  }

  /** Làm tròn lên một mốc "đẹp" cho trục: 1, 2, 2,5, 5 × 10ⁿ */
  function niceCeil(v) {
    if (!(v > 0)) return 1;
    const magnitude = 10 ** Math.floor(Math.log10(v));
    const step = [1, 2, 2.5, 5, 10].find((s) => s * magnitude >= v);
    return step * magnitude;
  }

  /** Tên mô hình trong baseline_comparison của metrics.json → nhãn tiếng Việt */
  function modelLabel(name) {
    if (LABELS.model[name]) return LABELS.model[name];
    if (name === "mô hình rỗng") return "Mô hình rỗng (luôn đoán hợp lệ)";
    const exported = name.match(/^(\w+) \+ (\w+) \((.+)\)$/);
    if (exported) {
      return `${LABELS.model[exported[1]] || exported[1]} + ${(LABELS.strategy[exported[2]] || exported[2]).toLowerCase()} (${exported[3]})`;
    }
    return name.charAt(0).toUpperCase() + name.slice(1);
  }

  const logit = (p) => Math.log(p / (1 - p));
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
  const emit = (name, detail) => window.dispatchEvent(new CustomEvent(name, { detail }));

  // --------------------------------------------------------------------------
  // Gọi API — mô hình lỗi thống nhất của docs/05 §4
  // --------------------------------------------------------------------------

  class ApiError extends Error {
    constructor(status, code, message, details) {
      super(message);
      this.status = status;
      this.code = code;
      this.details = details || null;
    }
  }

  async function api(path, { method = "GET", json, form, signal } = {}) {
    let response;
    try {
      response = await fetch(API_BASE + path, {
        method,
        headers: json === undefined ? undefined : { "Content-Type": "application/json" },
        body: json === undefined ? form : JSON.stringify(json),
        signal,
      });
    } catch (error) {
      if (error.name === "AbortError") throw error;
      throw new ApiError(0, "NETWORK", `Không kết nối được API tại ${API_BASE}. API đã chạy chưa?`);
    }
    const text = await response.text();
    let body = null;
    try {
      body = text ? JSON.parse(text) : null;
    } catch {
      body = null;
    }
    if (!response.ok) {
      const err = (body && body.error) || {};
      throw new ApiError(response.status, err.code || `HTTP_${response.status}`,
        err.message || `Lỗi HTTP ${response.status}`, err.details);
    }
    return body;
  }

  const query = (params) => {
    const q = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) if (v !== null && v !== undefined && v !== "") q.set(k, v);
    const s = q.toString();
    return s ? `?${s}` : "";
  };

  // --------------------------------------------------------------------------
  // Điểm của tập kiểm thử — UI-D1. Giữ ngoài Alpine (không phản ứng).
  // --------------------------------------------------------------------------

  const Scores = {
    prepared: null,
    dataset: null,
    metrics(tau, costFn, costFp) {
      return FraudThreshold.metricsAt(this.prepared, tau, {
        costFn,
        costFp,
        sampleFraction: this.dataset.test_fraction,
        days: this.dataset.days,
      });
    },
    /** Quy đổi một đại lượng đếm trên tập kiểm thử (20% của hai ngày) ra mỗi ngày trên toàn luồng. */
    perDay(value) {
      return value / this.dataset.test_fraction / this.dataset.days;
    },
  };

  // --------------------------------------------------------------------------
  // Trạng thái toàn cục (07 §9)
  // --------------------------------------------------------------------------

  document.addEventListener("alpine:init", () => {
    const Alpine = window.Alpine;

    Alpine.store("app", {
      route: routeFromHash(),
      apiBase: API_BASE,
      health: { state: "loading", code: null, message: "", db: null },
      modelVersion: null,
      modelChanged: false,
      trainedAt: null,
      // GET /threshold: current, source, default, cost_fn, cost_fp, alternatives, model_version,
      // block_threshold (từ điểm này trở lên, và ≥ current, API trả decision = "block" — chọn theo
      // precision trên OOF), block_min_precision
      threshold: null,
      thresholdError: null,
      scoresReady: false,
      scoresError: null,
      toasts: [],

      get title() {
        return TITLES[this.route];
      },
      get tau() {
        return this.threshold ? this.threshold.current : null;
      },
      /** Tóm tắt của ngưỡng hiện hành trên tập kiểm thử — dòng đầu UI-01, ma trận ở UI-04. */
      get summary() {
        if (!this.scoresReady || !this.threshold) return null;
        return Scores.metrics(this.threshold.current, this.threshold.cost_fn, this.threshold.cost_fp);
      },
      get nTest() {
        return this.scoresReady ? Scores.prepared.n : null;
      },

      async boot() {
        window.addEventListener("hashchange", () => {
          this.route = routeFromHash();
        });
        await this.checkHealth();
        await Promise.allSettled([this.loadThreshold(), this.loadScores(), this.loadModelInfo()]);
        setInterval(() => this.checkHealth(), 15000);
      },

      async checkHealth() {
        try {
          const h = await api("/health");
          this.health = { state: "ok", code: null, message: "", db: h.db };
          this.noteModel(h.model_version);
          if (!this.threshold) this.loadThreshold();
          if (!this.scoresReady) this.loadScores();
          if (!this.trainedAt) this.loadModelInfo();
        } catch (e) {
          const details = e.details || {};
          this.health = { state: "error", code: e.code, message: e.message, db: details.db || null, reason: details.reason };
          if (details.model_version) this.noteModel(details.model_version);
        }
      },

      noteModel(version) {
        if (!version) return;
        if (this.modelVersion && this.modelVersion !== version) this.modelChanged = true;
        this.modelVersion = version;
      },

      async loadThreshold() {
        try {
          this.threshold = await api("/threshold");
          this.thresholdError = null;
          this.noteModel(this.threshold.model_version);
        } catch (e) {
          this.thresholdError = e.message;
        }
      },

      async loadScores() {
        try {
          const [scores, dataset] = await Promise.all([
            api("/metrics?section=test_scores"),
            api("/metrics?section=dataset"),
          ]);
          Scores.dataset = dataset.dataset;
          Scores.prepared = FraudThreshold.prepare(scores.test_scores.y_true, scores.test_scores.y_score);
          this.scoresReady = true;
          this.scoresError = null;
        } catch (e) {
          this.scoresError = e.message;
        }
      },

      async loadModelInfo() {
        try {
          const info = await api("/metrics?section=trained_at");
          this.trainedAt = info.trained_at;
          this.noteModel(info.model_version);
        } catch {
          /* chân thanh bên để trống; /health báo lỗi riêng */
        }
      },

      /** Sau PUT /threshold thành công: cập nhật toàn ứng dụng (UI-D4). */
      setThreshold(state) {
        this.threshold = state;
        emit("threshold-changed", state);
      },

      openTx(id, returnFocus) {
        emit("open-tx", { id, returnFocus });
      },

      toast(message, kind = "info") {
        const id = Date.now() + Math.random();
        this.toasts.push({ id, message, kind });
        setTimeout(() => {
          this.toasts = this.toasts.filter((t) => t.id !== id);
        }, kind === "error" ? 8000 : 4500);
      },
    });

    // ------------------------------------------------------------------------
    // UI-01 — Hàng đợi thẩm định
    // ------------------------------------------------------------------------

    Alpine.data("queueScreen", () => {
      let requestId = 0;
      return {
        items: [],
        total: 0,
        page: 1,
        pageSize: 25,
        sort: "-risk_score",
        status: "all",
        batchId: null,
        loading: false,
        loaded: false,
        error: null,
        hasAnyData: null,       // bảng transactions có dòng nào không — phân biệt "rỗng" và "lọc hết"
        activeIndex: 0,
        uploading: false,
        uploadStarted: 0,
        uploadResult: null,
        uploadError: null,

        init() {
          this.$watch("$store.app.route", (route) => {
            if (route === "queue") this.load();
          });
          window.addEventListener("threshold-changed", () => {
            this.page = 1;
            this.load();
          });
          window.addEventListener("queue-changed", () => this.load());
          window.addEventListener("review-saved", (e) => {
            const item = this.items.find((it) => it.id === e.detail.id);
            if (item) {
              item.reviewed = true;
              item.review_decision = e.detail.decision;
            }
          });
          this.load();
        },

        get pages() {
          return Math.max(1, Math.ceil(this.total / this.pageSize));
        },

        /** ‹ 1 2 3 … 8 › — luôn có trang đầu, trang cuối và hai bên trang hiện tại */
        get pageItems() {
          const n = this.pages;
          const p = this.page;
          if (n <= 7) return Array.from({ length: n }, (_, i) => i + 1);
          const keep = [...new Set([1, 2, p - 1, p, p + 1, n - 1, n])].filter((x) => x >= 1 && x <= n).sort((a, b) => a - b);
          const out = [];
          keep.forEach((x, i) => {
            if (i && x - keep[i - 1] > 1) out.push(`gap-${x}`);
            out.push(x);
          });
          return out;
        },

        async load() {
          const id = ++requestId;
          this.loading = true;
          this.error = null;
          const params = { sort: this.sort, page: this.page, page_size: this.pageSize, batch_id: this.batchId };
          if (this.status !== "all") params.review_status = this.status;
          try {
            const data = await api(`/transactions${query(params)}`);
            if (id !== requestId) return;
            this.items = data.items;
            this.total = data.total;
            this.activeIndex = 0;
            if (data.total === 0) {
              const any = await api(`/transactions${query({ min_score: 0, page_size: 1 })}`);
              if (id !== requestId) return;
              this.hasAnyData = any.total > 0;
            } else {
              this.hasAnyData = true;
            }
          } catch (e) {
            if (id !== requestId) return;
            this.error = e.message;
          } finally {
            if (id === requestId) {
              this.loading = false;
              this.loaded = true;
            }
          }
        },

        setStatus(status) {
          this.status = status;
          this.page = 1;
          this.load();
        },

        setSort(key) {
          this.sort = this.sort === `-${key}` ? key : `-${key}`;
          this.page = 1;
          this.load();
        },

        ariaSort(key) {
          if (this.sort === key) return "ascending";
          if (this.sort === `-${key}`) return "descending";
          return "none";
        },

        sortArrow(key) {
          if (this.sort === key) return "▲";
          if (this.sort === `-${key}`) return "▼";
          return "";
        },

        goto(page) {
          if (typeof page !== "number" || page < 1 || page > this.pages || page === this.page) return;
          this.page = page;
          this.load();
        },

        clearBatch() {
          this.batchId = null;
          this.page = 1;
          this.load();
        },

        focusRow(i) {
          const row = this.$refs.tbody && this.$refs.tbody.querySelector(`[data-row="${i}"]`);
          if (row) {
            this.activeIndex = i;
            row.focus();
          }
        },

        /** ↑/↓ giữa các dòng, Enter mở chi tiết (07 §3). Chỉ một dòng nằm trong thứ tự Tab. */
        onRowKey(event, i) {
          const last = this.items.length - 1;
          const moves = { ArrowDown: i + 1, ArrowUp: i - 1, Home: 0, End: last };
          if (event.key in moves) {
            event.preventDefault();
            this.focusRow(clamp(moves[event.key], 0, last));
          } else if (event.key === "Enter") {
            event.preventDefault();
            this.open(this.items[i], event.currentTarget);
          }
        },

        open(item, element) {
          Alpine.store("app").openTx(item.id, element);
        },

        risk(item) {
          return riskLevel(item.risk_score);
        },

        statusOf(item) {
          return item.review_decision || "pending";
        },

        // --- Tải CSV (API-04) ---

        pickFile() {
          this.$refs.file.click();
        },

        async upload(event) {
          const file = event.target.files && event.target.files[0];
          event.target.value = "";
          if (!file) return;
          const form = new FormData();
          form.append("file", file);
          this.uploading = true;
          this.uploadStarted = performance.now();
          this.uploadError = null;
          this.uploadResult = null;
          try {
            const result = await api("/score/upload", { method: "POST", form });
            this.uploadResult = { ...result, fileName: file.name, clientMs: performance.now() - this.uploadStarted };
            Alpine.store("app").toast(`Đã chấm ${fmt.int(result.count)} giao dịch, ${fmt.int(result.alerts)} vượt ngưỡng`, "success");
            this.page = 1;
            this.load();
          } catch (e) {
            this.uploadError = e;
          } finally {
            this.uploading = false;
          }
        },

        showBatch(batchId) {
          this.batchId = batchId;
          this.status = "all";
          this.page = 1;
          this.load();
        },

        openSamples() {
          emit("open-samples");
        },
      };
    });

    // ------------------------------------------------------------------------
    // Thư viện giao dịch mẫu (API-14 → API-02)
    // ------------------------------------------------------------------------

    Alpine.data("samplesDialog", () => {
      let returnFocus = null;
      return {
        items: [],
        categories: {},
        category: "fraud_easy",
        loading: false,
        error: null,
        results: {},             // id mẫu → {transaction_id, risk_score, decision} | {error}
        busy: false,

        init() {
          window.addEventListener("open-samples", () => this.show());
        },

        async show() {
          returnFocus = document.activeElement;
          this.$refs.dialog.showModal();
          if (this.items.length || this.loading) return;
          this.loading = true;
          try {
            const data = await api("/samples");
            this.items = data.items;
            this.categories = data.categories;
          } catch (e) {
            this.error = e.message;
          } finally {
            this.loading = false;
          }
        },

        close() {
          this.$refs.dialog.close();
        },

        onClose() {
          if (returnFocus && document.contains(returnFocus)) returnFocus.focus();
          returnFocus = null;
        },

        get visible() {
          return this.items.filter((it) => it.category === this.category);
        },

        get pendingInCategory() {
          return this.visible.filter((it) => !this.results[it.id] || this.results[it.id].error);
        },

        async scoreOne(item) {
          try {
            const r = await api("/score", {
              method: "POST",
              json: { transaction: item.features, sample_id: item.id, persist: true },
            });
            this.results[item.id] = r;
            return r;
          } catch (e) {
            this.results[item.id] = { error: e.message };
            return null;
          }
        },

        async scoreSingle(item) {
          this.busy = true;
          const r = await this.scoreOne(item);
          this.busy = false;
          if (r) {
            emit("queue-changed");
            Alpine.store("app").toast(`${item.id} → ${r.transaction_id}: ${fmt.score(r.risk_score)}, ${LABELS.decision[r.decision].toLowerCase()}`, "success");
          }
        },

        /** Nạp cả nhóm, bốn yêu cầu song song — mỗi mẫu một lời gọi /score để giữ nhãn thật. */
        async scoreCategory() {
          const queue = [...this.pendingInCategory];
          if (!queue.length) return;
          this.busy = true;
          let done = 0;
          let alerts = 0;
          const worker = async () => {
            while (queue.length) {
              const r = await this.scoreOne(queue.shift());
              if (r) {
                done++;
                if (r.decision !== "allow") alerts++;
              }
            }
          };
          await Promise.all([worker(), worker(), worker(), worker()]);
          this.busy = false;
          emit("queue-changed");
          Alpine.store("app").toast(`Đã nạp ${done} mẫu nhóm "${LABELS.category[this.category]}", ${alerts} vào hàng đợi`, "success");
        },
      };
    });

    // ------------------------------------------------------------------------
    // UI-02 — Chi tiết giao dịch (ngăn kéo bên phải)
    // ------------------------------------------------------------------------

    Alpine.data("txDrawer", () => {
      let returnFocus = null;
      let requestId = 0;
      return {
        id: null,
        tx: null,
        loading: false,
        error: null,
        exp: null,
        expLoading: false,
        expError: null,
        saved: null,           // phản hồi POST /reviews trong lần mở này
        saving: false,
        note: "",

        init() {
          window.addEventListener("open-tx", (e) => this.show(e.detail.id, e.detail.returnFocus));
        },

        async show(id, focusElement) {
          returnFocus = focusElement || document.activeElement;
          const req = ++requestId;
          Object.assign(this, {
            id, tx: null, loading: true, error: null, exp: null, expLoading: true, expError: null,
            saved: null, saving: false, note: "",
          });
          const dialog = this.$refs.dialog;
          if (!dialog.open) dialog.showModal();
          this.$refs.closeButton.focus();

          // Hai lời gọi song song: điểm rủi ro hiện ngay, SHAP (60–100 ms) theo sau (07 §4)
          const detail = api(`/transactions/${encodeURIComponent(id)}`);
          const explanation = api("/explain", { method: "POST", json: { transaction_id: id } });
          try {
            const tx = await detail;
            if (req !== requestId) return;
            this.tx = tx;
            this.note = (tx.review && tx.review.note) || "";
          } catch (e) {
            if (req === requestId) this.error = e;
          } finally {
            if (req === requestId) this.loading = false;
          }
          try {
            const exp = await explanation;
            if (req === requestId) this.exp = exp;
          } catch (e) {
            if (req === requestId) this.expError = e;
          } finally {
            if (req === requestId) this.expLoading = false;
          }
        },

        retry() {
          this.show(this.id, returnFocus);
        },

        close() {
          this.$refs.dialog.close();
        },

        onClose() {
          requestId++;
          if (returnFocus && document.contains(returnFocus)) returnFocus.focus();
          returnFocus = null;
        },

        /** F xác nhận gian lận, A báo động sai (07 §4) — trừ khi đang gõ ghi chú. */
        onKey(event) {
          if (event.ctrlKey || event.metaKey || event.altKey) return;
          if (event.target.closest("textarea, input, select")) return;
          const key = event.key.toLowerCase();
          if (key === "f") {
            event.preventDefault();
            this.decide("confirmed_fraud");
          } else if (key === "a") {
            event.preventDefault();
            this.decide("false_alarm");
          }
        },

        get decided() {
          if (this.saved) return this.saved.decision;
          return this.tx && this.tx.review ? this.tx.review.decision : null;
        },

        /** Nhãn thật chỉ lộ SAU khi người dùng đã quyết định — để tự đối chiếu mà không bị mồi (07 §4). */
        get revealed() {
          return Boolean(this.decided);
        },

        get trueLabel() {
          if (!this.revealed) return null;
          return this.saved ? this.saved.true_label : this.tx.true_label;
        },

        get matches() {
          if (!this.revealed || this.trueLabel == null) return null;
          return (this.decided === "confirmed_fraud") === (this.trueLabel === 1);
        },

        get level() {
          return this.tx ? riskLevel(this.tx.risk_score) : null;
        },

        async decide(decision) {
          if (!this.tx || this.saving) return;
          this.saving = true;
          try {
            const r = await api("/reviews", {
              method: "POST",
              json: { transaction_id: this.tx.id, decision, note: this.note.trim() || null },
            });
            this.saved = r;
            emit("review-saved", { id: r.transaction_id, decision: r.decision });
            Alpine.store("app").toast(`${r.transaction_id}: đã ghi "${LABELS.review[r.decision]}"`, "success");
          } catch (e) {
            Alpine.store("app").toast(`Không ghi được kết luận: ${e.message}`, "error");
          } finally {
            this.saving = false;
          }
        },

        /**
         * Biểu đồ thác nước SHAP ở thang log-odds: base_value + Σ SHAP = margin (docs/05 §7).
         * Mỗi dòng là một đoạn [from, to]; phần trăm vị trí tính trên miền chung của cả biểu đồ.
         */
        get waterfall() {
          const e = this.exp;
          if (!e) return null;
          const tau = this.tx ? this.tx.threshold : Alpine.store("app").tau;
          const rows = [];
          let cum = e.base_value;
          const step = (kind, c, label) => {
            const from = cum;
            cum += c.shap;
            rows.push({ kind, key: `${kind}-${c.feature || label}`, feature: c.feature, label, value: c.value, shap: c.shap, from, to: cum });
          };
          rows.push({ kind: "base", key: "base", label: "Giá trị cơ sở E[f(x)]", from: cum, to: cum });
          if (e.top_positive.length) rows.push({ kind: "group", key: "g-pos", label: "Yếu tố đẩy điểm rủi ro lên" });
          e.top_positive.forEach((c) => step("pos", c));
          if (e.top_negative.length) rows.push({ kind: "group", key: "g-neg", label: "Yếu tố kéo điểm rủi ro xuống" });
          e.top_negative.forEach((c) => step("neg", c));
          const others = e.contributions.length - e.top_positive.length - e.top_negative.length;
          if (others > 0) step("rest", { shap: e.remaining_shap }, `${others} đặc trưng còn lại`);
          rows.push({ kind: "total", key: "total", label: "Kết quả f(x)", from: e.margin, to: e.margin });

          const tauLogit = tau > 0 && tau < 1 ? logit(tau) : null;
          const ends = rows.filter((r) => r.kind !== "group").flatMap((r) => [r.from, r.to]);
          if (tauLogit !== null) ends.push(tauLogit);
          let lo = Math.min(...ends);
          let hi = Math.max(...ends);
          const pad = (hi - lo) * 0.06 || 1;
          lo -= pad;
          hi += pad;
          const pos = (v) => ((v - lo) / (hi - lo)) * 100;
          rows.forEach((r) => {
            if (r.kind === "group") return;
            r.left = pos(Math.min(r.from, r.to));
            r.width = Math.max(Math.abs(pos(r.to) - pos(r.from)), r.kind === "base" || r.kind === "total" ? 0 : 0.6);
          });
          return {
            rows,
            tauPos: tauLogit === null ? null : pos(tauLogit),
            tauLogit,
            nPositive: e.top_positive.length,
            nNegative: e.top_negative.length,
            probability: sigmoid(e.margin),
          };
        },
      };
    });

    // ------------------------------------------------------------------------
    // UI-03 — Cấu hình ngưỡng
    // ------------------------------------------------------------------------

    Alpine.data("thresholdScreen", () => {
      let chart = null;
      let chartFrame = 0;
      let optimizeTimer = null;
      let optimizeController = null;
      let optimizeKey = null;                  // thân yêu cầu lần gần nhất — không gọi lại khi không đổi
      const LOG_MIN = -4;                      // τ = 0,0001
      const LOG_MAX = Math.log10(0.999);
      return {
        initialized: false,
        tau: null,
        costFnInput: "",
        costFpInput: "",
        constraintType: "none",
        constraintValueInput: "",
        m: null,                // chỉ số tại τ đang xem (tập kiểm thử, tính tại máy khách)
        appliedM: null,         // chỉ số tại ngưỡng đang áp dụng
        latencyMs: null,
        opt: null,              // phản hồi POST /threshold/optimize
        optLoading: false,
        optError: null,
        applying: false,
        logMin: LOG_MIN,
        logMax: LOG_MAX,

        init() {
          const tryInit = () => {
            const s = Alpine.store("app");
            if (this.initialized || !s.threshold || !s.scoresReady) return;
            this.initialized = true;
            this.tau = s.threshold.current;
            this.costFnInput = String(s.threshold.cost_fn);
            this.costFpInput = String(s.threshold.cost_fp);
            this.recalc();
            this.optimize();
          };
          this.$watch("$store.app.threshold", tryInit);
          this.$watch("$store.app.scoresReady", tryInit);
          this.$watch("$store.app.route", (route) => {
            if (route === "threshold") this.$nextTick(() => this.drawChart());
          });
          this.$watch("costFnInput", () => this.onCostsChanged());
          this.$watch("costFpInput", () => this.onCostsChanged());
          this.$watch("constraintType", () => this.scheduleOptimize());
          this.$watch("constraintValueInput", () => this.scheduleOptimize());
          window.addEventListener("threshold-changed", () => this.recalc());
          tryInit();
        },

        get store() {
          return Alpine.store("app");
        },

        get sliderValue() {
          return clamp(Math.log10(this.tau), LOG_MIN, LOG_MAX);
        },

        get costs() {
          const fn = Number(this.costFnInput);
          const fp = Number(this.costFpInput);
          const valid = this.costFnInput !== "" && this.costFpInput !== "" && Number.isFinite(fn) && Number.isFinite(fp) && fn > 0 && fp >= 0;
          return { fn, fp, valid };
        },

        get constraint() {
          if (this.constraintType === "none") return { valid: true, body: null };
          const value = Number(this.constraintValueInput);
          if (this.constraintValueInput === "" || !Number.isFinite(value)) return { valid: false, body: null };
          if (this.constraintType === "min_recall") {
            const recall = value / 100;           // nhập theo phần trăm
            return { valid: recall > 0 && recall <= 1, body: { type: "min_recall", value: recall } };
          }
          return { valid: value > 0, body: { type: "max_alerts_per_day", value } };
        },

        get dirty() {
          const t = this.store.threshold;
          if (!t || this.tau === null) return false;
          const c = this.costs;
          return this.tau !== t.current || (c.valid && (c.fn !== t.cost_fn || c.fp !== t.cost_fp));
        },

        recalc() {
          if (!this.initialized) return;
          const c = this.costs;
          const t = this.store.threshold;
          const fn = c.valid ? c.fn : t.cost_fn;
          const fp = c.valid ? c.fp : t.cost_fp;
          this.m = Scores.metrics(this.tau, fn, fp);
          this.appliedM = Scores.metrics(t.current, fn, fp);
          this.scheduleChart();
        },

        setTau(value) {
          if (!Number.isFinite(value) || value <= 0 || value >= 1) return;
          this.tau = value;
          this.recalc();
        },

        /** Từ lúc kéo tới lúc khung hình kế tiếp đã vẽ xong — hai lần requestAnimationFrame. */
        measure(started) {
          requestAnimationFrame(() => requestAnimationFrame(() => {
            this.latencyMs = performance.now() - started;
          }));
        },

        /** Thanh trượt theo thang log10: vùng quyết định 0,0005…0,97 trải ba bậc độ lớn. */
        onSlider(event) {
          const started = event.timeStamp || performance.now();
          this.setTau(10 ** Number(event.target.value));
          this.measure(started);
        },

        /** ←/→ một bước nhỏ, PgUp/PgDn một bước lớn, Home/End hai đầu (07 §5). */
        onSliderKey(event) {
          const steps = { ArrowLeft: -0.01, ArrowDown: -0.01, ArrowRight: 0.01, ArrowUp: 0.01, PageDown: -0.1, PageUp: 0.1 };
          let next = null;
          if (event.key in steps) next = this.sliderValue + steps[event.key];
          else if (event.key === "Home") next = LOG_MIN;
          else if (event.key === "End") next = LOG_MAX;
          if (next === null) return;
          event.preventDefault();
          const started = performance.now();
          this.setTau(10 ** clamp(next, LOG_MIN, LOG_MAX));
          this.measure(started);
        },

        onTauInput(event) {
          const value = Number(String(event.target.value).replace(",", "."));
          if (Number.isFinite(value) && value > 0 && value < 1) this.setTau(value);
          else event.target.value = this.tauInputValue;
        },

        get tauInputValue() {
          return this.tau === null ? "" : String(Number(this.tau.toPrecision(6)));
        },

        onCostsChanged() {
          this.recalc();
          this.scheduleOptimize();
        },

        scheduleOptimize() {
          clearTimeout(optimizeTimer);
          optimizeTimer = setTimeout(() => this.optimize(), 300);      // debounce 300 ms (07 §5)
        },

        async optimize() {
          const c = this.costs;
          const k = this.constraint;
          if (!c.valid || !k.valid) return;
          const body = { cost_fn: c.fn, cost_fp: c.fp };
          if (k.body) body.constraint = k.body;
          const key = JSON.stringify(body);
          if (key === optimizeKey) return;      // trước khi huỷ: yêu cầu đang chạy có thể chính là yêu cầu này
          optimizeKey = key;
          if (optimizeController) optimizeController.abort();
          optimizeController = new AbortController();
          this.optLoading = true;
          this.optError = null;
          try {
            this.opt = await api("/threshold/optimize", { method: "POST", json: body, signal: optimizeController.signal });
            this.scheduleChart();
          } catch (e) {
            if (e.name !== "AbortError") {
              this.optError = e.message;
              optimizeKey = null;
            }
          } finally {
            this.optLoading = false;
          }
        },

        useOptimal() {
          if (this.opt) this.setTau(this.opt.optimal_threshold);
        },

        /** Ngưỡng gợi ý sẵn (GET /threshold → alternatives) và nghiệm tối ưu theo tham số đang nhập. */
        get presets() {
          const t = this.store.threshold;
          if (!t || !this.initialized) return [];
          const c = this.costs;
          const fn = c.valid ? c.fn : t.cost_fn;
          const fp = c.valid ? c.fp : t.cost_fp;
          const list = ALTERNATIVES.filter((a) => a.key in t.alternatives).map((a) => ({ ...a, value: t.alternatives[a.key] }));
          if (this.opt && this.opt.optimal_threshold !== t.alternatives.min_expected_cost) {
            list.push({
              key: "optimized",
              label: this.constraintType === "none" ? "Tối ưu với tham số chi phí đang nhập" : "Tối ưu với tham số và ràng buộc đang nhập",
              hint: `Chi phí bỏ lọt ${fmt.num(this.opt.cost_fn)} EUR, thẩm định ${fmt.num(this.opt.cost_fp)} EUR — chọn trên out-of-fold`,
              value: this.opt.optimal_threshold,
            });
          }
          if (t.source === "user" && !list.some((p) => p.value === t.current)) {
            list.push({ key: "applied", label: "Ngưỡng đang áp dụng", hint: "Người dùng đã đặt", value: t.current });
          }
          return list.map((p) => {
            const m = Scores.metrics(p.value, fn, fp);
            return { ...p, recall: m.recall, alertsPerDay: m.alerts_per_day, costPerDay: Scores.perDay(m.expected_cost), tp: m.tp, nPos: m.tp + m.fn };
          });
        },

        delta(field) {
          if (!this.m || !this.appliedM || this.tau === this.store.tau) return null;
          return this.m[field] - this.appliedM[field];
        },

        /** Chênh lệch so với ngưỡng đang áp dụng — thứ khiến kịch bản kéo 0,5 → τ* dễ thấy (07 §5). */
        deltaInfo(field, { goodIf = null, perDay = false, unit = "" } = {}) {
          const d = this.delta(field);
          if (d === null) return null;
          const v = perDay ? Scores.perDay(d) : d;
          if (Math.abs(v) < 0.5) return { text: "như ngưỡng đang áp dụng", cls: "" };
          const text = `${v > 0 ? "▲ +" : "▼ −"}${fmt.int(Math.abs(v))}${unit} so với ngưỡng đang áp dụng`;
          const cls = goodIf === null ? "" : (v > 0) === (goodIf === "up") ? "delta-good" : "delta-bad";
          return { text, cls };
        },

        perDay(value) {
          return value == null ? null : Scores.perDay(value);
        },

        async apply() {
          const c = this.costs;
          if (!c.valid || this.applying) return;
          this.applying = true;
          try {
            const state = await api("/threshold", { method: "PUT", json: { value: this.tau, cost_fn: c.fn, cost_fp: c.fp } });
            this.store.setThreshold(state);
            this.store.toast(`Đã áp dụng ngưỡng ${fmt.tau(state.current)} cho toàn ứng dụng`, "success");
          } catch (e) {
            this.store.toast(`Không áp dụng được: ${e.message}`, "error");
          } finally {
            this.applying = false;
          }
        },

        resetToDefault() {
          if (this.store.threshold) this.setTau(this.store.threshold.default);
        },

        // --- Đường cong chi phí (Chart.js) ---

        scheduleChart() {
          if (chartFrame) return;
          chartFrame = requestAnimationFrame(() => {
            chartFrame = 0;
            this.drawChart();
          });
        },

        /** Điểm của đường cong: chi phí out-of-fold quy ra EUR/ngày, cùng thang với ô "Chi phí mỗi ngày". */
        curvePoints() {
          const curve = this.opt ? this.opt.curve : [];
          const ref = curve.find((p) => p.alerts > 0);
          const perDay = ref ? ref.alerts_per_day / ref.alerts : 0;   // = 1 / (tỷ lệ mẫu × số ngày)
          return curve
            .filter((p) => p.threshold >= 1e-4)
            .map((p) => ({ x: p.threshold, y: p.cost * perDay, apd: p.alerts_per_day, recall: p.recall }));
        },

        drawChart() {
          if (this.store.route !== "threshold" || !this.opt || !this.$refs.costCanvas) return;
          if (!chart) chart = FraudCharts.costCurve(this.$refs.costCanvas, fmt);
          if (chart.$source !== this.opt) {
            const data = this.curvePoints();
            chart.data.datasets[0].data = data;
            // Hai đầu đường cong vọt lên hàng chục nghìn EUR/ngày và đè bẹp vùng đáy chữ U —
            // chỗ duy nhất có quyết định. Cắt trục tung ở khoảng 2,5 lần cực tiểu.
            const ys = data.map((p) => p.y);
            const lowest = Math.min(...ys);
            chart.options.scales.y.max = niceCeil(lowest > 0 ? lowest * 2.5 : Math.max(...ys));
            chart.$source = this.opt;
          }
          chart.$markers = [
            { value: this.tau, color: FraudCharts.token("--ink"), width: 1.5, label: `đang xem ${fmt.tau(this.tau)}` },
            { value: this.opt.optimal_threshold, color: FraudCharts.token("--series-2"), width: 2, label: `tối ưu ${fmt.tau(this.opt.optimal_threshold)}` },
          ];
          chart.update("none");
        },

        get curveSummary() {
          if (!this.opt) return "";
          const o = this.opt;
          const m = o.metrics_at_optimal;
          return `Đường cong chi phí mỗi ngày theo ngưỡng, trục hoành thang log. Cực tiểu tại τ = ${fmt.tau(o.optimal_threshold)}; `
            + `tại đó trên tập kiểm thử bắt ${m.tp}/${m.tp + m.fn} gian lận với ${fmt.int(m.alerts_per_day)} cảnh báo mỗi ngày.`;
        },
      };
    });

    // ------------------------------------------------------------------------
    // UI-04 — Hiệu năng mô hình
    // ------------------------------------------------------------------------

    Alpine.data("performanceScreen", () => {
      const charts = [];
      const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5"];
      return {
        loaded: false,
        loading: false,
        error: null,
        s: {},                  // các mục của metrics.json

        init() {
          this.$watch("$store.app.route", (route) => {
            if (route === "performance") this.load();
          });
          if (Alpine.store("app").route === "performance") this.load();
        },

        async load() {
          if (this.loaded || this.loading) return;
          this.loading = true;
          this.error = null;
          const names = ["headline", "grid_results", "strategy_pr_curves", "baseline_comparison", "shap_global",
            "split_comparison", "pr_curve", "roc_curve", "training", "dataset"];
          try {
            const parts = await Promise.all(names.map((n) => api(`/metrics?section=${n}`)));
            const s = {};
            names.forEach((n, i) => {
              s[n] = parts[i][n];
            });
            this.s = s;
            this.loaded = true;
            this.$nextTick(() => this.drawCharts());
          } catch (e) {
            this.error = e.message;
          } finally {
            this.loading = false;
          }
        },

        get grid() {
          const rows = [...(this.s.grid_results || [])].sort((a, b) => b.pr_auc_mean - a.pr_auc_mean);
          const t = this.s.training || {};
          return rows.map((r, i) => ({ ...r, rank: i + 1, exported: r.model === t.model && r.strategy === t.strategy }));
        },

        get strategies() {
          const block = this.s.strategy_pr_curves;
          if (!block) return [];
          // Màu đi theo chiến lược, không theo thứ hạng — lọc hay sắp lại không đổi màu
          const order = ["class_weight", "smote", "smote_tomek", "none", "undersample"];
          return [...block.curves].sort((a, b) => order.indexOf(a.strategy) - order.indexOf(b.strategy)).map((c) => ({
            ...c,
            label: LABELS.strategy[c.strategy] || c.strategy,
            colorVar: SERIES[order.indexOf(c.strategy)] || "--ink-3",
          }));
        },

        get baselines() {
          const rows = this.s.baseline_comparison || [];
          return rows.map((r) => ({ ...r, label: modelLabel(r.model) }));
        },

        get shapRows() {
          return [...(this.s.shap_global || [])].sort((a, b) => a.rank_shap - b.rank_shap);
        },

        get splits() {
          const s = this.s.split_comparison || {};
          return Object.entries(s).map(([key, v]) => ({ key, label: LABELS.split[key] || key, ...v }));
        },

        get ciRows() {
          const h = this.s.headline;
          if (!h) return [];
          return [
            { key: "pr_auc", label: "PR-AUC", ...h.pr_auc },
            { key: "roc_auc", label: "ROC-AUC", ...h.roc_auc },
            { key: "recall", label: "Recall tại τ*", ...h.recall },
            { key: "precision", label: "Precision tại τ*", ...h.precision },
          ];
        },

        get modelCiRows() {
          return this.baselines.map((b) => ({ key: b.model, label: b.label, value: b.pr_auc, ci_low: b.pr_auc_ci_low, ci_high: b.pr_auc_ci_high }));
        },

        get confusion() {
          return Alpine.store("app").summary;
        },

        bar(v) {
          return `${clamp(v, 0, 1) * 100}%`;
        },

        rank(v) {
          return v == null || Number.isNaN(v) ? "—" : fmt.int(v);
        },

        drawCharts() {
          if (charts.length || !this.loaded) return;
          const token = FraudCharts.token;
          const block = this.s.strategy_pr_curves;
          if (block && this.$refs.prStrategies) {
            charts.push(FraudCharts.unitCurves(this.$refs.prStrategies, {
              fmt,
              xTitle: "Recall",
              yTitle: "Precision",
              series: this.strategies.map((c) => ({ label: c.label, x: c.recall, y: c.precision, color: token(c.colorVar) })),
              reference: { label: `Đường cơ sở ${fmt.num(block.baseline, 5)}`, points: [{ x: 0, y: block.baseline }, { x: 1, y: block.baseline }] },
            }));
          }
          const baseline = this.s.headline ? this.s.headline.baseline_pr_auc : null;
          if (this.s.pr_curve && this.$refs.prTest) {
            charts.push(FraudCharts.unitCurves(this.$refs.prTest, {
              fmt,
              legend: false,
              xTitle: "Recall",
              yTitle: "Precision",
              series: [{ label: "Mô hình xuất — tập kiểm thử", x: this.s.pr_curve.recall, y: this.s.pr_curve.precision, color: token("--series-1") }],
              reference: baseline === null ? null : { label: "Đường cơ sở", points: [{ x: 0, y: baseline }, { x: 1, y: baseline }] },
            }));
          }
          if (this.s.roc_curve && this.$refs.rocTest) {
            charts.push(FraudCharts.unitCurves(this.$refs.rocTest, {
              fmt,
              legend: false,
              xTitle: "FPR",
              yTitle: "TPR",
              series: [{ label: "Mô hình xuất — tập kiểm thử", x: this.s.roc_curve.fpr, y: this.s.roc_curve.tpr, color: token("--series-1") }],
              reference: { label: "Đoán ngẫu nhiên", points: [{ x: 0, y: 0 }, { x: 1, y: 1 }] },
            }));
          }
        },
      };
    });

    // ------------------------------------------------------------------------
    // UI-05 — Chế độ phát lại (SSE, API-15)
    // ------------------------------------------------------------------------

    Alpine.data("replayScreen", () => {
      let source = null;           // EventSource — ngoài Alpine
      let seen = new Set();        // mã đã đếm: tiếp tục sau tạm dừng phát lại đúng giây cuối
      let buffer = [];
      let frame = 0;
      let reconnectTimer = null;
      let clockTimer = null;
      let clockAnchor = null;      // {sim, wall, speed} — nội suy đồng hồ giữa hai sự kiện stats
      const SPEEDS = [1, 10, 30, 60, 120, 300, 600, 1800, 3600];
      const FEED = 30;
      const ALERTS = 60;
      return {
        speeds: SPEEDS,
        speedIndex: 3,              // 60 lần — 1 giờ dữ liệu ≈ 1 phút thực (07 §7)
        status: "idle",             // idle | connecting | playing | paused | reconnecting | ended
        simSeconds: 0,
        lastEventSeconds: 0,        // sim_seconds của giao dịch cuối cùng đã nhận
        processed: 0,
        alerts: 0,
        dayTotal: null,
        threshold: null,
        feed: [],
        alertList: [],
        message: "",

        init() {
          window.addEventListener("beforeunload", () => this.disconnect());
          // Đồng hồ mô phỏng: 10 lần mỗi giây là đủ mượt cho HH:MM:SS, và chỉ chạy khi đang phát
          this.$watch("status", (status) => {
            clearInterval(clockTimer);
            if (status !== "playing") return;
            clockTimer = setInterval(() => {
              if (!clockAnchor) return;
              const sim = clockAnchor.sim + ((performance.now() - clockAnchor.wall) / 1000) * clockAnchor.speed;
              this.simSeconds = Math.min(sim, 86400);
            }, 100);
          });
        },

        get speed() {
          return SPEEDS[this.speedIndex];
        },

        get alertsPerHour() {
          const hours = this.simSeconds / 3600;
          return hours > 0 ? this.alerts / hours : null;
        },

        get progress() {
          return this.dayTotal ? this.processed / this.dayTotal : 0;
        },

        play() {
          if (this.status === "playing" || this.status === "connecting") return;
          if (this.status === "ended") this.reset();
          this.connect(this.lastEventSeconds);
        },

        /** Tạm dừng là đóng kết nối; máy chủ không giữ trạng thái nào (docs/05 §7, API-15). */
        pause() {
          this.disconnect();
          if (this.status !== "idle" && this.status !== "ended") this.status = "paused";
        },

        reset() {
          this.disconnect();
          seen = new Set();
          buffer = [];
          Object.assign(this, {
            status: "idle", simSeconds: 0, lastEventSeconds: 0, processed: 0, alerts: 0, dayTotal: null,
            feed: [], alertList: [], message: "", threshold: null,
          });
          clockAnchor = null;
        },

        setSpeed(index) {
          this.speedIndex = Number(index);
          if (["playing", "connecting", "reconnecting"].includes(this.status)) {
            this.disconnect();
            this.connect(this.lastEventSeconds);
          }
        },

        disconnect() {
          clearTimeout(reconnectTimer);
          if (source) {
            source.close();
            source = null;
          }
        },

        connect(start) {
          this.disconnect();
          this.status = "connecting";
          this.message = "";
          const url = `${API_BASE}/replay/stream${query({ speed: this.speed, start: Math.min(start, 86399) })}`;
          const es = new EventSource(url);
          source = es;
          const parse = (e) => JSON.parse(e.data);

          es.addEventListener("start", (e) => {
            const d = parse(e);
            this.status = "playing";
            this.threshold = d.threshold;
            if (this.dayTotal === null) this.dayTotal = d.start === 0 ? d.total : this.processed + d.total;
            clockAnchor = { base: d.start, sim: d.start, wall: performance.now(), speed: d.speed };
          });
          const onTx = (alert) => (e) => {
            buffer.push({ ...parse(e), alert });
            if (!frame) frame = requestAnimationFrame(() => this.flush());
          };
          es.addEventListener("transaction", onTx(false));
          es.addEventListener("alert", onTx(true));
          es.addEventListener("stats", (e) => {
            const d = parse(e);
            this.threshold = d.threshold;
            // Đồng hồ máy chủ là chuẩn; giữa hai nhịp stats, trình duyệt tự nội suy theo tốc độ
            if (clockAnchor) {
              clockAnchor = { ...clockAnchor, sim: clockAnchor.base + (d.elapsed_sim_seconds || 0), wall: performance.now() };
            }
          });
          es.addEventListener("end", () => {
            this.flush();
            this.disconnect();
            this.status = "ended";
            this.simSeconds = 86400;
            this.message = "Đã phát hết ngày 2 của tập kiểm thử.";
            emit("queue-changed");
          });
          es.onerror = () => {
            if (source !== es) return;
            // Tự nối lại với start = giây cuối đã nhận, không để trình duyệt phát lại từ đầu
            this.disconnect();
            this.status = "reconnecting";
            this.message = "Mất kết nối với API — thử nối lại sau 3 giây…";
            reconnectTimer = setTimeout(() => this.connect(this.lastEventSeconds), 3000);
          };
        },

        /** Gộp sự kiện theo khung hình: ở tốc độ 3.600 lần có hơn nghìn giao dịch mỗi giây. */
        flush() {
          frame = 0;
          if (!buffer.length) return;
          const batch = buffer;
          buffer = [];
          const fresh = [];
          for (const ev of batch) {
            this.lastEventSeconds = Math.max(this.lastEventSeconds, ev.sim_seconds);
            if (seen.has(ev.id)) continue;
            seen.add(ev.id);
            fresh.push(ev);
          }
          if (!fresh.length) return;
          const alerts = fresh.filter((ev) => ev.alert);
          this.processed += fresh.length;
          this.alerts += alerts.length;
          const newestFirst = fresh.slice(-FEED).reverse().map((ev) => ({ ...ev, key: ev.id }));
          this.feed = [...newestFirst, ...this.feed].slice(0, FEED);
          if (alerts.length) {
            const items = alerts.reverse().map((ev) => ({ ...ev, key: ev.id }));
            this.alertList = [...items, ...this.alertList].slice(0, ALERTS);
          }
        },

        openAlert(item, element) {
          Alpine.store("app").openTx(item.id, element);
        },

        level(score) {
          return riskLevel(score);
        },
      };
    });
  });

  // Dùng trong biểu thức của index.html
  window.fmt = fmt;
  window.LABELS = LABELS;
})();
