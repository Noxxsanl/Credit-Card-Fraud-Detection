// Luồng chính của bản Next.js trên Chrome thật: AC-A1…A3, AC-A5, UI-02, UI-05, UI-04 (21 bước).
//
//   cd scripts/ui && npm install && node flow.js
//
// Cần hệ thống đang chạy ở BASE (mặc định http://localhost:3000/ — docker compose up) và tệp CSV
// 10.000 dòng ở data/giao-dich-10000.csv (lệnh tạo: docs/lenh-chay.md §8.1.3).
// GHI vào cơ sở dữ liệu (nạp mẫu, tải CSV, thẩm định, phát lại): chạy trên dữ liệu demo hoặc
// sau `python scripts/demo_db.py reset`, không chạy trên dữ liệu cần giữ. Ảnh chụp vào shots/.
// Bước "nhãn thật ẩn trước khi quyết định" trượt nếu dòng thứ 3 của hàng đợi đã được thẩm định
// từ trước (ví dụ ngay sau `demo_db.py seed`) — khi đó ngăn kéo hiện kết luận cũ là đúng.
const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");
const OUT = path.join(__dirname, "shots");
fs.mkdirSync(OUT, { recursive: true });
const BASE = (process.env.BASE || "http://localhost:3000").replace(/\/?$/, "/");
const CHROME = process.env.CHROME_PATH || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const CSV = process.env.CSV || path.join(__dirname, "..", "..", "data", "giao-dich-10000.csv");
if (!fs.existsSync(CSV)) {
  console.error(`Thiếu ${CSV} — tạo theo docs/lenh-chay.md §8.1.3`);
  process.exit(2);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${detail}`); };

// Hàm trợ giúp chạy trong trang
const HELPERS = () => {
  window.__tile = (label) => {
    const el = [...document.querySelectorAll("main div")].find((d) => d.children.length === 0 && d.textContent.trim() === label);
    return el ? el.nextElementSibling.textContent.replace(/\s+/g, " ").trim() : null;
  };
  window.__button = (text) => [...document.querySelectorAll("button, a")].find((b) => b.textContent.replace(/\s+/g, " ").trim().includes(text));
  window.__setRange = (el, value) => {
    Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set.call(el, String(value));
    el.dispatchEvent(new Event("input", { bubbles: true }));
  };
  window.__tau = () => document.querySelector('[aria-label^="Nhập ngưỡng chính xác"]').previousElementSibling.textContent;
};

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new" });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1000 });
  const problems = [];
  page.on("console", (m) => { if (["error", "warning"].includes(m.type())) problems.push(`[console.${m.type()}] ${m.text()}`); });
  page.on("pageerror", (e) => problems.push(`[pageerror] ${e.message}`));
  page.on("load", () => page.evaluate(HELPERS).catch(() => {}));

  // ---------------- UI-03 (AC-A3) ----------------
  await page.goto(`${BASE}threshold/`, { waitUntil: "networkidle0" });
  await page.evaluate(HELPERS);
  await page.waitForFunction(() => window.__tile("Gian lận bắt được") && document.querySelector("#tau-slider"));
  const timings = await page.evaluate(async () => {
    const slider = document.querySelector("#tau-slider");
    const out = [];
    for (let i = 0; i <= 20; i++) {
      const before = window.__tile("Chi phí mỗi ngày");
      const t0 = performance.now();
      window.__setRange(slider, Math.log10(0.5) + (i / 20) * (Math.log10(0.023173) - Math.log10(0.5)));
      await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
      out.push({ ms: performance.now() - t0, changed: window.__tile("Chi phí mỗi ngày") !== before });
    }
    return out;
  });
  const worst = Math.max(...timings.map((t) => t.ms));
  check("AC-A3 thanh trượt cập nhật < 200 ms (21 lần kéo 0,5 → τ*)", worst < 200 && timings.filter((t) => t.changed).length > 10,
    `lâu nhất ${worst.toFixed(1)} ms, ${timings.filter((t) => t.changed).length}/21 lần đổi số`);

  const pick = async (label) => {
    await page.evaluate((label) => [...document.querySelectorAll("label")].find((l) => l.textContent.includes(label)).querySelector("input").click(), label);
    await sleep(200);
    return page.evaluate(() => ["Cảnh báo mỗi ngày", "Gian lận bắt được", "Precision", "Chi phí mỗi ngày"].map((l) => window.__tile(l)).join(" | "));
  };
  const at05 = await pick("Mặc định 0,5");
  await page.screenshot({ path: path.join(OUT, "nflow-threshold-05.png"), fullPage: true });
  const atStar = await pick("Cực tiểu chi phí");
  check("Kịch bản 0,5 → τ*", at05 !== atStar, `0,5: ${at05}  →  τ*: ${atStar}`);

  await page.focus("#tau-slider");
  const kb0 = await page.evaluate(() => window.__tau());
  await page.keyboard.press("PageUp");
  await page.keyboard.press("ArrowRight");
  const kb1 = await page.evaluate(() => window.__tau());
  check("Thanh trượt nhận PgUp và →", kb0 !== kb1, `${kb0} → ${kb1}`);
  await pick("Cực tiểu chi phí");

  // ---------------- AC-A5 ----------------
  const optText = () => page.evaluate(() => [...document.querySelectorAll("p")].find((p) => p.textContent.startsWith("Ngưỡng tối ưu:"))?.querySelector("strong")?.textContent);
  await page.waitForFunction(() => [...document.querySelectorAll("p")].some((p) => p.textContent.startsWith("Ngưỡng tối ưu:")));
  const opt0 = await optText();
  await page.click("#cost-fn", { clickCount: 3 });
  await page.type("#cost-fn", "600");
  await sleep(3000);
  const opt1 = await optText();
  check("AC-A5 chi phí bỏ lọt 600 EUR dịch ngưỡng tối ưu", opt0 !== opt1, `${opt0} → ${opt1}`);
  await page.select("#constraint-type", "max_alerts_per_day");
  await page.type("#constraint-value", "150");
  await sleep(3000);
  const binding = await page.evaluate(() => document.querySelector("form [aria-live]").textContent.replace(/\s+/g, " "));
  check("Ràng buộc 150 cảnh báo/ngày: báo đang chặn nghiệm", binding.includes("đang chặn"), binding.slice(0, 120));
  await page.screenshot({ path: path.join(OUT, "nflow-threshold-cost.png"), fullPage: true });
  await page.select("#constraint-type", "none");
  await page.click("#cost-fn", { clickCount: 3 });
  await page.type("#cost-fn", "122.21");
  await sleep(1500);

  // Trạng thái nháp sống qua lần chuyển trang (đi bằng liên kết của thanh bên)
  await pick("F1 lớn nhất");
  await page.evaluate(() => [...document.querySelectorAll("nav a")].find((a) => a.textContent.includes("Hàng đợi")).click());
  await sleep(800);
  await page.evaluate(() => [...document.querySelectorAll("nav a")].find((a) => a.textContent.includes("Ngưỡng")).click());
  await page.waitForSelector("#tau-slider");
  const kept = await page.evaluate(() => window.__tau());
  check("Ngưỡng đang xem giữ nguyên sau khi chuyển trang", kept === "0,721", kept);
  await pick("Cực tiểu chi phí");

  // ---------------- Thư viện mẫu → hàng đợi (AC-A2) ----------------
  await page.evaluate(() => [...document.querySelectorAll("nav a")].find((a) => a.textContent.includes("Hàng đợi")).click());
  await page.waitForFunction(() => window.__button("Chọn mẫu có sẵn"));
  await page.evaluate(() => window.__button("Chọn mẫu có sẵn").click());
  await page.waitForFunction(() => document.querySelectorAll("dialog.modal tbody tr").length > 0);
  for (const cat of ["Gian lận dễ", "Gian lận khó", "Hợp lệ khó"]) {
    await page.evaluate((cat) => [...document.querySelectorAll("dialog.modal .segmented button")].find((b) => b.textContent.startsWith(cat)).click(), cat);
    await sleep(150);
    await page.evaluate(() => [...document.querySelectorAll("dialog.modal button")].find((b) => b.textContent.startsWith("Nạp ")).click());
    await page.waitForFunction(() => ![...document.querySelectorAll("dialog.modal button")].some((b) => b.textContent.includes("Đang chấm")), { timeout: 60000 });
  }
  await page.screenshot({ path: path.join(OUT, "nflow-samples.png") });
  await page.keyboard.press("Escape");
  await sleep(1200);
  const values = await page.$$eval("table.data-table tbody tr[data-row] td:first-child", (tds) => tds.map((td) => {
    const t = td.textContent;
    return t.includes(">") ? 99.95 : t.includes("<") ? 0.05 : parseFloat(t.replace(/[^\d,]/g, "").replace(",", "."));
  }));
  check("AC-A2 hàng đợi sắp giảm dần", values.length > 0 && values.every((v, i) => i === 0 || values[i - 1] >= v), `${values.length} dòng`);

  // ---------------- UI-02 bằng bàn phím ----------------
  await page.focus('table.data-table tbody tr[tabindex="0"]');
  await page.keyboard.press("ArrowDown");
  await page.keyboard.press("ArrowDown");
  const focusedRow = await page.evaluate(() => document.activeElement.dataset.row);
  check("↓ di chuyển giữa các dòng", focusedRow === "2", `dòng ${focusedRow}`);
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("dialog.drawer").open);
  await page.waitForFunction(() => document.querySelector("dialog.drawer caption"), { timeout: 10000 });
  const drawer = await page.evaluate(() => {
    const d = document.querySelector("dialog.drawer");
    const rows = [...d.querySelectorAll("table tbody tr")].filter((tr) => tr.querySelector("th.font-mono"));
    const signs = rows.map((tr) => tr.lastElementChild.textContent.trim()[0]);
    return {
      pca: d.querySelector('[role="note"]').textContent.includes("không diễn giải được"),
      labelHidden: !d.textContent.includes("Nhãn thật:"),
      pos: signs.filter((s) => s === "+").length,
      neg: signs.filter((s) => s === "−").length,
      id: d.querySelector("#drawer-title").textContent,
    };
  });
  check("UI-02 chú thích PCA cố định có mặt", drawer.pca);
  check("UI-02 nhãn thật ẩn trước khi quyết định", drawer.labelHidden);
  check("UI-02 thác nước 5 dương + 3 âm", drawer.pos === 5 && drawer.neg === 3, `${drawer.pos} dương, ${drawer.neg} âm (${drawer.id})`);
  await page.screenshot({ path: path.join(OUT, "nflow-drawer.png") });
  await page.keyboard.press("f");
  await page.waitForFunction(() => document.querySelector("dialog.drawer").textContent.includes("Kết luận của bạn"));
  const reveal = await page.evaluate(() => [...document.querySelectorAll('dialog.drawer [role="status"]')].map((e) => e.textContent.replace(/\s+/g, " ")).join(" "));
  check("Phím F ghi kết luận và lộ nhãn thật", reveal.includes("Nhãn thật"), reveal.trim());
  await page.screenshot({ path: path.join(OUT, "nflow-drawer-decided.png") });
  await page.keyboard.press("Escape");
  await sleep(300);
  const back = await page.evaluate(() => ({ open: document.querySelector("dialog.drawer").open, row: document.activeElement.dataset.row }));
  check("Esc đóng ngăn kéo và trả trọng tâm về dòng cũ", !back.open && back.row === "2", JSON.stringify(back));
  const status = await page.$eval('table.data-table tbody tr[data-row="2"]', (tr) => tr.children[5].textContent);
  check("Trạng thái ở UI-01 cập nhật ngay", status.includes("Xác nhận gian lận"), status);

  // ---------------- AC-A1: tải CSV ----------------
  const input = await page.$('input[type="file"]');
  const t0 = Date.now();
  await input.uploadFile(CSV);
  await page.waitForFunction(() => document.body.textContent.includes("Đã chấm tệp"), { timeout: 60000 });
  check("AC-A1 tải CSV 10.000 dòng < 30 giây", Date.now() - t0 < 30000, `${((Date.now() - t0) / 1000).toFixed(1)} giây`);
  await sleep(800);
  await page.screenshot({ path: path.join(OUT, "nflow-queue.png"), fullPage: true });

  // ---------------- UI-05 ----------------
  await page.evaluate(() => [...document.querySelectorAll("nav a")].find((a) => a.textContent.includes("Phát lại")).click());
  await page.waitForSelector("#replay-speed");
  await page.evaluate(() => window.__setRange(document.querySelector("#replay-speed"), 6));
  await page.evaluate(() => window.__button("▶ Phát").click());
  await sleep(6000);
  const clock = () => page.evaluate(() => [...document.querySelectorAll("main span")].find((s) => /^\d\d:\d\d:\d\d$/.test(s.textContent)).textContent);
  const processed = () => page.evaluate(() => window.__tile("Đã xử lý"));
  const r1 = { clock: await clock(), processed: await processed(), alerts: await page.evaluate(() => window.__tile("Cảnh báo")) };
  // Chuyển sang hàng đợi rồi quay lại: luồng vẫn chạy
  await page.evaluate(() => [...document.querySelectorAll("nav a")].find((a) => a.textContent.includes("Hàng đợi")).click());
  await sleep(2000);
  await page.evaluate(() => [...document.querySelectorAll("nav a")].find((a) => a.textContent.includes("Phát lại")).click());
  await page.waitForSelector("#replay-speed");
  const r1b = await processed();
  check("UI-05 phát: đồng hồ và bộ đếm chạy", r1.processed !== "0" && r1.clock !== "00:00:00", `${r1.clock}, ${r1.processed} giao dịch, ${r1.alerts} cảnh báo`);
  check("UI-05 luồng vẫn chạy khi sang màn hình khác", r1b !== r1.processed, `${r1.processed} → ${r1b}`);
  await page.evaluate(() => window.__button("Tạm dừng").click());
  await sleep(1500);
  const r2 = await processed();
  await sleep(1500);
  const r3 = await processed();
  check("UI-05 tạm dừng: bộ đếm đứng yên", r2 === r3, `${r2} → ${r3}`);
  await page.evaluate(() => window.__button("Tiếp tục").click());
  await sleep(4000);
  const r4 = await processed();
  check("UI-05 tiếp tục: đếm tiếp", r4 !== r3, `${r3} → ${r4}`);
  await page.screenshot({ path: path.join(OUT, "nflow-replay.png"), fullPage: true });
  await page.evaluate(() => window.__button("Đặt lại").click());
  await sleep(300);
  const r5 = { clock: await clock(), processed: await processed() };
  check("UI-05 đặt lại về 00:00:00", r5.clock === "00:00:00" && r5.processed === "0", JSON.stringify(r5));

  // ---------------- UI-04 ----------------
  await page.evaluate(() => [...document.querySelectorAll("nav a")].find((a) => a.textContent.includes("Hiệu năng")).click());
  await page.waitForFunction(() => document.body.textContent.includes("20 tổ hợp"));
  await sleep(800);
  const legend = await page.evaluate(() => [...document.querySelectorAll(".recharts-legend-item-text")].map((e) => e.textContent).join(" | "));
  check("UI-04 chú giải theo thứ tự chiến lược", legend.startsWith("Trọng số lớp"), legend);
  await page.screenshot({ path: path.join(OUT, "nflow-performance.png"), fullPage: true });

  console.log(problems.length ? problems.join("\n") : "no console problems");
  console.log(`${results.filter((r) => r.ok).length}/${results.length} PASS`);
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
