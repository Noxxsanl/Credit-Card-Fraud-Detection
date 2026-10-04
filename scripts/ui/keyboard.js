// AC-A9 — đi khắp bản Next.js chỉ bằng bàn phím (24 bước).
//
//   cd scripts/ui && npm install && node keyboard.js
//
// Phím dùng: Tab, Shift+Tab, Enter. Thêm mũi tên ở đúng ba chỗ mà mẫu WAI-ARIA quy định một điểm
// dừng Tab cho cả nhóm: bảng hàng đợi (↓ giữa các dòng, docs/07 §10), thanh trượt, nhóm radio.
// Mỗi lần nhấn đều kiểm phần tử nhận trọng tâm có viền :focus-visible và không thoát khỏi hộp
// thoại đang mở. GHI vào cơ sở dữ liệu (nạp nhóm "Gian lận khó", một kết luận "báo động sai").
const puppeteer = require("puppeteer-core");
const BASE = (process.env.BASE || "http://localhost:3000").replace(/\/$/, "");
const CHROME = process.env.CHROME_PATH || "C:/Program Files/Google/Chrome/Application/chrome.exe";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const results = [];
const check = (name, ok, detail = "") => { results.push({ name, ok }); console.log(`${ok ? "PASS" : "FAIL"}  ${name}  ${detail}`); };

const describe = () => {
  const e = document.activeElement;
  if (!e || e === document.body) return { key: "body", text: "body", visible: true, sized: true, inDialog: null };
  const text = (e.getAttribute("aria-label") || e.textContent || e.value || "").replace(/\s+/g, " ").trim().slice(0, 50);
  const cs = getComputedStyle(e);
  const outline = cs.outlineStyle !== "none" && parseFloat(cs.outlineWidth) > 0;
  const ring = cs.boxShadow && cs.boxShadow !== "none";
  const r = e.getBoundingClientRect();
  const dialog = document.querySelector("dialog[open]");
  return {
    key: `${e.tagName.toLowerCase()}${e.id ? "#" + e.id : ""}${e.dataset.row ? "[row " + e.dataset.row + "]" : ""} "${text}"`,
    text, full: (e.textContent || "").replace(/\s+/g, " "), tag: e.tagName.toLowerCase(), type: e.type || null, visible: outline || ring, sized: r.width > 0 && r.height > 0,
    inDialog: dialog ? dialog.contains(e) : null,
  };
};

(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new" });
  const page = await browser.newPage();
  await page.setViewport({ width: 1440, height: 1000 });
  const problems = [];
  page.on("pageerror", (e) => problems.push(`[pageerror] ${e.message}`));
  page.on("console", (m) => { if (m.type() === "error") problems.push(`[console.error] ${m.text()}`); });

  const noFocusRing = new Set();
  const dialogEscapes = [];
  const press = async (key, shift = false) => {
    if (shift) await page.keyboard.down("Shift");
    await page.keyboard.press(key);
    if (shift) await page.keyboard.up("Shift");
    const d = await page.evaluate(describe);
    if (d.key !== "body" && !d.visible) noFocusRing.add(d.key);
    if (d.inDialog === false) dialogEscapes.push(d.key);
    return d;
  };
  // Nhấn Tab tới khi phần tử đang có trọng tâm thỏa điều kiện; trả số lần nhấn (hoặc -1)
  const tabTo = async (pred, max = 160) => {
    for (let i = 1; i <= max; i++) {
      const d = await press("Tab");
      if (pred(d)) return i;
    }
    return -1;
  };
  const textStarts = (s) => (d) => d.text.startsWith(s);
  const route = () => page.evaluate(() => location.pathname);
  const fresh = async (path) => {
    await page.goto(BASE + path, { waitUntil: "networkidle0" });
    await sleep(300);
  };

  // ---- 1. Quét Tab trọn vòng trên từng màn hình ----
  for (const path of ["/", "/threshold/", "/performance/", "/replay/"]) {
    await fresh(path);
    const seen = [];
    let first = null;
    for (let i = 0; i < 200; i++) {
      const d = await press("Tab");
      if (first && d.key === first) break;
      if (!first && d.key !== "body") first = d.key;
      if (d.key !== "body") seen.push(d);
    }
    const nav = ["Hàng đợi", "Ngưỡng", "Hiệu năng", "Phát lại"].filter((n) => seen.some((d) => d.tag === "a" && d.text.startsWith(n)));
    const hidden = seen.filter((d) => !d.sized).map((d) => d.key);
    check(`Tab trọn vòng ${path}`, nav.length === 4 && hidden.length === 0,
      `${seen.length} điểm dừng, đủ ${nav.length}/4 mục thanh bên${hidden.length ? ", ẩn: " + hidden.join("; ") : ""}`);
    // Shift+Tab đi ngược được (không kẹt)
    const back1 = await press("Tab", true);
    const back2 = await press("Tab", true);
    check(`Shift+Tab đi ngược ${path}`, back1.key !== back2.key, `${back1.key} ← ${back2.key}`);
  }

  // ---- 2. Liên kết bỏ qua ----
  await fresh("/");
  const skip = await press("Tab");
  await page.keyboard.press("Enter");
  const afterSkip = await page.evaluate(() => document.activeElement.id);
  check("Tab đầu tiên là liên kết bỏ qua; Enter đưa trọng tâm vào nội dung chính", skip.text.startsWith("Bỏ qua") && afterSkip === "main", `${skip.key} → #${afterSkip}`);

  // ---- 3. Điều hướng bốn màn hình bằng Tab + Enter ----
  for (const [label, path] of [["Ngưỡng", "/threshold/"], ["Hiệu năng", "/performance/"], ["Phát lại", "/replay/"], ["Hàng đợi", "/"]]) {
    const n = await tabTo((d) => d.tag === "a" && d.text.startsWith(label));
    await page.keyboard.press("Enter");
    await sleep(700);
    check(`Tab ×${n} + Enter → ${label}`, n > 0 && (await route()) === path, await route());
  }

  // ---- 4. Thư viện mẫu: mở, chọn nhóm, nạp, đóng ----
  await fresh("/");
  let n = await tabTo((d) => d.tag === "button" && (d.text.startsWith("Chọn mẫu có sẵn") || d.text.startsWith("Mở thư viện mẫu")));
  const opener = (await page.evaluate(describe)).key;
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("dialog[open]") && document.querySelectorAll("dialog[open] tbody tr").length > 0, { timeout: 15000 });
  n = await tabTo(textStarts("Gian lận khó"), 40);
  await page.keyboard.press("Enter");
  const pressed = await page.evaluate(() => document.activeElement.getAttribute("aria-pressed"));
  n = await tabTo(textStarts("Nạp "), 40);
  const loadLabel = (await page.evaluate(describe)).text;
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => ![...document.querySelectorAll("dialog[open] button")].some((b) => b.textContent.includes("Đang chấm")), { timeout: 60000 });
  await sleep(300);
  // Tab trong hộp thoại phải quay vòng bên trong (showModal), rồi tới nút Đóng
  for (let i = 0; i < 30; i++) await press("Tab");
  n = await tabTo(textStarts("Đóng"), 300);
  await page.keyboard.press("Enter");
  await sleep(300);
  const afterClose = await page.evaluate(() => ({ open: !!document.querySelector("dialog[open]") }));
  const focusBack = (await page.evaluate(describe)).key;
  check("Thư viện mẫu bằng bàn phím: mở → chọn nhóm → nạp → đóng", pressed === "true" && n > 0 && !afterClose.open,
    `"${loadLabel}"; nhóm aria-pressed=${pressed}`);
  check("Đóng hộp thoại trả trọng tâm về nút đã mở nó", focusBack === opener, focusBack);

  // ---- 5. Hàng đợi → ngăn kéo chi tiết → ghi kết luận → đóng ----
  await sleep(800);
  // Bảng hàng đợi là MỘT điểm dừng Tab (roving tabindex); ↓ đi giữa các dòng (07 §10, mẫu lưới WAI-ARIA)
  n = await tabTo((d) => d.tag === "tr");
  let downs = 0;
  while (!(await page.evaluate(describe)).full.includes("Chờ xử lý") && downs < 25) { await press("ArrowDown"); downs++; }
  const row = (await page.evaluate(describe)).key;
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("dialog.drawer")?.open && document.querySelector("dialog.drawer caption"), { timeout: 15000 });
  const n2 = await tabTo((d) => d.tag === "button" && d.text.toLowerCase().includes("báo động sai"), 80);
  await page.keyboard.press("Enter");
  await page.waitForFunction(() => document.querySelector("dialog.drawer").textContent.includes("Kết luận của bạn"), { timeout: 10000 });
  const verdict = await page.evaluate(() => [...document.querySelectorAll('dialog.drawer [role="status"]')].map((e) => e.textContent.replace(/\s+/g, " ")).join(" ").trim());
  const n3 = await tabTo(textStarts("Đóng"), 80);
  await page.keyboard.press("Enter");
  await sleep(300);
  const drawerBack = await page.evaluate(() => ({ open: document.querySelector("dialog.drawer").open }));
  const rowBack = (await page.evaluate(describe)).key;
  check("Hàng đợi: Tab vào bảng, ↓ tới dòng chờ xử lý, Enter mở chi tiết", n > 0, `${row} — Tab ×${n}, ↓ ×${downs}`);
  check("Ngăn kéo: Tab + Enter ghi \"Báo động sai\", lộ nhãn thật", n2 > 0 && verdict.includes("Báo động sai") && verdict.includes("Nhãn thật"), `sau ${n2} lần Tab: ${verdict.slice(0, 110)}`);
  check("Ngăn kéo: Tab tới Đóng + Enter, trọng tâm về đúng dòng", n3 > 0 && !drawerBack.open && rowBack === row, rowBack);

  // ---- 6. Màn hình ngưỡng: thanh trượt, ô nhập, nhóm phương án ----
  await fresh("/threshold/");
  n = await tabTo((d) => d.key.startsWith("input#tau-slider"));
  const tau0 = await page.evaluate(() => document.querySelector('[aria-label^="Nhập ngưỡng chính xác"]').value);
  await press("ArrowLeft"); await press("ArrowLeft");
  await sleep(200);
  const tau1 = await page.evaluate(() => document.querySelector('[aria-label^="Nhập ngưỡng chính xác"]').value);
  check("Tab tới thanh trượt; ← đổi ngưỡng", n > 0 && tau0 !== tau1, `${tau0} → ${tau1}`);
  n = await tabTo((d) => d.tag === "input" && d.text.startsWith("Nhập ngưỡng chính xác"), 20);
  check("Tab tới ô nhập ngưỡng chính xác", n > 0, `sau ${n} lần Tab`);
  n = await tabTo((d) => d.type === "radio", 40);
  const radio0 = await page.evaluate(() => document.activeElement.type === "radio" ? document.activeElement.closest("label")?.textContent.replace(/\s+/g, " ").trim().slice(0, 30) : null);
  await press("ArrowDown");
  const radio1 = await page.evaluate(() => document.activeElement.type === "radio" ? document.activeElement.closest("label")?.textContent.replace(/\s+/g, " ").trim().slice(0, 30) : null);
  check("Nhóm phương án ngưỡng: Tab vào, ↓ đổi lựa chọn", radio0 && radio1 && radio0 !== radio1, `${radio0} → ${radio1}`);

  // ---- 7. Phát lại: Phát / Tạm dừng / Đặt lại bằng Tab + Enter ----
  await fresh("/replay/");
  n = await tabTo(textStarts("▶ Phát"));
  await page.keyboard.press("Enter");
  await sleep(2500);
  const afterPlay = (await page.evaluate(describe)).key;
  n = await tabTo(textStarts("❚❚ Tạm dừng"), 60);
  await page.keyboard.press("Enter");
  await sleep(500);
  const paused = await page.evaluate(() => [...document.querySelectorAll("button")].some((b) => b.textContent.includes("Tiếp tục")));
  check("Phát lại: Tab + Enter phát, rồi tạm dừng", n > 0 && paused, `trọng tâm sau khi bấm Phát: ${afterPlay}`);
  n = await tabTo(textStarts("↺ Đặt lại"), 20);
  await page.keyboard.press("Enter");
  await sleep(300);

  check("Mọi điểm dừng có viền trọng tâm nhìn thấy (:focus-visible)", noFocusRing.size === 0, noFocusRing.size ? [...noFocusRing].join("; ") : "");
  check("Trọng tâm không thoát khỏi hộp thoại đang mở", dialogEscapes.length === 0, dialogEscapes.slice(0, 3).join("; "));
  console.log(problems.length ? problems.join("\n") : "no console errors");
  console.log(`${results.filter((r) => r.ok).length}/${results.length} PASS`);
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
