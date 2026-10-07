(() => {
  const chat = document.getElementById("chat");
  const form = document.getElementById("composer");
  const input = document.getElementById("input");
  const sendBtn = document.getElementById("sendBtn");
  const statusEl = document.getElementById("status");
  const welcome = document.getElementById("welcome");
  const contextBar = document.getElementById("contextBar");
  const contextChips = document.getElementById("contextChips");

  // ---------------------------------------------------------------- session (server-side conversation memory)
  const newId = () => (crypto.randomUUID ? crypto.randomUUID() : String(Date.now()) + Math.random().toString(16).slice(2));
  let sessionId = null;
  try { sessionId = sessionStorage.getItem("sessionId"); } catch (_) {}
  if (!sessionId) {
    sessionId = newId();
    try { sessionStorage.setItem("sessionId", sessionId); } catch (_) {}
  }

  const STARTERS = [
    ["Find", "Gaming laptop under €1200 with 16GB RAM"],
    ["Sort", "Lightest ultrabook with an i7"],
    ["Count", "How many Lenovo laptops have an SSD?"],
    ["Average", "Average price of 2-in-1 laptops by brand"],
    ["Compare", "Compare Dell XPS 13 vs MacBook Pro"],
    ["Use case", "Cheap laptop for students, not Apple"],
  ];

  // ---------------------------------------------------------------- theme
  const root = document.documentElement;
  try {
    const saved = localStorage.getItem("theme");
    if (saved) root.dataset.theme = saved;
  } catch (_) {}
  document.getElementById("themeBtn").addEventListener("click", () => {
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (_) {}
  });

  // ---------------------------------------------------------------- helpers
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const el = (tag, cls, html) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (html !== undefined) n.innerHTML = html;
    return n;
  };
  const inline = (s) => esc(s).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>").replace(/~~(.+?)~~/g, "<del>$1</del>");

  // Minimal markdown: paragraphs, "- " bullet lists, **bold**, ~~strike~~
  function renderText(text) {
    const out = [];
    let list = null;
    for (const line of String(text || "").split("\n")) {
      if (line.startsWith("- ")) {
        if (!list) { list = []; out.push(list); }
        list.push(`<li>${inline(line.slice(2))}</li>`);
      } else {
        list = null;
        if (line.trim()) out.push(`<p>${inline(line)}</p>`);
      }
    }
    return out.map((x) => (Array.isArray(x) ? `<ul>${x.join("")}</ul>` : x)).join("");
  }

  function scrollDown() { chat.scrollTop = chat.scrollHeight; }

  // ---------------------------------------------------------------- rendering
  function addUser(text) {
    const m = el("div", "msg user");
    m.appendChild(el("div", "bubble", esc(text)));
    chat.appendChild(m);
    scrollDown();
  }

  function addBot() {
    const m = el("div", "msg bot");
    m.appendChild(el("div", "avatar", "AI"));
    const b = el("div", "bubble", '<div class="typing"><span></span><span></span><span></span></div>');
    m.appendChild(b);
    chat.appendChild(m);
    scrollDown();
    return b;
  }

  function renderCard(c) {
    const card = el("article", "card");
    const tags = (c.tags || []).map((t) => `<span class="tag">${esc(t)}</span>`).join("");
    card.innerHTML = `
      <div class="card-head">
        <div>
          <div class="card-title">${c.rank ? `<span class="rank">#${c.rank}</span>` : ""}${esc(c.title)}</div>
          <div class="card-sub">${esc(c.category)}</div>
        </div>
        <div class="price">${esc(c.price_text)}</div>
      </div>
      <dl class="specs">
        <dt>CPU</dt><dd>${esc(c.cpu)}</dd>
        <dt>RAM</dt><dd>${esc(c.ram)}</dd>
        <dt>Storage</dt><dd>${esc(c.storage)}</dd>
        <dt>GPU</dt><dd>${esc(c.gpu)}</dd>
        <dt>Screen</dt><dd>${esc(c.screen)}</dd>
        <dt>OS</dt><dd>${esc(c.os)}</dd>
        <dt>Weight</dt><dd>${esc(c.weight)}</dd>
      </dl>
      ${tags ? `<div class="tags">${tags}</div>` : ""}`;
    return card;
  }

  function renderTable(t) {
    const wrap = el("div", "table-wrap");
    const head = t.columns.map((c) => `<th>${esc(c)}</th>`).join("");
    const rows = t.rows.map((r) => `<tr>${r.map((v) => `<td>${esc(v)}</td>`).join("")}</tr>`).join("");
    wrap.innerHTML = `<table><thead><tr>${head}</tr></thead><tbody>${rows}</tbody></table>`;
    return wrap;
  }

  function renderTrace(trace) {
    if (!trace || !trace.parsed) return null;
    const p = trace.parsed;
    const pills = [`turn: ${trace.plan || "run"}`, `intent: ${p.intent}`, `retrieval: ${trace.retrieval_mode}`];
    (p.constraints || []).forEach((c) => pills.push(`filter: ${c.label}`));
    if (p.sort) pills.push(`sort: ${p.sort.label}`);
    (p.preferences || []).forEach((x) => pills.push(`prefer: ${x}`));
    if (trace.candidates !== undefined) pills.push(`candidates: ${trace.candidates}`);
    pills.push(`${trace.ms} ms`);
    const d = el("details", "trace");
    d.innerHTML = `<summary>How this answer was computed</summary><div class="trace-body">${pills.map((x) => `<span class="pill">${esc(x)}</span>`).join("")}</div>`;
    return d;
  }

  function renderAnswer(bubble, data) {
    bubble.innerHTML = "";
    bubble.appendChild(el("div", "answer", renderText(data.text)));
    if (data.table && data.table.rows && data.table.rows.length) bubble.appendChild(renderTable(data.table));
    if (data.cards && data.cards.length) {
      const grid = el("div", "cards");
      data.cards.forEach((c) => grid.appendChild(renderCard(c)));
      bubble.appendChild(grid);
    }
    if (data.suggestions && data.suggestions.length) {
      const s = el("div", "suggestions");
      data.suggestions.forEach((q) => {
        const b = el("button", "chip", esc(q));
        b.type = "button";
        b.addEventListener("click", () => ask(q));
        s.appendChild(b);
      });
      bubble.appendChild(s);
    }
    const tr = renderTrace(data.trace);
    if (tr) bubble.appendChild(tr);
    renderContext(data.context);
    scrollDown();
  }

  // Active filters the server remembers for this conversation; × removes one.
  function renderContext(ctx) {
    if (!ctx) return;
    contextChips.innerHTML = "";
    (ctx.filters || []).forEach((label) => {
      const chip = el("span", "ctx-chip", `${esc(label)}`);
      const x = el("button", "", "&times;");
      x.type = "button";
      x.setAttribute("aria-label", `Remove ${label}`);
      x.addEventListener("click", () => ask(`remove filter: ${label}`));
      chip.appendChild(x);
      contextChips.appendChild(chip);
    });
    if (ctx.sort) contextChips.appendChild(el("span", "ctx-chip sort", `sorted: ${esc(ctx.sort)}`));
    if (ctx.focus) contextChips.appendChild(el("span", "ctx-chip sort", `talking about: ${esc(ctx.focus)}`));
    contextBar.hidden = !contextChips.children.length;
  }

  // ---------------------------------------------------------------- api
  let busy = false;
  async function ask(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    busy = true;
    sendBtn.disabled = true;
    welcome.style.display = "none";
    addUser(text);
    const bubble = addBot();
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text.slice(0, 500), session_id: sessionId }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      renderAnswer(bubble, await res.json());
    } catch (err) {
      renderAnswer(bubble, { text: `Something went wrong: ${err.message}. Is the server running?` });
    } finally {
      busy = false;
      sendBtn.disabled = false;
      input.focus();
    }
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const text = input.value;
    input.value = "";
    autosize();
    ask(text);
  });
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      form.requestSubmit();
    }
  });
  function autosize() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 140) + "px";
  }
  input.addEventListener("input", autosize);

  function clearConversation() {
    fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId }),
    }).catch(() => {});
    chat.querySelectorAll(".msg").forEach((n) => n.remove());
    renderContext({ filters: [] });
    welcome.style.display = "";
    input.focus();
  }
  document.getElementById("clearBtn").addEventListener("click", clearConversation);
  document.getElementById("contextReset").addEventListener("click", () => ask("start over"));

  const starters = document.getElementById("starters");
  STARTERS.forEach(([label, q]) => {
    const b = el("button", "starter", `<b>${esc(label)}</b><span>${esc(q)}</span>`);
    b.type = "button";
    b.addEventListener("click", () => ask(q));
    starters.appendChild(b);
  });

  // ---------------------------------------------------------------- status polling
  async function pollStatus() {
    try {
      const s = await (await fetch("/api/status")).json();
      const text = statusEl.querySelector(".status-text");
      statusEl.className = "status";
      if (s.stage === "error") {
        statusEl.classList.add("status-error");
        text.textContent = "Error";
      } else if (s.stage === "ready") {
        statusEl.classList.add(s.dense ? "status-ready" : "status-partial");
        text.textContent = s.dense ? `Ready · ${s.stats.laptops} laptops` : "Ready · keyword only";
      } else {
        statusEl.classList.add(s.ready ? "status-partial" : "status-loading");
        text.textContent = s.ready ? "Keyword search · embedding…" : "Indexing…";
      }
      statusEl.title = s.message;
      if (s.stage !== "ready" && s.stage !== "error") setTimeout(pollStatus, 2000);
    } catch (_) {
      setTimeout(pollStatus, 3000);
    }
  }
  pollStatus();
  input.focus();
})();
