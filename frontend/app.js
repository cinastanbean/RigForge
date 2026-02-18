const chatBox = document.getElementById("chatBox");
const chatForm = document.getElementById("chatForm");
const messageInput = document.getElementById("messageInput");
const interactionMode = document.getElementById("interactionMode");
const enthusiasmLevel = document.getElementById("enthusiasmLevel");
const buildDataMode = document.getElementById("buildDataMode");
const requirementsBox = document.getElementById("requirements");
const buildBox = document.getElementById("build");
const risksBox = document.getElementById("risks");

const sessionId = crypto.randomUUID();
const DEFAULT_REQUIREMENTS = {
  budget_min: 5000,
  budget_max: 10000,
  use_case: [],
  resolution: "1080p",
  need_monitor: null,
  storage_target_gb: 0,
  cpu_preference: "",
  priority: "balanced",
  case_size: "ATX",
  need_quiet: false,
  need_wifi: false,
};
const EMPTY_BUILD = {
  cpu: null,
  motherboard: null,
  memory: null,
  storage: null,
  gpu: null,
  psu: null,
  case: null,
  cooler: null,
};

function resolveApiBase() {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("api_base");
  if (override) return override.replace(/\/$/, "");

  const isLocal = ["localhost", "127.0.0.1"].includes(window.location.hostname);
  if (isLocal && window.location.port && window.location.port !== "8000") {
    return "http://127.0.0.1:8000";
  }
  return "";
}

const API_BASE = resolveApiBase();

function addMessage(role, content) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = content;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
}

function setRequirements(req) {
  const lines = [
    ["预算区间", `${req.budget_min} - ${req.budget_max} 元`],
    ["用途", (req.use_case || []).join(", ") || "未明确"],
    ["分辨率", req.resolution || "未明确"],
    ["存储目标", req.storage_target_gb ? `${req.storage_target_gb}GB` : "未确认"],
    ["CPU偏好", req.cpu_preference || "未明确"],
    ["推荐策略", req.priority || "balanced"],
    ["机箱规格", req.case_size],
    ["静音", req.need_quiet ? "是" : "否"],
    ["WiFi", req.need_wifi ? "是" : "否"],
  ];
  requirementsBox.innerHTML = `<h3>需求画像</h3>${lines
    .map(([k, v]) => `<div class="line"><span>${k}</span><span>${v}</span></div>`)
    .join("")}`;
}

function setBuild(build) {
  const current = build || EMPTY_BUILD;
  const slots = [
    ["CPU", current.cpu],
    ["主板", current.motherboard],
    ["内存", current.memory],
    ["SSD", current.storage],
    ["显卡", current.gpu],
    ["电源", current.psu],
    ["机箱", current.case],
    ["散热", current.cooler],
  ];
  const knownParts = slots.filter(([, part]) => !!part).length;
  const partialTotal = slots.reduce((sum, [, part]) => sum + (part?.price || 0), 0);
  buildBox.innerHTML = `<h3>推荐配置</h3>${slots
    .map(
      ([name, part]) =>
        `<div class="line"><span>${name}</span><span>${part ? `${part.name} (${part.price}元)` : "待推荐"}</span></div>`,
    )
    .join("")}
    <div class="line"><span>已匹配部件</span><span>${knownParts}/8</span></div>
    <div class="line"><span>当前合计</span><span>${partialTotal} 元</span></div>`;
}

function setRisks(payload) {
  const issues = payload.compatibility_issues || [];
  const modeMap = {
    jd_newegg: "京东 + Newegg",
    jd: "仅京东",
    newegg: "仅 Newegg",
  };
  const providerMap = {
    zhipu: "智谱",
    openrouter: "OpenRouter",
    rules: "规则模式",
  };
  const sessionProvider = payload.model_name || providerMap[payload.session_model_provider] || payload.session_model_provider || "-";
  const turnProvider = providerMap[payload.turn_model_provider] || payload.turn_model_provider || "-";
  let fallbackHtml = "";
  if (payload.response_mode === "fallback") {
    const reasonMap = {
      rate_limited: "模型限流，已切换到规则模式回复。",
      auth_error: "模型鉴权失败，已切换到规则模式回复。",
      timeout: "模型请求超时，已切换到规则模式回复。",
      no_model_config: "当前会话未选到可用模型，已使用规则模式回复。",
      model_error: "模型服务异常，已切换到规则模式回复。",
    };
    const reason = payload.model_status_detail || reasonMap[payload.fallback_reason] || reasonMap.model_error;
    fallbackHtml = `<div class="line warn"><span>模型状态</span><span>${reason}</span></div>`;
  }
  const issueHtml =
    issues.length > 0
      ? issues.map((x) => `<div class="line bad"><span>风险</span><span>${x}</span></div>`).join("")
      : `<div class="line good"><span>兼容性</span><span>通过</span></div>`;
  risksBox.innerHTML = `
    <h3>风险与估算</h3>
    <div class="line"><span>预计功耗</span><span>${payload.estimated_power || 0}W</span></div>
    <div class="line"><span>性能预估</span><span>${payload.estimated_performance || "-"}</span></div>
    <div class="line"><span>构建数据源</span><span>${payload.build_data_source || "-"}</span></div>
    <div class="line"><span>数据版本</span><span>${payload.build_data_version || "-"}</span></div>
    <div class="line"><span>数据接入</span><span>${modeMap[payload.build_data_mode] || payload.build_data_mode || "-"}</span></div>
    <div class="line"><span>会话模型</span><span>${sessionProvider}</span></div>
    <div class="line"><span>本轮使用</span><span>${turnProvider}</span></div>
    ${fallbackHtml}
    ${issueHtml}
  `;
}

async function sendMessage(text) {
  addMessage("user", text);
  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        message: text,
        interaction_mode: interactionMode?.value || "chat",
        enthusiasm_level: enthusiasmLevel?.value || "standard",
        build_data_mode: buildDataMode?.value || "jd",
      }),
    });

    if (!res.ok) {
      const detail = await res.text();
      const suffix = detail ? `（${res.status}）` : "";
      addMessage("assistant", `请求失败，请稍后重试${suffix}。`);
      return;
    }

    const payload = await res.json();
    addMessage("assistant", payload.reply);
    setRequirements(payload.requirements);
    setBuild(payload.build);
    setRisks(payload);
  } catch (_err) {
    addMessage("assistant", "网络异常，请确认后端服务已启动（http://127.0.0.1:8000）。");
  }
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  messageInput.value = "";
  await sendMessage(text);
});

addMessage(
  "assistant",
  "你好呀，我们一起来设计你的组装机吧！先从方向开始：这台电脑你主要想拿来做什么？游戏、办公、剪辑，还是 AI 开发？",
);
setRequirements(DEFAULT_REQUIREMENTS);
setBuild(EMPTY_BUILD);
  setRisks({
  estimated_power: 0,
  estimated_performance: "-",
  build_data_source: "csv(jd+newegg)",
  build_data_version: "latest",
  build_data_mode: buildDataMode?.value || "jd",
  session_model_provider: "-",
  turn_model_provider: "-",
  response_mode: "fallback",
  fallback_reason: null,
  model_status_detail: "等待你输入需求后开始分析。",
  compatibility_issues: [],
});
