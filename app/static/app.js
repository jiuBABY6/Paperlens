"use strict";

const $ = selector => document.querySelector(selector);
const escapeHtml = value => { const node = document.createElement("div"); node.textContent = value ?? ""; return node.innerHTML; };
const state = { paperId: "", paper: null, conversationId: "", conversations: [], learningMemory: {}, memoryItems: [], activeRunId: "", source: null, pdf: null, page: 1, bbox: null, bboxes: [] };
const pdfScale = 1.35;

async function request(url, options = {}) {
  const response = await fetch(url, options);
  if (response.status === 204) return null;
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`);
  return data;
}

$("#upload").onclick = async () => {
  const file = $("#file").files[0]; if (!file) return;
  setStatus("正在解析、建立索引并生成阅读卡片…");
  const form = new FormData(); form.append("file", file);
  try {
    const data = await request("/api/papers", { method: "POST", body: form });
    if (data.status === "processing") return setStatus(`论文仍在处理中：${data.id}`);
    await openPaper(data);
    setStatus(`${data.reused ? "已复用" : "已完成"}：版本 ${data.analysis_version}，${data.chunk_count} 个证据块。`);
  } catch (error) { setStatus(`失败：${error.message}`, true); }
};

$("#reanalyze").onclick = async () => {
  if (!state.paperId) return;
  setStatus("正在重新分析论文…");
  try { await openPaper(await request(`/api/papers/${state.paperId}/reanalyze`, { method: "POST" })); setStatus("重新分析完成。"); }
  catch (error) { setStatus(`重新分析失败：${error.message}`, true); }
};

async function openPaper(data) {
  state.paperId = data.id; state.paper = data; state.conversationId = "";
  await loadLearningMemory(); renderCard(data); await loadPdf(data.pdf_url); await loadConversations();
  $("#workspace").classList.remove("hidden"); $("#reanalyze").classList.remove("hidden");
  if (state.conversations.length) await selectConversation(state.conversations[0].id); else await createConversation();
}

async function loadConversations() {
  const data = await request(`/api/papers/${state.paperId}/conversations`);
  state.conversations = data.items; renderConversationList();
}

async function loadLearningMemory() {
  const [value, items] = await Promise.all([
    request(`/api/papers/${state.paperId}/learning-memory`),
    request(`/api/papers/${state.paperId}/learning-memory/items`),
  ]);
  state.learningMemory = value.memory || {};
  state.memoryItems = items.items || [];
}

function renderConversationList() {
  const query = $("#conversationSearch").value.trim().toLowerCase();
  const items = state.conversations.filter(item => `${item.title} ${item.last_message || ""}`.toLowerCase().includes(query));
  $("#conversationList").innerHTML = items.length ? items.map(item => `<div class="conversation-item ${item.id === state.conversationId ? "active" : ""}" data-id="${item.id}"><strong>${escapeHtml(item.title)}</strong><small>${escapeHtml(item.last_message || "尚无消息")}</small></div>`).join("") : '<p class="empty-chat">暂无对话</p>';
  document.querySelectorAll(".conversation-item").forEach(node => node.onclick = () => selectConversation(node.dataset.id));
}

$("#conversationSearch").oninput = renderConversationList;
$("#newConversation").onclick = createConversation;
async function createConversation() {
  if (!state.paperId) return;
  const value = await request(`/api/papers/${state.paperId}/conversations`, { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({title:"新对话"}) });
  await loadConversations(); await selectConversation(value.id);
}

async function selectConversation(id) {
  if (state.activeRunId) return setStatus("请等待当前回答结束或先停止。", true);
  const conversation = await request(`/api/papers/${state.paperId}/conversations/${id}`);
  state.conversationId = id; renderConversationList(); renderConversation(conversation);
}

function renderConversation(conversation) {
  $("#conversationTitle").textContent = conversation.title;
  const learned = (state.learningMemory.interactions || []).length;
  $("#contextState").textContent = `${conversation.messages.filter(m => m.role === "user").length} 轮 · 当前会话上下文 · 论文级记忆 ${learned} 条`;
  $("#renameConversation").disabled = false; $("#archiveConversation").disabled = false;
  $("#question").disabled = false; $("#send").disabled = false;
  $("#messages").innerHTML = conversation.messages.length ? conversation.messages.map(messageHtml).join("") : '<div class="empty-chat">从论文中的方法、图表或实验结果开始提问。</div>';
  bindCitationButtons(); scrollMessages();
}

function messageHtml(message) {
  const citations = (message.citations || []).map(citationButton).join("");
  const trace = message.metadata?.trace;
  const visualFailed = message.metadata?.visual_answer_check?.supported === false || message.metadata?.visual_verification_status === "failed";
  const decision = visualFailed ? "视觉核验未通过" : message.metadata?.answerable === false ? "证据不足，已拒答" : message.metadata?.evidence_sufficient === false ? "任务证据不完整" : "";
  const warning = decision ? `<p class="status ${visualFailed ? "error" : ""}">${escapeHtml(decision)}</p>` : "";
  return `<article class="message ${message.role}" data-message-id="${message.id}"><div class="message-meta">${message.role === "user" ? "你" : "PaperLens"} · 第 ${message.turn_index} 轮 · ${escapeHtml(message.status)}</div><div class="bubble">${escapeHtml(message.content)}</div>${warning}${citations ? `<div class="citations">${citations}</div>` : ""}${trace ? traceHtml(trace) : ""}</article>`;
}
function citationButton(item) { const id = item.evidence_id || item.id || ""; return `<button class="citation" data-evidence-id="${escapeHtml(id)}">${escapeHtml(item.type || "text")} · 第 ${item.page || "?"} 页 · ${escapeHtml(item.section || id)}</button>`; }
function traceHtml(trace) { const lines = (trace.steps || []).map(step => `${step.step}. ${step.tool} → ${(step.result_ids || []).join(", ") || "无结果"}`).join("\n"); return `<details class="trace"><summary>Agent Trace</summary><pre>${escapeHtml(lines)}</pre></details>`; }
function bindCitationButtons() { document.querySelectorAll(".citation").forEach(node => node.onclick = () => showEvidence(node.dataset.evidenceId)); }

$("#renameConversation").onclick = async () => {
  const title = prompt("新的对话标题：", $("#conversationTitle").textContent); if (!title) return;
  await request(`/api/papers/${state.paperId}/conversations/${state.conversationId}`, {method:"PATCH", headers:{"Content-Type":"application/json"}, body:JSON.stringify({title})});
  await loadConversations(); await selectConversation(state.conversationId);
};
$("#archiveConversation").onclick = async () => {
  if (!confirm("归档当前对话？消息不会被物理删除。")) return;
  await request(`/api/papers/${state.paperId}/conversations/${state.conversationId}`, {method:"DELETE"});
  state.conversationId = ""; await loadConversations(); if (state.conversations.length) await selectConversation(state.conversations[0].id); else await createConversation();
};

$("#composer").onsubmit = async event => {
  event.preventDefault(); const question = $("#question").value.trim();
  if (!question || !state.conversationId || state.activeRunId) return;
  setRunning(true); $("#question").value = "";
  appendLiveMessage("user", question, "completed");
  const assistant = appendLiveMessage("assistant", "", "running");
  try {
    const submitted = await request(`/api/papers/${state.paperId}/conversations/${state.conversationId}/messages`, { method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({question, client_message_id:crypto.randomUUID()}) });
    state.activeRunId = submitted.run_id; connectEvents(submitted.run_id, assistant);
  } catch (error) { assistant.querySelector(".bubble").textContent = `提交失败：${error.message}`; setRunning(false); }
};
$("#question").onkeydown = event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); $("#composer").requestSubmit(); } };

function connectEvents(runId, assistant) {
  if (state.source) state.source.close();
  const url = `/api/papers/${state.paperId}/conversations/${state.conversationId}/runs/${runId}/events`;
  const source = new EventSource(url); state.source = source; const bubble = assistant.querySelector(".bubble");
  const statuses = {"run.started":"正在加载会话上下文…","context.loading":"正在读取会话与论文级记忆…","query.resolved":"已解析追问，准备检索…","router.started":"Router 正在识别任务…","router.completed":"Router 已完成路由…","langgraph.node.started":"LangGraph 节点执行中…","specialist.started":"专业智能体开始检索…","function_call.requested":"模型正在选择工具…","tool.started":"工具正在执行…","tool.completed":"工具执行完成…","repair.dispatched":"证据不足，正在定向修复…","answer.verified":"答案已完成证据核验…","answer.started":"正在输出已验证答案…"};
  Object.entries(statuses).forEach(([name, label]) => source.addEventListener(name, () => $("#runState").textContent = label));
  source.addEventListener("answer.delta", event => { bubble.textContent += JSON.parse(event.data).delta || ""; scrollMessages(); });
  ["run.completed", "run.partial", "run.failed", "run.cancelled", "run.timed_out"].forEach(name => source.addEventListener(name, async event => {
    const payload = JSON.parse(event.data); source.close(); state.source = null; state.activeRunId = ""; setRunning(false);
    if (payload.error && !bubble.textContent) bubble.textContent = payload.error;
    await loadLearningMemory(); renderCard(state.paper); await loadConversations(); await selectConversation(state.conversationId);
  }));
  source.onerror = () => { $("#runState").textContent = "连接中断，正在等待浏览器自动重连…"; };
}

$("#cancelRun").onclick = async () => { if (state.activeRunId) await request(`/api/papers/${state.paperId}/conversations/${state.conversationId}/runs/${state.activeRunId}/cancel`, {method:"POST"}); };
function setRunning(value) { $("#question").disabled = value || !state.conversationId; $("#send").disabled = value || !state.conversationId; $("#cancelRun").classList.toggle("hidden", !value); if (!value) $("#runState").textContent = ""; }
function appendLiveMessage(role, content, status) { const node = document.createElement("div"); node.innerHTML = messageHtml({id:"live", role, content, status, turn_index:"…", citations:[], metadata:{}}); const message = node.firstElementChild; const empty = $("#messages").querySelector(".empty-chat"); if (empty) empty.remove(); $("#messages").append(message); scrollMessages(); return message; }
function scrollMessages() { $("#messages").scrollTop = $("#messages").scrollHeight; }

function renderCard(data) {
  const card = data.card || {};
  const field = (name, item) => `<section class="field"><h3>${name}</h3><div>${escapeHtml(item?.text || "未生成")}</div></section>`;
  const claims = (name, items) => `<section class="field"><h3>${name}</h3>${(items || []).map(item => `<div class="claim">${escapeHtml(item.text)}</div>`).join("") || "未找到明确原文证据。"}</section>`;
  const learned = (state.learningMemory.interactions || []).length;
  const sections = (state.learningMemory.explored_sections || []).slice(-8).join("、") || "尚未形成";
  const unresolved = (state.learningMemory.unresolved_questions || []).length;
  const memories = state.memoryItems.slice(0, 8).map(item => `<div class="claim memory-item" data-memory-id="${escapeHtml(item.id)}"><strong>${item.pinned ? "📌 " : ""}${escapeHtml(item.resolved_question)}</strong><small>重要性 ${Number(item.importance).toFixed(2)} · ${escapeHtml(item.status)}${item.user_note ? ` · ${escapeHtml(item.user_note)}` : ""}</small><div><button class="secondary memory-pin" type="button">${item.pinned ? "取消置顶" : "置顶"}</button><button class="secondary memory-note" type="button">备注</button><button class="danger memory-forget" type="button">忘记</button></div></div>`).join("");
  const progress = `<section class="field"><h3>跨会话学习记忆</h3><div>已验证问答 ${learned} 条 · 已涉及章节：${escapeHtml(sections)} · 待解决 ${unresolved} 条</div><div class="memory-list">${memories || "尚无可管理的长期记忆。"}</div></section>`;
  $("#summary").innerHTML = `<h1 class="paper-title">${escapeHtml(data.title)}</h1><p class="abstract">${escapeHtml(data.abstract)}</p>${progress}${field("研究问题", card.research_question)}${field("方法概览", card.method_overview)}${field("实验解读", card.experiment_summary)}${claims("核心贡献", card.contributions)}${claims("局限性", card.limitations)}`;
  const figures = (data.figures || []).filter(item => item.image_url);
  $("#figurePanel").classList.toggle("hidden", !figures.length);
  $("#figures").innerHTML = figures.map(item => `<figure class="figure" data-evidence-id="${item.id}"><img loading="lazy" src="${item.image_url}" alt="${escapeHtml(item.caption)}"><figcaption>第 ${item.page} 页 · ${escapeHtml(item.caption)}</figcaption></figure>`).join("");
  document.querySelectorAll(".figure").forEach(node => node.onclick = () => showEvidence(node.dataset.evidenceId));
  bindMemoryActions();
}

function bindMemoryActions() {
  document.querySelectorAll(".memory-item").forEach(node => {
    const item = state.memoryItems.find(value => value.id === node.dataset.memoryId);
    node.querySelector(".memory-pin").onclick = () => updateMemory(item.id, {pinned: !item.pinned});
    node.querySelector(".memory-note").onclick = () => { const note = prompt("为这条记忆添加个人备注：", item.user_note || ""); if (note !== null) updateMemory(item.id, {user_note: note}); };
    node.querySelector(".memory-forget").onclick = () => { if (confirm("忘记这条派生记忆？原始对话不会被删除。")) updateMemory(item.id, {status: "forgotten"}); };
  });
}

async function updateMemory(memoryId, changes) {
  await request(`/api/papers/${state.paperId}/learning-memory/items/${memoryId}`, {method:"PATCH", headers:{"Content-Type":"application/json"}, body:JSON.stringify(changes)});
  await loadLearningMemory(); renderCard(state.paper);
}

async function showEvidence(id) {
  const item = await request(`/api/papers/${state.paperId}/evidence/${encodeURIComponent(id)}`);
  $("#evidenceDetail").innerHTML = `<strong>${escapeHtml(item.type)} · 第 ${item.page} 页 · ${escapeHtml(item.section || "")}</strong><p>${escapeHtml(item.content || item.text || item.metadata?.caption || "")}</p>`;
  state.page = item.page; state.bbox = item.bbox; state.bboxes = item.bboxes || []; $("#evidenceDrawer").classList.remove("hidden"); await renderPage();
}
$("#openPdf").onclick = async () => { $("#evidenceDrawer").classList.remove("hidden"); await renderPage(); };
$("#closeDrawer").onclick = () => $("#evidenceDrawer").classList.add("hidden");
$("#previousPage").onclick = () => { if (state.page > 1) { state.page--; state.bbox = null; state.bboxes = []; renderPage(); } };
$("#nextPage").onclick = () => { if (state.pdf && state.page < state.pdf.numPages) { state.page++; state.bbox = null; state.bboxes = []; renderPage(); } };

async function loadPdf(url) { if (!window.pdfjsLib) throw new Error("PDF.js 加载失败"); pdfjsLib.GlobalWorkerOptions.workerSrc = "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js"; state.pdf = await pdfjsLib.getDocument(url).promise; state.page = 1; await renderPage(); }
async function renderPage() {
  if (!state.pdf) return; const page = await state.pdf.getPage(state.page); const viewport = page.getViewport({scale:pdfScale}); const canvas = $("#pdfCanvas"); canvas.width = viewport.width; canvas.height = viewport.height; $("#pdfPage").style.width = `${viewport.width}px`; $("#pdfPage").style.height = `${viewport.height}px`; await page.render({canvasContext:canvas.getContext("2d"), viewport}).promise;
  const layer = $("#textLayer"); layer.replaceChildren(); layer.style.width = `${viewport.width}px`; layer.style.height = `${viewport.height}px`; if (pdfjsLib.renderTextLayer) await pdfjsLib.renderTextLayer({textContentSource:await page.getTextContent(), container:layer, viewport, textDivs:[]}).promise;
  const marks = $("#highlight"); marks.replaceChildren(); const boxes = state.bboxes.length ? state.bboxes : (state.bbox ? [state.bbox] : []); boxes.forEach(([x0,y0,x1,y1]) => { const mark = document.createElement("div"); mark.className = "highlight-box"; Object.assign(mark.style,{left:`${x0*pdfScale}px`,top:`${y0*pdfScale}px`,width:`${Math.max(3,(x1-x0)*pdfScale)}px`,height:`${Math.max(3,(y1-y0)*pdfScale)}px`}); marks.append(mark); }); $("#pageInfo").textContent = `第 ${state.page} / ${state.pdf.numPages} 页`;
}
function setStatus(text, error = false) { $("#status").textContent = text; $("#status").className = `status global-status${error ? " error" : ""}`; }
