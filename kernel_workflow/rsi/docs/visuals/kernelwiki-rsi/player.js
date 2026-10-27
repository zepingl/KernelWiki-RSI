"use strict";

(() => {
  const data = window.RSI_CASE;
  const find = identifier => document.getElementById(identifier);
  const scene = find("scene");
  const demo = scene.closest(".demo");
  const token = find("token");
  const phaseButtons = [...document.querySelectorAll("[data-phase-button]")];
  const trialGroups = [...scene.querySelectorAll("[data-trial-card]")];
  const nodes = [...scene.querySelectorAll("[data-node]")];
  const scores = [...scene.querySelectorAll("[data-score-node]")];
  const steps = [...scene.querySelectorAll("[data-step]")];
  const edges = [0, 1, 2, 3, 4].map(index => find("edge-" + index));
  const phases = [
    { title: "在线探索", description: "工作策略控制固定 GEAK workflow 的探索；Agent、参考知识与 evaluator 不变。每个结果绑定父源码、上下文和独立测量。这里的线上动作只是教学编排。", duration: 4200 },
    { title: "逆向追溯", description: "从保留结果追溯源码、直接父上下文、原始调用与知识版本，按 gap 调查。原始事实和 analyst 判断分开，未知不能被填成成功。", duration: 4000 },
    { title: "冻结 buffer", description: "历史和逆向调查完成后，冻结一次新的 buffer。正式 primary 支持先于候选确定；本图的构造 world 不进入真实 buffer。", duration: 3300 },
    { title: "梦境回放", description: "三条手写策略在同一构造 buffer 上回放。分数由现有 Python simulator 预计算；策略只读已揭示的 prefix，不预测结果，不发起新执行。", duration: 6600 },
    { title: "择优反馈", description: "validation 择优的工作副本控制下一轮探索。第二轮没有更好候选，保留 θ₂；真实发布另需 fresh replay、paired online 与 CAS。", duration: 4600 }
  ];
  const state = {
    round: 0, phase: 0, elapsed: 0, speed: 1,
    playing: !window.matchMedia("(prefers-reduced-motion: reduce)").matches,
    trial: "neutral", lastFrame: null
  };
  const percent = value => (value * 100).toFixed(2) + "%";
  const score = value => (value >= 0 ? "+" : "−") + Math.abs(value).toFixed(5);
  const currentRound = () => data.rounds[state.round];
  const trial = identifier => currentRound().trials.find(item => item.id === identifier);
  const visibleWorld = () => data.sources.worlds[state.round];
  const worldTrace = identifier => trial(identifier).result.traces.find(item => item.world_id === "constructed/" + visibleWorld().world_id);
  const subscript = index => "₁₂"[index];

  function showEvidence() {
    const panel = find("evidence-panel");
    panel.open = true;
    panel.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "instant" : "smooth", block: "start" });
  }

  // The token carries the phase's output along its outgoing edge as the phase ends.
  function moveToken(fraction) {
    const travel = Math.max(0, Math.min(1, (fraction - .55) / .45));
    const edge = edges[state.phase];
    const point = edge.getPointAtLength(edge.getTotalLength() * travel);
    token.setAttribute("cx", point.x.toFixed(1));
    token.setAttribute("cy", point.y.toFixed(1));
    token.setAttribute("fill", state.phase < 2 ? "#3273dc" : "#7a52c7");
    token.style.opacity = travel > 0 && travel < 1 ? "1" : "0";
  }

  function render() {
    const round = currentRound();
    const phase = phases[state.phase];
    const selected = trial(round.selected_id);
    const source = visibleWorld();
    const baseline = worldTrace("neutral");
    const candidate = worldTrace("continue");
    const fraction = Math.min(1, state.elapsed / phase.duration);
    if (state.phase === 3) state.trial = fraction < .28 ? "neutral" : fraction < .58 ? "stop" : "continue";
    else state.trial = state.phase === 4 ? round.selected_id : round.incumbent_id;
    scene.dataset.phase = String(state.phase);
    scene.dataset.round = String(state.round + 1);
    scene.dataset.playing = String(state.playing);
    scene.dataset.trial = state.trial;
    demo.dataset.playing = String(state.playing);
    find("round-label").textContent = "round t = " + (state.round + 1);
    find("round-select").value = String(state.round);
    find("phase-title").textContent = phase.title;
    find("phase-description").textContent = phase.description;
    find("world-count").textContent = String(round.worlds);
    find("objective-readout").textContent = score(trial(state.trial).result.summary.objective);
    find("selected-policy-label").textContent = "θ₂ · " + selected.title;
    find("selected-in-scene").textContent = "θ₂ · " + selected.title;
    find("retention-note").textContent = round.policy_changed ? "首轮择优工作副本；下一轮使用 θ₂。" : "没有更优候选：保留 θ₂，不强行更新。";
    find("feedback-label").textContent = round.policy_changed ? "validation 择优 → 控制下一轮" : "无更优候选 → 保留 θ₂";
    find("working-policy").textContent = (state.phase === 4 || round.incumbent_id === "continue") ? "θ₂ · 续探已测优叶" : "θ₀ · 基线轮转";
    find("tree-title").textContent = "DISCOVERY TREE · 构造世界 T" + subscript(state.round);
    find("node-a1-quality").textContent = "+" + percent(source.measurements.A1.quality);
    find("node-a2-quality").textContent = "+" + percent(source.measurements.A2.quality);
    find("baseline-quality").textContent = percent(baseline.best_quality);
    find("candidate-quality").textContent = percent(candidate.best_quality);
    find("baseline-bar").setAttribute("width", String(220 * baseline.best_quality / .1));
    find("candidate-bar").setAttribute("width", String(220 * candidate.best_quality / .1));
    find("quality-gain").textContent = "+" + ((candidate.best_quality - baseline.best_quality) * 100).toFixed(2) + "pp";
    find("case-gain").innerHTML = "+" + ((data.sources.worlds[0].measurements.A2.quality - data.sources.worlds[0].measurements.A1.quality) * 100).toFixed(2) + "<span>pp</span>";
    for (const group of trialGroups) {
      const identifier = group.dataset.trialCard;
      group.classList.toggle("active", identifier === state.trial && state.phase >= 3);
      find("score-" + identifier).textContent = score(trial(identifier).result.summary.objective);
    }
    const revealed = new Set(["root"]);
    const trace = worldTrace(state.trial);
    const events = trace.events.filter(event => event.revealed.length);
    const count = state.phase === 3 ? Math.min(events.length, fraction < .58 ? 1 : 2) : state.phase >= 1 ? events.length : fraction < .33 ? 0 : fraction < .7 ? 1 : 2;
    for (const event of events.slice(0, count)) for (const item of event.revealed) revealed.add(item.node_id);
    for (const node of nodes) node.classList.toggle("revealed", revealed.has(node.dataset.node));
    for (const label of scores) label.classList.toggle("revealed", revealed.has(label.dataset.scoreNode));
    const reached = state.phase === 0 ? 0 : state.phase > 1 ? steps.length : Math.ceil(fraction * 1.25 * steps.length);
    for (const step of steps) step.classList.toggle("reached", Number(step.dataset.step) < reached);
    phaseButtons.forEach((button, index) => {
      if (index === state.phase) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
      button.style.setProperty("--p", index < state.phase ? 1 : index === state.phase ? fraction : 0);
    });
    moveToken(fraction);
    find("play-button").setAttribute("aria-label", state.playing ? "暂停播放" : "播放循环");
    find("speed-button").textContent = state.speed + "×";
  }

  function setPhase(index, playing = false) {
    state.phase = Math.max(0, Math.min(phases.length - 1, index));
    state.elapsed = playing ? 0 : phases[state.phase].duration * .78;
    state.playing = playing;
    render();
  }

  function ended() {
    return state.round === data.rounds.length - 1
      && state.phase === phases.length - 1
      && state.elapsed >= phases[state.phase].duration;
  }

  function togglePlay() {
    if (!state.playing && ended()) {
      state.round = 0;
      setPhase(0, true);
      return;
    }
    state.playing = !state.playing;
    render();
  }

  // A static frame shows the whole loop: no dimming, no token, no motion.
  function exportSvg() {
    const clone = scene.cloneNode(true);
    clone.dataset.playing = "false";
    clone.dataset.focus = "false";
    clone.querySelector("#token").remove();
    clone.setAttribute("width", "1140");
    clone.setAttribute("height", "640");
    clone.setAttribute("role", "img");
    clone.querySelectorAll("[tabindex]").forEach(node => node.removeAttribute("tabindex"));
    clone.querySelectorAll(".node").forEach(node => node.removeAttribute("role"));
    return new XMLSerializer().serializeToString(clone);
  }

  function frame(timestamp) {
    const delta = state.lastFrame === null ? 0 : Math.min(timestamp - state.lastFrame, 100);
    state.lastFrame = timestamp;
    if (state.playing && !document.hidden) {
      state.elapsed += delta * state.speed;
      if (state.elapsed >= phases[state.phase].duration) {
        state.elapsed = 0;
        if (state.phase < phases.length - 1) state.phase += 1;
        else if (state.round < data.rounds.length - 1) { state.round += 1; state.phase = 0; }
        else { state.elapsed = phases[state.phase].duration; state.playing = false; }
      }
      render();
    }
    window.requestAnimationFrame(frame);
  }

  for (const button of phaseButtons) button.addEventListener("click", () => setPhase(Number(button.dataset.phaseButton)));
  for (const node of nodes) {
    node.addEventListener("click", event => { event.stopPropagation(); showEvidence(); });
    node.addEventListener("keydown", event => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); showEvidence(); } });
  }
  find("play-button").addEventListener("click", togglePlay);
  scene.addEventListener("click", togglePlay);
  find("speed-button").addEventListener("click", () => { state.speed = state.speed === 1 ? 2 : 1; render(); });
  find("round-select").addEventListener("change", event => { state.round = Number(event.target.value); setPhase(3); });
  find("export-button").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([exportSvg()], { type: "image/svg+xml;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = "kernelwiki-rsi-loop-t" + (state.round + 1) + ".svg";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });
  document.addEventListener("keydown", event => {
    if (["INPUT", "SELECT", "SUMMARY"].includes(event.target.tagName)) return;
    if (event.key === "ArrowRight") { event.preventDefault(); setPhase(state.phase + 1); }
    if (event.key === "ArrowLeft") { event.preventDefault(); setPhase(state.phase - 1); }
    if (event.key === " " && event.target === document.body) { event.preventDefault(); togglePlay(); }
  });
  const history = data.historical;
  find("coverage-attempts").textContent = data.current.attempts.toLocaleString("en-US");
  find("coverage-diagnostic").textContent = String(data.current.worlds);
  find("coverage-primary").textContent = String(data.current.primary_worlds);
  find("archived-validation").textContent = score(history["incumbent-validation"].objective) + " → " + score(history["candidate-1-validation"].objective);
  find("archived-quality").textContent = history["candidate-1-validation"].full_budget_quality.toFixed(8);
  find("worlds-hash").textContent = data.sources.worlds_sha256;
  for (let index = 0; index < data.sources.worlds.length; index += 1) {
    const source = data.sources.worlds[index];
    const row = document.createElement("tr");
    for (const value of ["T" + (index + 1), source.world_id, String(source.line)]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    find("source-rows").append(row);
  }
  window.RSI_VIEW = {
    setRound(index) { state.round = Math.max(0, Math.min(data.rounds.length - 1, index)); setPhase(3); },
    setPhase(index) { setPhase(index); },
    exportSvg,
    snapshot() { return { round: state.round + 1, phase: state.phase, playing: state.playing, selected: currentRound().selected_id }; }
  };
  render();
  window.requestAnimationFrame(frame);
})();
