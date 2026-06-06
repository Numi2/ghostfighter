import "./styles.css";

type Action = "guard" | "dash" | "jab" | "cross" | "hook" | "kick" | "push" | "recover";
type Side = "red" | "blue";
type Driver = "human1" | "human2" | "bot";

interface Fighter {
  x: number;
  y: number;
  vx: number;
  vy: number;
  theta: number;
  health: number;
  stamina: number;
  balance: number;
  guard: number;
  score: number;
  falls: number;
  damage: number;
  cooldown: number;
  lastAction: Action;
  hitFlash: number;
  safetyOverride: number;
}

interface BotGenome {
  name: string;
  aggression: number;
  guard: number;
  evasiveness: number;
  stamina: number;
  boundary: number;
  recovery: number;
  kickBias: number;
  elo: number;
  generation: number;
}

interface HitEvent {
  text: string;
  ttl: number;
  kind: "hit" | "fall" | "safety" | "round" | "train";
}

interface Controls {
  up: boolean;
  down: boolean;
  left: boolean;
  right: boolean;
  guard: boolean;
  jab: boolean;
  cross: boolean;
  kick: boolean;
  push: boolean;
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("Missing app root");

app.innerHTML = `
  <main class="shell">
    <section class="arenaPanel">
      <div class="topbar">
        <div>
          <h1>GhostFighter Arena</h1>
          <p>Browser PvP robot combat with trainable policy ghosts. Play same-keyboard, run bot leagues, export a model DNA string, and challenge imported bots.</p>
        </div>
        <div class="roundBox">
          <span>Round</span>
          <strong id="round">1</strong>
        </div>
      </div>
      <canvas id="arena" width="1120" height="760" aria-label="GhostFighter playable arena"></canvas>
      <div class="controls">
        <button id="start">Start</button>
        <button id="reset">Reset</button>
        <select id="mode" aria-label="Match mode">
          <option value="pvp">Local PvP</option>
          <option value="humanBot">Human vs Bot</option>
          <option value="botBot">Bot League</option>
        </select>
        <div class="hint">P1 WASD · Guard Shift · J/K/L/Space. P2 arrows · Guard / · 1/2/3/0.</div>
      </div>
    </section>
    <aside class="sidePanel">
      <section>
        <h2>Match Telemetry</h2>
        <div class="metrics">
          <div><span>Range</span><strong id="range">0.00m</strong></div>
          <div><span>Risk</span><strong id="risk">0.00</strong></div>
          <div><span>Mode</span><strong id="modeText">PvP</strong></div>
          <div><span>Winner</span><strong id="winner">active</strong></div>
        </div>
      </section>
      <section>
        <h2>Red Pilot</h2>
        <div id="redBars" class="bars"></div>
      </section>
      <section>
        <h2>Blue Pilot</h2>
        <div id="blueBars" class="bars"></div>
      </section>
      <section>
        <h2>Train Your Ghost</h2>
        <div class="trainGrid">
          <button id="train">Train 30 sims</button>
          <button id="exportBot">Export DNA</button>
        </div>
        <textarea id="botCode" spellcheck="false" aria-label="Bot DNA import and export"></textarea>
        <button id="importBot">Import Blue Bot</button>
        <div id="botStats" class="botStats"></div>
      </section>
      <section>
        <h2>Event Feed</h2>
        <div id="feed" class="feed"></div>
      </section>
    </aside>
  </main>
`;

const canvas = document.querySelector<HTMLCanvasElement>("#arena")!;
const ctx = canvas.getContext("2d")!;
const startButton = document.querySelector<HTMLButtonElement>("#start")!;
const resetButton = document.querySelector<HTMLButtonElement>("#reset")!;
const modeSelect = document.querySelector<HTMLSelectElement>("#mode")!;
const trainButton = document.querySelector<HTMLButtonElement>("#train")!;
const exportButton = document.querySelector<HTMLButtonElement>("#exportBot")!;
const importButton = document.querySelector<HTMLButtonElement>("#importBot")!;
const botCode = document.querySelector<HTMLTextAreaElement>("#botCode")!;
const arenaRadius = 3.85;
const events: HitEvent[] = [{ text: "Arena initialized. Choose a mode and start the round.", ttl: 5, kind: "round" }];
const trailRed: Array<[number, number]> = [];
const trailBlue: Array<[number, number]> = [];
const p1: Controls = makeControls();
const p2: Controls = makeControls();

let red = makeFighter(-1.25, 0);
let blue = makeFighter(1.25, 0);
let running = false;
let round = 1;
let last = performance.now();
let winner = "active";
let shake = 0;
let redBot = baseBot("Red Ghost", 0);
let blueBot = baseBot("Blue Ghost", 1);
let champion = baseBot("Champion Ghost", 2);

function makeControls(): Controls {
  return { up: false, down: false, left: false, right: false, guard: false, jab: false, cross: false, kick: false, push: false };
}

function baseBot(name: string, seed: number): BotGenome {
  const offset = seed * 0.11;
  return {
    name,
    aggression: 0.58 + offset,
    guard: 0.45,
    evasiveness: 0.36 + offset * 0.3,
    stamina: 0.62,
    boundary: 0.7,
    recovery: 0.56,
    kickBias: 0.42,
    elo: 1000,
    generation: 0,
  };
}

function makeFighter(x: number, y: number): Fighter {
  return { x, y, vx: 0, vy: 0, theta: 0, health: 100, stamina: 1, balance: 1, guard: 0, score: 0, falls: 0, damage: 0, cooldown: 0, lastAction: "guard", hitFlash: 0, safetyOverride: 0 };
}

function resetRound(full = false) {
  if (full) {
    round = 1;
    winner = "active";
  }
  red = makeFighter(-1.25, 0);
  blue = makeFighter(1.25, 0);
  trailRed.length = 0;
  trailBlue.length = 0;
  events.unshift({ text: full ? "Match reset." : `Round ${round} reset.`, ttl: 3, kind: "round" });
}

function drivers() {
  const mode = modeSelect.value;
  if (mode === "pvp") return { red: "human1" as Driver, blue: "human2" as Driver };
  if (mode === "humanBot") return { red: "human1" as Driver, blue: "bot" as Driver };
  return { red: "bot" as Driver, blue: "bot" as Driver };
}

function distance(a: Fighter, b: Fighter) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function face(a: Fighter, b: Fighter) {
  a.theta = Math.atan2(b.y - a.y, b.x - a.x);
}

function edgeClearance(f: Fighter) {
  return arenaRadius - Math.hypot(f.x, f.y);
}

function riskScore(f: Fighter) {
  const boundary = Math.max(0, 1 - edgeClearance(f) / 0.9);
  const fatigue = 1 - f.stamina;
  const unstable = 1 - f.balance;
  return Math.min(1, boundary * 0.42 + fatigue * 0.22 + unstable * 0.28 + f.damage * 0.08);
}

function applyMovement(f: Fighter, dx: number, dy: number, dt: number, action: Action) {
  const mag = Math.hypot(dx, dy) || 1;
  const sprint = action === "dash" ? 1.55 : 1;
  const speed = 1.85 * sprint * (0.72 + f.stamina * 0.34);
  f.vx += (dx / mag) * speed * dt * 8;
  f.vy += (dy / mag) * speed * dt * 8;
  f.stamina = Math.max(0, f.stamina - (action === "dash" ? 0.24 : 0.08) * dt);
  f.lastAction = action;
}

function strike(attacker: Fighter, defender: Fighter, action: Action, side: Side) {
  if (attacker.cooldown > 0 || attacker.stamina < 0.08) return;
  const spec = {
    jab: { range: 0.72, damage: 5.5, balance: 0.05, cost: 0.07, cd: 0.24 },
    cross: { range: 0.82, damage: 9, balance: 0.09, cost: 0.12, cd: 0.36 },
    hook: { range: 0.64, damage: 12, balance: 0.13, cost: 0.16, cd: 0.46 },
    kick: { range: 0.98, damage: 8, balance: 0.16, cost: 0.18, cd: 0.5 },
    push: { range: 0.7, damage: 2.5, balance: 0.24, cost: 0.14, cd: 0.42 },
  }[action as "jab" | "cross" | "hook" | "kick" | "push"];
  if (!spec) return;
  attacker.cooldown = spec.cd;
  attacker.stamina = Math.max(0, attacker.stamina - spec.cost);
  attacker.lastAction = action;
  const angle = Math.atan2(defender.y - attacker.y, defender.x - attacker.x);
  const alignment = Math.cos(angle - attacker.theta);
  const inRange = distance(attacker, defender) < spec.range && alignment > 0.18;
  if (!inRange) {
    attacker.balance = Math.max(0.12, attacker.balance - 0.025);
    return;
  }
  const guarded = defender.guard > 0.52;
  const guardScale = guarded ? 0.38 : 1;
  const damage = spec.damage * guardScale * (0.75 + attacker.stamina * 0.35);
  defender.health = Math.max(0, defender.health - damage);
  defender.balance = Math.max(0, defender.balance - spec.balance * (guarded ? 0.55 : 1));
  defender.damage = Math.min(1, defender.damage + damage / 220);
  defender.vx += Math.cos(angle) * spec.balance * 3.5;
  defender.vy += Math.sin(angle) * spec.balance * 3.5;
  defender.hitFlash = 0.22;
  attacker.score += damage * 0.08 + (guarded ? 0.05 : 0.18);
  shake = Math.max(shake, guarded ? 3 : 7);
  events.unshift({ text: `${side.toUpperCase()} ${action.toUpperCase()} ${guarded ? "checked" : "landed"} for ${damage.toFixed(1)}`, ttl: 4, kind: "hit" });
}

function recover(f: Fighter, dt: number) {
  f.stamina = Math.min(1, f.stamina + 0.18 * dt);
  f.balance = Math.min(1, f.balance + 0.16 * dt);
  f.guard = Math.max(0, f.guard - 1.8 * dt);
  f.cooldown = Math.max(0, f.cooldown - dt);
  f.hitFlash = Math.max(0, f.hitFlash - dt);
  f.safetyOverride = Math.max(0, f.safetyOverride - dt);
}

function clampArena(f: Fighter) {
  const r = Math.hypot(f.x, f.y);
  if (r > arenaRadius) {
    f.x = (f.x / r) * arenaRadius;
    f.y = (f.y / r) * arenaRadius;
    f.balance = Math.max(0, f.balance - 0.035);
    f.score -= 0.04;
  }
}

function maybeFall(f: Fighter, name: "RED" | "BLUE") {
  if (f.balance > 0.05) return;
  f.falls += 1;
  f.balance = 0.48;
  f.stamina = Math.max(0.18, f.stamina - 0.18);
  f.health = Math.max(0, f.health - 5);
  events.unshift({ text: `${name} knockdown: balance recovered by safety reset`, ttl: 4.5, kind: "fall" });
  shake = Math.max(shake, 10);
}

function humanAction(f: Fighter, opponent: Fighter, dt: number, controls: Controls, side: Side) {
  let dx = 0;
  let dy = 0;
  if (controls.up) dy -= 1;
  if (controls.down) dy += 1;
  if (controls.left) dx -= 1;
  if (controls.right) dx += 1;
  if (dx || dy) applyMovement(f, dx, dy, dt, controls.guard ? "dash" : "guard");
  f.guard = controls.guard ? Math.min(1, f.guard + 5 * dt) : f.guard;
  if (controls.jab) strike(f, opponent, "jab", side);
  if (controls.cross) strike(f, opponent, "cross", side);
  if (controls.kick) strike(f, opponent, "kick", side);
  if (controls.push) strike(f, opponent, "push", side);
}

function botAction(f: Fighter, opponent: Fighter, dt: number, bot: BotGenome, side: Side) {
  const d = distance(f, opponent);
  const risk = riskScore(f);
  let dx = opponent.x - f.x;
  let dy = opponent.y - f.y;
  let action: Action = "guard";
  if (risk > bot.boundary * 0.72 || f.balance < bot.recovery * 0.5 || f.stamina < (1 - bot.stamina) * 0.34) {
    f.safetyOverride = 0.55;
    f.guard = 1;
    dx = -f.x;
    dy = -f.y;
    action = "recover";
    if (Math.random() < 0.03) events.unshift({ text: `${side.toUpperCase()} ${bot.name} shielded into recovery`, ttl: 2.8, kind: "safety" });
  } else if (d > 0.62 + bot.aggression * 0.42) {
    action = "dash";
  } else if (bot.evasiveness > 0.55 && opponent.cooldown <= 0.08) {
    const tx = dx;
    dx = -dy;
    dy = tx;
    action = "dash";
  }
  if (action === "recover") {
    f.balance = Math.min(1, f.balance + 0.28 * dt);
    f.stamina = Math.min(1, f.stamina + 0.20 * dt);
    applyMovement(f, dx, dy, dt, "dash");
    return;
  }
  if (d > 0.58 || action === "dash") applyMovement(f, dx, dy, dt, action);
  const attackChance = (bot.aggression * 0.82 + (opponent.cooldown > 0.12 ? 0.22 : 0) + (1 - risk) * 0.16) * dt * 3.2;
  if (d < 1.02 && Math.random() < attackChance) {
    const roll = Math.random();
    const chosen: Action = roll < bot.kickBias * 0.34 ? "kick" : roll < 0.48 ? "jab" : roll < 0.82 ? "cross" : "hook";
    strike(f, opponent, chosen, side);
  }
  f.guard = Math.max(f.guard, Math.min(1, bot.guard * (risk + 0.35)));
}

function step(now: number) {
  const dt = Math.min(0.033, (now - last) / 1000);
  last = now;
  if (running && winner === "active") {
    const d = drivers();
    if (d.red === "human1") humanAction(red, blue, dt, p1, "red");
    else botAction(red, blue, dt, redBot, "red");
    if (d.blue === "human2") humanAction(blue, red, dt, p2, "blue");
    else botAction(blue, red, dt, blueBot, "blue");
    for (const f of [red, blue]) {
      recover(f, dt);
      f.x += f.vx * dt;
      f.y += f.vy * dt;
      f.vx *= 0.88;
      f.vy *= 0.88;
      clampArena(f);
    }
    face(red, blue);
    face(blue, red);
    maybeFall(red, "RED");
    maybeFall(blue, "BLUE");
    trailRed.push([red.x, red.y]);
    trailBlue.push([blue.x, blue.y]);
    if (trailRed.length > 42) trailRed.shift();
    if (trailBlue.length > 42) trailBlue.shift();
    for (const event of events) event.ttl -= dt;
    while (events.length > 9 || (events.length && events[events.length - 1].ttl <= 0)) events.pop();
    if (red.health <= 0 || blue.health <= 0) finishRound(red.health > blue.health ? "RED" : "BLUE");
  }
  draw();
  requestAnimationFrame(step);
}

function finishRound(result: "RED" | "BLUE") {
  winner = result;
  events.unshift({ text: `${winner} wins round ${round}`, ttl: 6, kind: "round" });
  round += 1;
  running = false;
  startButton.textContent = "Next Round";
}

function simulate(botA: BotGenome, botB: BotGenome, steps = 520) {
  const a = makeFighter(-1.25, 0);
  const b = makeFighter(1.25, 0);
  for (let i = 0; i < steps && a.health > 0 && b.health > 0; i++) {
    const dt = 1 / 45;
    face(a, b);
    face(b, a);
    botActionSilent(a, b, dt, botA);
    botActionSilent(b, a, dt, botB);
    for (const f of [a, b]) {
      recover(f, dt);
      f.x += f.vx * dt;
      f.y += f.vy * dt;
      f.vx *= 0.88;
      f.vy *= 0.88;
      clampArena(f);
      if (f.balance <= 0.05) {
        f.falls += 1;
        f.balance = 0.48;
        f.health = Math.max(0, f.health - 5);
      }
    }
  }
  return (a.health - b.health) + (a.score - b.score) * 4 - a.falls * 8 + b.falls * 8;
}

function botActionSilent(f: Fighter, opponent: Fighter, dt: number, bot: BotGenome) {
  const oldLength = events.length;
  botAction(f, opponent, dt, bot, "blue");
  events.length = oldLength;
}

function mutate(bot: BotGenome, i: number): BotGenome {
  const n = (x: number) => Math.max(0.05, Math.min(0.98, x + (Math.random() - 0.5) * (0.18 + i * 0.002)));
  return {
    ...bot,
    name: `${bot.name.split(" ")[0]} G${bot.generation + 1}`,
    aggression: n(bot.aggression),
    guard: n(bot.guard),
    evasiveness: n(bot.evasiveness),
    stamina: n(bot.stamina),
    boundary: n(bot.boundary),
    recovery: n(bot.recovery),
    kickBias: n(bot.kickBias),
    generation: bot.generation + 1,
  };
}

function trainChampion() {
  const pool = [champion, blueBot, redBot, ...Array.from({ length: 30 }, (_, i) => mutate(champion, i))];
  const scored = pool.map((bot) => {
    const opponents = [baseBot("Pressure", 2), baseBot("Counter", 4), baseBot("Evasive", 6), blueBot];
    const score = opponents.reduce((sum, opp) => sum + simulate(bot, opp) - simulate(opp, bot), 0);
    return { bot, score };
  }).sort((a, b) => b.score - a.score);
  champion = { ...scored[0].bot, elo: Math.round(1000 + Math.max(-250, Math.min(650, scored[0].score))) };
  blueBot = champion;
  botCode.value = encodeBot(champion);
  events.unshift({ text: `Training complete: ${champion.name} elo ${champion.elo}`, ttl: 6, kind: "train" });
  updateHud();
}

function encodeBot(bot: BotGenome) {
  return btoa(JSON.stringify(bot)).replaceAll("+", "-").replaceAll("/", "_").replaceAll("=", "");
}

function decodeBot(code: string): BotGenome | null {
  try {
    const normalized = code.trim().replaceAll("-", "+").replaceAll("_", "/");
    const parsed = JSON.parse(atob(normalized));
    const keys: Array<keyof BotGenome> = ["name", "aggression", "guard", "evasiveness", "stamina", "boundary", "recovery", "kickBias", "elo", "generation"];
    if (!keys.every((key) => key in parsed)) return null;
    return parsed as BotGenome;
  } catch {
    return null;
  }
}

function world(x: number, y: number) {
  const size = Math.min(canvas.width * 0.62, canvas.height * 0.78);
  return [canvas.width * 0.49 + (x / arenaRadius) * size * 0.5, canvas.height * 0.52 + (y / arenaRadius) * size * 0.5] as const;
}

function drawTrail(points: Array<[number, number]>, color: string) {
  points.forEach((p, i) => {
    const [x, y] = world(p[0], p[1]);
    ctx.globalAlpha = i / points.length * 0.35;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(x, y, 4 + (i / points.length) * 8, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.globalAlpha = 1;
}

function drawFighter(f: Fighter, color: string, label: string) {
  const [x, y] = world(f.x, f.y);
  ctx.save();
  ctx.shadowColor = color;
  ctx.shadowBlur = f.hitFlash > 0 ? 34 : 16;
  ctx.fillStyle = f.hitFlash > 0 ? "#fff4d8" : color;
  ctx.beginPath();
  ctx.arc(x, y, 25, 0, Math.PI * 2);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.strokeStyle = "rgba(255,255,255,.9)";
  ctx.lineWidth = 3;
  ctx.stroke();
  const hx = x + Math.cos(f.theta) * 42;
  const hy = y + Math.sin(f.theta) * 42;
  ctx.strokeStyle = "#f8fbff";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(hx, hy);
  ctx.stroke();
  if (f.guard > 0.35) {
    ctx.strokeStyle = "#64e092";
    ctx.lineWidth = 7;
    ctx.beginPath();
    ctx.arc(x, y, 39, -1.1, 1.1);
    ctx.stroke();
  }
  if (f.safetyOverride > 0) {
    ctx.fillStyle = "#ffcf65";
    ctx.fillRect(x - 34, y - 58, 68, 22);
    ctx.fillStyle = "#241a05";
    ctx.font = "700 12px Inter, system-ui";
    ctx.fillText("SAFE", x - 17, y - 43);
  }
  ctx.fillStyle = "#081018";
  ctx.font = "800 13px Inter, system-ui";
  ctx.fillText(label, x - 5, y + 5);
  ctx.restore();
}

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const jitterX = shake ? (Math.random() - 0.5) * shake : 0;
  const jitterY = shake ? (Math.random() - 0.5) * shake : 0;
  shake *= 0.88;
  ctx.save();
  ctx.translate(jitterX, jitterY);
  const [cx, cy] = world(0, 0);
  const radius = Math.min(canvas.width * 0.62, canvas.height * 0.78) * 0.5;
  const grd = ctx.createRadialGradient(cx, cy, radius * 0.15, cx, cy, radius * 1.05);
  grd.addColorStop(0, "#1f2b38");
  grd.addColorStop(1, "#101720");
  ctx.fillStyle = grd;
  ctx.beginPath();
  ctx.arc(cx, cy, radius + 34, 0, Math.PI * 2);
  ctx.fill();
  for (const r of [1, 0.72, 0.46]) {
    ctx.strokeStyle = r === 1 ? "#d8e1ec" : "#35485f";
    ctx.lineWidth = r === 1 ? 4 : 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, radius * r, 0, Math.PI * 2);
    ctx.stroke();
  }
  for (let a = 0; a < Math.PI * 2; a += Math.PI / 8) {
    ctx.strokeStyle = "#253345";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx + Math.cos(a) * radius * 0.45, cy + Math.sin(a) * radius * 0.45);
    ctx.lineTo(cx + Math.cos(a) * radius, cy + Math.sin(a) * radius);
    ctx.stroke();
  }
  drawTrail(trailRed, "#ef4b48");
  drawTrail(trailBlue, "#4f8cff");
  const [rx, ry] = world(red.x, red.y);
  const [bx, by] = world(blue.x, blue.y);
  ctx.setLineDash([8, 8]);
  ctx.strokeStyle = distance(red, blue) < 0.9 ? "#ffbf5b" : "rgba(214,226,242,.45)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(rx, ry);
  ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.setLineDash([]);
  drawFighter(red, "#ef4b48", "R");
  drawFighter(blue, "#4f8cff", "B");
  ctx.restore();
  updateHud();
}

function bar(label: string, value: number, color: string) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return `<label>${label}<span>${pct.toFixed(0)}%</span></label><div class="bar"><i style="width:${pct}%;background:${color}"></i></div>`;
}

function botSummary(bot: BotGenome) {
  return `
    <div><strong>${bot.name}</strong> · Elo ${bot.elo} · Gen ${bot.generation}</div>
    <div class="dnaGrid">
      <span>agg ${bot.aggression.toFixed(2)}</span><span>guard ${bot.guard.toFixed(2)}</span><span>evade ${bot.evasiveness.toFixed(2)}</span>
      <span>stam ${bot.stamina.toFixed(2)}</span><span>edge ${bot.boundary.toFixed(2)}</span><span>recover ${bot.recovery.toFixed(2)}</span>
    </div>`;
}

function updateHud() {
  const d = drivers();
  document.querySelector("#round")!.textContent = String(round);
  document.querySelector("#range")!.textContent = `${distance(red, blue).toFixed(2)}m`;
  document.querySelector("#risk")!.textContent = `${riskScore(red).toFixed(2)} / ${riskScore(blue).toFixed(2)}`;
  document.querySelector("#modeText")!.textContent = modeSelect.selectedOptions[0].textContent || "mode";
  document.querySelector("#winner")!.textContent = winner;
  document.querySelector("#redBars")!.innerHTML = `<div class="driver">${d.red === "bot" ? redBot.name : "Human P1"}</div>` + bar("Health", red.health / 100, "#ef4b48") + bar("Stamina", red.stamina, "#ffbf5b") + bar("Balance", red.balance, "#64e092") + bar("Damage", red.damage, "#ff7a6b");
  document.querySelector("#blueBars")!.innerHTML = `<div class="driver">${d.blue === "bot" ? blueBot.name : "Human P2"}</div>` + bar("Health", blue.health / 100, "#4f8cff") + bar("Stamina", blue.stamina, "#ffbf5b") + bar("Balance", blue.balance, "#64e092") + bar("Damage", blue.damage, "#ff7a6b");
  document.querySelector("#botStats")!.innerHTML = botSummary(champion);
  document.querySelector("#feed")!.innerHTML = events.map((e) => `<div class="${e.kind}">${e.text}</div>`).join("");
}

const p1Map: Record<string, keyof Controls> = { w: "up", W: "up", s: "down", S: "down", a: "left", A: "left", d: "right", D: "right", Shift: "guard", j: "jab", J: "jab", k: "cross", K: "cross", l: "kick", L: "kick", " ": "push" };
const p2Map: Record<string, keyof Controls> = { ArrowUp: "up", ArrowDown: "down", ArrowLeft: "left", ArrowRight: "right", "/": "guard", "1": "jab", "2": "cross", "3": "kick", "0": "push" };

window.addEventListener("keydown", (event) => {
  const key1 = p1Map[event.key];
  const key2 = p2Map[event.key];
  if (key1) {
    event.preventDefault();
    p1[key1] = true;
  }
  if (key2) {
    event.preventDefault();
    p2[key2] = true;
  }
});
window.addEventListener("keyup", (event) => {
  const key1 = p1Map[event.key];
  const key2 = p2Map[event.key];
  if (key1) {
    event.preventDefault();
    p1[key1] = false;
  }
  if (key2) {
    event.preventDefault();
    p2[key2] = false;
  }
});

startButton.addEventListener("click", () => {
  if (winner !== "active") {
    winner = "active";
    resetRound();
  }
  running = !running;
  startButton.textContent = running ? "Pause" : "Start";
});
resetButton.addEventListener("click", () => {
  running = false;
  startButton.textContent = "Start";
  resetRound(true);
});
modeSelect.addEventListener("change", () => {
  resetRound(true);
  updateHud();
});
trainButton.addEventListener("click", trainChampion);
exportButton.addEventListener("click", async () => {
  const code = encodeBot(champion);
  botCode.value = code;
  await navigator.clipboard?.writeText(code).catch(() => undefined);
  events.unshift({ text: "Champion DNA exported. Share it as a global bot challenge.", ttl: 5, kind: "train" });
});
importButton.addEventListener("click", () => {
  const imported = decodeBot(botCode.value);
  if (!imported) {
    events.unshift({ text: "Import failed: invalid bot DNA.", ttl: 4, kind: "safety" });
    return;
  }
  blueBot = imported;
  champion = imported;
  modeSelect.value = "humanBot";
  resetRound(true);
  events.unshift({ text: `Imported ${imported.name}. Fight it worldwide by sharing DNA.`, ttl: 5, kind: "train" });
});

botCode.value = encodeBot(champion);
draw();
requestAnimationFrame(step);
