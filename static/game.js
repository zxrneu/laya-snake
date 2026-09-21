// Laya Neural Snake Game Engine & Telemetry Dashboard

class SnakeGame {
  constructor() {
    this.canvas = document.getElementById('gameCanvas');
    this.ctx = this.canvas.getContext('2d');

    // UI elements
    this.scoreDisplay = document.getElementById('scoreDisplay');
    this.lengthDisplay = document.getElementById('lengthDisplay');
    this.highScoreDisplay = document.getElementById('highScoreDisplay');
    this.stepsDisplay = document.getElementById('stepsDisplay');
    this.latencyDisplay = document.getElementById('latencyDisplay');
    this.gameOverlay = document.getElementById('gameOverlay');
    this.choiceBadge = document.getElementById('choiceBadge');
    this.confidenceFill = document.getElementById('confidenceFill');
    this.confidenceVal = document.getElementById('confidenceVal');
    this.situationText = document.getElementById('situationText');
    this.criteriaList = document.getElementById('criteriaList');
    this.jsonInspector = document.getElementById('jsonInspector');
    this.logStream = document.getElementById('logStream');
    this.toggleAiBtn = document.getElementById('toggleAiBtn');
    this.aiBtnText = document.getElementById('aiBtnText');
    this.stepAiBtn = document.getElementById('stepAiBtn');
    this.restartBtn = document.getElementById('restartBtn');
    this.overlayRestartBtn = document.getElementById('overlayRestartBtn');
    this.modeAiBtn = document.getElementById('modeAiBtn');
    this.modeManualBtn = document.getElementById('modeManualBtn');
    this.speedSlider = document.getElementById('speedSlider');
    this.speedValText = document.getElementById('speedValText');
    this.gridSizeSelect = document.getElementById('gridSizeSelect');
    this.toggleInspectorBtn = document.getElementById('toggleInspectorBtn');
    this.clearLogBtn = document.getElementById('clearLogBtn');
    this.connectionStatus = document.getElementById('connectionStatus');
    this.connectionText = document.getElementById('connectionText');

    // Config & Game State
    this.gridSize = parseInt(this.gridSizeSelect.value, 10);
    this.stepInterval = parseInt(this.speedSlider.value, 10);
    this.mode = 'ai'; // 'ai' or 'manual'
    this.isAiRunning = false;
    this.isGameOver = false;
    this.isWaitingForModel = false;

    this.snake = [];
    this.food = [0, 0];
    this.direction = 'RIGHT';
    this.nextDirection = 'RIGHT';
    this.score = 0;
    this.steps = 0;
    this.highScore = parseInt(localStorage.getItem('laya_snake_highscore') || '0', 10);
    this.highScoreDisplay.textContent = this.highScore;

    // Telemetry cache
    this.currentProbabilities = { UP: 0.25, DOWN: 0.25, LEFT: 0.25, RIGHT: 0.25 };
    this.currentChoice = 'RIGHT';
    this.lastInferenceTime = 0;
    this.lastDecisionData = null;

    // WebSocket
    this.ws = null;
    this.wsConnected = false;
    this.stepId = 0;
    this.pendingCallbacks = new Map();

    // Visual effect arrays
    this.particles = [];
    this.floatTexts = [];

    this.initCanvasDPI();
    this.initWebSocket();
    this.bindEvents();
    this.resetGame();

    // Start render loop
    this.lastFrameTime = performance.now();
    this.renderLoop = this.renderLoop.bind(this);
    requestAnimationFrame(this.renderLoop);
  }

  initCanvasDPI() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = 560 * dpr;
    this.canvas.height = 560 * dpr;
    this.ctx.scale(dpr, dpr);
    this.displayWidth = 560;
    this.displayHeight = 560;
  }

  initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/play`;

    try {
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = () => {
        this.wsConnected = true;
        this.connectionStatus.classList.add('active');
        this.connectionText.textContent = 'WS 实时极速连接 (Ready)';
        this.addLog('系统', 'WebSocket 引擎直连建立成功，支持毫秒级推演。');
      };

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const callback = this.pendingCallbacks.get(data.step_id);
          if (callback) {
            this.pendingCallbacks.delete(data.step_id);
            callback(data);
          }
        } catch (e) {
          console.error('WS Parse error:', e);
        }
      };

      this.ws.onclose = () => {
        this.wsConnected = false;
        this.connectionStatus.classList.remove('active');
        this.connectionText.textContent = 'WS 断开，使用 REST API 模式';
        setTimeout(() => this.initWebSocket(), 3000);
      };

      this.ws.onerror = () => {
        this.wsConnected = false;
        this.connectionText.textContent = 'REST 回退模式';
      };
    } catch (e) {
      this.wsConnected = false;
      this.connectionText.textContent = 'REST 回退模式';
    }
  }

  bindEvents() {
    // Buttons
    this.toggleAiBtn.addEventListener('click', () => this.toggleAiAutopilot());
    this.stepAiBtn.addEventListener('click', () => this.stepModelOnce());
    this.restartBtn.addEventListener('click', () => this.resetGame());
    this.overlayRestartBtn.addEventListener('click', () => this.resetGame());

    // Mode Toggle
    this.modeAiBtn.addEventListener('click', () => this.setMode('ai'));
    this.modeManualBtn.addEventListener('click', () => this.setMode('manual'));

    // Speed Slider
    this.speedSlider.addEventListener('input', (e) => {
      this.stepInterval = parseInt(e.target.value, 10);
      this.speedValText.textContent = `${this.stepInterval}ms`;
    });

    // Grid Size
    this.gridSizeSelect.addEventListener('change', (e) => {
      this.gridSize = parseInt(e.target.value, 10);
      this.resetGame();
    });

    // Inspector JSON Toggle
    this.toggleInspectorBtn.addEventListener('click', () => {
      const show = this.jsonInspector.style.display === 'none';
      this.jsonInspector.style.display = show ? 'block' : 'none';
      this.toggleInspectorBtn.textContent = show ? '收起 JSON' : '展开 JSON';
    });

    // Clear Log
    this.clearLogBtn.addEventListener('click', () => {
      this.logStream.innerHTML = '';
      this.addLog('系统', '日志流已清空。');
    });

    // Keyboard Controls for Manual Mode
    window.addEventListener('keydown', (e) => {
      const keyMap = {
        ArrowUp: 'UP', KeyW: 'UP',
        ArrowDown: 'DOWN', KeyS: 'DOWN',
        ArrowLeft: 'LEFT', KeyA: 'LEFT',
        ArrowRight: 'RIGHT', KeyD: 'RIGHT'
      };

      if (keyMap[e.code]) {
        e.preventDefault();
        const opposite = { UP: 'DOWN', DOWN: 'UP', LEFT: 'RIGHT', RIGHT: 'LEFT' };
        if (keyMap[e.code] !== opposite[this.direction]) {
          this.nextDirection = keyMap[e.code];
          if (this.mode === 'manual' && !this.isGameOver) {
            this.executeStep(this.nextDirection);
          }
        }
      } else if (e.code === 'Space') {
        e.preventDefault();
        if (this.mode === 'ai') {
          this.toggleAiAutopilot();
        } else {
          this.stepModelOnce();
        }
      }
    });

    window.addEventListener('resize', () => this.initCanvasDPI());
  }

  setMode(mode) {
    this.mode = mode;
    if (mode === 'ai') {
      this.modeAiBtn.classList.add('active');
      this.modeManualBtn.classList.remove('active');
      this.addLog('模式', '切换为 Laya AI 全自动托管驾驶。');
    } else {
      this.pauseAi();
      this.modeManualBtn.classList.add('active');
      this.modeAiBtn.classList.remove('active');
      this.addLog('模式', '切换为玩家手动操控模式（WASD 或 方向键）。');
    }
  }

  resetGame() {
    this.pauseAi();
    this.isGameOver = false;
    this.isWaitingForModel = false;
    this.score = 0;
    this.steps = 0;
    this.direction = 'RIGHT';
    this.nextDirection = 'RIGHT';

    const mid = Math.floor(this.gridSize / 2);
    this.snake = [
      [mid, mid],
      [mid - 1, mid],
      [mid - 2, mid]
    ];

    this.spawnFood();
    this.updateStats();
    this.gameOverlay.classList.remove('show');
    this.addLog('游戏', `新对局开始。网格: ${this.gridSize}x${this.gridSize}，初始长度: 3。`);

    // Fetch initial model evaluation for current state
    this.queryModel(this.getGameState(), (res) => {
      this.applyModelTelemetry(res);
    });
  }

  spawnFood() {
    const occupied = new Set(this.snake.map(([x, y]) => `${x},${y}`));
    const emptyCells = [];
    for (let x = 0; x < this.gridSize; x++) {
      for (let y = 0; y < this.gridSize; y++) {
        if (!occupied.has(`${x},${y}`)) {
          emptyCells.push([x, y]);
        }
      }
    }

    if (emptyCells.length === 0) {
      this.triggerWin();
      return;
    }

    const randIdx = Math.floor(Math.random() * emptyCells.length);
    this.food = emptyCells[randIdx];
  }

  getGameState() {
    return {
      head: this.snake[0],
      food: this.food,
      body: this.snake.slice(1),
      grid_size: [this.gridSize, this.gridSize],
      current_direction: this.direction
    };
  }

  toggleAiAutopilot() {
    if (this.isGameOver) {
      this.resetGame();
    }
    if (this.isAiRunning) {
      this.pauseAi();
    } else {
      this.startAi();
    }
  }

  startAi() {
    if (this.mode !== 'ai') {
      this.setMode('ai');
    }
    this.isAiRunning = true;
    this.toggleAiBtn.classList.add('running');
    this.aiBtnText.textContent = '暂停 Laya 自动驾驶';
    this.addLog('AI', 'Laya 自动推演循环已启动。');
    this.runAiLoop();
  }

  pauseAi() {
    this.isAiRunning = false;
    this.toggleAiBtn.classList.remove('running');
    this.aiBtnText.textContent = '启动 Laya 自动驾驶';
  }

  async runAiLoop() {
    if (!this.isAiRunning || this.isGameOver) return;

    if (!this.isWaitingForModel) {
      this.isWaitingForModel = true;
      const state = this.getGameState();

      this.queryModel(state, (decision) => {
        this.isWaitingForModel = false;
        if (!this.isAiRunning || this.isGameOver) return;

        this.applyModelTelemetry(decision);
        this.executeStep(decision.choice);

        // Schedule next step respecting user's interval
        setTimeout(() => this.runAiLoop(), this.stepInterval);
      });
    } else {
      setTimeout(() => this.runAiLoop(), 20);
    }
  }

  stepModelOnce() {
    if (this.isGameOver) return;
    this.pauseAi();
    const state = this.getGameState();
    this.queryModel(state, (decision) => {
      this.applyModelTelemetry(decision);
      this.executeStep(decision.choice);
    });
  }

  queryModel(state, callback) {
    const stepId = ++this.stepId;

    if (this.wsConnected && this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.pendingCallbacks.set(stepId, callback);
      this.ws.send(JSON.stringify({ ...state, step_id: stepId }));
    } else {
      // Fallback HTTP POST
      fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(state)
      })
      .then(r => r.json())
      .then(data => callback(data))
      .catch(err => {
        console.error('Model query failed:', err);
        this.isWaitingForModel = false;
      });
    }
  }

  executeStep(moveDir) {
    if (this.isGameOver) return;

    const opposite = { UP: 'DOWN', DOWN: 'UP', LEFT: 'RIGHT', RIGHT: 'LEFT' };
    if (this.snake.length > 1 && moveDir === opposite[this.direction]) {
      // Physical guard: strictly prevent neck snap
      moveDir = this.direction;
    }

    this.direction = moveDir;
    const moves = {
      UP: [0, -1],
      DOWN: [0, 1],
      LEFT: [-1, 0],
      RIGHT: [1, 0]
    };

    const delta = moves[moveDir] || [0, 0];
    const newHead = [this.snake[0][0] + delta[0], this.snake[0][1] + delta[1]];
    this.steps++;

    // Check Wall Collision
    if (
      newHead[0] < 0 || newHead[0] >= this.gridSize ||
      newHead[1] < 0 || newHead[1] >= this.gridSize
    ) {
      this.triggerGameOver('撞击墙壁边缘');
      return;
    }

    // Check Body Collision (excluding tail that moves unless eating food)
    const isEating = (newHead[0] === this.food[0] && newHead[1] === this.food[1]);
    const bodyToCheck = isEating ? this.snake : this.snake.slice(0, -1);
    const hitBody = bodyToCheck.some(([x, y]) => x === newHead[0] && y === newHead[1]);

    if (hitBody) {
      this.triggerGameOver('撞击自身蛇躯');
      return;
    }

    // Advance snake
    this.snake.unshift(newHead);

    if (isEating) {
      this.score += 10;
      if (this.score > this.highScore) {
        this.highScore = this.score;
        localStorage.setItem('laya_snake_highscore', this.highScore);
      }
      this.spawnEatParticles(newHead[0], newHead[1], this.displayWidth / this.gridSize);
      this.addLog('进食', `精准捕获食物！长度增至 ${this.snake.length}，当前得分: ${this.score}。`, 'eat');
      this.spawnFood();
    } else {
      this.snake.pop();
    }

    this.updateStats();
  }

  triggerGameOver(reason) {
    this.isGameOver = true;
    this.pauseAi();
    document.getElementById('overlayTitle').textContent = 'GAME OVER';
    document.getElementById('overlayDesc').textContent = `${reason}！生存步数: ${this.steps}，最终得分: ${this.score}。`;
    this.gameOverlay.classList.add('show');
    this.addLog('终局', `对局结束（${reason}）。总生存步数: ${this.steps}，得分: ${this.score}。`);
  }

  triggerWin() {
    this.isGameOver = true;
    this.pauseAi();
    document.getElementById('overlayTitle').textContent = 'VICTORY!';
    document.getElementById('overlayDesc').textContent = `完美填满所有网格！Laya 模型通关成功！`;
    this.gameOverlay.classList.add('show');
    this.addLog('胜利', '贪吃蛇已填满全图！');
  }

  updateStats() {
    this.scoreDisplay.textContent = this.score;
    this.lengthDisplay.textContent = this.snake.length;
    this.highScoreDisplay.textContent = this.highScore;
    this.stepsDisplay.textContent = this.steps;
  }

  applyModelTelemetry(data) {
    if (!data) return;

    this.currentChoice = data.choice;
    this.currentProbabilities = data.probabilities;
    this.lastInferenceTime = data.inference_ms;
    this.lastDecisionData = data;

    // Latency
    this.latencyDisplay.textContent = `${data.inference_ms} ms`;

    // Choice Badge
    this.choiceBadge.textContent = `CHOSEN: ${data.choice}`;
    this.choiceBadge.classList.add('chosen');

    // Confidence
    const conf = (data.confidence || 0);
    this.confidenceFill.style.width = `${Math.min(100, Math.max(0, conf * 100))}%`;
    this.confidenceVal.textContent = conf.toFixed(3);

    // Probability Bars
    const dirs = ['UP', 'DOWN', 'LEFT', 'RIGHT'];
    dirs.forEach(d => {
      const prob = data.probabilities[d] || 0;
      const pct = (prob * 100).toFixed(1);
      const row = document.getElementById(`probRow-${d}`);
      const fill = document.getElementById(`fill-${d}`);
      const pctText = document.getElementById(`pct-${d}`);
      const statusText = document.getElementById(`status-${d}`);

      fill.style.width = `${pct}%`;
      pctText.textContent = `${pct}%`;

      // Status indicator
      const analysis = data.analysis?.[d];
      if (analysis) {
        if (analysis.reason === 'Physically Forbidden (Reverse)') {
          statusText.className = 'dir-status';
          statusText.style.opacity = '0.5';
          statusText.textContent = 'LOCKED (来时路)';
        } else if (!analysis.safe) {
          statusText.className = 'dir-status danger';
          statusText.style.opacity = '1';
          statusText.textContent = analysis.reason || 'DANGER';
        } else if (analysis.is_food) {
          if (analysis.can_reach_tail) {
            statusText.className = 'dir-status best';
            statusText.textContent = 'EAT (SAFE)';
          } else {
            statusText.className = 'dir-status danger';
            statusText.textContent = 'EAT (TRAP)';
          }
        } else if (analysis.can_reach_tail) {
          if (analysis.food_path_len !== null) {
            statusText.className = 'dir-status best';
            statusText.textContent = `CLOSER (${analysis.food_path_len}s)`;
          } else {
            statusText.className = 'dir-status safe';
            statusText.textContent = 'CHASE TAIL';
          }
        } else {
          statusText.className = 'dir-status danger';
          statusText.textContent = 'POCKET TRAP';
        }
      }

      // Active choice highlight
      if (d === data.choice) {
        row.classList.add('active-choice');
      } else {
        row.classList.remove('active-choice');
      }
    });

    // Inspector: Situation & Criteria
    const sit = data.state_sent?.snake_game?.situation || '空间局势解析正常';
    this.situationText.textContent = `[态势描述]: ${sit}`;

    if (data.criteria_sent) {
      let html = '';
      for (const [d, desc] of Object.entries(data.criteria_sent)) {
        html += `
          <div class="criteria-item">
            <span class="criteria-dir">${d}</span>
            <span class="criteria-desc">${desc}</span>
          </div>
        `;
      }
      this.criteriaList.innerHTML = html;
    }

    // JSON Inspector
    this.jsonInspector.textContent = JSON.stringify({
      step: this.steps,
      choice: data.choice,
      probabilities: data.probabilities,
      confidence: data.confidence,
      inference_ms: data.inference_ms,
      state_sent: data.state_sent
    }, null, 2);

    // Add to stream log
    const probPct = ((data.probabilities[data.choice] || 0) * 100).toFixed(1);
    this.addLog(data.choice, `置信度 ${conf.toFixed(2)} | 概率 ${probPct}% | 耗时 ${data.inference_ms}ms`);
  }

  addLog(action, text, type = 'normal') {
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;

    const now = new Date();
    const timeStr = `${String(now.getMinutes()).padStart(2, '0')}:${String(now.getSeconds()).padStart(2, '0')}.${String(Math.floor(now.getMilliseconds() / 100))}`;

    entry.innerHTML = `
      <span class="time">[${timeStr}]</span>
      <span class="action">${action}</span>
      <span class="text">${text}</span>
    `;

    this.logStream.appendChild(entry);
    this.logStream.scrollTop = this.logStream.scrollHeight;

    // Limit log size to 100 items
    if (this.logStream.children.length > 100) {
      this.logStream.removeChild(this.logStream.children[0]);
    }
  }

  renderLoop(time) {
    this.render();
    requestAnimationFrame(this.renderLoop);
  }

  spawnEatParticles(x, y, cellSize) {
    const cx = x * cellSize + cellSize / 2;
    const cy = y * cellSize + cellSize / 2;
    const colors = ['#00f2fe', '#4facfe', '#10b981', '#f59e0b', '#ec4899', '#ffffff'];
    for (let i = 0; i < 18; i++) {
      const angle = Math.random() * Math.PI * 2;
      const speed = 1.2 + Math.random() * 3.5;
      this.particles.push({
        x: cx,
        y: cy,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        radius: 1.5 + Math.random() * 2.5,
        color: colors[Math.floor(Math.random() * colors.length)],
        alpha: 1.0,
        decay: 0.02 + Math.random() * 0.025
      });
    }
    // Floating score
    this.floatTexts.push({
      x: cx,
      y: cy - 10,
      text: '+10',
      alpha: 1.0,
      color: '#10b981'
    });
  }

  updateEffects() {
    // Update particles
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx;
      p.y += p.vy;
      p.vx *= 0.95;
      p.vy *= 0.95;
      p.alpha -= p.decay;
      if (p.alpha <= 0) {
        this.particles.splice(i, 1);
      }
    }
    // Update floating texts
    for (let i = this.floatTexts.length - 1; i >= 0; i--) {
      const ft = this.floatTexts[i];
      ft.y -= 0.7;
      ft.alpha -= 0.025;
      if (ft.alpha <= 0) {
        this.floatTexts.splice(i, 1);
      }
    }
  }

  render() {
    this.updateEffects();
    const ctx = this.ctx;
    const w = this.displayWidth;
    const h = this.displayHeight;
    const cellSize = w / this.gridSize;
    const now = performance.now();

    // Clear background
    ctx.fillStyle = '#060910';
    ctx.fillRect(0, 0, w, h);

    // Subtle tactical cyber grid
    ctx.strokeStyle = 'rgba(0, 242, 254, 0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= this.gridSize; i++) {
      const pos = i * cellSize;
      ctx.beginPath();
      ctx.moveTo(pos, 0);
      ctx.lineTo(pos, h);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(0, pos);
      ctx.lineTo(w, pos);
      ctx.stroke();
    }

    // Grid intersection dots
    ctx.fillStyle = 'rgba(255, 255, 255, 0.08)';
    for (let x = 0; x <= this.gridSize; x += 2) {
      for (let y = 0; y <= this.gridSize; y += 2) {
        ctx.fillRect(x * cellSize - 1, y * cellSize - 1, 2, 2);
      }
    }

    // Draw Food (Pulsing neon fruit with orbiting particles)
    const [fx, fy] = this.food;
    const foodCenterX = fx * cellSize + cellSize / 2;
    const foodCenterY = fy * cellSize + cellSize / 2;
    const pulse = Math.sin(now / 150) * (cellSize * 0.06);
    const foodRadius = cellSize * 0.38 + pulse;

    // Food outer glowing aura
    const grad = ctx.createRadialGradient(
      foodCenterX, foodCenterY, 2,
      foodCenterX, foodCenterY, foodRadius * 2.5
    );
    grad.addColorStop(0, 'rgba(255, 65, 108, 0.85)');
    grad.addColorStop(0.4, 'rgba(255, 107, 107, 0.3)');
    grad.addColorStop(1, 'rgba(255, 75, 43, 0)');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.arc(foodCenterX, foodCenterY, foodRadius * 2.5, 0, Math.PI * 2);
    ctx.fill();

    // Orbiting mini stars
    for (let i = 0; i < 3; i++) {
      const orbitAngle = (now / 400) + (i * (Math.PI * 2 / 3));
      const orbitDist = foodRadius * 1.35;
      const ox = foodCenterX + Math.cos(orbitAngle) * orbitDist;
      const oy = foodCenterY + Math.sin(orbitAngle) * orbitDist;
      ctx.fillStyle = 'rgba(255, 215, 0, 0.85)';
      ctx.beginPath();
      ctx.arc(ox, oy, 1.8, 0, Math.PI * 2);
      ctx.fill();
    }

    // Food core
    ctx.fillStyle = '#ff3366';
    ctx.beginPath();
    ctx.arc(foodCenterX, foodCenterY, foodRadius, 0, Math.PI * 2);
    ctx.fill();

    // Food glossy highlight
    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.beginPath();
    ctx.arc(foodCenterX - foodRadius * 0.32, foodCenterY - foodRadius * 0.32, foodRadius * 0.3, 0, Math.PI * 2);
    ctx.fill();

    // Draw Snake Body
    for (let i = this.snake.length - 1; i >= 1; i--) {
      const [bx, by] = this.snake[i];
      const progress = 1 - (i / this.snake.length); // 1 at neck, 0 at tail
      const cx = bx * cellSize + cellSize / 2;
      const cy = by * cellSize + cellSize / 2;
      const radius = (cellSize * 0.42) * (0.65 + 0.35 * progress);

      // Connecting joints between segments for sleek continuous worm look
      const prev = this.snake[i - 1];
      if (prev) {
        const px = prev[0] * cellSize + cellSize / 2;
        const py = prev[1] * cellSize + cellSize / 2;
        ctx.strokeStyle = `rgba(0, 242, 254, ${0.4 + 0.5 * progress})`;
        ctx.lineWidth = radius * 1.8;
        ctx.lineCap = 'round';
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(px, py);
        ctx.stroke();
      }

      // Segment core with gradient from Cyan (#00f2fe) to Violet (#9d4edd)
      const r = Math.floor(0 + 157 * (1 - progress));
      const g = Math.floor(242 - 164 * (1 - progress));
      const b = Math.floor(254 - 33 * (1 - progress));
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;

      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fill();

      // Top glossy highlight
      ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
      ctx.beginPath();
      ctx.arc(cx, cy - radius * 0.2, radius * 0.45, 0, Math.PI * 2);
      ctx.fill();
    }

    // Draw Snake Head
    if (this.snake.length > 0) {
      const [hx, hy] = this.snake[0];
      const cx = hx * cellSize + cellSize / 2;
      const cy = hy * cellSize + cellSize / 2;
      const headRadius = cellSize * 0.48;

      // Flickering tongue
      const tongueCycle = (now % 1400);
      if (tongueCycle < 250) {
        const dirVectors = {
          UP: [0, -1], DOWN: [0, 1], LEFT: [-1, 0], RIGHT: [1, 0]
        };
        const vec = dirVectors[this.direction] || [1, 0];
        const tx1 = cx + vec[0] * headRadius;
        const ty1 = cy + vec[1] * headRadius;
        const tLen = cellSize * 0.45 * Math.sin((tongueCycle / 250) * Math.PI);
        const tx2 = tx1 + vec[0] * tLen;
        const ty2 = ty1 + vec[1] * tLen;

        ctx.strokeStyle = '#ff3366';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(tx1, ty1);
        ctx.lineTo(tx2, ty2);
        // Forked tongue tip
        const perpX = -vec[1] * 3;
        const perpY = vec[0] * 3;
        ctx.lineTo(tx2 + vec[0] * 3 + perpX, ty2 + vec[1] * 3 + perpY);
        ctx.moveTo(tx2, ty2);
        ctx.lineTo(tx2 + vec[0] * 3 - perpX, ty2 + vec[1] * 3 - perpY);
        ctx.stroke();
      }

      // Head glow
      ctx.shadowColor = '#00f2fe';
      ctx.shadowBlur = 18;
      ctx.fillStyle = '#00f2fe';
      ctx.beginPath();
      ctx.arc(cx, cy, headRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0; // reset shadow

      // Dynamic eyes looking towards food
      const eyeOffsets = {
        RIGHT: [{ x: 4, y: -5 }, { x: 4, y: 5 }],
        LEFT: [{ x: -4, y: -5 }, { x: -4, y: 5 }],
        UP: [{ x: -5, y: -4 }, { x: 5, y: -4 }],
        DOWN: [{ x: -5, y: 4 }, { x: 5, y: 4 }]
      };
      const offsets = eyeOffsets[this.direction] || eyeOffsets.RIGHT;

      // Angle from head to food
      const angleToFood = Math.atan2(this.food[1] - hy, this.food[0] - hx);
      const lookDist = 1.6;
      const lookX = Math.cos(angleToFood) * lookDist;
      const lookY = Math.sin(angleToFood) * lookDist;

      offsets.forEach(off => {
        const eyeX = cx + off.x * (cellSize / 28);
        const eyeY = cy + off.y * (cellSize / 28);
        const eyeRadius = cellSize * 0.14;

        // Eye whites
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(eyeX, eyeY, eyeRadius, 0, Math.PI * 2);
        ctx.fill();

        // Eye pupil glancing at food
        ctx.fillStyle = '#050b14';
        ctx.beginPath();
        ctx.arc(eyeX + lookX, eyeY + lookY, eyeRadius * 0.55, 0, Math.PI * 2);
        ctx.fill();

        // Eye glint
        ctx.fillStyle = '#ffffff';
        ctx.beginPath();
        ctx.arc(eyeX + lookX - 1, eyeY + lookY - 1, eyeRadius * 0.2, 0, Math.PI * 2);
        ctx.fill();
      });

      // Overlay Direction Probability vectors around head!
      this.drawDirectionCompass(cx, cy, cellSize);
    }

    // Draw Particles
    for (const p of this.particles) {
      ctx.save();
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = p.color;
      ctx.shadowColor = p.color;
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    // Draw Floating Scores
    for (const ft of this.floatTexts) {
      ctx.save();
      ctx.globalAlpha = ft.alpha;
      ctx.fillStyle = ft.color;
      ctx.font = `bold ${Math.max(14, Math.floor(cellSize * 0.65))}px Outfit, sans-serif`;
      ctx.textAlign = 'center';
      ctx.shadowColor = ft.color;
      ctx.shadowBlur = 8;
      ctx.fillText(ft.text, ft.x, ft.y);
      ctx.restore();
    }
  }

  drawDirectionCompass(headX, headY, cellSize) {
    if (!this.currentProbabilities) return;
    const ctx = this.ctx;

    const vectors = {
      UP: { dx: 0, dy: -1 },
      DOWN: { dx: 0, dy: 1 },
      LEFT: { dx: -1, dy: 0 },
      RIGHT: { dx: 1, dy: 0 }
    };

    const maxLen = cellSize * 1.1;

    for (const [d, vec] of Object.entries(vectors)) {
      const prob = this.currentProbabilities[d] || 0;
      if (prob < 0.04) continue;

      const isChosen = (d === this.currentChoice);
      const len = maxLen * Math.sqrt(prob);
      const startDist = cellSize * 0.55;
      const x1 = headX + vec.dx * startDist;
      const y1 = headY + vec.dy * startDist;
      const x2 = x1 + vec.dx * len;
      const y2 = y1 + vec.dy * len;

      ctx.strokeStyle = isChosen ? '#00f2fe' : 'rgba(255, 255, 255, 0.25)';
      ctx.lineWidth = isChosen ? 3.5 : 1.5;

      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();

      // Arrow head or glowing tip
      ctx.fillStyle = isChosen ? '#10b981' : 'rgba(255, 255, 255, 0.4)';
      ctx.beginPath();
      ctx.arc(x2, y2, isChosen ? 4.5 : 2.2, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

// Initialize on DOM load
window.addEventListener('DOMContentLoaded', () => {
  window.snakeGame = new SnakeGame();
});
