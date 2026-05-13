// Receives postMessage from parent frame; updates the in-iframe scatterplot.
(function () {
  const dpr = window.devicePixelRatio || 1;

  // Create a transparent canvas above the scatterplot for path overlay.
  const canvas = document.createElement("canvas");
  canvas.style.cssText = "position:fixed;top:0;left:0;width:100vw;height:100vh;z-index:5;pointer-events:none";
  document.body.appendChild(canvas);

  function resize() {
    canvas.width = window.innerWidth * dpr;
    canvas.height = window.innerHeight * dpr;
  }
  resize();
  window.addEventListener("resize", resize);

  let pathNodes = null;

  function getSP() { return window._tmap_scatterplot || null; }

  function draw() {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!pathNodes || pathNodes.length < 2) return;
    const sp = getSP();
    if (!sp) return;
    const view = sp.get("view");
    if (!view) return;
    const W = canvas.width, H = canvas.height;
    const toScr = (nx, ny) => [
      ((nx - view[0]) / (view[2] - view[0])) * W,
      (1 - (ny - view[1]) / (view[3] - view[1])) * H,
    ];
    // 1. Draw the line connecting all nodes.
    ctx.strokeStyle = "rgba(255, 51, 51, 0.85)";
    ctx.lineWidth = 3 * dpr;
    ctx.lineJoin = "round";
    ctx.beginPath();
    const [sx, sy] = toScr(pathNodes[0].nx, pathNodes[0].ny);
    ctx.moveTo(sx, sy);
    for (let i = 1; i < pathNodes.length; i++) {
      const [px, py] = toScr(pathNodes[i].nx, pathNodes[i].ny);
      ctx.lineTo(px, py);
    }
    ctx.stroke();
    // 2. Draw a red ball at every intermediate node.
    for (let i = 1; i < pathNodes.length - 1; i++) {
      const [px, py] = toScr(pathNodes[i].nx, pathNodes[i].ny);
      ctx.beginPath();
      ctx.arc(px, py, 5 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = "#ff3333";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.9)";
      ctx.lineWidth = 1.5 * dpr;
      ctx.stroke();
    }
    // 3. Draw larger highlighted endpoints.
    for (const node of [pathNodes[0], pathNodes[pathNodes.length - 1]]) {
      const [px, py] = toScr(node.nx, node.ny);
      ctx.beginPath();
      ctx.arc(px, py, 9 * dpr, 0, Math.PI * 2);
      ctx.fillStyle = "#ff0033";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2.5 * dpr;
      ctx.stroke();
    }
  }

  // Trigger a resize when the iframe is loaded — regl-scatterplot needs to recompute its canvas.
  window.addEventListener("load", () => {
    setTimeout(() => window.dispatchEvent(new Event("resize")), 100);
  });

  function attach() {
    const sp = getSP();
    if (!sp) { setTimeout(attach, 300); return; }
    sp.set({ opacityInactiveScale: 0.9, pointSizeSelected: 12 });
    sp.subscribe("view", () => requestAnimationFrame(draw));
  }
  setTimeout(attach, 1500);

  window.addEventListener("message", (ev) => {
    const msg = ev.data;
    if (!msg || !msg.type) return;
    const sp = getSP();
    if (msg.type === "select" && sp) sp.select(msg.indices || []);
    if (msg.type === "draw-path") { pathNodes = msg.nodes; draw(); }
    if (msg.type === "add-marker") {
      if (sp && typeof msg.idx === "number") sp.select([msg.idx]);
    }
    if (msg.type === "clear") {
      pathNodes = null;
      draw();
      if (sp) sp.select([]);
    }
  });
})();
