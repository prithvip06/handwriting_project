const canvas = document.getElementById("drawing-canvas");
const ctx = canvas.getContext("2d");
const brushSize = document.getElementById("brush-size");
const resultEl = document.getElementById("result");

let drawing = false;
let lastPos = null;
let color = "#000000";

function clearCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}
clearCanvas();

function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * (canvas.width / rect.width),
    y: (e.clientY - rect.top) * (canvas.height / rect.height),
  };
}

canvas.addEventListener("pointerdown", (e) => {
  drawing = true;
  lastPos = getPos(e);
  canvas.setPointerCapture(e.pointerId);
});

canvas.addEventListener("pointermove", (e) => {
  if (!drawing) return;
  const pos = getPos(e);
  ctx.strokeStyle = color;
  ctx.lineWidth = brushSize.value;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.beginPath();
  ctx.moveTo(lastPos.x, lastPos.y);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  lastPos = pos;
});

function stopDrawing() {
  drawing = false;
  lastPos = null;
}
canvas.addEventListener("pointerup", stopDrawing);
canvas.addEventListener("pointercancel", stopDrawing);
canvas.addEventListener("pointerleave", stopDrawing);

document.querySelectorAll(".color").forEach((swatch) => {
  swatch.addEventListener("click", () => {
    document.querySelectorAll(".color").forEach((s) => s.classList.remove("selected"));
    swatch.classList.add("selected");
    color = swatch.dataset.color;
  });
});

document.getElementById("clear-btn").addEventListener("click", () => {
  clearCanvas();
  resultEl.innerHTML = "";
});

document.getElementById("predict-btn").addEventListener("click", async () => {
  resultEl.textContent = "Thinking…";
  const image = canvas.toDataURL("image/png");
  const res = await fetch("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image }),
  });
  const data = await res.json();
  if (!data.letter) {
    resultEl.textContent = "Draw a letter first!";
    return;
  }
  resultEl.innerHTML = `<span class="letter">${data.letter}</span><br>${data.confidence.toFixed(1)}% confidence`;
});
