"""
Training Space web server — shows live training status at port 7860.
Launches train.py in a background thread on startup.
"""
import threading
import train as trainer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

app = FastAPI()

HTML = """<!DOCTYPE html>
<html>
<head>
  <title>Dead Internet Detective — Training</title>
  <meta http-equiv="refresh" content="10">
  <style>
    body {{ font-family: monospace; background: #111; color: #eee; padding: 2rem; }}
    h1 {{ color: #4af; }}
    .status {{ font-size: 1.4rem; margin: 1rem 0; }}
    .bar {{ background: #333; border-radius: 4px; height: 24px; width: 100%; max-width: 600px; }}
    .fill {{ background: #4af; height: 24px; border-radius: 4px; transition: width 0.5s; }}
    pre {{ background: #1a1a1a; padding: 1rem; max-height: 400px; overflow-y: auto;
           border-radius: 4px; font-size: 0.85rem; }}
  </style>
</head>
<body>
  <h1>Dead Internet Detective — GRPO Training (A10G)</h1>
  <div class="status">Phase: <b>{phase}</b></div>
  <div class="status">Step: {step} / {total} &nbsp;|&nbsp; Last reward: {reward:.3f}</div>
  <div class="bar"><div class="fill" style="width:{pct:.1f}%"></div></div>
  <br>
  <pre>{logs}</pre>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    s = trainer.STATUS
    pct = (s["step"] / max(s["total"], 1)) * 100
    logs = "\n".join(s["log"][-50:])
    return HTML.format(
        phase=s["phase"],
        step=s["step"],
        total=s["total"],
        reward=s["mean_reward"],
        pct=pct,
        logs=logs,
    )


@app.get("/status")
def status():
    return JSONResponse(trainer.STATUS)


@app.get("/health")
def health():
    return {"status": "ok"}


def _start_training():
    try:
        trainer.run_training()
    except Exception as e:
        trainer._log(f"TRAINING ERROR: {e}")
        trainer.STATUS["phase"] = f"error: {e}"


if __name__ == "__main__":
    t = threading.Thread(target=_start_training, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=7860)
