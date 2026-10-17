# Pick it Easy

## Minimal robot-arm trainer

This demonstration uses NVIDIA Nemotron 3.5 Super VL as a **teacher** for a
two-joint simulated arm. It collects the teacher's joint commands and fits a
tiny local linear policy by imitation learning. It does not fine-tune the
120B-parameter Nemotron model.

## Run it

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

First run the entire pipeline offline:

```powershell
python train.py --mock
```

To call Nemotron through Crusoe, create an **Intelligence API key** in the
Crusoe Cloud console and set it in the environment. Point `CRUSOE_BASE_URL` at
the OpenAI-compatible URL for your Crusoe-hosted Super VL deployment:

```powershell
$env:CRUSOE_API_KEY = "your-key"
$env:CRUSOE_BASE_URL = "https://your-deployment.example/v1"
python train.py
```

The default run makes at most 36 API calls (6 episodes x 6 steps) and writes
the learned weights to `policy.npz`. Use fewer calls for a quick smoke test:

```powershell
python train.py --episodes 1 --steps 2
```

As of October 17, 2026, Crusoe's documented shared serverless catalog includes
Nemotron 3.5 Lightning but does not list the newly released Super VL. Use a
Crusoe self-serve/NIM deployment for the exact model until it appears in the
shared catalog. Confirm the models exposed by whichever endpoint you use:

```powershell
$headers = @{ Authorization = "Bearer $env:CRUSOE_API_KEY" }
(Invoke-RestMethod "$env:CRUSOE_BASE_URL/models" -Headers $headers).data.id
```

If the ID differs, pass it with `--model MODEL_ID` or set `CRUSOE_MODEL`.

## What the pipeline does

1. Randomizes the arm and a reachable target.
2. Sends the joint angles, end-effector position, and target to Nemotron.
3. Applies and records Nemotron's two joint-angle deltas.
4. Fits a seven-input, two-output linear policy with least squares. The inputs
   include simple sine/cosine geometry features for the two-link arm.
5. Evaluates the learned policy on fresh targets and saves it.

This is intentionally small and educational. A real robotics pipeline would
add collision checks, actuator limits, image observations, a stronger policy,
many more trajectories, and hardware safety controls.
