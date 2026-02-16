# AWS_FAST_SETUP.md
## Goal
Run a **FAST inference endpoint** on an AWS GPU EC2 VM and call it from your notebook/router using:

- `FAST_URL` (POST): `http://<PUBLIC_IP>:8000/generate`
- `FAST_HEALTH_URL` (GET): `http://<PUBLIC_IP>:8000/health`

This guide sets up:
1) GPU EC2 VM (Ubuntu)
2) NVIDIA driver + Docker + NVIDIA Container Toolkit
3) A simple FastAPI inference server (HTTP) inside the VM
4) Security group rules
5) Colab/Notebook secrets setup (FAST_URL, FAST_HEALTH_URL)

---

## 1) Launch the AWS EC2 GPU VM

### Recommended instance types
- **Budget / demo:** `g4dn.xlarge` (T4 GPU)
- **Better:** `g5.xlarge` (A10G GPU)

### AMI
- **Ubuntu 22.04 LTS** (recommended)

### Storage
- 80–120 GB gp3 is usually enough.

### Security Group (Inbound Rules)
Create/attach a Security Group with:

| Type | Protocol | Port | Source |
|------|----------|------|--------|
| SSH  | TCP      | 22   | **Your IP** (recommended) |
| Custom TCP | TCP | 8000 | **Your IP** (recommended) |

> For quick testing only you can set port 8000 source to `0.0.0.0/0`,
> but lock it down afterward.

### Get the Public IP
After the instance is running, note:
- `Public IPv4 address` (example: `34.213.111.92`)

---

## 2) SSH into the instance

From your local machine:

```bash
chmod 400 your-key.pem
ssh -i your-key.pem ubuntu@<PUBLIC_IP>
Example:

ssh -i your-key.pem ubuntu@34.213.111.92
3) Install NVIDIA driver (Ubuntu) + verify GPU
Update packages
sudo apt update && sudo apt upgrade -y
Install recommended driver (common choice)
sudo apt install -y nvidia-driver-535
sudo reboot
Reconnect after reboot, then verify:

nvidia-smi
You should see the GPU (T4/A10G etc.).

4) Install Docker
sudo apt install -y docker.io
sudo systemctl enable docker
sudo systemctl start docker
sudo usermod -aG docker ubuntu
Log out and SSH back in so group changes apply.

Verify:

docker ps
5) Install NVIDIA Container Toolkit (nvidia-docker)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker.gpg

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-docker.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo systemctl restart docker
Test GPU inside Docker:

docker run --rm --gpus all nvidia/cuda:12.2.0-base nvidia-smi
6) Run a FAST inference server (FastAPI) on port 8000
This is a minimal HTTP server for routing demos. It exposes:

GET /health

POST /generate

Create a working directory
mkdir -p ~/fast_server && cd ~/fast_server
Create server.py
cat > server.py << 'EOF'
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

app = FastAPI()

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

class GenerateReq(BaseModel):
    prompt: str
    max_new_tokens: int = 120
    temperature: float = 0.2

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate")
def generate(req: GenerateReq):
    prompt = req.prompt
    max_new_tokens = req.max_new_tokens
    temperature = req.temperature

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    t0 = time.perf_counter()
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=(temperature > 0),
        temperature=temperature if temperature > 0 else 1.0,
    )
    t1 = time.perf_counter()

    # Decode full output (prompt+gen). Router can still compute prompt tokens separately.
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    gen_tokens = int(out.shape[1] - inputs["input_ids"].shape[1])
    total_latency_s = t1 - t0

    return {
        "text": text,
        # TTFT not measured in this minimal server; leave None.
        # If you implement streaming, you can compute real TTFT.
        "ttft_s": None,
        "gen_tokens": gen_tokens,
        "total_s": total_latency_s,
    }
EOF
Install dependencies
python3 -m pip install --upgrade pip
pip install fastapi uvicorn "transformers>=4.45" torch --extra-index-url https://download.pytorch.org/whl/cu121
If you already have torch installed via AMI, you can skip reinstalling torch.

Run the server
uvicorn server:app --host 0.0.0.0 --port 8000
Leave this running in the terminal. (Use tmux for persistence.)

7) Test from your local machine (or another terminal)
Replace <PUBLIC_IP>:

Health:

curl http://<PUBLIC_IP>:8000/health
Generate:

curl -X POST http://<PUBLIC_IP>:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Say hello in one sentence.","max_new_tokens":32,"temperature":0.2}'
If both work, the VM endpoint is reachable.

8) Set FAST_URL + FAST_HEALTH_URL in your notebook (Colab Secrets)
In Colab:

Left sidebar → 🔐 Secrets

Add:

FAST_URL         = http://<PUBLIC_IP>:8000/generate
FAST_HEALTH_URL  = http://<PUBLIC_IP>:8000/health
Then in your notebook:

from google.colab import userdata
FAST_URL = userdata.get("FAST_URL")
FAST_HEALTH_URL = userdata.get("FAST_HEALTH_URL")
print("FAST_URL loaded:", bool(FAST_URL))
print("FAST_HEALTH_URL loaded:", bool(FAST_HEALTH_URL))
9) Common issues + fixes
A) Colab cannot reach localhost
If FAST_URL is http://localhost:8000/..., it will NOT work from Colab.
Use the EC2 public IP.

B) Security group blocked
If health check times out:

Ensure inbound rule exists for TCP 8000

Restrict to your IP, or allow 0.0.0.0/0 temporarily to debug.

C) Server running but not listening publicly
Make sure uvicorn uses:

--host 0.0.0.0 (not 127.0.0.1)

D) Keep server alive after SSH disconnect
Use tmux:

sudo apt install -y tmux
tmux new -s fast
uvicorn server:app --host 0.0.0.0 --port 8000
# detach: Ctrl+b then d
# reattach: tmux attach -t fast
10) Production hardening (recommended later)
Lock port 8000 source to your IP (not world)

Put NGINX in front for rate limiting

Add HTTPS via a load balancer / cert

Add structured logs + metrics (CloudWatch/Prometheus)

Use Triton/TensorRT-LLM for true “FAST” acceleration

Quick reference
FAST_HEALTH_URL: http://<PUBLIC_IP>:8000/health (GET)

FAST_URL: http://<PUBLIC_IP>:8000/generate (POST)
