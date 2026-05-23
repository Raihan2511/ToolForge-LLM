# ToolForge-LLM: End-to-End RunPod & LangGraph Integration

This document outlines the entire architecture and the exact step-by-step commands we executed to take your fine-tuned `Qwen2.5-7B` adapter, serve it on a RunPod GPU, and seamlessly connect it to your local `IHCL-POC` LangGraph agents.

---

## 1. The Architecture (How it all connects)

When you use OpenAI, the `gpt-4` servers automatically parse the AI's response and package it into a neat `tool_calls` object for LangChain. 
However, an open-source model served via **vLLM** just outputs raw text (in your case, a raw JSON string like `{"tool_calls": [...]}`). 

Here is how we bridged that gap:
1. **The Server (RunPod)**: Runs vLLM. It takes your custom `template.jinja` file, which forces the model to read your `[BEGIN OF AVAILABLE TOOLS]` prompt and output raw JSON.
2. **The Tunnel (Cloudflare)**: Punches a secure hole through RunPod's firewall so your laptop can talk to the server.
3. **The Parser (`vllm_chat.py`)**: Sits inside your local `IHCL-POC` code. It receives the raw JSON text from RunPod, extracts the `tool_calls`, and converts them into standard LangChain `ToolCall` objects. LangGraph sees these objects and routes them exactly as if they came from OpenAI!

---

## 2. Server Setup & Merging on RunPod

Because uploading a 15GB model from a laptop is too slow, we used RunPod to download the base model and merge it with your tiny adapter.

### Fixing the Directory Structure
Your `setup.sh` hardcoded paths to `/workspace/training`, so we copied all the files from your cloned repo into that folder so the scripts wouldn't break:
```bash
cp -r /workspace/ToolForge-LLM/training/* /workspace/training/
cd /workspace/training
```

### Installing Dependencies
We had to install the specific libraries required for merging and serving:
```bash
# Installs Unsloth, Transformers, PEFT, etc.
bash setup.sh

# Installs the vLLM server
pip install vllm
```

### Executing the Merge
With the adapter files uploaded to `./runs`, we pointed the merge script directly to them. This downloaded the 15GB base Qwen model at 130MB/s and merged it with your adapter:
```bash
python scripts/merge_and_save.py \
    --adapter_path ./runs \
    --output_dir ./merged
```

---

## 3. Starting the vLLM Server

Newer versions of vLLM require explicit flags if you are sending "tools" in your API request. Instead of using the old `serve_vllm.sh`, we ran this exact command to start the server:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model "/workspace/training/merged" \
    --served-model-name qwen-tools \
    --chat-template "/workspace/training/template.jinja" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser xlam
```
*Note: We used the `xlam` parser flag to satisfy vLLM's strict requirements, while relying on our custom Jinja template to do the actual structural formatting.*

---

## 4. Networking: The Cloudflare Tunnel

RunPod blocked port `8000` because it wasn't exposed when the Pod was created. To fix this instantly without losing our GPU state, we opened a second Terminal tab and created a free Cloudflare tunnel:

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
./cloudflared-linux-amd64 tunnel --url http://127.0.0.1:8000
```
This generated a public URL (e.g., `https://jose-alto-need-infrared.trycloudflare.com`).

---

## 5. Local Integration & Testing

### The CLI Test Client
To prove the model worked before touching complex agent code, we updated `C:\Users\raiha\OneDrive\Desktop\Msc_proj\ToolForge LLM\training\scripts\chat_client.py`:
```python
VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "https://jose-alto-need-infrared.trycloudflare.com/v1")
```
Running this file locally allowed us to interactively test tool-calling (e.g., asking for the weather) and see the JSON payload returned instantly.

### The LangGraph Integration (`IHCL-POC`)
Finally, to plug this into your real agents, you simply update the agent files (like `faq.py`) to use your new URL:

```python
from bot.models.vllm_chat import create_vllm_model as ChatOpenAI

llm = ChatOpenAI(
    model="qwen-tools",
    base_url="https://jose-alto-need-infrared.trycloudflare.com/v1",
    api_key="empty",
    temperature=0
)
```

**How `vllm_chat.py` works under the hood:**
1. Your LangGraph agent calls `llm.invoke()`.
2. `vllm_chat.py` intercepts the call, formats your Python tools into JSON schemas, and sends them to the RunPod URL.
3. RunPod (vLLM) generates a raw string like `{"tool_calls": [{"name": "get_faq", "arguments": {"query": "..."}}]}`.
4. `vllm_chat.py` parses that string, maps it to a LangChain `AIMessage(tool_calls=[...])`, and returns it to LangGraph.
5. LangGraph routes to the correct tool node seamlessly!
