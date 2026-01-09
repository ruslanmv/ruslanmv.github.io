---
title: "Why I Rewrote My AI Agent Swarm in Rust (and Built an Agent Kernel)"
excerpt: "Building Scalable AI Agent Systems with Nexus — a complete, reproducible engineering guide for running thousands of agents with kernel-level semantics."
description: "Building Scalable AI Agent Systems with Nexus"
date: 2025-03-03
categories: [Engineering, Rust, AI]
tags: [rust, tokio, actor-model, agents, systems, performance, wasm]
header:
  image: "./../assets/images/posts/2025-03-03-why-ai-agents-need-a-kernel-not-a-framework/nexus-hero.jpg"
  teaser: "./../assets/images/posts/2025-03-03-why-ai-agents-need-a-kernel-not-a-framework/nexus-hero.jpg"
  caption: "Agents aren’t just scripts; they’re long-lived concurrent processes. Treat them like threads, not functions."
---

I tried to run an agent swarm in Python. At 10 agents it was cute. At 100 it was "fine". At 1,000 it started doing what large concurrent systems do when they are held together with string: latency spikes, memory climbs, and your laptop fan starts auditioning for a jet engine.

The failure wasn't some exotic bug. It was architectural.

I kept treating agents like scripts.

But agents are not scripts.

Agents are long-lived, concurrent processes that communicate, fail, restart, and contend for resources. That is operating-systems territory. When you scale agents, you are scaling the kernel problems: mailboxes, routing, timeouts, shutdown semantics, and observability.

So I built **Nexus**: a small Rust **agent kernel** that gives agent systems the primitives they should have had from day one.

This post is a complete, reproducible guide for [Nexus](https://github.com/ruslanmv/Nexus)

---


## The problem: Python at scale

When you scale autonomous agents beyond toy demos, you run into the same recurring issues:

- **Unpredictable latency** as agent count grows
- **No enforced backpressure** (unbounded queues become unbounded RAM)
- **Ad-hoc concurrency** (coordination is DIY)
- **Poor observability** (print() is not a tracing system)
- **Fragile lifecycle** (shutdown and cancellation are afterthoughts)

The core insight that finally snapped the problem into focus:

> Agents are not "just scripts". They are concurrent processes.
> 
> If you want them to scale, you need kernel primitives.

---

## The realization: agents need a kernel

A real agent system needs enforced semantics, not suggestions.

Nexus is built around a few kernel invariants:

- **Isolation**: each agent gets its own mailbox and handler loop
- **Backpressure**: bounded mailboxes, enforced at send time
- **Timeouts**: send and handle operations are time-bounded
- **Routing**: O(1) agent-id to mailbox lookup via a concurrent map
- **Lifecycle**: explicit spawn and graceful shutdown
- **Observability**: structured logs via `tracing`

You can duct-tape these into a Python app. Many do.

But duct tape does not become a kernel just because you believe in it harder.

---

## What Nexus actually is


**Nexus** is a high-performance agent kernel written in Rust for building and running scalable AI agent systems. It treats agents as long-lived concurrent processes rather than scripts, providing kernel-level primitives such as isolated mailboxes, deterministic message routing, enforced backpressure, timeouts, and graceful lifecycle management. By grounding agent execution in the actor model and Tokio’s async runtime, Nexus delivers predictable latency, bounded resource usage, and production-grade observability.

Unlike traditional agent frameworks that focus on orchestration or prompts, Nexus focuses on **execution**. It does not prescribe how agents think or which models they use; instead, it provides the stable, low-level substrate that allows thousands of agents—written in Rust or bridged from other languages—to run reliably under load.



In the project Nexus is a Rust crate (`nexus`) plus a demo runtime binary (`nexus-demo`).

Under the hood:

- Each agent runs as a Tokio task.
- Each agent processes messages **serially** from its mailbox (ordering is preserved).
- The kernel maintains a concurrent routing table (`DashMap<Uuid, Sender<Message>>`).
- Sending and handling are protected by timeouts.
- Shutdown is coordinated via a cancellation token.

If you want to read the "metal" first, these are the important files:

- `src/kernel.rs` - spawn, route, timeouts, shutdown
- `src/agent.rs` - message schema and `Agent` trait
- `src/config.rs` - TOML config + default locations
- `src/bin/demo.rs` - the real CLI flags and demo behavior



 **Rust** is the systems programming language that provides low-level performance with strong safety guarantees, allowing agents to run as long-lived concurrent processes without risking memory corruption or data races, while **Tokio** is the asynchronous runtime that schedules and executes those agents efficiently as lightweight tasks, handling timers, message passing, and concurrency so thousands of agents can run simultaneously without one thread per agent.


---

## Environment setup

### Supported OS

- Linux or macOS (recommended)
- Windows: use WSL2

### Required tools

- Rust toolchain (via rustup)
- A C toolchain (for common crates)
- Python 3 (for baseline scripts and Python bridge example)
- Node 18+ (for JavaScript bridge example)

### Step 1: install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

rustc --version
cargo --version
```

### Step 2: install system dependencies

Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y build-essential pkg-config libssl-dev python3 python3-pip nodejs npm
```

macOS (Homebrew):

```bash
brew update
brew install openssl python node
```

### Step 3: get the code

If you are using the GitHub repo:

```bash
git clone https://github.com/ruslanmv/Nexus.git
cd Nexus
```

If you are using the zip you provided:

```bash
unzip Nexus-master.zip
cd Nexus-master
```

---

## Build and run Nexus (from source)

### Build release binaries

```bash
cargo build --release
```

This produces:

- `target/release/nexus-demo`

### Run the demo

`nexus-demo` is the actual binary defined in `Cargo.toml`.

```bash
./target/release/nexus-demo --help
```

The real flags (from `src/bin/demo.rs`) are:

- `--config <FILE>`
- `--log-level <trace|debug|info|warn|error>`
- `--log-format <compact|json|pretty>`
- `-n, --num-agents <N>`
- `-d, --duration <SECONDS>` (0 means run until Ctrl-C)
- `--generate-config <FILE>`

Run a short demo:

```bash
./target/release/nexus-demo -n 4 -d 10
```

Run indefinitely until Ctrl-C:

```bash
./target/release/nexus-demo -n 100 -d 0
```

Enable debug logging:

```bash
./target/release/nexus-demo --log-level debug -n 10 -d 10
```

---

## Install `nexus-runtime` (recommended)

The repository includes an installer script that:

- builds Nexus
- runs tests
- installs `nexus-demo` as `nexus-runtime`
- creates config + data directories

### User-local install (no sudo)

From the repo root:

```bash
chmod +x install.sh
./install.sh
```

This installs:

- binary: `$HOME/.local/bin/nexus-runtime`
- config: `$HOME/.config/nexus/config.toml`
- data:   `$HOME/.local/share/nexus/`

If `$HOME/.local/bin` is not in your PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Now run:

```bash
nexus-runtime --help
nexus-runtime -n 4 -d 10
```

### System-wide install (requires sudo)

```bash
sudo ./install.sh --system
```

Installs:

- binary: `/usr/local/bin/nexus-runtime`
- config: `/etc/nexus/config.toml`
- data:   `/var/lib/nexus/`

---

## Configuration

### Generate a config template

The demo binary can generate a config file directly:

```bash
./target/release/nexus-demo --generate-config config.toml
```

Or with the installed runtime:

```bash
nexus-runtime --generate-config config.toml
```

### Default config locations

Nexus loads configuration from these paths in order (see `src/config.rs`):

1. `./config.toml`
2. `~/.config/nexus/config.toml`
3. `/etc/nexus/config.toml`

If none exist, it uses defaults.

### Example config

```toml
[kernel]
mailbox_capacity = 1024
send_timeout_ms = 500
handle_timeout_ms = 30000

[logging]
level = "info"
format = "compact"

[runtime]
worker_threads = 0
max_blocking_threads = 512
```

Run with a specific config:

```bash
nexus-runtime --config ./config.toml -n 50 -d 30
```

---

## Writing your first Rust agent

Nexus agents implement the `Agent` trait (see `src/agent.rs`).

Important detail: in this repo, `handle_message` takes `&self` (not `&mut self`), so if you want mutable state you should use interior mutability (`AtomicUsize`, `Mutex`, etc.).

### Create an example agent

Create `examples/ping_agent.rs`:

```rust
use async_trait::async_trait;
use nexus::{Agent, Kernel, KernelConfig, Message, MessageKind};
use serde_json::json;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

struct PingAgent {
    id: Uuid,
    name: String,
    seen: Arc<AtomicUsize>,
}

impl PingAgent {
    fn new(name: &str) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.to_string(),
            seen: Arc::new(AtomicUsize::new(0)),
        }
    }
}

#[async_trait]
impl Agent for PingAgent {
    fn id(&self) -> Uuid {
        self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn handle_message(&self, msg: Message) -> Option<Message> {
        let n = self.seen.fetch_add(1, Ordering::Relaxed) + 1;

        // Simulate tiny work
        sleep(Duration::from_millis(10)).await;

        if msg.kind == MessageKind::Command && msg.payload.get("ping").is_some() {
            return Some(msg.reply(self.id, json!({"pong": true, "count": n, "agent": self.name})));
        }

        None
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt().with_target(false).compact().init();

    let kernel = Kernel::new(KernelConfig::default());

    let a = PingAgent::new("A");
    let b = PingAgent::new("B");

    let a_id = kernel.spawn_agent(Box::new(a)).await?;
    let b_id = kernel.spawn_agent(Box::new(b)).await?;

    let system = Uuid::new_v4();

    // System -> A
    kernel
        .send(Message::new(system, a_id, MessageKind::Command, json!({"ping": true})))
        .await?;

    // A -> B
    kernel
        .send(Message::new(a_id, b_id, MessageKind::Command, json!({"ping": true})))
        .await?;

    // Give agents a moment to process
    sleep(Duration::from_secs(1)).await;

    kernel.shutdown().await;
    Ok(())
}
```

Run it:

```bash
cargo run --example ping_agent
```

If you see logs and a clean shutdown, you have a working agent system.

---

## Benchmarking performance

Nexus ships Criterion benchmarks (see `benches/message_throughput.rs`).

Run them:

```bash
cargo bench --bench message_throughput
```

Criterion writes HTML reports under:

- `target/criterion/`

Open the report in your browser (path will vary by benchmark):

- `target/criterion/message_send/report/index.html`

### Python asyncio baseline (apples-to-apples-ish)

Create `scripts/python_baseline_asyncio.py`:

```python
#!/usr/bin/env python3
import asyncio
import time

NUM_AGENTS = 1000
NUM_MESSAGES = 20000
MAILBOX = 1024

class Agent:
    def __init__(self):
        self.q = asyncio.Queue(maxsize=MAILBOX)

    async def run(self):
        while True:
            try:
                await self.q.get()
                # minimal work
                await asyncio.sleep(0)
            except asyncio.CancelledError:
                break

async def main():
    agents = [Agent() for _ in range(NUM_AGENTS)]
    tasks = [asyncio.create_task(a.run()) for a in agents]

    # warmup
    for i in range(200):
        await agents[i % NUM_AGENTS].q.put(i)

    t0 = time.perf_counter()
    for i in range(NUM_MESSAGES):
        await agents[i % NUM_AGENTS].q.put(i)
    t1 = time.perf_counter()

    dt = t1 - t0
    print(f"python_asyncio: {NUM_MESSAGES/dt:.0f} msg/s, {dt*1000:.1f} ms total")

    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
mkdir -p scripts
python3 scripts/python_baseline_asyncio.py
```

Now you have real measurements from your machine:

- Rust: `cargo bench ...`
- Python: `python3 scripts/python_baseline_asyncio.py`

That is the honest comparison.

---

## Flame graphs (the money shot)

Benchmarks give you numbers.

Flame graphs tell you the story behind the numbers.

### Rust flamegraph (Linux)

```bash
cargo install flamegraph

# You may need to adjust perf permissions:
# sudo sysctl -w kernel.perf_event_paranoid=1

cargo flamegraph --bin nexus-demo --release -- -n 2000 -d 10
```

This produces an SVG flamegraph you can open in a browser.

### Python flamegraph

```bash
python3 -m pip install --user py-spy
py-spy record -o python.svg -- python3 scripts/python_baseline_asyncio.py
```

When you compare them, you are not arguing about languages.

You are looking at where the CPU actually went.

---

## Multi-language agents: Python and JavaScript bridges

This repo includes **agent-side bridges** for Python and Node.js.

They implement a simple newline-delimited JSON protocol over stdin/stdout.

- Python: `bridges/python/nexus_bridge.py`
- JS: `bridges/javascript/nexus-bridge.js`

### Run the included example agents

Python example:

```bash
python3 examples/python_agent_example.py
```

JavaScript example:

```bash
node examples/javascript_agent_example.js
```

These examples expect to speak to a kernel process over stdin/stdout.

In this repo version, the demo runtime (`nexus-demo`) does **not** spawn external bridge processes yet.

So here is a tiny harness you can use today to verify the protocol.

### Protocol harness (spawn Python agent and send one message)

Create `scripts/bridge_harness.py`:

```python
#!/usr/bin/env python3
import json
import subprocess
import sys
import time

AGENT_CMD = [sys.executable, "examples/python_agent_example.py"]

p = subprocess.Popen(
    AGENT_CMD,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1,
)

# Read the agent's registration message
line = p.stdout.readline().strip()
print("agent ->", line)

# Parse registration
reg = json.loads(line)
agent_id = reg["payload"]["id"]

# Send one message
msg = {
    "type": "message",
    "payload": {
        "from": "00000000-0000-0000-0000-000000000000",
        "to": agent_id,
        "kind": "Command",
        "payload": {"operation": "add", "a": 40, "b": 2},
        "timestamp": "2025-03-03T00:00:00Z",
    },
}

p.stdin.write(json.dumps(msg) + "\n")
p.stdin.flush()

# Read response
resp = p.stdout.readline().strip()
print("agent ->", resp)

# Ask agent to shutdown
p.stdin.write(json.dumps({"type": "shutdown", "payload": {}}) + "\n")
p.stdin.flush()

# Give it a moment and exit
time.sleep(0.2)
p.terminate()
```

Run it:

```bash
python3 scripts/bridge_harness.py
```

If you see a `response` message with `result: 42`, the bridge is working.

That bridge protocol is the foundation for running Python/JS agents under a Rust kernel.

---

## Production deployment

This repo includes:

- `Dockerfile`
- `scripts/nexus.service` (systemd unit)
- `scripts/deploy.sh`

### Docker

Build and run:

```bash
docker build -t nexus:local .
docker run --rm -it nexus:local nexus-runtime -n 100 -d 30
```

### systemd (Linux)

Copy the unit file:

```bash
sudo cp scripts/nexus.service /etc/systemd/system/nexus.service
sudo systemctl daemon-reload
sudo systemctl enable --now nexus

sudo systemctl status nexus
sudo journalctl -u nexus -f
```

Before enabling systemd, make sure `nexus-runtime` is installed in a stable path, e.g. `/usr/local/bin/nexus-runtime` (use `./install.sh --system`).

---

## Troubleshooting

### OpenSSL / pkg-config errors

Ubuntu/Debian:

```bash
sudo apt-get install -y pkg-config libssl-dev
```

macOS:

```bash
brew install openssl
export OPENSSL_DIR="$(brew --prefix openssl)"
```

### Too many open files (large agent counts)

Temporary:

```bash
ulimit -n 65536
```

Persistent (Linux): adjust limits in `/etc/security/limits.conf`.

### "nexus-runtime not found"

Ensure your PATH includes `$HOME/.local/bin` after running `./install.sh`:

```bash
export PATH="$HOME/.local/bin:$PATH"
which nexus-runtime
```

---

## Conclusion
![](./../assets/images/posts/2025-03-03-why-ai-agents-need-a-kernel-not-a-framework/nexus.jpg)


I started with a Python swarm.

I ended up building a kernel.

Because the moment you scale agents, you stop writing "scripts" and start writing a runtime. Nexus makes that runtime explicit:

- bounded mailboxes
- timeouts
- deterministic shutdown
- fast routing
- structured observability

If you care about scalable agent systems, stop hoping your framework grows into an operating system.

Build on kernel semantics from the start.
