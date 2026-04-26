---
title: Dead Internet Detective
emoji: 🕵️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

Dead Internet Detective is a reinforcement learning environment
for training LLM agents to detect AI-generated disinformation.
The server exposes /reset, /step, /state, and /health endpoints.
It is used as the training environment for a GRPO-trained agent
built on Llama-3.1-8B-Instruct.
