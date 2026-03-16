# Andrej Karpathy 关于 Auto Research 的推文分析

Andrej Karpathy 是专注大规模深度神经网络的 AI 研究者，曾任特斯拉自动驾驶 AI 负责人、OpenAI 联合创始人，斯坦福博士。通过 nanochat 等项目探索低成本 LLM 训练，在单机两小时内达到 GPT-2 水平，并推进用于自主研究的 agent 工作流。

---

## 一、按时间顺序的推文总结（2026-03-07 至 03-15）

**2026-03-07 19:53**  
将「autoresearch」打包成极简自包含仓库（GitHub: karpathy/autoresearch）。核心是把 nanochat LLM 训练代码精简到单 GPU、单文件约 630 行：人类只迭代 `prompt.md`，AI 代理在 git feature branch 上自主循环迭代训练代码，每次实验固定 5 分钟，只保留能降低验证损失的 commit。目标是「让代理无限自主推进研究，无需人类介入」。他形容这是「部分代码、部分科幻、一点精神错乱」。

**2026-03-08 18:00**  
展望 autoresearch 的下一个方向：要变成异步大规模协作（类似 SETI@home），不再是模拟单个 PhD，而是模拟整个「研究社区」。当前实现是单线程 commit，他认为 GitHub 几乎合适但需改进。分享了用 GitHub Discussion（代理写总结）或 PR（精确 commit）的轻量原型，代理可先读他人成果再贡献。强调未来代理能处理数千个分支/commit，现有抽象会因智能与注意力不再成为瓶颈而承受压力。

**2026-03-08 22:46**  
确认过去两天（约 650 次实验）在 depth=12 模型上找到的改进，成功迁移到 depth=24 模型，nanochat 的「Time to GPT-2」排行榜即将更新。他感慨：「谁知道早期奇点居然这么好玩？」

**2026-03-09 19:03**  
回复关于 Codex 的问题：当前 autoresearch 设置下 Codex 无法正常工作（GitHub issue #57，已联系 OpenAI）。他不喜欢「-p + ralph 无头循环」，更希望用 tmux 交互式会话，能实时看到代理在做什么并随时介入。

**2026-03-09 22:28（主帖）**  
详细报告：3 天前让 autoresearch 在 depth=12 上自主运行约 2 天，累计约 700 次改动，得到约 20 个真正降低验证损失的改进。昨天手动验证，所有改进可叠加，且迁移到 depth=24 后有效——Time to GPT-2 从 2.02 小时降到 1.80 小时（约 11%），将更新排行榜。  
他强调这是 20 年来第一次看到代理端到端完成整个研究循环（想 idea → 实现 → 验证 → 规划下一轮）。列举了部分发现（QKnorm 缺 scaler、Value Embeddings 需要正则化、banded attention 太保守、AdamW betas、weight decay 调度、初始化等）。  
未来计划：启动 round 2 + 多代理协作；预测「所有 LLM 前沿实验室都会这么做」，并推广到任意可高效评估的指标。

**2026-03-09 22:38**  
补充：应早点给出 autoresearch GitHub 链接——这是给代理的 recipe/idea，让代理用到自己关心的项目上，而非直接「用」的成品工具。

**2026-03-09 23:04**  
解释之前 plot 的困惑：当前版本没有做「时间控制」，有些点虽然验证损失更低但训练时间更长，因此被拒绝。只有损失更好或相等，且时间更好或相等的改动才会被接受。

**2026-03-11 18:01**  
autoresearch 的「实验室」在 OAuth 故障中被清空，需要考虑 failover。他感慨：前沿 AI 一卡顿，整个星球就会「智商 brownout」。

**2026-03-15 15:53**  
回复某博客：「Nice! My autoresearch would love some markdown version of this - pool of ideas.」（希望把博客导出为 markdown，加入 autoresearch 的 idea pool）。

**2026-03-15 23:07**  
感谢回复：已用 Obsidian 插件把该博客导出为 markdown，排队加入 autoresearch loop 作为新 idea。

**总结**：从 3 月 7 日发布极简仓库开始，Karpathy 逐步展示 autoresearch 的实际效果（真实性能提升）、未来愿景（多代理协作社区）、以及遇到的问题（故障、工具兼容），并实际用它优化了 nanochat。这是他近期最核心的「AI 代理做研究」实验系列。

---

## 二、Autoresearch 核心机制详解

### 1. 需要的 GPU 和显存

- **GPU**：单张 NVIDIA GPU（不支持多 GPU、CPU 或 MPS）。README 写明：「This code currently requires that you have a single NVIDIA GPU.」他在 H100 上跑通整个流程。
- **显存**：README 未给精确数字。nanochat 小模型 + 固定 5 分钟训练 + 单 GPU，实际需求较低（H100 80GB 足够，推测 24–40GB 即可，消费级卡如 4090 在调低 DEPTH 后也可能运行）。
- **运行方式**：本地或云端单卡均可，按仓库说明启动训练即可。

### 2. 使用的是 Codex 还是 Claude？具体实验用的什么模型？

**代理 LLM（AI 研究员）**  
README 未强制指定，可用任意支持工具调用的 LLM（Claude、GPT-4o、o1 等）。Karpathy 反馈 Codex 当前无法正常工作（GitHub issue #57）。他更倾向用 tmux 交互式会话，能实时看到代理并随时介入。实际推荐 Claude（如 Claude-3.5-Sonnet 或更新版本），工具调用稳定、上下文长，适合长时间自主迭代。

**具体实验用的 nanochat 模型**  
- 来自 nanochat 项目的简化版（https://github.com/karpathy/nanochat）。  
- 实验模型：depth=12（初期约 700 次实验）、depth=24（迁移验证用）。  
- DEPTH 是主要旋钮（默认 8），层数、宽度、头数等随其自动缩放。  
- 目标：从零训练（BPE tokenizer），追求最低 val_bpb，对标「Time to GPT-2」排行榜（如 1.8 小时内达到 GPT-2 水平）。  
- 从随机初始化的小模型训练（参数量级为百万级，单次实验约 5 分钟）。

### 3. 如何界定损失下降是改进贡献，还是训练步数增多？

这是 autoresearch 的核心设计之一，避免「刷时长」。

**当前版本（固定时间）**  
- 每个实验严格固定 5 分钟 wall-clock（不含启动/编译）。  
- 无论代码如何改，训练器只跑固定时长，然后测量 val_bpb。  
- 只有 val_bpb 降低或持平时才 commit，否则丢弃。  
- 这样所有实验的计算预算一致，性能提升只能来自架构/优化器/初始化等代码改动。

**补充规则（他曾发推解释）**  
- 接受条件：loss 更好或相等，且 time 更好或相等（二者都需满足）。  
- 有些 commit 虽然 loss 更低但训练时间变长，会被拒绝（「must improve one, the other or both」）。

**实际效果**  
在 depth=12 上约 700 次实验，得到约 20 个真实改进（QKnorm scaler、Value Embeddings 正则化、banded attention、AdamW betas、weight decay 调度、初始化等），全部可叠加；迁移到 depth=24 后，Time to GPT-2 从 2.02 小时降到 1.80 小时（约 11%），说明是真正的端到端自主研究循环，而非刷步数。

**总结**：固定时间预算 + 双指标（loss + time）是关键防作弊手段；若自行运行，可先把 DEPTH 调到 4–8 试跑，Claude + tmux 是较舒适的搭配。
