# 流量套餐智能客服 · 知识库问答系统

> 面向「流量套餐推荐 + 知识库问答」场景的智能客服系统。
> FastAPI 提供对话接口，Gradio 提供 Web 界面，两者通过 HTTP 解耦；
> 检索链路包含**文档相关性判断**与**多轮查询重写**；
> 会话状态与用户记忆持久化到 **PostgreSQL**；模型调用层抽象后可切换
> DeepSeek / 通义千问 / 本地 Ollama。

---

## 1. 项目定位

这个系统要解决两个具体问题，而不是简单地接一个大模型接口。

**问题一：答非所问。**
朴素 RAG 的做法是「无脑检索 → 把 top-k 片段塞给模型」，它会稳定地犯两种错：

- 用户问的内容根本不在知识库范围内（比如问天气），检索器仍然会返回「最相关」的几个片段 ——
  因为总有东西排在前面。模型拿到无关片段后，要么答非所问，要么直接编。
- 用户问的是依赖上文的追问（上一轮问「A 套餐多少钱」，这一轮只问「那 B 呢」）。
  拿「那 B 呢」去做向量检索，几乎没有有效信息，召回必然很差。

**问题二：会话无状态。**
多轮对话必须记住上下文，用户偏好（预算、流量需求）最好能跨会话保留，
否则每次都要用户重复描述一遍。

整套系统的结构：

```
                    ┌──────────────────────┐
   浏览器 ─────────▶│  Gradio 界面  :7860  │   只发 HTTP，不碰业务代码
                    └──────────┬───────────┘
                               │ HTTP
                    ┌──────────▼───────────┐
                    │  FastAPI 服务 :8000  │   /chat /conversations /providers
                    └──────────┬───────────┘
                               │
                    ┌──────────▼───────────┐
                    │  SupportAgent 编排层 │   检索 → 提示词 → 生成 → 工具回灌
                    └───┬──────────┬───────┘
                        │          │
         ┌──────────────▼──┐   ┌───▼────────────────┐
         │  RAG 检索链路   │   │  工具层            │
         │  范围判断       │   │  套餐查询 / 推荐    │
         │  多轮查询重写   │   └────────────────────┘
         │  向量检索       │
         └────────┬────────┘
                  │
      ┌───────────▼────────────┐
      │  抽象模型调用层         │   deepseek / qwen / zhipu / ollama
      └───────────┬────────────┘
                  │
      ┌───────────▼────────────┐
      │  持久化层 PostgreSQL   │   会话 / 消息 / 用户记忆 / 引用溯源
      └────────────────────────┘
```

---

## 2. 五分钟上手

```bash
# 1. 建环境
uv venv .venv --python 3.12
uv pip install --python .venv/Scripts/python.exe -r requirements.txt

# 2. 配置（把 .env.example 复制为 .env 并填入自己的模型 key）
cp .env.example .env

# 3. （可选）启动 PostgreSQL。不启动也能跑 —— 会自动降级为 SQLite
docker compose up -d

# 4. 知识库入库
.venv/Scripts/python.exe scripts/ingest.py

# 5. 启动 API 服务
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000

# 6. 另开一个终端，启动 Web 界面（它只通过 HTTP 调后端）
.venv/Scripts/python.exe app/ui/gradio_app.py
```

接口文档：http://127.0.0.1:8000/docs

---

## 3. 实施步骤

下面九步是本项目的完整实施顺序。每一步给出：**做什么 → 落在哪个文件 → 为什么这么做 → 怎么验收**。

### 步骤 1 · 配置层与模型抽象层

**产出**：`core/config.py`、`core/llm.py`

配置全部走「环境变量 > `.env` > 代码默认值」三级，
代码里不出现任何硬编码的密钥或连接串。

模型抽象层的做法是**一个统一接口 + 一组预置**：

| provider | base_url | 默认模型 | 需要 key |
| --- | --- | --- | --- |
| deepseek | api.deepseek.com/v1 | deepseek-chat | 是 |
| qwen | dashscope.aliyuncs.com/compatible-mode/v1 | qwen-plus | 是 |
| zhipu | open.bigmodel.cn/api/paas/v4 | glm-4-flash | 是 |
| ollama | 127.0.0.1:11434/v1 | qwen2.5:7b | 否 |

四家的差异被压缩成三个字段。**它们全部走 OpenAI 兼容协议**，
所以本地模型和云端厂商共用同一套调用代码。

**为什么这么做**：`RAGPipeline`、`SupportAgent`、HTTP 接口、Gradio 界面
这四层完全感知不到背后用的是哪一家。要新增一家厂商，只需要在
`PROVIDER_PRESETS` 里加一行，上层代码零改动。

**验收**：`GET /providers` 返回四个可用提供方；`POST /providers/switch` 可运行时切换。

---

### 步骤 2 · 持久化层

**产出**：`core/memory/db.py`（ORM 模型）、`core/memory/store.py`（读写接口）

三张表：

| 表 | 存什么 |
| --- | --- |
| `conversations` | 会话元信息：标题、创建/更新时间、消息数、会话级摘要 |
| `messages` | 每轮对话，含 `citations`（引用溯源）与 `retrieval_meta`（检索诊断） |
| `user_memories` | 跨会话的长期记忆：预算上限、关注套餐、所在城市等 |

三个设计取舍：

1. **用 SQLAlchemy ORM，而不是手写 SQL。** 因为「PostgreSQL 还是 SQLite」
   这件事对业务代码应当透明 —— 换库只改连接串，不改一行查询逻辑。
2. **消息与会话分表。** 「加载历史会话」因此变成一次索引查询，
   而不是把一整坨 JSON 读出来再解析。
3. **用户记忆单独一张表。** 记忆是跨会话的长期信息，与会话生命周期解耦 ——
   会话可以删，记忆要留下。

`messages` 表里的 `citations` 与 `retrieval_meta` 是这个项目「可追溯」的基础：
回答完之后能查出这一轮引用了哪些知识片段、检索链路的每一步决策是什么。

**验收**：`tests/test_memory.py`，含「模拟重启后历史仍在」的用例。

---

### 步骤 3 · 知识库切片与向量化

**产出**：`scripts/ingest.py`、`core/rag/embedding.py`、`core/rag/vectorstore.py`

**切片策略**（不是无脑按字数切）：

1. 先按 Markdown 二级标题分节 —— 知识库的每一节本身就是语义完整的单元
   （例如「如何退订」），从中间切断会让片段缺头少尾
2. 节内再按段落聚合到约 400 字 —— 太短信息量不足，太长会稀释向量、降低区分度
3. 保留来源与章节名作为元数据 —— 回答时能给出引用出处，
   排查「为什么答非所问」时也能立刻知道是哪段被召回了

**向量化**：默认用本地实现（中文字符 bigram + 特征哈希 + 余弦相似度）。
理由很直接：零依赖、完全离线、可复现，且对「套餐名、资费数字、关键词」
这类词面匹配为主的场景效果足够。想换成语义嵌入，实现一个 `embed()` 函数
并改配置即可 —— 上层检索代码一行都不用动。

**为什么不用 Chroma / FAISS**：知识库是几百个片段这个量级，
在这个量级下暴力余弦相似度就是毫秒级。引入一个带索引构建、带持久化格式的
向量库，增加的是依赖和排查难度，换来的性能收益为零。
`VectorStore` 的接口语义与 LangChain 的 Retriever 保持一致，
真要换的时候替换 `search()` 即可。

**验收**：`ingest.py` 结尾会跑检索自检，输出 4 条典型查询的命中来源与相似度。

---

### 步骤 4 · 检索链路：范围判断 + 多轮查询重写

**产出**：`core/rag/pipeline.py` · `RAGPipeline`

这是整个项目最核心的一段。流程是：

```
用户提问
  ↓
① 范围判断 ── 不在知识库范围内 → 直接拒答（不检索、不生成）
  ↓ 在范围内
② 原句检索
  ↓
③ 召回质量不足 → 查询重写 → 重新检索（最多 N 轮）
  ↓
④ 仍无命中 → 承认不知道，而不是硬答
```

**为什么"先判断范围"能省成本**：拒答发生在检索**之前**。
一次超范围提问，省掉的不只是一次向量检索，还有一次完整的模型生成
（这才是 token 消耗的大头）。更重要的是它省掉了一次**幻觉的机会** ——
模型没拿到无关片段，就没有素材去编。

**为什么查询重写不是每轮都做**：改写本身要花一次模型调用。
如果每次提问都改写，等于把成本翻倍，而绝大多数问题（比如「畅享39 多少钱」）
本身就很完整，根本不需要改写。所以策略是：
**先用原句检索，只有召回质量不达标时才改写**。
这是一个典型的「用一次便宜的操作换掉一次昂贵的操作」的取舍。

**两个防御性设计**：

- 范围判断失败时**默认放行**。宁可多检索一次，也不要因为判断器抽风
  把正常问题挡在门外 —— 客服场景下「答不了」比「答得慢」严重得多。
- 改写结果异常（为空或过长）时**退回原句**，保证链路不会因为改写失败而停摆。

**验收**：`tests/test_rag.py` 覆盖范围外拒答、范围内放行、判断失败放行、
改写触发、改写不触发、改写用尽后拒答六种路径。

---

### 步骤 5 · 工具层

**产出**：`core/agent/tools.py`

工具是「知识库问答」之外的第二条能力线：知识库负责把**已有信息的解释**讲清楚，
工具负责回答**需要精确筛选或计算**的问题。

**为什么要把筛选写成代码而不是交给模型**：
模型适合做表达和推理，不适合做精确的数值筛选。
让它自己从 8 个套餐里挑出「20 元以内且流量最多的」，它可能挑错。

实现上踩过两个真实的坑，都已修掉并加了回归测试：

1. **模型把「流量尽量多」翻译成硬性数字门槛。** 用户说「40 元以内、流量尽量多」，
   模型调工具时传了 `min_data_gb: 100`，结果把本来合适的套餐全筛掉了。
   → 修法是在工具描述与系统提示里明确：只传用户明确说出的条件，
   「尽量」是偏好不是门槛。
2. **定向流量包混进了性价比排序。** 「视频定向包」9 元 30GB，折合 0.3 元/GB，
   在按单位价格排序时排到了第一 —— 但它的 30GB 只能在指定视频 App 内使用，
   和通用流量比单价会得出严重失真的结论。
   → 修法是在 `recommend()` 里排除定向与日付套餐。

这两个坑说明同一件事：**工具的正确性必须是确定的、可回归的**，
这也正是把筛选逻辑从模型手里拿回来写进代码的原因。

**验收**：`tests/test_tools.py`，含上述两个坑对应的回归用例。

---

### 步骤 6 · Agent 编排

**产出**：`core/agent/chain.py` · `SupportAgent`

一次回答经过的完整步骤：

```
读历史 + 读记忆
  → 范围判断（超出直接拒答，省掉检索与生成）
  → 多轮查询重写 + 检索
  → 组装 Prompt（LangChain 的 ChatPromptTemplate）
  → 模型生成（可调用工具）
  → 工具执行回灌 → 再生成（最多 3 轮）
  → 落库（消息 + 引用 + 检索元数据）
  → 更新长期记忆
```

**LangChain 在这里做什么**：用来组织提示词模板与步骤编排。
底层 HTTP 调用仍走统一的 `LLMClient` —— 这样「换厂商」和「换编排方式」
两件事互不干扰。

**记忆抽取用规则而不是模型**：规则便宜、可预测、不会把用户没说的话「记」下来。
真要上模型抽取，也应当只在规则命中失败时才启用。

**工具轮数用尽时给兜底话术**，而不是把空字符串返回给用户。

**验收**：`scripts/smoke.py` 会跑 8 个真实用例（见下一节）。

---

### 步骤 7 · HTTP 接口

**产出**：`app/main.py`

FastAPI 提供对话、会话管理、记忆查询、模型切换等接口，
用 `lifespan` 在启动时完成建表、加载索引、装配 Agent。

`POST /chat` 的响应里带一个 `retrieval` 字段：

```json
{
  "answer": "畅享39 月费 39 元，含 20GB 通用流量……",
  "conversation_id": "a1b2c3d4e5f6",
  "citations": [{"source": "01-套餐总览.md", "section": "畅享39", "score": 0.71}],
  "retrieval": {
    "in_scope": true,
    "scope_reason": "询问套餐资费",
    "rewritten_query": "",
    "rewrite_rounds": 0,
    "hit_count": 2,
    "best_score": 0.71
  },
  "tool_calls": []
}
```

**这个字段是排查「为什么答非所问」的关键**：一眼就能看出是范围判断错了、
改写没生效、还是检索本身没命中。

**验收**：`tests/test_api.py`，用 TestClient 直接打接口，不启动服务、不调模型。

---

### 步骤 8 · Web 界面

**产出**：`app/ui/gradio_app.py`

**这个文件不 import 任何业务代码**，只通过 HTTP 调用 FastAPI 的接口。
这就是「接口与界面解耦」的实际含义：

- 换前端（Vue / 小程序 / 桌面端）→ 直接调同一批接口，后端零改动
- 后端可以脱离界面独立测试 → pytest + TestClient 就够
- 界面的 bug 影响不到业务逻辑的稳定性

界面上除了对话区，还并排显示**引用来源**与**检索链路诊断**
（范围判断结果、改写后的查询、命中数、最高相似度、工具调用），
让排查过程可视化。

**验收**：同时启动 API 与界面两个进程，在界面上问一个知识库问题，
右侧应当出现引用与诊断信息。

---

### 步骤 9 · 测试与验收

**产出**：`tests/`（25 个用例）+ `scripts/smoke.py`

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_rag.py` | 范围判断、查询重写、检索命中、索引持久化 |
| `tests/test_memory.py` | 会话生命周期、历史恢复、引用落库、记忆隔离 |
| `tests/test_tools.py` | 推荐筛选、排序、参数错误回灌、schema 合法性 |
| `tests/test_api.py` | 全部 HTTP 接口、错误码、对话装配与落库 |

全部**不联网、不花钱**：用脚本化的假模型替代真实调用，
因为这一层测的是链路逻辑（什么时候改写、什么时候放弃），不是模型质量。

---

## 4. 完整验收流程

```bash
# 1. 单元测试：25 个用例
.venv/Scripts/python.exe -m pytest tests -v

# 2. 知识库入库 + 检索自检
.venv/Scripts/python.exe scripts/ingest.py

# 3. 真实端到端：8 个用例（会调用真实模型）
.venv/Scripts/python.exe scripts/smoke.py

# 4. 启动服务，人工验证接口与界面
.venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
.venv/Scripts/python.exe app/ui/gradio_app.py
```

`scripts/smoke.py` 的 8 个用例及预期：

| # | 用例 | 预期 |
| --- | --- | --- |
| ① | 服务健康检查 | 返回数据库后端、知识库片段数、当前模型 |
| ② | 知识库问答「畅享39 多少钱」 | 命中检索，回答含正确资费，带引用 |
| ③ | 多轮追问「那它超出怎么算？」 | 能结合上文作答 |
| ④ | 推荐「40 元以内流量尽量多」 | **调用工具**筛选，而不是模型自己心算 |
| ⑤ | 超范围「今天上海天气怎么样」 | **直接拒答**，`in_scope=false`，命中 0 |
| ⑥ | 历史会话恢复 | 从库里读回全部消息，含引用与 seq 顺序 |
| ⑦ | 用户长期记忆 | 抽取到关注套餐与预算上限 |
| ⑧ | 模型提供方列表 | 四家可选，当前为配置项 |

---

## 5. 接口参考

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/chat` | 对话主接口，返回答案 + 引用 + 检索诊断 |
| GET | `/conversations` | 会话列表 |
| GET | `/conversations/{id}` | 会话详情 |
| GET | `/conversations/{id}/messages` | 加载历史消息 |
| DELETE | `/conversations/{id}` | 删除会话（级联删消息） |
| GET | `/memories?user_id=` | 查看某用户的长期记忆 |
| DELETE | `/memories?user_id=` | 清空长期记忆 |
| GET | `/packages?keyword=` | 套餐列表 / 按关键词查询 |
| GET | `/providers` | 可用的模型提供方 |
| POST | `/providers/switch` | 运行时切换模型提供方 |
| GET | `/health` | 健康检查 |

---

## 6. 项目结构

```
customer-service-agent/
├── app/
│   ├── main.py            FastAPI 入口与全部路由
│   └── ui/gradio_app.py   Gradio 界面（只发 HTTP，不 import 业务代码）
├── core/
│   ├── config.py          配置：环境变量 > .env > 默认值
│   ├── llm.py             抽象模型调用层（deepseek/qwen/zhipu/ollama）
│   ├── rag/
│   │   ├── embedding.py   中文词面嵌入（可切换为厂商 embedding 接口）
│   │   ├── vectorstore.py 轻量向量库 + JSON 持久化
│   │   └── pipeline.py    范围判断 + 多轮查询重写 + 检索
│   ├── agent/
│   │   ├── chain.py       编排：检索 → 提示词 → 生成 → 工具回灌
│   │   └── tools.py       套餐查询 / 推荐工具
│   └── memory/
│       ├── db.py          SQLAlchemy 模型（PostgreSQL / SQLite）
│       └── store.py       会话与记忆的读写接口
├── data/
│   ├── knowledge/         知识库文档（套餐总览 / 办理退订 / 常见问题）
│   └── packages.json      套餐结构化数据（工具用）
├── scripts/
│   ├── ingest.py          知识库切片入库
│   ├── smoke.py           真实端到端冒烟测试
│   └── make_env.py        从 .env.example 生成 .env
├── tests/                 25 个用例
└── docker-compose.yml     一键启动 PostgreSQL
```

---

## 7. 功能点与实现位置

| 功能点 | 实现位置 |
| --- | --- |
| 对话 API | `app/main.py` · `POST /chat` |
| 会话管理 API | `app/main.py` · `/conversations` 系列 |
| Gradio 界面 | `app/ui/gradio_app.py` |
| 接口与界面解耦 | 界面不 import 业务代码，只发 HTTP |
| 多轮对话 | `core/agent/chain.py` · 每轮加载历史拼进提示词 |
| 历史会话加载 | `core/memory/store.py` · `load_history()` / `load_messages_full()` |
| 先判断范围 | `core/rag/pipeline.py` · `judge_scope()` |
| 多轮查询重写 | `core/rag/pipeline.py` · `rewrite_query()` |
| 向量检索 | `core/rag/vectorstore.py` |
| 向量化 | `core/rag/embedding.py` |
| 提示词编排 | `core/agent/chain.py` · `ChatPromptTemplate` |
| 工具调用与回灌 | `core/agent/tools.py` + `chain.py` 的循环 |
| PostgreSQL 持久化 | `core/memory/db.py` · 三张表 |
| 用户记忆 | `core/memory/store.py` · `upsert_memory()` / `get_memories()` |
| 抽象模型调用层 | `core/llm.py` · `PROVIDER_PRESETS` |

---

## 8. 关于 PostgreSQL 与 SQLite

代码里用的是 SQLAlchemy ORM，所以**换库不需要改任何业务逻辑**。

- 生产 / 演示：`docker compose up -d` 起 PostgreSQL，在 `.env` 里配 `DATABASE_URL`
- 本地开发：PostgreSQL 连不上时会自动降级为 SQLite（启动日志会打印一行提示）

降级是**开发便利**，不是替代方案。启动日志与 `/health` 接口都会明确标出
当前实际使用的是哪个后端，不会出现「以为在用 PG、其实在用 SQLite」的情况。

---

## 9. 已知局限与后续方向

**局限**

- 知识库规模小（3 篇文档 / 13 个片段），检索链路的效果验证样本有限
- 嵌入默认用词面匹配而非语义嵌入，对同义改写（「便宜点」vs「资费低」）的召回能力有限
- 记忆抽取用规则实现，覆盖面窄（只识别套餐名、预算、流量、城市）
- 未做并发与限流，界面与接口都是单进程同步处理
- 未接入真正的评测体系（回答质量目前靠人工抽检）

**后续方向**

- 把嵌入切换到厂商 embedding 接口，并做一轮「词面 vs 语义」的召回对比
- 建立一套问答评测集，把「范围判断准确率」「改写命中率」变成可回归的数字
- 记忆抽取改为「规则优先、模型兜底」，并加入记忆的时效衰减
- 引入流式输出与并发处理

---

## 10. 许可

MIT
