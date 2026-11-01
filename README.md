# 人格工坊

把一个具体的人做成**可长期维护的数据资产**：人格库 + 记忆库是本体，素材台与蒸馏台负责往里添砖，试聊负责验证。它不连任何聊天通道——QQ、DSH、自写脚本都是库的消费者。

> 一句话：**人格是资产，程序只是手。** 换通道、换模型、换账号，库一个字节都不用动。

---

## 一、它和 QQ 通道器是什么关系

```
人格工坊（本程序）                        QQ 通道器（另一个程序）
├─ 总端：多库管理                        ├─ 接 QQ 收发消息
├─ 人格库读写（wiki / 母本 / raw）        ├─ 触发 / 合流 / 门控 / 限速
├─ 记忆库读写（sessions.db）              ├─ 人格漂移
├─ 素材台：按母本直出反应卡               └─ 读同一个库（只读人格内容）
├─ 蒸馏台：小说 → 对话 → 反应卡
├─ 试聊：不接通道也能验证语气
└─ contract/：库契约 ← 两边唯一的耦合面
```

两边共用的只有**库**。工坊写人格内容，通道器只读；会话记录由通道器写、工坊读（蒸馏与统计的来源）。谁都不碰对方的配置。

---

## 二、启动

**环境要求：Python 3.11+，零第三方依赖。** 不需要 pip install，装个 Python 就能跑（3.11 是 `tomllib` 的底线；开发环境 3.13）。任何 OpenAI 兼容端点都能接：DeepSeek / 通义 / Kimi / 智谱 / 本地 Ollama。

三选一：

```bash
python run.py              # 开控制台，自动弹浏览器
python run.py --port 8780  # 指定端口（被占用会自动往后找）
python run.py --check      # 只自检：配置 / 人格库 / 库契约 / 归档器
```

Windows 下双击 `启动.bat` 即可。

`config.toml` 不随仓库发布。首次启动若它不存在，程序会提示你从 `config.example.toml` 复制一份（凭证位留空），填好再启动。人格库目录不存在时会自动建一套空骨架，不用手动建。

---

## 三、第一次用

1. **建库**：总端点「新建人格库」，起个名字。一个库就是一个完整的人，各带自己的会话库与偏好，互不污染。
2. **写身份**：「身份」页是身份母本，七节骨架按节填。母本是所有素材的上游——不写，后面生成的反应跟身份无关。
3. **填模型**：「配置」页 → 模型组。任何 OpenAI 兼容端点都行（DeepSeek / 通义 / Kimi / 智谱 / 本地 Ollama），填 API Key 后保存。
4. **试聊**：不接任何通道直接聊，每条回复能展开看这一轮注入了什么、命中了哪些人格页。语气不对就先修母本或 wiki，别急着上 QQ。
5. **出素材**：「素材台」选刺激与档位，一次出三张反应卡，改到满意再过审收稿；成品 md 丢进库的 `raw/` 即参与归纳。
6. **蒸馏**：有小说/长文原文时，「蒸馏」页切出对话、抽成反应卡再收进库。
7. **归纳**：归档器把 `raw/` 与实战对话归进 `wiki/`、`index.md`、`memory/`。配置页可设周期与是否允许新建页。

---

## 四、目录与数据

```
人格工坊/
├── config.toml            全局配置（库侧：库路径、模型、记忆、素材台、蒸馏）
├── config.example.toml    空凭证模板
├── run.py                 启动入口
├── run_tests.py           测试矩阵（9 个套件）
├── workshop/              库 + 素材台 + 蒸馏台 + 控制台
│   ├── library.py         **库读写唯一实现**
│   ├── persona.py seed.py 人格库 / 身份母本
│   ├── ingest*.py indexer.py memwiki.py   归纳与记忆 wiki
│   ├── lint.py            库体检 + 契约校验
│   ├── memory.py          sessions.db 访问
│   ├── sim/ distill/      素材台 / 蒸馏台
│   ├── webui.py ui/       控制台
│   └── templates/         建库骨架模板
├── tools/
│   ├── mem_ingest.py      归档器
│   └── mem_doctor.py      记忆库修复 / hub 重建
├── contract/
│   ├── 库契约.md          ★ 外部工具照这个接库
│   └── sessions_schema.sql 会话表冻结版 v1
├── data/                  日志、库注册表 personas.index.toml、运行锁
├── personas/              人格库默认落点
├── sim/  distill/         素材与蒸馏的中间产物
└── .gitignore 等          本地配置与私人库均已被排除，不入版本库
```

一个人格库的内部结构：

```
personas/<库名>/
├── agents.md       人格库契约（给 AI 的读写规则）
├── index.md        总括页：一句话人设 + 路由规则
├── wiki/           八页：identity / speech / traits / values / boundaries /
│                   reactions / relationships / exemplars
├── raw/            原始素材（母本、实战对话、收稿的反应卡）
├── memory/         归纳产出（按主题的记忆条目 + ingest_stats.jsonl）
├── data/sessions.db 会话记录
├── preferences.toml 这个库的偏好覆盖（库侧段落，工坊写）
└── log.md          变更日志
```

---

## 五、库契约（这是它能"通用"的原因）

`contract/库契约.md` 定义了外部程序怎么接库，只有四条：

1. **目录结构与读写权限**：工坊写人格内容，消费者只读；会话记录由消费者写 `data/sessions.db`。`lint.py` 能机器校验。
2. **sessions.db schema 冻结 v1**：不改表结构就实现通用化——靠命名空间约定。
   - `session_id = <命名空间>:<范围ID>`：`private:` / `group:` 永久兼容，新工具注册自己的前缀（`dsh:`、`workbuddy:`、`qq-group:`），不冲突就互不干扰；
   - `role` 词表：`user` / `assistant` 必需，`system` 允许，其他值原样存原样读；
   - `user_id` 开放字符串：QQ openid、DSH 用户 ID 都合法。
3. **preferences.toml 归属**：`[persona] [bot] [owner] [llm] [memory] [sim] [distill]` 归工坊；`[adapter] [trigger] [batch] [gate] [rate_limit] [reply] [drift]` 归通道器。工坊保存时不写通道段，通道器保存时不写库段。
4. **一库一消费者**：同一个库同时只允许一个通道器连，避免两个进程抢写会话记录。

接一个新工具不需要懂工坊内部，照这四条做就行。

---

## 六、测试

```bash
python run_tests.py              # 跑全部 9 个套件
python run_tests.py test_sim     # 只跑一个
```

覆盖：关系分档、库体检与契约校验、index 重建、多库注册/切库/偏好隔离、记忆 wiki 判定、归纳管线、归档器、素材台、蒸馏台。

通道侧（QQ、DSH 等外部消费者）的触发 / 合流 / 门控 / 限速套件不在本仓库。本仓库只管库：通道器是独立程序，按 `contract/` 接入即可，不需要也不应该和工坊一起发布。

---

## 七、常见问题

**启动报"persona_dir 没填"** — `config.toml` 的 `[bot] persona_dir` 没填人格库路径。留空时填一个目录名即可，不存在会自动建库。

**提示缺 API Key** — 配置页模型组填 `api_key`，或设环境变量（默认 `DEEPSEEK_API_KEY`）。

**改完配置没生效** — 工坊是保存即生效（没有通道可重启）。若改了 `persona_dir`，需要重启进程。

**新建库报 `agents.md 不存在`** — `workshop/templates/` 缺失，请确认该目录完整（建库骨架全靠它）。

**归档没跑** — 看配置页归纳组：`wiki_enabled` 要开，`ingest_interval_hours > 0` 才定时跑；也可以手动点归档。归档器不可用时启动日志会写原因。

**两个库会不会串** — 不会。会话库、偏好、wiki 都在各自库目录里；启动库由全局 `config.toml` 的 `[bot] persona_dir` 决定，切库时写回。

---

## 八、许可证

MIT，见 [LICENSE](LICENSE)。

## 九、注意

本仓库**不含**任何人格库实例。`personas/`、`data/`、`sim/`、`distill/`、`config.toml` 全部在 `.gitignore` 里——人格库是使用者的私人数据，不该跟着代码走。第一次运行请从 `config.example.toml` 复制出 `config.toml` 再填自己的 Key。
