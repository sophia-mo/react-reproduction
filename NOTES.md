# HotpotQA
模型一边推理，一边主动查 Wikipedia，最后给答案：Reasoning + Acting
## Data
多跳问答数据集 `hotpot_dev_v1_simplified.json`

里面一共有 7405 条 dev 数据，每条被简化成：
```json
{
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "answer": "yes",
  "type": "comparison"
}
```
代码真正用到的只有两个字段：
```
d["question"]
d["answer"]
```
也就是说这个实验没有用 HotpotQA 原始 supporting facts/context，而是只拿问题和标准答案，证据来源改成了实时 Wikipedia 搜索


## `prompts_naive.json` 讲解
### webthink_simple
是完整 ReAct prompt，有 8 个示例
```
Question: ...
Thought 1: ...
Action 1: Search[...]
Observation 1: ...
Thought 2: ...
Action 2: Lookup[...]
Observation 2: ...
Thought 3: ...
Action 3: Finish[...]
```
#### webthink_simple6
只有 6 个示例，而且 Observation 被压短了
#### webthink_simple_3
只有 3 个示例，用途是做 ablation 或省 token

### cotqa_simple
CoT baseline，只推理，不行动
```
Question: ...
Thought: Let's think step by step. ...
Answer: ...
```
#### cotqa_simple6
和 cotqa_simple 一样，但只有 6 个示例

### webqa_simple
direct QA baseline
```
Question: ...
Answer: ...
```
#### webqa_simple6

### webact_simple6
Act-only，只有行动，没有显式 Thought，用来比较显式写 Thought 到底有没有帮助
```
Question: ...
Action 1: Search[...]
Observation 1: ...
Action 2: Lookup[...]
Observation 2: ...
Action 3: Finish[...]
```


## Prompt
在 `prompts_naive.json` 中使用 `webthink_simple6`

在 `hotpotqa.ipynb` 中，最终 prompt 由两部分拼起来：`instruction + webthink_examples`

instruction 告诉模型任务规则：
`Solve a question answering task with interleaving Thought, Action, Observation steps.`
并定义三种动作：
```
Search[entity]   # 搜索 Wikipedia 实体
Lookup[keyword]  # 在当前页面里找包含关键词的下一句话
Finish[answer]   # 给最终答案并结束
```

webthink_examples 是 few-shot 示例：
```
Question: ...
Thought 1: ...
Action 1: Search[...]
Observation 1: ...
Thought 2: ...
Action 2: Lookup[...]
Observation 2: ...
Thought 3: ...
Action 3: Finish[...]
```


## 实验流程
在 `hotpotqa.ipynb` 中环境初始化：
```python
env = wikienv.WikiEnv() # 提供 search/lookup/finish 动作
env = wrappers.HotPotQAWrapper(env, split="dev") # 把 HotpotQA 问题喂给环境，并负责评分
env = wrappers.LoggingWrapper(env) # 记录轨迹
```

每一题从这里开始：
```python
question = env.reset(idx=idx)
prompt += question + "\n"
```
`reset(idx=i)` 会取第 i 条 HotpotQA 数据，返回 `Question: ...`
位置在 `wrappers.py`：
```python
observation = f"Question: {self.data[self.data_idx][0]}"
```

然后进入最多 7 轮循环：
```python
for i in range(1, 8):
    thought_action = llm(prompt + f"Thought {i}:", stop=[f"\nObservation {i}:"])
```
让模型补全
```
Thought i: ...
Action i: ...
```

例如：
```
Thought 1: I need to search the album first.
Action 1: Search[Guitars for Wounded Warriors]
```
然后代码解析出 action：
```python
thought, action = thought_action.strip().split(f"\nAction {i}: ")
```
再交给环境执行：
```python
obs, r, done, info = step(env, action[0].lower() + action[1:])
```

### Search 是怎么工作的
在 `wikienv.py` 中访问 Wikipedia 搜索页

```python
search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
response_text = requests.get(search_url).text
soup = BeautifulSoup(response_text, features="html.parser")
```

如果找到的是搜索结果列表，就返回类似 `Could not find X. Similar: [...]`
如果直接进入页面，就抽取页面里的 `<p>` 和 `<ul>`，然后返回前 5 句话：
`self.obs = self.get_page_obs(self.page)`

### Lookup 是怎么工作的
在 `wikienv.py` 中
`elif action.startswith("lookup[") and action.endswith("]"):`
lookup[keyword] 会在当前 Wikipedia 页面里找包含关键词的句子

比如当前页面是 Milhouse，模型执行 `Lookup[named after]`

环境会返回 `(Result 1 / 1) Milhouse was named after U.S. president Richard Nixon...`

如果重复 lookup 同一个 keyword，它会返回下一条匹配句子

### 评分
当模型输出 `Finish[Richard Nixon]`，在 `wikienv.py` 中
```python
self.answer = answer
done = True
```
然后 HotPotQAWrapper 会拿模型答案和标准答案比对

评分在 `wrappers.py`：
```python
pred = normalize_answer(self.data[self.data_idx][1]) # 标准答案
gt = normalize_answer(info['answer']) # 模型答案
score = (pred == gt)
```

计算 EM 时，`normalize_answer` 会做：
```
小写
去标点
去 a/an/the
规整空格
```

最后在 `hotpotqa.ipynb` 中
```python
idxs = list(range(7405)) # HotpotQA dev 总共 7405 条
random.Random(233).shuffle(idxs) # 用随机种子 233 打乱

for i in idxs[:500]:
    r, info = webthink(i, to_print=True) # 取前 500 条，逐题跑 ReAct，统计 EM
```

每跑完一题打印 `print(sum(rs), len(rs), sum(rs) / len(rs), avg_time)`
分别是答对数量、已跑题数、当前 EM、平均每题耗时

## Execution
```bash
conda create -n react python=3.9 -y
pip install openai gym numpy requests beautifulsoup4 jupyter notebook
conda activate react
$env:OPENAI_API_KEY="sk-proj-..."
```
模型由源代码的 text-davinci-002 改成 gpt-4.1-mini
```bash
jupyter notebook hotpotqa_reprod.ipynb
```
