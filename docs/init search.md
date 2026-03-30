1. Fact→Passage Prior o

做法：把 filtered triples 的分数直接回灌到这些 triples 来源的 passage 节点上，作为额外的 seed prior。
插槽：graph_search_with_fact_entities()
为什么值得先试：HippoRAG 2 的 online retrieval 已经把 query 链到 triples 和 passages，再用 filtered triples 选 phrase seeds，同时所有 passage nodes 也会作为 seeds；这说明“passage seed 权重”本来就是有效杠杆。你只是把 triple 信号更直接地传给对应 passage，不改图结构。HippoRAG 2 还显示 query-to-triple 相比 NER-to-node 平均提升了 Recall@5 12.5%，说明 triple 端本身就是高价值信号。

2. 自适应 passage-node weight

做法：别把 passage_node_weight 设成固定值；根据 query 复杂度动态调，比如：

实体数多 / filtered triples 多：降低 passage 先验，强调图传播
没有 triples / triples 很弱：提高 dense passage 权重
插槽：graph_search_with_fact_entities()
灵感来源：HippoRAG 2 本来就把 passage nodes 作为 seed，并专门讨论了 weight factor；Q-PRM 则说明 query rewriting/处理应当随 query 复杂度变化，而不是固定规则。
3. Dense + Graph 的 RRF 融合 x

做法：保留原始 PPR 排名，再和 dense passage retrieval 排名做一次 RRF / Borda / 加权和 融合。
插槽：PPR 输出 top passages 后
为什么容易涨：HippoRAG 2 已经证明 passage 节点和图搜索是互补的；RankRAG 的核心也说明“在初始检索不完美时，context ranking 仍然很关键”。你这一步不训练，只做后融合。

4. Query Decomposition 多查询检索 x

做法：先让 LLM 把问题拆成 2 个子问句或 hop 目标，然后：

原 query 跑一遍
每个子 query 跑一遍
把 passage 排名做 union + fusion
插槽：retrieve 前面套一层 query rewrite
为什么适合 MuSiQue：多跳题常常不是“找不到”，而是一个 query 压不出两跳证据。Q-PRM 说明 query rewriting 本身就是 RAG 关键部件；RAT 则展示了“按步骤生成/修正 thought，再做 retrieval”这个思路可行。
5. Step-back Query + 原 Query 双路检索

做法：除了原问题，再生成一个更抽象的“上位问题”，例如把
“X 的出生地属于哪个 county”
改成
“先找 X 的出生地，再找该地所属 county”
然后两路检索结果融合。
插槽：query rewrite 层
优点：比完整 decomposition 更轻，通常只多一次 LLM 调用。
适合场景：HotpotQA / MuSiQue 里那种“词面命中但逻辑链不完整”的题。Q-PRM 和 RAT 都支持这种“中间步骤引导检索”的方向。

6. Predicate-aware Phrase Weighting

做法：现在 HippoRAG 2 主要根据 filtered triples 给 phrase node 分权。你可以进一步把 predicate 词 也引入打分，比如：

query 和 predicate 的 embedding 相似度
query 中关系词与 predicate 的 lexical overlap
包含 wh-word 对应关系模板的加分
插槽：rerank_facts() 或 phrase seed weight 计算
为什么合理：HippoRAG 原论文就把“允许 relation 直接引导 graph traversal”列为后续改进方向；HopRAG 的标题和路线也是 logic-aware multi-hop retrieval。你不需要改图，只是把 relation 信息更充分用于 seed 计算。
7. HopRAG-lite：局部扩展 + 剪枝 x

做法：先拿原 HippoRAG 2 的 top-M passages，再基于它们共享的 phrase/triple 邻居做 一跳局部扩展，然后用简单规则剪枝：

fact overlap
entity coverage
dense similarity
是否形成两跳链
插槽：PPR 之后，QA 之前
为什么值得试：KG²RAG 明确采用“seed chunks -> KG-guided chunk expansion -> chunk organization”；HopRAG 也是 logic-aware 的多跳推理路线。你不重建 KG，只是在已有结果上做局部扩展。
8. Coverage-aware Rerank o

做法：对 top-K passages 重新打分，奖励“覆盖更多 filtered triples / 更多 query entity / 更多 hop bridge entity”的 passage。
一个简单分数可以是：
final = ppr_score + α*triple_coverage + β*entity_coverage
插槽：PPR 后 rerank
为什么容易实现：需要的数据你基本都有：filtered triples、query entities、top passages。
为什么可能有效：HippoRAG 2 的剩余错误很多来自 triple filtering 和 graph search 后的证据未对齐；coverage rerank 是最轻量的纠偏。

9. Path-consistency Rerank

做法：不只看单 passage 分数，而看 top passages 之间能不能拼成链。
例如：

passage A 含实体 e1,e2
passage B 含 e2,e3
那 A+B 组合优先。
插槽：PPR 后组合打分
本质：让最终送进 QA 的证据集更像“链”而不是“散点”。
灵感来源：HopRAG 是 logic-aware multi-hop retrieval；KG²RAG 强调用图关系组织 chunks。
10. MAIN-RAG 风格的后过滤

做法：对 top-10 / top-20 passages 做一次轻量过滤，只保留“高相关、非噪声”的文档。
可以有三个版本：

单 LLM 判别
双 LLM 投票
规则 + LLM 混合
插槽：PPR 后，QA 前
为什么值得试：MAIN-RAG 明确针对 noisy retrieval documents，且是 training-free。HippoRAG 2 论文也承认 graph search / filtering 还会引入错误证据。
11. BRIEF-style Evidence Compression x

做法：不要把 top-5 passages 原样拼给 reader；先把它们压成一个短证据块：

Fact 1
Fact 2
Bridge
Candidate answer
再让 QA 模型只读这个压缩证据。
插槽：qa() 前
为什么很稳：BRIEF 直接指出，多跳问题里检索文档一多，输入长度线性增长，长上下文理解反而下降；这正适合你现有 pipeline 的“retrieval 还行但 QA 读不明白”的情况。
12. 一次性 RAT-lite 第二轮检索

做法：只做 一轮 额外 retrieval，不做完整 agent。流程是：

首轮检索 + 读 top passages
让 LLM 写一句“还缺什么信息”
用这句话当新 query 再检一次
融合两轮结果
插槽：PPR/QA 之间
为什么合适：RAT 的核心就是“用当前 thought 修正后续 retrieval”。你不用做完整 iterative reasoning，只做一轮就够当初步研究。
我最建议你先试的 5 个

按“实现快 + 最像会涨”的顺序，我建议先试：

Fact→Passage Prior
Dense + Graph RRF 融合
Coverage-aware Rerank
MAIN-RAG 风格后过滤
BRIEF-style Evidence Compression

这 5 个有一个共同点：都不动图结构，只改在线打分和证据组织。其中前 3 个更可能先涨 Recall@5，后 2 个更可能先涨 EM/F1。HippoRAG 2 在 MuSiQue、2Wiki、HotpotQA 上的在线检索本来就依赖 query-to-triple、triple filtering、phrase/passages seed 与 PPR 的组合，所以这些改法都和它的主干方向一致。

你可以怎么安排实验顺序

我建议别乱试，按这个梯度来：

第一梯队：纯打分改动
Fact→Passage Prior、Adaptive passage weight、RRF、Coverage rerank

第二梯队：轻量 reasoning 改动
Query decomposition、Step-back query、Predicate-aware weighting、Path-consistency

第三梯队：读者端改动
MAIN-RAG filter、BRIEF compression、RAT-lite second retrieval

这样做的好处是，你能很快判断问题到底主要卡在：

seed 初始化
graph search 后排序
还是 QA 读者
一个很重要的提醒

这些方案里，不是每个都适合先看完整 QA 指标。

更稳的做法是分两类看：

改 seed / rerank 的方案，先看 Recall@2 / Recall@5
改 compression / filtering / second-pass QA 的方案，重点看 EM / F1

不然你会把“检索涨了但 reader 没吃进去”和“检索没涨但 reader 更会读了”混在一起。