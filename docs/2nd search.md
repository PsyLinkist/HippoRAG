我先给你一个判断：你现在这 1.3% overall F1 提升，说明你抓到的方向是对的。对 HippoRAG / HippoRAG 2 这条线来说，真正决定多跳表现的，往往不是“再召回一点相似文本”，而是让最终 top-K 更完整地覆盖 reasoning chain。原始 HippoRAG 本来就主要在 MuSiQue、2Wiki、HotpotQA 上验证，且作者明确说 2Wiki 更偏实体型、HotpotQA 的知识整合要求相对更弱；HippoRAG 2 又进一步证明了 query-to-triple、passage node、triple filter 都是有效增益点，但 recognition memory 本身仍然脆弱，甚至会出现过滤后零 triples 的情况。

我也看了你上传的代码，你这版现在有几个很明显的瓶颈：linking_top_k=5，DSPy filter prompt 里还写了“最多保留 4 条 fact”，graph_search_with_fact_entities() 只给 subject/object 设 seed、不用 predicate，passage_node_weight 固定成 0.05，而且你现在的 coverage-aware rerank 只对前 50 个 graph candidates 生效。基于这些点，我给你 15 条更值得做的改法，按“简单有效优先”排：

自适应扩大 fact 候选池。x
把 linking_top_k 从固定 5 改成动态 5/10/15：当 top fact 分数很平、或者 query 比较长、或者实体数>2 时，自动放大候选池。你现在 top-5 太紧了，容易把真正的 bridge fact 挤掉。这个改动最小，通常最先见效。
把 hard filter 改成 soft filter。x
不要只让 LLM 输出“保留/丢弃”，而是输出每条 fact 的 relevance score 或者 3 档标签，再把这个分数直接乘到 seed weight 上。因为 HippoRAG 2 自己的分析就显示，triple filter 经常把有用信息筛没，18% 样本过滤后变成 0 triples。
去掉“最多保留 4 条 fact”的 prompt 上限。x
这是你当前代码里最不合理的硬约束之一。MuSiQue 里不少题本来就不止 2-hop，4 条 fact 很容易不够。建议改成“保留所有高相关 fact，但上限设为 8 或 10”，再交给后续 graph search 自然衰减。
把 predicate 也纳入 seed，而不是只 seed subject/object。
你现在 graph_search_with_fact_entities() 只用头尾实体，等于把 relation 信息浪费了。原始 HippoRAG 论文结尾就明确点过，后续一个自然方向就是“让 relation 直接引导 graph traversal”。
最省事的实现是：先不给 relation 真建节点，只额外算一个 predicate-query 相似度，用来放大/缩小对应 fact 的 subject/object seed weight。
把静态 PPR 改成 query-aware edge weighting。
不是所有邻边都该被同样传播。你可以在每次 query 时，对边权做一次轻量重标定：
w(u,v|q)=base_w * sim(q, edge/predicate/u,v)。
这和最近的 QAFD-RAG 很像，核心是让扩散跟着 query 语义走，而不是盲目沿图散开。对 MuSiQue 这类“错一跳就全错”的题尤其有用。
把 passage_node_weight=0.05 改成 query-adaptive。
HippoRAG 2 论文里 0.05 是 validation 上的默认好点，但不是每道题都适合固定值。
你的实现里可以用一个很简单的门控：
当 fact filter 很自信时，减小 passage node weight；
当 fact filter 很不自信或 top facts 很散时，增大 passage node weight。
这会比全局固定 0.05 更稳。
不要只在“0 facts”时 fallback dense retrieval，要做 confidence-gated fallback。
REAR 的核心就是“判断当前检索结果到底靠不靠谱，再决定怎么用外部知识”。
你这里可以在三种情况下直接混回 dense 排序：
fact 分数过平；
filter 后剩余 triples 太少；
top passages 的 bridge 覆盖太低。
这会比现在“非空就硬走图”更稳。
把 passage rerank 从 pointwise 改成 setwise / set-cover。
你现在仍然是给每个 passage 单独打分。更好的方式是直接优化“前 K 篇合起来覆盖多少 triples / query entities / bridge entities”。
很简单的贪心版就够：每次从剩余 passage 里选那个“新增覆盖最多 + base score 最高”的。
这通常比单篇 rerank 更贴近多跳 QA 的真实目标。
在你现有 coverage-aware rerank 上，再加一个“bridge centrality”项。
不只是看 passage 命中了多少 entity，还要看它命中的 entity 是否同时出现在多个 retained facts 里。
也就是把
entity_coverage
升级成
bridge_entity_coverage + bridge_entity_centrality。
这样可以更强地区分“普通相关实体”和“真正连通两跳的桥”。
做一个轻量版 PropRAG：在 graph shortlist 上加 beam-search path discovery。
PropRAG 的关键不是 PPR 本身，而是它在局部子图里显式找 reasoning path，再用 path 反过来更新 seed 和 passage 排名。
你完全没必要整篇重写，只要：
先用现有 graph/PPR 召回局部子图；
再在局部图上做 2-hop/3-hop beam search；
把参与高分路径的 passages 再加一层 bonus。
这是非常像论文点的改法。
把 triple 表示升级成 proposition / statement 表示。
PropRAG 的一个核心结论是：triple 容易 context collapse，而 proposition 更保留语义上下文。
你不用把全库都 propositionize，先只对 top facts 对应的 source passages 做 statement extraction 即可。
这是“小范围加上下文”的版本，成本可控。
做 chunk + statement 的双层索引，而不是只靠 passage + triple。
HLG/StatementGraphRAG 这类新方法的思路，是把 atomic proposition、topic、entity/relation 分层组织，再在层间检索。
对你来说，最实用的简化版是：
第一层还是原始 passage；
第二层只给 top candidate passages 生成 statements；
第二层只在 hard query 上触发。
这样能增强多跳细粒度，不用重建全图。
加 query rewrite / missing-topic augmentation。
Chain-of-Rewrite 的思路很适合你这类图检索：先分析问题缺了什么 topic，再补成更适合检索的 query，甚至可以把 rewrite 结果再反馈给 rerank。
你可以做最轻版本：
原 query 检索一次；
如果 coverage 太低，就生成 1 个 bridge-oriented rewrite；
两次结果做 union 或 rerank。
这类改动对 Hotpot 往往也有帮助。
强化 synonym / alias canonicalization，而不是只靠 embedding 阈值。
原始 HippoRAG 已经显示 synonym edges 尤其对 2Wiki 很关键。
你可以额外加几条廉价规则：
标题去括号；
罗马数字/缩写归一；
日期和地名别名归一；
人名首字母展开；
Wikipedia title redirect 风格归一。
这类规则经常比继续调 embedding threshold 更划算。
离线 pruning：先删低价值 chunk / fact，再建更干净的图。
最近有一类工作专门做 graph-guided concept selection，核心结论就是：不是所有 chunk 都值得进图，先挑 salient chunks 再建图，成本和噪声都能一起降。
你这条线也适用，尤其 HotpotQA 里 distractor 很多。
最简实现：
对每个 chunk 先算 entity density、relation density、IDF-weighted concept score；
低于阈值的 chunk 不进 fact graph，只保留 dense 检索用。
这属于“删冗余代码/冗余图节点也能提分”的典型方案。

如果你现在要我帮你排一个“最可能继续涨分、且改动不大”的执行顺序，我建议是：先做 1、2、4、6、8；第二批做 10、11、13。这样最像是在你现有 coverage-aware rerank 基础上继续往上抬，而不是重开一套系统。