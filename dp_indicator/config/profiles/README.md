# Target Profile 说明

## 用途

`config/profiles/` 目录存放靶点专属知识文件。系统启动时会自动扫描并热加载，为该靶点的检索、证据分级、推理各阶段提供背景知识和引导。

## 命名规则

- 文件名: `{TargetName}.yaml`(大小写敏感)
- 示例: `Kv1.3.yaml`, `TRPV1.yaml`, `BRAF.yaml`

## 完整模板

见 `TEMPLATE.yaml`，复制后重命名为 `{YourTarget}.yaml` 即可。

```yaml
target_id: "YourTarget"
official_name: "GeneName (full name)"

cell_type_expression:
  cell_type_1:
    level: high
    dependency: target-dominant

functional_chain:
  - Step 1 → Step 2 → Step 3

therapeutic_window:
  suppressed: [cell_type_1]
  spared: [cell_type_2]

mechanistic_bridges:
  - axis: mechanism_name
    cell_types: [cell_type_1]
    mechanism: "描述该机制轴如何连接靶点到疾病"

mechanism_scope:
  description: "该靶点在本profile中建模的生物学角色范围(见下方字段说明)"
  out_of_scope_keywords:
    - "超出建模范围的关键词1"

intersection_guidance: >-
  指导LLM如何进行交集分析的文本

reasoning_guidance:
  - "推理指导 1"
  - "推理指导 2"

supplemental_target_disease_context: >-
  补充给推理agent的靶点背景知识(自然语言段落)

excluded_indications: []
key_publications: []
```

## 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `target_id` | ✅ | 靶点标识符 |
| `official_name` | ✅ | 官方名称/基因名 |
| `cell_type_expression` | ✅ | 靶点在不同细胞类型中的表达水平和功能依赖，用于判断"治疗窗" |
| `functional_chain` | ✅ | 靶点激活后的功能级联链(5-7步)，从分子事件到细胞表型 |
| `therapeutic_window` | ✅ | 被抑制的细胞类型 vs 被保留的细胞类型，体现靶点选择性 |
| `mechanistic_bridges` | 可选 | 机制桥接轴：从靶点的细胞类型选择性推导到疾病层面的中间机制描述，供Reasoner在无直接证据时进行机制外推。同时是L6检索层查询词的唯一来源(见下方说明)，不再有单独的检索关键词字段需要维护 |
| `mechanism_scope` | 可选 | 声明本profile实际建模的生物学角色范围，在两处生效：(1) `description` 注入`build_biological_context()`渲染为"Mechanism Scope Boundary"区块，直接进入Reasoner的prompt，明确告知其本profile只建模某一种生物学角色(例如免疫调节)，靶点的另一独立角色(例如神经兴奋性)不在建模范围内，Reasoner被要求不得提出主要病理机制依赖于该"未建模角色"的候选适应症；(2) `out_of_scope_keywords` 列出与"超出范围角色"相关的关键词，用于在检索阶段过滤基因层面的疾病关联数据库(OpenTargets/KEGG等)返回的、实际上来自靶点另一独立功能角色的疾病关联。两处生效范围完全一致、均为目标机制无关的通用过滤，不针对任何特定候选疾病 |
| `intersection_guidance` | 可选 | 交集分析引导文本：告诉Reasoner如何寻找"机制强相关、但靶点未被直接研究过"的疾病方向 |
| `reasoning_guidance` | 可选 | LLM推理引导(列表)，每条是一个具体的推理原则，直接拼入Reasoner的prompt |
| `supplemental_target_disease_context` | 可选 | 自由格式的补充背景知识段落，直接注入Reasoner的prompt，用于提供本profile结构化字段之外的机制细节 |
| `excluded_indications` | 可选 | 已被广泛研究/临床验证过的适应症列表，Reasoner会避免重复提出这些方向，聚焦真正新颖的假设 |
| `key_publications` | 可选 | 关键文献列表，每条包含 `pmid` 和 `claim`(该文献支持的具体论断) |

## 知识库合成(KnowledgeSynthesizer)与本文件的关系

Explore阶段完成证据检索和分级后，会读取*全部*入选证据(不只是评分最高的一小部分)，逐条抽取机制事实，汇总成一份知识库，随后连同本profile的 `reasoning_guidance` / `supplemental_target_disease_context` 一起交给Reasoner。这意味着 profile 里的字段是"引导"而非"限定"——真正决定候选适应症的是证据本身覆盖到的机制广度，profile 字段只是帮助Reasoner用什么视角去解读这些证据。

## 加载优先级

1. `config/profiles/{Target}.yaml` — 热加载，无需重启，是靶点特异知识的唯一来源
2. `default` — 未匹配到任何YAML文件时(包括故意删除YAML做消融测试的情况)，退化为完全通用、不含任何机制/细胞类型特异内容的空模式

`core/target_knowledge.py` 不含靶点特异的硬编码fallback。删除某个靶点的YAML文件即可实现干净的"无先验知识"消融，不会有代码层面的隐藏内容顶替。`agents/core_agents.py` 中 `EvidenceFilter` 的证据预过滤机制关键词同样从当前YAML profile动态派生(`functional_chain`/`cell_type_expression`/`mechanistic_bridges`)；YAML缺失时只使用通用生物医学词表，不含细胞类型或机制特异词。

L6机制桥接检索层的查询词由 `get_bridge_search_terms()` 从 `mechanistic_bridges` 的结构化字段(axis名称 + cell_types列表)按固定规则程序化生成。检索查询词与Reasoner读到的机制桥接描述来自同一份结构化数据，不存在检索层单独调优用词、使其偏向某个特定疾病词汇的空间。

L6使用短查询：每条axis生成 `{靶点} {axis名称}` 一条，并对该axis下每个cell_type各生成 `{靶点} {单个cell_type}` 一条。PubMed/Europe PMC把多词查询视为所有词的隐式AND，因此短查询能避免要求一篇文献同时包含多种细胞类型。该规则与具体靶点或疾病无关，并保持程序化派生、无人工疾病措辞的边界。

## `mechanistic_bridges` 的收录原则(重要)

`mechanistic_bridges` 是本文件中信息密度最高、也最容易被写成"指向性内容"的字段——因为每条bridge的`mechanism`描述都是自然语言，理论上可以写到任意具体程度。撰写/审查这个字段时应遵循以下原则：

1. **每条axis必须是该靶点某个通用生物学功能维度的描述，不能是某个候选疾病的机制片段。** 判断标准：把axis名称和`mechanism`文本单独拿出来，如果读者能直接猜出它对应哪个具体疾病(尤其是猜出组织/器官/免疫复合物类型/抗体亚型)，这条axis的措辞就过细了，应改写为更上游、更通用的生物学描述，或直接删除。
2. **优先保留数量少、跨疾病通用的axis，而不是数量多、逐一贴合某类候选疾病的axis。** 例如"体液免疫(抗体分泌细胞选择性)"和"巨噬细胞极化(组织炎症选择性)"这两条适用于数十种抗体介导/炎症性疾病，属于应当保留的通用axis；而"某类T细胞辅助信号如何决定特定抗体同种型选择"这种描述已经把机制收窄到接近某一个具体疾病的程度，应当删除或至少大幅泛化改写。
3. **允许(且推荐)加入完全抽象、不含任何疾病或器官名称的"汇聚原则"作为`reasoning_guidance`条目**，用来表达"同时命中N个独立机制维度的疗法，在机制上区别于只命中单一维度的现有疗法"这一类通用药理学判断标准，而不是"疾病X具备维度A和维度B所以应该被提出"这种反向工程式的表述。汇聚原则本身必须能同等适用于任意候选疾病，不能只为了让某一个特定疾病更容易被提出而设计。
4. 是否遵循了以上原则，可以用消融测试来检验：删除某条axis后，如果某个特定候选疾病消失而其他候选疾病基本不受影响，说明这条axis对该疾病的贡献接近"专属通道"，需要重新审视其措辞是否过细；如果某条axis删除后多个不同类型的候选疾病同时受到类似程度的影响，说明它确实是通用机制维度。

## 清除缓存

修改 YAML 后，如需立即生效，删除 `checkpoints/stages/` 目录下的 checkpoint 文件后重新运行(这些文件是运行时缓存，不需要随代码一起分发)。
