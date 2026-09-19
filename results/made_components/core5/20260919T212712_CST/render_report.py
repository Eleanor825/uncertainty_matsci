"""Render descriptive tables from the fixed local analysis; no scientific calls."""
from pathlib import Path
import csv
import json

D=Path(__file__).resolve().parent
s=json.loads((D/'summary.json').read_text())
v=json.loads((D/'study_inputs.json').read_text())
rows=lambda name:list(csv.DictReader((D/name).open()))
BASE='baseline_reference';UQ='uq_only_support_aware';ES='es_only_independent';FULL='full_support_aware'
ARMS=[BASE,UQ,ES,FULL];LABEL={BASE:'Baseline',UQ:'UQ/controller',ES:'ES-only',FULL:'Full'}
fmt=lambda x:'NA' if x in (None,'') else f'{float(x):.6f}'
signed=lambda x:f'{float(x):+.6f}'
runs={(r['arm'],r['task_id'],int(r['seed'])):r for r in rows('runs.csv')}
systems={(r['arm'],r['task_id']):r for r in rows('per_system_seed_statistics.csv')}
aggregate={r['arm']:r for r in rows('aggregate_seed_statistics.csv')}
contrast={r['contrast']:r for r in rows('contrast_aggregate_statistics.csv')}

lines=['# MADE固定五体系：B10四臂×三评价seed，60/60完整消融','',
f"实际观测截至 **{s['observed_CST']}**。这60条核心评价已全部完成；整个新研究为 **279/1,080完成**，并未结束。原56/60快照保留不变，其56条已完成结果的SUN、AUDC、曲线、成本和receipt/result SHA与本次完全一致；本次新增4条原先缺失的UQ结果。",'',
'**UQ/controller在该固定核心集合的平均结果高于baseline，但完整方法仍低于baseline；ES的算法级对比为负。不能据此声称完整方法有效、内部特征NN贡献已独立成立，或UQ带来了稳定的跨seed改善。**','',
'## 完整四臂结果','',
'模型为同一个Qwen3.5-4B初始化，固定5体系与评价seeds2/3/4，每条B10。每臂15条。SUN总数是逐轨迹计数之和，不是跨seed去重后的材料并集；mean AUDC是等权15条均值，也等于先对每seed固定5体系求宏平均、再对3seed求均值。','',
'| 方法 | 完成 | 15条SUN总和 | 每条平均SUN | 平均AUDC |','|---|---:|---:|---:|---:|']
for a in ARMS:
    r=s['arms'][a];lines.append(f"|{LABEL[a]}|15/15|{r['SUN_total']}|{fmt(r['mean_per_run']['SUN'])}|{fmt(r['mean_per_run']['AUDC'])}|")
lines+=['',
'UQ−baseline的SUN总差为+3，AUDC均值差为+0.036667；full−baseline为SUN−2、AUDC−0.012000。UQ−full为SUN+5、AUDC+0.048667。以上为同15个task/seed的配对描述，不是不同分母间的比较。','',
'## 完整逐体系、逐seed原值','',
'每格为 `SUN / AUDC`，未删除零发现或负对比。原job ID、receipt/result SHA和660个官方曲线点见 [runs.csv](runs.csv)、[curves.csv](curves.csv)。','',
'| 体系 | seed | Baseline | UQ/controller | ES-only | Full | UQ−B AUDC | Full−B AUDC |',
'|---|---:|---:|---:|---:|---:|---:|---:|']
for task in s['systems']:
    for seed in s['seeds']:
        rr=[runs[a,task,seed]for a in ARMS]
        cells=[f"{r['SUN']} / {float(r['AUDC']):.2f}"for r in rr]
        lines.append('|'+task+'|'+str(seed)+'|'+'|'.join(cells)+f"|{signed(float(rr[1]['AUDC'])-float(rr[0]['AUDC']))}|{signed(float(rr[3]['AUDC'])-float(rr[0]['AUDC']))}|")
lines+=['','## 各体系跨3seed均值与样本方差','',
'以下每格为 `均值 (样本方差s²)`；分母n−1=2。SD及全部三seed向量另见 [per_system_seed_statistics.csv](per_system_seed_statistics.csv)。这些方差沿同一体系、预算、方法的评价seed变化计算，不是体系之间的方差，也不是重复训练的方差。','']
for m in ['SUN','AUDC']:
    lines += [f'**{m}**','', '| 体系 | Baseline | UQ/controller | ES-only | Full |','|---|---:|---:|---:|---:|']
    for task in s['systems']:
        cells=[f"{fmt(systems[a,task][m+'_mean'])} ({fmt(systems[a,task][m+'_sample_variance'])})"for a in ARMS]
        lines.append('|'+task+'|'+'|'.join(cells)+'|')
    lines.append('')
lines += ['**正负体系均须保留。** UQ−baseline的SUN和AUDC体系均值均为2正/2平/1负：Au–K–Tb与Co–Dy–W改善，Al–Li–V下降，Al–V–Zn与Co–Mg–Na打平。Al–V–Zn内部seed2/3改善、seed4下降，平均恰好抵消。', '',
'UQ的SUN总增量分解为Au–K–Tb +2、Co–Dy–W +3、Al–Li–V −2，其余0；AUDC总增量为+0.36、+0.33、−0.14，其余0，合计+0.55，再除15得到+0.036667。', '',
'Full−baseline的体系平均AUDC为4正/1负，但Al–V–Zn的3seed AUDC总差−0.62大于其余4体系合计+0.44，整体仍为−0.18/15=−0.012。SUN为2正/1平/2负。因此“改善体系更多”不等于平均发现效率改善。','',
'## 固定五体系宏平均：跨评价seed的方差','',
'每seed聚合相同5体系，再对三个聚合值计算均值、样本方差和SD。这里没有把15个task/seed当作15次独立训练。','',
'| 方法 | seed2/3/4 SUN总数 | 总SUN均值 | 总SUN s² | 总SUN SD |',
'|---|---|---:|---:|---:|']
for a in ARMS:
    r=aggregate[a];lines.append(f"|{LABEL[a]}|{s['arms'][a]['per_seed_SUN_total']}|{fmt(r['SUN_total_mean'])}|{fmt(r['SUN_total_sample_variance'])}|{fmt(r['SUN_total_sample_SD'])}|")
lines+=['', '| 方法 | seed2/3/4 AUDC宏平均 | 均值 | s² | SD |','|---|---|---:|---:|---:|']
for a in ARMS:
    r=aggregate[a];vector=', '.join(f'{x:.3f}'for x in s['arms'][a]['per_seed_AUDC_macro']);lines.append(f"|{LABEL[a]}|[{vector}]|{fmt(r['AUDC_macro_mean_mean'])}|{fmt(r['AUDC_macro_mean_sample_variance'])}|{fmt(r['AUDC_macro_mean_sample_SD'])}|")
lines+=['',
'UQ的三seed总SUN为[10,11,4]，baseline为[8,7,7]，配对差[+2,+4,−3]；AUDC宏平均为[0.224,0.238,0.116]，baseline为[0.152,0.154,0.162]，差[+0.072,+0.084,−0.046]。seed4明显退步，UQ的样本方差也更大；这里不能使用“稳定提升”的结论。n=3仍很小，这些数是描述性重复评价，未给出显著性或泛化保证。','',
'## UQ-on/off与ES-on/off：算法级配对','',
'B=θ0无controller；U=θ0加完整support-aware UQ/controller；E=从θ0独立训练的无UQ ES策略θE；F=在UQ/controller下训练的策略θF。E与F分别训练，权重不同。因此F−E同时包含训练信号和测试controller变化，不能当成“固定LLM权重只切换uncertainty”的因果消融。U−B可比较固定θ0下整套controller，但仍把NN、支持域判定、schema/重试/筛选等合在一起。','',
'下面平均差均按相同15个task/seed计算；AUDC差的s²沿3个固定五体系seed宏平均差计算。完整135条对比、45组体系统计和27组seed聚合见相关CSV。','',
'| 对比 | 每条平均SUN差 | 平均AUDC差 | seed2/3/4 AUDC宏差 | 配对宏差s² |',
'|---|---:|---:|---|---:|']
for name in ['UQ_minus_base','ES_minus_base','full_minus_UQ','full_minus_ES','full_minus_base','UQ_minus_ES']:
    r=contrast[name];c=s['contrasts'][name];vector=', '.join(f'{x:+.3f}'for x in c['per_seed_AUDC_macro_delta']);lines.append(f"|{name}|{signed(c['mean_per_run_delta']['SUN'])}|{signed(c['mean_per_run_delta']['AUDC'])}|[{vector}]|{fmt(r['AUDC_macro_mean_delta_sample_variance'])}|")
lines+=['',
'- 无UQ条件下，E−B的AUDC均值差−0.024667；带UQ/controller条件下，F−U为−0.048667。本固定集合没有显示ES更新提高最终平均效率。',
'- U−B为+0.036667；F−E为+0.012667。后者的两套学习权重不同，不能把它全部归因于测试时的风险controller。',
'- Full−B依然为−0.012000；完整方法效果未被这一核心消融证明。','',
'若把四个完整算法写成2×2编码，可作以下纯代数描述。这里的“平均UQ/ES contrast”不是独立因果主效应，也不是同权重NN开关试验。','',
'| 描述性算法对比 | 定义 | 每条平均SUN差 | 平均AUDC差 |',
'|---|---|---:|---:|']
definitions={'average_UQ_algorithm_contrast':'0.5[(U−B)+(F−E)]','average_ES_algorithm_contrast':'0.5[(E−B)+(F−U)]','algorithm_interaction':'(F−U)−(E−B)'}
for name,definition in definitions.items():
    c=s['contrasts'][name];lines.append(f"|{name}|{definition}|{signed(c['mean_per_run_delta']['SUN'])}|{signed(c['mean_per_run_delta']['AUDC'])}|")
lines+=['',
'交互对比的SUN平均为0、AUDC为−0.024；这只是四个算法成绩的差分。它不能证明NN与ES存在某种因果拮抗，尤其两个ES分支只各训练了一套已固定策略，缺乏重复训练以及固定权重的controller交叉试验。','',
'**NN贡献仍未隔离。** 当前“UQ/controller”整个算法包的正平均结果，并不能证明内部activation/attribution特征已带来独立增益；还可能涉及确定性规则、支持域abstention、retry或筛选变化。这个结果快照也没有逐决策risk触发统计，不能补写未观测的机制。这里没有用test重校准风险、改阈值、选择新的checkpoint或重跑失败轨迹。','',
'## 已记录成本与全量进度','',
'每臂候选预算相同，但不等于计算/工具成本相同。下表是15条rollout记录的求和，wall时间是轨迹耗时总和，不是多卡研究的日历耗时；graph时间与wall时间不能相加。','',
'| 方法 | 候选attempts | 初始化attempts | Σrollout wall秒 | Σgraph秒 | surrogate计数有值的行 |','|---|---:|---:|---:|---:|---:|']
for r in rows('cost_summary.csv'):
    lines.append(f"|{LABEL[r['arm']]}|{r['candidate_oracle_attempts_sum']}|{r['initialization_oracle_attempts_sum']}|{float(r['wall_seconds_sum']):.2f}|{float(r['graph_seconds_sum']):.2f}|{r['surrogate_oracle_attempts_observed_n']}/15|")
lines+=['',
'完整核心候选attempts为600。部分原行缺 `surrogate_oracle_attempts`，四臂完整surrogate总量保持NA，不能把缺字段当0或拿可见子集和当完整MACE成本。不同attempt计数也不能直接相加宣称独立物理查询总数。详见 [timings_and_costs.csv](timings_and_costs.csv)、[cost_summary.csv](cost_summary.csv)。','',
'| 新1,080矩阵的方法 | 已完成 /270 | 已领取未完成 | 尚未领取 |','|---|---:|---:|---:|']
for a in v['arms']:
    p=a['all1080_arm_progress'];lines.append(f"|{LABEL[a['arm']]}|{p['complete']}|{p['claimed_no_result']}|{p['pending']}|")
lines+=['',
'本次为279完成、10已领取未完成、791待领取、0已记录失败。ES-only的90个B10已完成，原队列已领取Al–Li–V/seed2/B30；不能把“核心60/60”表述为全30体系或B10/B30/B50主实验完成。原180条seed1研究、SnAr35条及开发集G0诊断均不并入这里。','',
'## 来源、重算与限制','',
'固定体系来自原五体系扩展和事先已有60个task/seed/arm ID，并非按本次成绩选取。原五体系扩展前已有两个seed1结果，因此不能声称该subset从研究起点就完全前瞻未见。当前评价seed2/3/4与训练seed职责区分；最终仍需报告全30体系的注册矩阵。','',
'[provenance.json](provenance.json) 固定源快照SHA和科学投影SHA。原始快照中的主机、进程、登录和资源操作信息未进入包。每条科学原值、曲线和receipt/result hash都保留；此次只复核本地已采投影，没有重新执行完整远端物理验收。','',
'执行 `python3 -B analyze.py` 可用Python标准库重算所有CSV/JSON并核对文件哈希；`--write`仅重新生成派生分析文件。`render_report.py`从这些表生成报告，不调用模型或oracle。样本方差另用成对差值恒等式复核，算法差分也检查代数恒等关系。原56/60与所有负结果不覆盖。本包已作为新的独立快照发布；原56/60报告与其他历史结果保持原样。原分析与导出哈希分别记录在 publication_provenance.json。','']
(D/'report.md').write_text('\n'.join(lines))
(D/'README.md').write_text('''# Complete fixed-core MADE B10 ablation: five systems × three seeds × four arms

Read [report.md](report.md) and [summary.json](summary.json). All60 core cells are complete, while the larger registered study is279/1,080. UQ/controller has a positive core mean difference; full remains below baseline. UQ seed4 declines. The two ES policies were trained separately, and the NN contribution is not isolated.

Run `python3 -B analyze.py` to recompute and byte-verify the local cached tables and manifest. No model, graph, oracle or network is invoked. `--write` recreates only derived tables; `python3 -B render_report.py` renders the report from them. All variances use ddof=1 over evaluation seeds within fixed systems, or over complete five-system seed aggregates. They are not between-system variance or repeated-training variance.

The prior56 completed core results are unchanged; the four formerly missing UQ rows are now present. This package remains separate from the older180 seed1 study, SnAr and development diagnostics.
''')
print(json.dumps({'report_rendered':True,'new_scientific_calls':0}))
