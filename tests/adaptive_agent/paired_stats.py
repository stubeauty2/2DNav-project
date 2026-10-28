"""Offline paired statistics for a fixed runtime and matched experiment settings."""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT=Path(__file__).resolve().parents[2]/'out/16_adaptive_agent/20261027'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def wilson(successes,n):
    if n<=0 or not 0<=successes<=n:raise ValueError('Invalid success count')
    z=1.959963984540054;p=successes/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.,center-half),min(1.,center+half)]


def summarize_pairs(rows,seed=20261102):
    pairs={}
    for row in rows:
        key=(row['dataset'],row['episode_id']);arm=row['arm']
        if arm not in ('before','after') or type(row['success']) is not bool:raise ValueError('Invalid outcome')
        if arm in pairs.setdefault(key,{}):raise ValueError('Repeated episode arm')
        pairs[key][arm]=row
    if not pairs or any(set(p)!= {'before','after'} for p in pairs.values()):raise ValueError('Complete pairs required')
    n=len(pairs);wins=losses=both=neither=0
    for pair in pairs.values():
        a,b=pair['before']['success'],pair['after']['success']
        wins+=not a and b;losses+=a and not b;both+=a and b;neither+=not a and not b
    discordant=wins+losses
    pvalue=min(1.,2*sum(math.comb(discordant,k) for k in range(min(wins,losses)+1))/2**discordant) if discordant else 1.
    rng=np.random.default_rng(seed)
    samples=rng.multinomial(n,[wins/n,losses/n,(both+neither)/n],size=50000)
    interval=np.quantile((samples[:,0]-samples[:,1])/n,[.025,.975]).tolist()
    result=dict(n=n,before_success=both+losses,after_success=both+wins,
        before_rate=(both+losses)/n,after_rate=(both+wins)/n,
        before_wilson95=wilson(both+losses,n),after_wilson95=wilson(both+wins,n),
        wins=wins,losses=losses,both_success=both,both_fail=neither,
        paired_rate_difference=(wins-losses)/n,paired_bootstrap95=interval,
        paired_bootstrap_degenerate=interval[0]==interval[1],
        mcnemar_exact_two_sided_p=pvalue)
    for arm in ('before','after'):
        rr=[p[arm] for p in pairs.values()]
        result[arm+'_requests']=sum(r.get('requests',0) for r in rr)
        result[arm+'_api_errors']=sum(r.get('api_errors',0) for r in rr)
        result[arm+'_confirmed_finals']=sum(r.get('termination')=='confirmed_final' for r in rr)
        result[arm+'_confirmed_metric_failures']=sum(r.get('termination')=='confirmed_final' and not r['success'] for r in rr)
        result[arm+'_cases_with_metric_passing_proposal']=sum(r.get('correct_proposal',False) for r in rr)
        result[arm+'_metric_passing_proposals_rejected']=sum(r.get('correct_proposal_rejected',0) for r in rr)
    return result


def condition(plan):
    return dict(scope=plan['scope'],sources=plan['sources'],budgets=plan['budgets'],
        decision_model=plan['decision_model'],visual_model=plan['visual_model'],url=plan['url'],
        thinking_budget=plan.get('thinking_budget',0))


def load_batch(folder):
    plan=read(folder/'plan.json');evaluation=read(folder/'evaluation.json')
    expected={(j['dataset'],j['episode_id'],arm) for j in plan['jobs'] for arm in ('before','after')}
    rows=evaluation['rows'];actual=[(r['dataset'],r['episode_id'],r['arm']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('Missing, repeated or unexpected trajectories')
    for rel,h in plan['sources'].items():
        if hashlib.sha256((folder/rel).read_bytes()).hexdigest()!=h:raise ValueError('Frozen runtime changed')
    return plan,rows


def tables(rows):
    return {ds:summarize_pairs([r for r in rows if r['dataset']==ds]) for ds in ('AVDN','CityNav')}


def report_html(result):
    def interval(x,delta=False):
        return f'[{100*x[0]:.1f}, {100*x[1]:.1f}] 个百分点' if delta else f'[{100*x[0]:.1f}%, {100*x[1]:.1f}%]'
    sections=[]
    for key,title in [('primary','主要结果：本次新增样例'),('combined','补充结果：相同固定版本的历史样例合并')]:
        if key not in result:continue
        body=''
        for ds,x in result[key].items():
            uncertainty=interval(x['paired_bootstrap95'],True)
            if x['paired_bootstrap_degenerate']:
                uncertainty+='；观测差值相同导致区间退化，不能据此证明等效或无不确定性'
            body+=f'''<tr><td>{ds}</td><td>{x['n']}</td><td>{x['before_success']}/{x['n']}<br>{interval(x['before_wilson95'])}</td>
<td>{x['after_success']}/{x['n']}<br>{interval(x['after_wilson95'])}</td><td>{x['wins']} / {x['losses']}</td>
<td>{100*x['paired_rate_difference']:+.1f} 个百分点<br>95% 区间 {uncertainty}</td><td>{x['mcnemar_exact_two_sided_p']:.4f}</td></tr>'''
        detail=''
        for ds,x in result[key].items():
            detail+=f'''<tr><td>{ds}</td><td>{x['before_confirmed_finals']} → {x['after_confirmed_finals']}</td>
<td>{x['before_confirmed_metric_failures']} → {x['after_confirmed_metric_failures']}</td>
<td>{x['before_metric_passing_proposals_rejected']} → {x['after_metric_passing_proposals_rejected']}</td>
<td>{x['before_requests']} → {x['after_requests']}</td><td>{x['before_api_errors']} → {x['after_api_errors']}</td></tr>'''
        sections.append(f'''<section><h2>{title}</h2><table><tr><th>数据集</th><th>配对样例数</th><th>优化前成功数<br>Wilson 95% 区间</th><th>优化后成功数<br>Wilson 95% 区间</th><th>新增成功 / 退步</th><th>配对成功率差</th><th>精确 McNemar p</th></tr>{body}</table>
<h3>模型结论与实际评测分开统计（优化前 → 优化后）</h3><table><tr><th>数据集</th><th>模型确认最终框数</th><th>已确认但指标失败数</th><th>达标候选被拒次数</th><th>模型请求数</th><th>模型 API 错误数</th></tr>{detail}</table></section>''')
    return '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定版本扩样配对验证</title>
<style>body{max-width:1320px;margin:32px auto;padding:0 24px;background:#10141c;color:#e9eef6;font:16px/1.6 system-ui}table{border-collapse:collapse;width:100%;margin:16px 0}th,td{border:1px solid #3b475c;padding:10px;text-align:left}th{background:#202a3a}section{margin:30px 0}h1,h2,h3{line-height:1.3}.note{padding:16px;background:#202a3a}</style>
<h1>固定版本扩样配对验证</h1><p>主要批次：'''+html.escape(result['primary_batch'])+'''。主决策、融合及复核均为 Qwen 3.5；3.8 仅作为可选视觉证据工具。两边使用相同观察点与预算。</p>
<p class="note">本实验限定于目标中心已入图的历史检查点。AVDN 成功阈值为评测方形框 IoU &gt; 0.4，CityNav 为中心距离 ≤ 20 米。距离达标可能落在相邻错误对象上，语义核查另见对应逐例图像报告。模型确认不等于定位正确。</p>'''+''.join(sections)+'''
<p>区间基于独立抽样例的近似假设；成功率差的 95% 区间采用预先约定的 50,000 次配对 bootstrap（种子 20261102），不能将两个独立比例的区间简单相减。全部观测差值相同会产生退化 bootstrap 区间，不能据此认为没有不确定性。McNemar 检验只使用结果不同的配对样例。</p>
<p>主要结果只含本次未使用的新样例。补充汇总纳入已被检查过的相同固定版本旧批次，仅作描述性参考；不同代码版本或推理设置不合并。以上数据不足以直接代表完整导航任务成功率。</p></html>'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary',required=True)
    parser.add_argument('--combine',nargs='*',default=[])
    args=parser.parse_args()
    names=[args.primary]+args.combine
    if len(names)!=len(set(names)) or any(not re.fullmatch('[a-z0-9_-]+',name) for name in names):parser.error('Unique safe batch names required')
    plan,rows=load_batch(ROOT/args.primary)
    result=dict(primary_batch=args.primary,primary=tables(rows),
        note='Metric outcomes only; semantic image audits are separate. Intervals assume independent sampled episodes and are exploratory at small n. Percentile bootstrap intervals can be degenerate when all observed paired differences are equal. Do not infer natural navigation success from visible checkpoints.')
    combined=list(rows);seen={r['episode_id'] for r in rows}
    for name in args.combine:
        other,rr=load_batch(ROOT/name)
        if condition(other)!=condition(plan):raise ValueError('Cannot pool different runtime/model/budget/scope settings')
        added={r['episode_id'] for r in rr}
        if seen & added:raise ValueError('Cannot pool repeated episodes')
        seen|=added;combined.extend(rr)
    if args.combine:
        result.update(combined_batches=names,combined=tables(combined),
            combined_note='Secondary descriptive analysis; the previous smaller batch was already inspected. The new batch is the primary held-out result.')
    path=ROOT/args.primary/'paired_statistics.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    path.with_suffix('.html').write_text(report_html(result),encoding='utf-8')
    print(path)
    for scope in ('primary','combined'):
        for dataset,row in result.get(scope,{}).items():
            print(scope,dataset,'n',row['n'],'success',row['before_success'],'->',row['after_success'],
                'wins/losses',row['wins'],row['losses'],'exact_p',row['mcnemar_exact_two_sided_p'])


if __name__=='__main__':main()
