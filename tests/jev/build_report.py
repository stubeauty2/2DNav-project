"""Rebuild the local Chinese Jev evidence report from saved artifacts; no API calls."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import html
import json
import math
import os
from pathlib import Path
from statistics import median
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2] / 'out/jev_study_20261022'
SPLITS = {'avdn_capacity20x3': 'AVDN 开发候选', 'citynav_primary14': 'CityNav 主诊断集',
          'citynav_targeted_posthoc1': 'CityNav 事后单例'}
STAGES = [('fixed_review_08b', '0.8B 原始注意力'),
          ('fixed_review_08b_efficient', '0.8B efficient 复跑'),
          ('fixed_review_4b', '4B 固定候选')]


def esc(value):
    return html.escape(str(value if value is not None else '—'), quote=True)


def pct(value):
    return f'{value:.1%}' if isinstance(value, (int, float)) and math.isfinite(value) else '—'


def num(value, digits=2):
    return f'{value:.{digits}f}' if isinstance(value, (int, float)) and math.isfinite(value) else '—'


def ci(value):
    return '[' + ', '.join(pct(v) for v in value) + ']' if value else '—'


def table(headers, rows):
    if not rows:
        return '<p class="muted">尚无可汇总记录。</p>'
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th>'+esc(h)+'</th>' for h in headers) + \
        '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows) + '</tbody></table></div>'


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.inputs, self.warnings = {}, []

    def path(self, path):
        return path if isinstance(path, Path) and path.is_absolute() else self.root / path

    def load(self, path, default=None, jsonl=False):
        path = self.path(path)
        if not path.exists():
            return default
        try:
            raw = path.read_bytes()
            data = [json.loads(x) for x in raw.decode('utf-8-sig').splitlines() if x.strip()] if jsonl else json.loads(raw)
            self.inputs[path] = hashlib.sha256(raw).hexdigest()
            return data
        except (OSError, ValueError) as exc:
            self.warnings.append(f'{path.relative_to(self.root)} 暂不可读：{type(exc).__name__}')
            return default

    def link(self, path, title=None):
        path = self.path(path)
        label = title or str(path.relative_to(self.root))
        if not path.exists():
            return '<span class="muted">'+esc(label)+'（尚未生成）</span>'
        relative = os.path.relpath(path, self.root).replace('\\', '/')
        return '<a href="'+quote(relative, safe='/')+'">'+esc(label)+'</a>'

    def register_text(self, path):
        path = self.path(path)
        if not path.exists():
            return False
        self.inputs[path] = hashlib.sha256(path.read_bytes()).hexdigest()
        return True


def fixed_stage(e, folder, title):
    cases = e.load(folder+'/public_requests.jsonl', [], jsonl=True)
    truth = e.load(folder+'/evaluator_only.json', {})
    manifest = e.load(folder+'/run_manifest.json', {})
    design = e.load(folder+'/design.json', {})
    statuses, groups = Counter(), defaultdict(Counter)
    raw_choices, raw_groups = Counter(), defaultdict(Counter)
    below_threshold, threshold_changed, confirm_probabilities = 0, Counter(), []
    latencies, response_links = [], []
    for case in cases:
        path = folder+'/responses/'+case['case_id']+'.json'
        response = e.load(path, {})
        status = response.get('status', 'unstarted')
        if response and response.get('input_sha256') != case.get('input_sha256'):
            status = 'input_mismatch'
        statuses[status] += 1
        group = groups[case['split']]
        group['planned'] += 1
        if status in ('pending', 'unstarted'):
            continue
        group['terminal'] += 1
        label = truth.get(case['case_id'], {}).get('proposal_score', {}).get('correct')
        if not isinstance(label, bool):
            group['unscored'] += 1
            continue
        group['positive' if label else 'negative'] += 1
        native = response.get('response', {})
        if status == 'ok' and native.get('choice') in ('confirm', 'reject', 'uncertain'):
            choice = native.get('choice')
            raw_choices[choice] += 1
            below_threshold += bool(native.get('abstained'))
            if response.get('verdict') != choice:
                threshold_changed[choice+' → '+str(response.get('verdict'))] += 1
            raw_group = raw_groups[case['split']]
            raw_group['TP' if label else 'FP'] += choice == 'confirm'
            raw_group['FN' if label else 'TN'] += choice == 'reject'
            raw_group['uncertain_positive' if label else 'uncertain_negative'] += choice == 'uncertain'
            probability = native.get('probabilities', {}).get('confirm')
            if isinstance(probability, (int, float)):
                confirm_probabilities.append(probability)
        verdict = response.get('verdict') if status == 'ok' else 'invalid'
        if verdict == 'confirm':
            group['TP' if label else 'FP'] += 1
        elif verdict == 'reject':
            group['FN' if label else 'TN'] += 1
        elif verdict == 'uncertain':
            group['uncertain_positive' if label else 'uncertain_negative'] += 1
        else:
            group['invalid'] += 1
        latency = response.get('response', {}).get('latency_ms')
        if isinstance(latency, (int, float)):
            latencies.append(latency)
        if len(response_links) < 2:
            response_links.append(e.link(path, case['case_id']))
    terminal = sum(n for s, n in statuses.items() if s not in ('pending', 'unstarted'))
    expected = len(cases) or sum(design.get('case_counts', {}).values())
    status_text = '未准备' if not expected else f'{terminal}/{expected} 已结束；ok {statuses["ok"]}；pending {statuses["pending"]}；未运行 {statuses["unstarted"]}'
    errors = sum(n for s, n in statuses.items() if s not in ('ok', 'pending', 'unstarted'))
    body = '<h3>'+esc(title)+'</h3><p>'+esc(status_text)+f'；错误 {errors}。</p>'
    body += table(['集合', '结束/计划', '几何正/负', 'TP', 'FN', 'FP', 'TN', 'uncertain 正/负', '无效', '明确判定覆盖', '合格未确认率'], [
        [esc(SPLITS.get(split, split)), f'{g["terminal"]}/{g["planned"]}', f'{g["positive"]}/{g["negative"]}',
         g['TP'], g['FN'], g['FP'], g['TN'], f'{g["uncertain_positive"]}/{g["uncertain_negative"]}', g['invalid'],
         pct(sum(g[k] for k in ('TP', 'FP', 'TN', 'FN'))/g['terminal']) if g['terminal'] else '—',
         pct((g['positive']-g['TP'])/g['positive']) if g['positive'] else '—'] for split, g in sorted(groups.items())])
    if latencies:
        body += '<p class="muted">已记录调用时延：中位数 '+num(median(latencies))+' ms，合计 '+num(sum(latencies)/1000)+' s；含 '+str(len(latencies))+' 条记录。</p>'
    if raw_choices:
        body += '<p>阈值前原始首选：confirm '+str(raw_choices['confirm'])+'，reject '+str(raw_choices['reject'])+'，uncertain '+str(raw_choices['uncertain'])+'。最高选项概率低于 0.7 的有 '+str(below_threshold)+' 次；confirm 概率最大值 '+num(max(confirm_probabilities, default=None), 6)+ '。</p>'
        body += '<p>最终判决与原始首选不同的有 '+str(sum(threshold_changed.values()))+' 次'+('（'+esc('，'.join(k+'：'+str(v) for k,v in sorted(threshold_changed.items())))+'）' if threshold_changed else '')+'。低于阈值的次数包含原本就选择 uncertain 的请求，不能全部当作阈值改变的判决。</p>'
        body += '<details><summary>离线移除额外阈值后的判决诊断（不是新增导航实验）</summary>'
        body += table(['集合', '原始TP', '原始FN', '原始FP', '原始TN', '原始uncertain 正/负'], [
            [esc(SPLITS.get(split, split)), g['TP'], g['FN'], g['FP'], g['TN'],
             f'{g["uncertain_positive"]}/{g["uncertain_negative"]}'] for split,g in sorted(raw_groups.items())])
        body += '<p>直接采用原始三选项 argmax 仍保留 uncertain 选项。本表只重算开发候选日志，不改变在线阈值，不推断取消阈值后的导航成功率。几何合格也不等于人工确认的视觉身份。概率未做导航域校准，固定 0.7 的结果不能代表所有 Jev 决策策略。</p></details>'
    body += '<p class="sources">'+ ' · '.join(e.link(folder+'/'+f) for f in ('design.json', 'run_manifest.json', 'comparison.json', 'public_requests.jsonl', 'evaluator_only.json'))+'</p>'
    if response_links:
        body += '<details><summary>抽查原始响应</summary><p>'+' · '.join(response_links)+'</p></details>'
    comparison = e.load(folder+'/comparison.json', {})
    history = [r for r in comparison.get('summaries', []) if r.get('method') != 'OneJev' and
               (r.get('split') == 'citynav_primary14' or r.get('split') == 'avdn_capacity20x3')]
    if history and folder == 'fixed_review_08b':
        body += '<details><summary>历史 MLLM 复核对照（不同来源与提示，非在线三臂实验）</summary>'
        body += table(['集合', '来源', '模型 / thinking', '来源子集', 'n', 'TP/FN/FP/TN', 'uncertain 正/负'], [
            [esc(SPLITS.get(r['split'], r['split'])), esc(r['method']), esc(r['model'])+' / '+str(r['thinking_budget']),
             esc(r['source_subset']), r['n'], '/'.join(str(r[k]) for k in ('TP','FN','FP','TN')),
             f'{r["uncertain_positive"]}/{r["uncertain_negative"]}'] for r in history])
        body += '<p>AVDN 各源模型产生的候选不同；只有相同来源子集可比较。CityNav fixed_input_mllm 使用同 14 个上下文。原 MLLM 可修框，本表仅评价其对原候选的判决。</p></details>'
    return dict(folder=folder, title=title, expected=expected, terminal=terminal, statuses=dict(statuses),
                errors=errors, groups=groups, body=body, manifest=manifest,
                below_threshold=below_threshold, threshold_changed=dict(threshold_changed))


def research_section(e):
    repositories, dates = {}, []
    for path in sorted((e.root/'research').glob('github_metadata*.json')):
        data = e.load(path, {})
        dates.append(data.get('queried_at_utc', ''))
        for repo in data.get('repositories', []):
            repositories[repo['full_name']] = repo
    notes = {
        'Valen-Team/Valen': ('公开 0.8B / 2B / 4B；不同决策头候选', 'https://huggingface.co/Valen-Team/Valen-2B'),
        'OmniJev/OneJev': ('本次本地测试 0.8B / 4B；4B 用于完整轨迹', 'https://huggingface.co/OmniJev/OneJev1.6-4B'),
        'Shanghua-Gao/RSI-Jev': ('公开视觉权重；后续需固定 effort', 'https://huggingface.co/shgao/rsi-jev-v6.1-vl-4b'),
        'SimpleJev/JevAny': ('公开 LoRA；另需基础模型；适配器许可需核对', 'https://huggingface.co/SimpleJev/JevAny-Qwen3.5-4B-LoRA'),
    }
    rows = []
    for name, repo in sorted(repositories.items(), key=lambda kv: -kv[1].get('stargazers_count', 0)):
        if name not in notes:
            continue
        note, weight = notes[name]
        rows.append(['<a href="'+esc(repo['html_url'])+'">'+esc(name)+'</a>', repo.get('stargazers_count'),
                     esc((repo.get('license') or {}).get('spdx_id')), esc(note)+' · <a href="'+weight+'">权重</a>'])
    return '<h2 id="research">模型筛选</h2><p>研究对象是 TypeSafe Jev 的开放实现。' \
        '<a href="https://typesafe.ai/blog/jev-1-14">TypeSafe 官方</a>提供托管接口，本次未找到原版公开 checkpoint。' \
        'JEPA 是另一条研究路线。stars 仅表示查询时关注度，不能代表导航性能。</p>'+ \
        table(['官方项目', 'Stars', '代码许可', '可获取性与定位'], rows)+ \
        '<p class="sources">快照时间：'+esc(', '.join(sorted(set(dates))))+'。'+e.link('research/jev_model_research_zh.md', '完整核验、限制与一手引用')+' · '+e.link('research/github_metadata.json', 'GitHub 原始快照')+' · '+e.link('research/github_metadata_additional.json', '补充快照')+'</p>'


def runtime_section(e):
    rows = []
    for path in sorted((e.root/'services').glob('*.manifest.json')):
        d = e.load(path, {})
        settings, runtime = d.get('settings', {}), d.get('runtime', {})
        rows.append([e.link(path, path.name), esc(d.get('served_model')), '<code>'+esc(d.get('checkpoint_revision'))+'</code>',
                     esc(' / '.join(f'{k} {v}' for k, v in runtime.items())), esc(settings.get('device')),
                     esc(settings.get('dtype'))+' / head '+esc(settings.get('head_dtype')),
                     esc(settings.get('attention', 'upstream'))+' / repeat KV '+esc(settings.get('expanded_kv', False))])
    body = '<h2 id="runtime">运行环境与一致性检查</h2>'+table(['冻结清单', '模型', '权重 revision', '依赖版本', '设备', '精度', '注意力'], rows)
    body += '<p>服务 manifest 记录源码、wrapper、权重文件哈希与运行参数。顺序分支、debias=1；概率使用上游默认温度，未做导航域校准。0.7 是模型最大选项概率阈值，不是官方 confidence，也不是已校准的正确率。</p>'
    sanity_rows = []
    for path in sorted((e.root/'sanity').glob('*.json')):
        data = e.load(path, {})
        response = data.get('response', {})
        answers = response.get('answers', {})
        for name, a in answers.items():
            sanity_rows.append([e.link(path, path.stem), esc(name), esc(a.get('choice', a.get('noul'))),
                                esc(json.dumps(a.get('probabilities', {}), ensure_ascii=False)) if a.get('probabilities') else '—'])
    body += '<h3>基础视觉 sanity</h3>'+table(['记录', '问题', '输出', '选项概率'], sanity_rows)
    body += '<p class="muted">简单形状题仅确认图像/API 基础可用；不能证明复杂航拍复核有效。图像替换对照的进度与结果见用法消融部分；输出对图像敏感也不等于正确理解目标身份。</p>'
    attention = e.load('attention_comparison.json', {})
    if attention:
        before, after = attention.get('before_total_s'), attention.get('after_total_s')
        body += '<h3>原始注意力 → efficient + repeat KV</h3>'+table(
            ['成对请求', 'choice 相同', 'verdict 相同', '最大概率差', '原总时延', '新总时延', '时延比'], [[
                attention.get('n'), attention.get('same_choice'), attention.get('same_verdict'), num(attention.get('max_probability_difference'), 6),
                num(before)+' s', num(after)+' s', num(before/after)+'×' if before and after else '—']])
        body += '<p>数学上等价的 K/V 重复允许 efficient 内核，缓解 Windows GQA 的二次方注意力内存开销；有限精度输出仍有差异。以上为同一批已保存调用时延之和，不是严格硬件基准，也不外推至 4B 或全部输入。'+e.link('attention_comparison.json', '逐例数值对照')+'</p>'
    concurrent = e.load('concurrency_preflight_v2/report.json', {})
    if concurrent:
        body += '<h3>4B 并发与健康检查</h3><p>在线服务使用独立单线程执行原始推理处理函数，推理串行、健康接口保持响应。开发集最长与最短输入并发复核结果：</p>'
        body += table(['候选', '状态', '概率逐项相同', '判决相同', '调用时延'], [
            [esc(r['case_id']), esc(r['status']), esc(r['same_probabilities']), esc(r['same_verdict']), num(r['latency_ms']/1000)+' s']
            for r in concurrent.get('cases', [])])
        body += '<p>'+esc(concurrent.get('health_checks'))+' 次健康检查，失败 '+esc(concurrent.get('health_failures'))+'；最长 '+num(concurrent.get('max_health_seconds'), 3)+' s。仅证明这次检查，不保证所有未来请求。'+e.link('concurrency_preflight_v2/report.json')+'</p>'
    replay = e.load('default_mllm_replay/report.json', {})
    if replay:
        body += '<h3>默认 MLLM 路径回放</h3><p>'+esc(replay.get('status'))+'；'+esc(replay.get('episodes'))+' 条历史轨迹，'+esc(replay.get('model_requests'))+' 个模型请求，'+esc(replay.get('review_requests'))+' 个复核请求。使用历史响应验证实现兼容性，不是新推理或成功率提升证据。'+e.link('default_mllm_replay/report.json')+'</p>'
    audit = e.load('online/metric_audit.json', {})
    if audit:
        body += '<h3>真实轨迹评分交叉核对</h3><p>'+esc(audit.get('status'))+'；已核对 '+esc(audit.get('checked'))+' / '+esc(audit.get('planned'))+' 条结果。独立调用项目此前的 IoU 实现并核对预测中心距离；检查字段和几何计算，不证明视觉身份判断正确。AVDN 主指标沿用 capacity20 的 &gt;0.4，同时保留此前通用评估的 &gt;0.3。'+e.link('online/metric_audit.json')+'</p>'
    return body


def online_section(e):
    plan = e.load('online/plan.json', {})
    analysis_path = e.root/'online/analysis.json'
    analysis = e.load(analysis_path, {})
    jobs = plan.get('jobs', [])
    statuses, newest = Counter(), 0
    for job in jobs:
        folder = e.root/'online/episodes'/job['dataset']/job['model']/job['arm']/job['episode_id']
        path = folder/'attempt.json'
        attempt = e.load(path, {})
        status = attempt.get('status', 'unstarted')
        statuses[status] += 1
        for file in (path, folder/'result.json'):
            if file.exists():
                newest = max(newest, file.stat().st_mtime_ns)
    terminal = statuses['complete']+statuses['failed']
    stale = bool(analysis) and (analysis_path.stat().st_mtime_ns < newest or
        analysis.get('plan_sha256') != e.inputs.get(e.root/'online/plan.json'))
    complete = bool(jobs) and terminal == len(jobs) and bool(analysis.get('all_terminal')) and not stale
    state = '未冻结计划' if not jobs else f'{terminal}/{len(jobs)} 已结束；complete {statuses["complete"]}，failed {statuses["failed"]}，pending {statuses["pending"]}，未运行 {statuses["unstarted"]}'
    body = '<h2 id="online">全轨迹配对实验</h2><p><strong>'+esc(state)+'</strong></p>'
    if not complete:
        body += '<p class="callout">完整在线对照尚未结束并获得最新分析，不能据此声称 Jev 已成功替换 MLLM 复核或缩小模型差距。</p>'
    else:
        body += '<p class="callout">完整配对轨迹已有分析；下面是所测样本上的系统效果，不能由 20 个样本推出正式等效结论。</p>'
    if stale:
        body += '<p class="warning">analysis.json 落后于当前轨迹或计划，以下为旧分析快照。请重新运行 online_experiment.py analyze 后更新本报告。</p>'
    body += infrastructure_block(e, analysis)
    body += '<p>每数据集从既有公开池均匀抽取 20 条、seed=20261022，排除开发轨迹与重构对照轨迹；同一缓存初始化。两个主控 Qwen3.8-flash / max，各比较 mllm、只裁决原候选的 mllm_fixed、OneJev1.6-4B 三臂。thinking=0、12 轮、每轮工具 2 次、每轨迹 MLLM 请求上限 48。Jev 阈值固定 p≥0.7。开发候选诊断不进入这里的测试样本。</p>'
    if plan:
        budget = plan.get('budget', {})
        body += '<p>冻结预算：'+esc(budget.get('episodes'))+' 条轨迹；最多 '+esc(budget.get('max_paid_mllm_requests'))+' 次轨迹内付费 MLLM 请求；并发上限 '+esc(budget.get('max_workers'))+'。初始化不重采样，pending/failed 不重试。</p>'
    preflight = e.load('online_preflight/results.json', {})
    if preflight:
        body += '<p>另有 API 连通性预检 '+str(len(preflight.get('results', [])))+' 次；它不属于导航成功率与轨迹内预算。'+e.link('online_preflight/results.json')+'</p>'
    rows = []
    for r in analysis.get('summaries', []):
        distances = [v for v in r.get('actual_stop_distances_m', []) if isinstance(v, (int, float))]
        rows.append([esc(r['dataset']), esc(r['model']), esc(r['arm']), f'{r["terminal"]}/{r["planned"]}',
            f'{r["successes"]} / '+pct(r.get('sr')), pct(r.get('secondary_sr')) if r['dataset']=='AVDN' else '—',
            ci(r.get('observed_sr_bounds')) if r.get('sr') is None else '—', r.get('model_requests',0), r.get('api_errors',0),
            r.get('jev_errors',0), r.get('tool_errors',0), r.get('grounder_unavailable_episodes',0),
            r.get('grounder_capability_unknown_episodes',0), num(median(distances)) if distances else '—'])
    body += table(['数据集', '主控', '复核', '结束/计划', '成功数 / 主SR', 'AVDN >.3', '未完成SR界', 'MLLM请求', 'API错', 'Jev错', '工具错', '工具缺失轨迹', '工具未知轨迹', '停止距离中位 m'], rows)
    jev_groups = [r for r in analysis.get('summaries', []) if r.get('arm') == 'jev']
    if complete and jev_groups and all(r.get('successes') == 0 for r in jev_groups):
        body += '<p class="warning"><strong>本轮结论：当前 OneJev 候选复核方案没有维持成功率，暂不支持替换现有复核流程。</strong>两个主控在 Jev 臂都降至零，模型间差距归零不能解释为能力对齐。该结论只覆盖当前权重、输入、阈值与导航协议，不能推广为所有 Jev 模型无效。</p>'
    effects = []
    for r in analysis.get('within_model_effects', []):
        s = r.get('primary', {})
        effects.append([esc(r['dataset']), esc(r['model']), esc(r['right']+' − '+r['left']),
            str(s.get('n'))+'/'+str(r.get('planned_pairs')), pct(s.get('difference_right_minus_left')), ci(s.get('difference_ci95')),
            f'{s.get("gained",0)}/{s.get("lost",0)}', num(s.get('exact_mcnemar_two_sided'),4)])
    body += '<h3>同一主控下的工具效应</h3>'+table(['数据集', '主控', '右臂 − 左臂', '配对数/计划', 'SR差', '配对95% CI', '获得/丢失', 'McNemar p'], effects)
    gaps = []
    for r in analysis.get('across_model_gaps', []):
        s = r.get('max_minus_flash', {})
        gaps.append([esc(r['dataset']), esc(r['arm']), str(s.get('n'))+'/'+str(r.get('planned_pairs')),
                     pct(s.get('difference_right_minus_left')), ci(s.get('difference_ci95')), f'{s.get("gained",0)}/{s.get("lost",0)}'])
    body += '<h3>同一复核方式下的主控差距</h3>'+table(['数据集', '复核', '配对数/计划', 'max − flash SR', '配对95% CI', '获得/丢失'], gaps)
    changes = [[esc(r['dataset']), esc(r['baseline']), r.get('n'), pct(r.get('signed_gap_change_jev_minus_baseline')), ci(r.get('ci95'))] for r in analysis.get('gap_changes', [])]
    body += table(['数据集', '基线复核', '完整六臂配对数', 'Jev后的有符号差距变化', '95% CI'], changes)
    body += '<p>有符号差距变化为负不一定表示绝对差距缩小；即使差距缩小，也可能是两个主控同时变差。应同时查看 Flash、Max 各自的成功率、Jev 相对各自基线的变化和绝对差距。</p>'
    diagnostic = e.load('online/gap_diagnostics.json', {})
    if diagnostic:
        gap_stale = diagnostic.get('plan_sha256') != e.inputs.get(e.root/'online/plan.json') or \
            diagnostic.get('analysis_sha256') != e.inputs.get(analysis_path)
        body += '<h3>绝对差距与两侧成功率（事后描述补充）</h3>'
        if gap_stale:
            body += '<p class="warning">绝对差距诊断的源分析已更新，请重跑 gap_report.py；旧值暂不展示。</p>'
        else:
            body += table(['数据集', '基线', '六臂齐全/计划', '基线 Flash / Max', 'Jev Flash / Max',
                           '基线绝对差', 'Jev绝对差', '绝对差变化', '配对95% CI'], [
                [esc(r['dataset']), esc(r['baseline']), str(r['n'])+'/'+str(r['planned_pairs']),
                 pct(r['baseline_flash_sr'])+' / '+pct(r['baseline_max_sr']),
                 pct(r['jev_flash_sr'])+' / '+pct(r['jev_max_sr']), pct(r['baseline_absolute_gap']),
                 pct(r['jev_absolute_gap']), pct(r['absolute_gap_change']), ci(r['ci95']['absolute_gap_change'])]
                for r in diagnostic.get('rows', [])])
        body += '<p>只使用同一任务六个模型/复核组合均结束的交集，未完成任务不补零。每次 bootstrap 联合重采样任务，再计算两侧均值的绝对差。它是描述补充，没有新增等效判据。'+e.link('online/gap_diagnostics.json')+'</p>'
    body += '<p>AVDN 主指标为最终正方框 IoU&gt;0.4，附指标 &gt;0.3；CityNav 为最终预测中心距 GT≤20 m。真实停止点距离单列，不替代定位指标。CI 为轨迹配对 bootstrap；获得/丢失表示右臂相对左臂的 discordant 轨迹。未完成时配对统计仅覆盖已结束的交集，不能视为最终结果。</p>'
    body += '<p>小样本或全为相同结果时，经验 bootstrap 区间可能退化为 [0,0]；这不代表总体差异没有不确定性。尤其两侧均无成功时，同分不足以证明保持了可用的导航性能。</p>'
    body += '<p>同样的 48 次 MLLM 上限下，Jev 可省去 review 请求并把额度用于更多规划；在线效应包含预算重新分配。剩余主控仍接收导航图像。这项设计验证候选复核工具的系统作用，不证明去除主控视觉能力后仍等效。</p>'
    body += '<p class="sources">'+' · '.join(e.link('online/'+p) for p in ('plan.json','run_manifest.json','analysis.json','analysis.md'))+'</p>'
    return dict(body=body, terminal=terminal, planned=len(jobs), complete=complete, stale=stale)


def recorded_infrastructure_error(row):
    return bool(row.get('api_errors',0) or row.get('jev_errors',0) or row.get('tool_errors',0)
                or row.get('unavailable_planned_grounders') or row.get('unknown_planned_grounders'))


def infrastructure_summary(analysis):
    """Post-hoc descriptive subset; original outcomes/denominators are never changed."""
    terminal = {'complete', 'failed'}
    rows = analysis.get('rows', [])
    affected = recorded_infrastructure_error
    def succeeded(row):
        return row['status']=='complete' and bool(row['success'])
    groups, pairs = [], []
    settings = sorted({(r['dataset'],r['model']) for r in rows})
    for dataset,model in settings:
        subset = [r for r in rows if (r['dataset'],r['model'])==(dataset,model)]
        for arm in ('mllm','mllm_fixed','jev'):
            selected = [r for r in subset if r['arm']==arm and r['status'] in terminal]
            groups.append(dict(dataset=dataset,model=model,arm=arm,terminal=len(selected),
                affected=sum(affected(r) for r in selected),
                api_interface_termination=sum(r.get('termination')=='interface_error' and r.get('api_errors',0)>0 for r in selected)))
        index = {(r['episode_id'],r['arm']):r for r in subset}
        for baseline in ('mllm','mllm_fixed'):
            ids = [r['episode_id'] for r in subset if r['arm']==baseline]
            joined = [(index[(ident,baseline)],index.get((ident,'jev'))) for ident in ids]
            complete = [(a,b) for a,b in joined if b is not None and a['status'] in terminal and b['status'] in terminal]
            clean = [(a,b) for a,b in complete if not affected(a) and not affected(b)]
            base_successes = sum(succeeded(a) for a,b in clean)
            jev_successes = sum(succeeded(b) for a,b in clean)
            gained = [(a,b) for a,b in complete if not succeeded(a) and succeeded(b)]
            lost = [(a,b) for a,b in complete if succeeded(a) and not succeeded(b)]
            def mean_change(field):
                values = [r.get(field) for pair in clean for r in pair]
                if not clean or any(not isinstance(v,int) or isinstance(v,bool) or v<0 for v in values):
                    return None
                return sum(b[field]-a[field] for a,b in clean)/len(clean)
            pairs.append(dict(dataset=dataset,model=model,baseline=baseline,planned=len(ids),
                complete_pairs=len(complete),unaffected_pairs=len(clean),
                baseline_successes=base_successes,jev_successes=jev_successes,
                all_gained=len(gained), all_lost=len(lost),
                affected_gained=sum(affected(a) or affected(b) for a,b in gained),
                affected_lost=sum(affected(a) or affected(b) for a,b in lost),
                mean_mllm_request_change=mean_change('model_requests'),
                mean_image_request_change=mean_change('all_image_mllm_requests'),
                sr_change=(jev_successes-base_successes)/len(clean) if clean else None))
    return dict(groups=groups,pairs=pairs,
        api_interface_terminations=sum(g['api_interface_termination'] for g in groups),
        affected_episodes=sum(g['affected'] for g in groups))


def infrastructure_block(e, analysis):
    data = infrastructure_summary(analysis)
    if not data['affected_episodes']:
        return ''
    body = '<p class="warning">当前已结束轨迹中 '+str(data['affected_episodes'])+' 条记录过基础设施错误，其中 '+str(data['api_interface_terminations'])+' 条含 API 失败且以 interface_error 终止。主结果保留这些轨迹；总成功率混有运行可靠性的影响，不能直接解释为纯视觉能力差异。</p>'
    body += '<details><summary>基础设施影响与无已记录错误的配对子集（事后描述）</summary>'
    body += table(['数据集','主控','复核','已结束','有基础设施错误','含API错误的接口终止'], [
        [esc(g['dataset']),esc(g['model']),esc(g['arm']),g['terminal'],g['affected'],g['api_interface_termination']]
        for g in data['groups']])
    body += table(['数据集','主控','基线','已齐配对/计划','全配对获得/丢失','其中涉及故障','双方无错误配对','基线成功','Jev成功','子集SR差'], [
        [esc(r['dataset']),esc(r['model']),esc(r['baseline']),str(r['complete_pairs'])+'/'+str(r['planned']),
         str(r['all_gained'])+'/'+str(r['all_lost']),str(r['affected_gained'])+'/'+str(r['affected_lost']),
         r['unaffected_pairs'],r['baseline_successes'],r['jev_successes'],pct(r['sr_change'])] for r in data['pairs']])
    body += '<p>获得/丢失均指 Jev 相对该基线；“涉及故障”表示配对任一侧记录过上述错误，不证明故障是该得失的唯一原因。后面三个成功率列仅属于双方无已记录错误的子集。</p>'
    body += '<p>子集只保留两臂都已结束，且双方均无记录的 API/Jev/工具错误、检测器不可用或能力未知。即使某次 API 错误后来恢复，该轨迹也不进入这个保守子集；无 API 错误的格式失败仍作为模型失败保留。未完成任务不补零。筛选是事后的，耗时和故障暴露可能因复核方式而异，不能把子集当作无偏的替代主结果，也不据此宣称等效。</p>'
    hold = e.load('online/operational_hold.json', {})
    guard = e.load('online/dispatch_guard.json', {})
    if hold or guard:
        body += '<p>派发控制快照：'+esc(hold.get('state',hold.get('status')))+'；时间 '+esc(hold.get('utc',hold.get('updated_utc')))+'。外部保护记录：'+esc(guard.get('status'))+'。这些是保存的操作记录，不是此页面实时查询的进程状态。'+e.link('online/operational_events.jsonl','操作时间线')+' · '+e.link('online/dispatch_guard.json','保护进程记录')+'</p>'
    body += '<p>连续连接错误曾未触发原调度熔断，因为运行时将 interface_error 标为普通 complete。随后通过外部进程控制暂停新派发、等待现有任务结束；确认连接恢复后继续未开始任务，并加独立派发保护。冻结模型、提示、阈值、样本和执行代码未改，已结束轨迹不重跑。</p></details>'
    return body


def discordant_pair_details(analysis, stages):
    """All terminal outcome disagreements, not a selection of illustrative failures."""
    rows = analysis.get('rows', [])
    index = {(r['dataset'],r['model'],r['episode_id'],r['arm']):r for r in rows}
    by_job = {r['job_id']:r for r in stages.get('rows', [])}
    output = []
    for base in rows:
        if base['arm'] not in ('mllm','mllm_fixed') or base['status'] not in ('complete','failed'):
            continue
        jev = index.get((base['dataset'],base['model'],base['episode_id'],'jev'))
        if not jev or jev['status'] not in ('complete','failed'):
            continue
        a = base['status']=='complete' and bool(base.get('success'))
        b = jev['status']=='complete' and bool(jev.get('success'))
        if a == b:
            continue
        output.append(dict(dataset=base['dataset'],model=base['model'],episode_id=base['episode_id'],
            baseline=base['arm'],direction='gained' if b else 'lost',
            involves_infrastructure=recorded_infrastructure_error(base) or recorded_infrastructure_error(jev),
            base=base,jev=jev,base_stage=by_job.get(base.get('job_id')),
            jev_stage=by_job.get(jev.get('job_id'))))
    return sorted(output,key=lambda r:(r['dataset'],r['model'],r['episode_id'],r['baseline']))


def discordant_pairs_block(e, analysis, stages):
    pairs = discordant_pair_details(analysis, stages)
    if not pairs:
        return '<p>当前已齐配对尚无成功/失败不一致案例。</p>'
    rows = []
    for p in pairs:
        s = p['jev_stage']
        raw = s.get('qualified_jev_raw_choices',{}) if s else {}
        choices = raw.get('choices',{})
        def result_link(row):
            path = row.get('result_path')
            return e.link(Path(path),str(row.get('termination'))) if path else esc(row.get('termination'))
        rows.append([esc(p['dataset']),esc(p['model']),esc(p['episode_id']),esc(p['baseline']),
            '获得' if p['direction']=='gained' else '丢失',
            '有记录' if p['involves_infrastructure'] else '无已记录错误',
            result_link(p['base']),num(p['base'].get('metric'),3),
            result_link(p['jev']),num(p['jev'].get('metric'),3),
            ('是' if s.get('geometric_center_covered') else '否') if s else '—',
            esc(s.get('qualified_tool_proposals') if s else None),
            esc(s.get('qualified_final_proposals') if s else None),
            esc(s.get('qualified_proposal_reviews') if s else None),
            '/'.join(str(choices.get(k,0)) for k in ('confirm','reject','uncertain')) if s else '—',
            esc(s.get('invalid_actions') if s else None),esc(s.get('invalid_formats') if s else None)])
    body = '<details><summary>全部已齐配对中的成功/失败不一致案例</summary>'
    body += table(['数据集','主控','任务','基线','Jev相对基线','配对基础设施',
                   '基线终止','基线指标','Jev终止','Jev指标','Jev臂覆盖目标中心',
                   'Jev臂检测器合格候选','Jev臂主控合格候选','Jev合格候选复核数',
                   '其原始confirm/reject/uncertain','Jev臂无效动作','Jev臂格式错误'],rows)
    body += '<p>每行是同任务、同主控的一组已结束配对；不同基线可重复同一 Jev 轨迹，不能把表行当作独立任务数。指标为 AVDN IoU（越大越好）或 CityNav 米制距离（越小越好）。原始选项计数仅覆盖几何合格候选、已校验的 Jev 日志，缺失或错误不补成 uncertain。</p>'
    body += '<p>两臂的路线、视图和候选可能不同；这张表定位共存现象，不隔离复核器的因果效应。基础设施无已记录错误不等于无模型格式/动作失败。几何合格也不证明候选满足全部语义条件。</p></details>'
    return body


def stage_section(e):
    report = e.load('online/stage_analysis.json', {})
    body = '<h2 id="stages">导航轨迹阶段诊断</h2>'
    if not report:
        return body+'<p>尚无阶段诊断快照。</p>'
    if report.get('source_analysis_sha256') != e.inputs.get(e.root/'online/analysis.json'):
        body += '<p class="warning">阶段诊断与当前 analysis.json 不是同一快照；下列覆盖数仅属于该诊断生成时。</p>'
    body += '<p>本节覆盖 '+esc(report.get('terminal'))+' / '+esc(report.get('planned'))+' 条已结束轨迹，包含成功和失败。每列统计出现该现象的轨迹数，列之间可以重叠；一次未确认之后仍可能最终成功，不能把这些计数当作互斥的失败原因。</p>'
    fields = ('geometric_center_covered','qualified_tool_proposals','qualified_final_proposals',
              'qualified_reviews_not_confirmed','qualified_reviews_service_error','refinement_damaged','refinement_repaired')
    rows = [[esc(r['dataset']),esc(r['model']),esc(r['arm']),r['terminal'],r['missing_trace'],
             *[r[field] for field in fields]] for r in report.get('groups', [])]
    body += table(['数据集','主控','复核','已结束','缺日志','几何覆盖目标中心','检测器有合格候选',
                   '主控提出合格候选','合格候选未确认','合格候选复核服务错','改框降级','改框修复'], rows)
    if any('invalid_action_reasons' in r for r in report.get('groups', [])):
        body += '<details><summary>规划动作与格式错误（事件数）</summary>'
        action_rows = []
        for r in report.get('groups', []):
            reasons = r.get('invalid_action_reasons',{})
            action_rows.append([esc(r['dataset']),esc(r['model']),esc(r['arm']),
                r.get('invalid_action_events',0),*[reasons.get(k,0) for k in
                    ('outside_map','invalid_raster','tool_budget_exhausted','other')],r.get('invalid_format_events',0)])
        body += table(['数据集','主控','复核','无效动作总数','越出地图','无效栅格点',
                       '工具额度耗尽后仍调用','其他动作校验','格式错误'], action_rows)
        body += '<p>按运行时记录的错误文字分类，每次事件只计一个动作类别；格式错误另计。这里是事件数，包含同轨迹重复尝试和成功轨迹，不是互斥失败轨迹数。此类请求仍占用全局 MLLM 请求额度；不能将它们归为 Jev 服务错误或视觉拒绝。</p></details>'
    raw = report.get('qualified_jev_raw_choices', {})
    if raw:
        choices = raw.get('choices', {})
        body += '<h3>几何合格候选的原始 Jev 首选</h3><p>这里统计复核事件，可来自同一轨迹、同一候选，不能当作独立样本。</p>'
        body += table(['合格候选复核事件', '原始confirm', '原始reject', '原始uncertain', '低于阈值'], [[
            raw.get('total'), choices.get('confirm',0), choices.get('reject',0), choices.get('uncertain',0),
            raw.get('threshold_abstentions')]])
        body += '<p>日志校验状态：'+esc(json.dumps(raw.get('journal_statuses', {}), ensure_ascii=False))+'。缺失、无效和服务失败不填成某个选项。即使降低门槛，原始 reject 或 uncertain 也不会自动成为 confirm；另一方面，几何合格不能代替完整语义身份标注。</p>'
    body += '<p>仅将最终目标候选与最终目标 GT 比较，辅助目标不参与。几何合格采用本次主指标，不能替代人工视觉身份标注；几何覆盖也不保证目标足够清晰。服务错误和正常判为 uncertain/reject 分开。合格候选未确认是观察到的瓶颈，不保证改成 confirm 后就会导航成功。改框降级/修复指跨越几何阈值，不等于语义判断变差/变好。'+e.link('online/stage_analysis.json','逐轨迹、逐候选和逐复核记录')+'</p>'
    analysis = e.load('online/analysis.json', {})
    if (analysis and report.get('source_analysis_sha256') == e.inputs.get(e.root/'online/analysis.json')
            and report.get('plan_sha256') == analysis.get('plan_sha256')):
        body += discordant_pairs_block(e, analysis, report)
    else:
        body += '<p>配对案例表等待分析与阶段诊断快照一致后显示。</p>'
    return body


def workload_section(e):
    report = e.load('online/workload.json', {})
    body = '<h2 id="workload">MLLM 工作量与耗时</h2>'
    if not report:
        return body+'<p>尚无请求日志统计。</p>'
    body += '<p>当前日志快照共 '+esc(report.get('observed_mllm_requests'))+' 次 MLLM 请求尝试（含进行中任务）。表中均值只覆盖已结束轨迹；样本未齐时不能当作最终节省比例。</p>'
    rows = []
    for group in report.get('groups', []):
        count = group['terminal']
        tokens = group['reported_tokens']['total_tokens']
        missing = group['token_fields_missing']['total_tokens']
        rows.append([esc(group['dataset']), esc(group['model']), esc(group['arm']),
            f"{count}/{group['planned']}", num(group['mean_mllm_requests']), num(group['mean_review_requests']),
            num(group['mean_jev_requests']), num(tokens/count, 0) if count and not missing else '缺失/未完成',
            missing, num(group['median_episode_seconds'])])
    body += table(['数据集', '主控', '复核', '结束/计划', '平均MLLM调用', '其中复核', '平均Jev调用', '平均上报总token', 'token缺失请求', '轨迹耗时中位秒'], rows)
    pairs = [[esc(r['dataset']), esc(r['model']), esc('jev − '+r['baseline']),
              f"{r['n']}/{r['planned']}", num(r['mean_mllm_request_change_jev_minus_baseline']),
              '['+', '.join(num(v) for v in r['descriptive_ci95'])+']' if r['descriptive_ci95'] else '—']
             for r in report.get('paired_request_changes', [])]
    body += table(['数据集', '主控', '对比', '配对数/计划', '平均MLLM调用差', '配对95% CI'], pairs)
    clean = infrastructure_summary(e.load('online/analysis.json', {}))['pairs']
    if clean:
        body += '<h3>双方无已记录基础设施错误的调用量（事后描述）</h3>'
        body += table(['数据集','主控','对比','配对数/计划','平均MLLM调用差','其中图像请求差'], [
            [esc(r['dataset']),esc(r['model']),esc('jev − '+r['baseline']),
             str(r['unaffected_pairs'])+'/'+str(r['planned']),num(r['mean_mllm_request_change']),
             num(r['mean_image_request_change'])] for r in clean])
        body += '<p>这里沿用基础设施诊断的事后筛选规则，仅比较同一任务两臂均已结束且均无已记录错误的交集。正值表示 Jev 臂使用更多主控请求。图像请求只统计携带图像的请求次数，不度量视觉推理难度；省去 review 并不保证总图像调用更少。更早的错误停止也可能让基线调用较少，应联合看两侧成功率和终止原因。</p>'
    body += '<p>负调用差表示 Jev 臂使用更少的 MLLM 请求。应同时查看成功率：失败更早结束也可能减少调用。token 是接口上报值，不是费用账单；图像和缓存 token 属于输入 token 的子集，不能再加到总量。轨迹耗时包含远程 API、CPU 检测器及本地 Jev 排队。'+e.link('online/workload.json', '逐轨迹调用与token日志统计')+'</p>'
    return body


def usage_section(e):
    body = '<h2 id="usage">当前用法审查</h2>'
    if not e.register_text('research/usage_audit_zh.md'):
        return body+'<p>尚无用法审查证据。</p>'
    for name in ('integration_usage_audit_zh.md', 'experiment_setup_audit_zh.md'):
        e.register_text('research/'+name)
    body += '<p>以下区分已经观察到的行为和仍需对照实验验证的原因。主实验保持冻结；这些发现不会在中途改变提示、阈值或抽样。</p>'
    body += table(['方面', '已核实的当前行为', '下一轮可验证的改法'], [
        ['判决门槛', '最大选项概率低于 0.7 时改为 uncertain；未做导航域校准。改判次数与原始分布见候选诊断。',
         '用独立开发数据选择风险与覆盖的取舍；保留原生 uncertain，并单列错误确认。'],
        ['复核问题', '单个 Choice 同时要求身份、唯一性、形状、空间关系、群组/序数与可见性判断。',
         '在相同图像和候选上比较原子问题，再由代码组合；检验拆题是否改善，而非预设改善。'],
        ['上下文', '固定候选 4B 的日志输入为 3,516–18,817 token，包含图文；最长例主要由 navigation 与 named_landmarks 扩大。',
         '保留原始目标描述、候选及必要关系，比较相关字段子集；不可机械截断路线或否定条件。'],
        ['导航闭环', '已有相同两图和候选反复复核的轨迹；通用 uncertain 反馈未强制主控获取新证据。',
         '按视觉证据和目标去重，在不确定后要求换视角、改变候选或补充明确的缺失条件。'],
        ['结论范围', '主控仍收到导航图像；当前只替换候选复核，OneJev 不是 TypeSafe 原版权重。',
         '先验证这一职责的效果；若要检验视觉负担是否移出主控，需另设受控的输入对照。'],
    ])
    body += '<p>TypeSafe 建议使用窄而明确的问题并仅提供相关上下文；本轮整体复核与其建议存在差异，但这不证明差异导致了当前结果，也不保证开放实现采用该建议后一定提升。'+ \
        '<a href="https://docs.typesafe.ai/concepts/how-to-build-with-system-one">官方构建指南</a> · '+ \
        e.link('research/usage_audit_zh.md','证据、官方引用与待验证假设')+' · '+ \
        e.link('research/integration_usage_audit_zh.md','输入、反馈与重复复核审查')+' · '+ \
        e.link('research/experiment_setup_audit_zh.md','选样、初始化与历史设置核对')+'</p>'
    usage = e.load('online/review_usage.json', {})
    if usage:
        stats = usage.get('summary', {})
        tokens = stats.get('input_tokens', {})
        body += '<h3>重复视觉证据的实时诊断快照</h3><p>UTC '+esc(usage.get('snapshot_utc'))+'；已结束复核请求 '+esc(stats.get('terminal_reviews'))+' 次（可包含进行中轨迹），首条之外的重复证据调用 '+esc(stats.get('repeated_calls'))+' 次。读取错误 '+str(len(usage.get('read_errors', [])))+'，pending 请求跳过 '+esc(usage.get('skipped_pending'))+' 次。输入 token 的 P50 / P90 / 最大值：'+num(tokens.get('p50'), 0)+' / '+num(tokens.get('p90'), 0)+' / '+num(tokens.get('max'), 0)+'。</p>'
        body += table(['服务状态', '次数'], [[esc(k),v] for k,v in sorted(stats.get('statuses', {}).items())])
        body += table(['原始 choice → 最终 verdict', '次数'], [[esc(k),v] for k,v in sorted(stats.get('raw_to_verdict', {}).items())])
        examples = sorted([r for r in usage.get('episodes', []) if r.get('repeated_calls', 0)],
                          key=lambda r: (-r['repeated_calls'], r['path']))[:4]
        body += table(['轨迹', '已结束复核', '首条之外重复', 'move事件', '终止状态'], [
            [e.link('online/'+r['path']+'/trace.json', r['path'].removeprefix('episodes/')),
             r['terminal_review_count'], r['repeated_calls'], esc(r.get('move_count')), esc(r.get('termination'))]
            for r in examples])
        body += '<p>重复键要求同一轨迹内有序两图哈希、精确候选几何、目标角色/描述及完整指令一致；不含轮次、剩余额度和局部视图编号。因此它表示重复的视觉证据，不表示完整请求相同，也不能直接作为普遍缓存键。未做近似框匹配，轻微变化会分入不同组。'+e.link('online/review_usage.json','逐请求与分组证据')+'</p>'
        if examples:
            episode = examples[0]
            group = max(episode['groups'], key=lambda g: g['count'])
            record = e.load('online/'+group['reviews'][0]['journal'], {})
            images = []
            for asset in record.get('assets', []):
                path = Path(asset['image_path'])
                if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256']:
                    e.inputs[path] = asset['sha256']
                    src = quote(os.path.relpath(path, e.root).replace('\\', '/'), safe='/')
                    images.append('<figure><img loading="lazy" src="'+src+'" alt="'+esc(asset['view_id'])+'"><figcaption>'+esc(asset['view_id'])+'</figcaption></figure>')
            if images:
                body += '<details><summary>重复最多轨迹的原始复核两图（'+str(group['count'])+' 次同组证据）</summary><div class="evidence-images">'+''.join(images)+'</div><p>这是已保存的原图与黄色候选标记，仅展示复核所见；这里没有新增语义真值标注。</p></details>'
    return body


STYLE = '''
:root{font-family:Inter,"Microsoft YaHei",system-ui,sans-serif;color:#172b3a;background:#f4f6f8;font-size:15px;line-height:1.65}
.evidence-images{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:14px}.evidence-images figure{margin:0}.evidence-images img{max-width:100%;height:auto}.evidence-images figcaption{font-size:12px;color:#61717d}
body{margin:0}main{max-width:1320px;margin:0 auto;padding:44px 34px 60px}header{padding:28px 32px;background:#142e3d;color:white;border-radius:16px}
h1{font-size:30px;letter-spacing:.02em;margin:4px 0 12px}header p{color:#d3e1e7;margin:8px 0}h2{font-size:23px;margin:38px 0 14px}h3{font-size:18px;margin:26px 0 10px}
nav{display:flex;gap:20px;flex-wrap:wrap;margin:22px 0}a{color:#146f7e;text-decoration:none}a:hover{text-decoration:underline}nav a{font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:20px 0}.card{background:white;border:1px solid #dce4e9;padding:18px;border-radius:10px}.card b{font-size:24px;display:block}.muted,.sources{color:#61717d;font-size:13px}.sources{overflow-wrap:anywhere}
section{background:white;border:1px solid #dce4e9;border-radius:14px;padding:0 24px 20px;margin:24px 0}.table-wrap{overflow:auto;border:1px solid #dce4e9;border-radius:8px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e3e9ed;vertical-align:top}th{background:#edf3f5;white-space:nowrap}td{min-width:38px}tr:last-child td{border-bottom:0}code{font-size:11px;overflow-wrap:anywhere}details{border-top:1px solid #e3e9ed;padding:12px 0;margin-top:14px}summary{cursor:pointer;font-weight:600}.callout{border-left:4px solid #168c91;background:#edf8f7;padding:13px 18px}.warning{border-left:4px solid #d29224;background:#fff7e8;padding:12px 18px}.small{font-size:13px}li{margin:8px 0}footer{font-size:12px;color:#61717d} @media(max-width:700px){main{padding:14px}header{padding:20px}section{padding:0 14px 16px}h1{font-size:24px}}@media print{body{background:white}main{padding:0;max-width:none}header{background:#fff;color:#172b3a;border:1px solid}header p{color:#172b3a}section{break-inside:avoid}nav{display:none}}
'''


def usage_ablation_section(e):
    folder = 'usage_ablation_4b'
    plan = e.load(folder+'/plan.json', {})
    if not plan:
        return ''
    body = '<h2 id="ablation">开发候选用法对照</h2><p>完整上下文、精简上下文、替换图像三组，共 '+esc(len(plan.get('jobs',[])))+' 次本地 Jev 请求。仅在全部在线轨迹结束后执行，不调用付费 MLLM。</p>'
    body += '<p>精简组保留完整指令、对话条件、初始化、位置方向、命名地标、候选和两张原图，删除规划反馈、导航汇总、工具能力和预算字段；删除的信息可能有用。替换图像组保持文字与候选元数据不变，改用同数据集其他轨迹的图像对；供图任务允许重复。这是输入敏感性诊断，不是导航成功率对照。</p>'
    result = e.load(folder+'/analysis.json', {})
    if not result:
        return body+'<p class="callout">设计已冻结，尚无对照分析结果。</p>'+e.link(folder+'/plan.json','冻结对照设计')
    if result.get('plan_sha256') != e.inputs.get(e.root/folder/'plan.json'):
        return body+'<p class="warning">对照分析与当前设计不一致，暂不展示结果。</p>'
    expected = {f"responses/{j['arm']}/{j['case_id']}.json" for j in plan.get('jobs',[])}
    saved_hashes = result.get('response_sha256',{})
    hashes = {relative.replace('\\','/'): digest for relative,digest in saved_hashes.items()}
    if len(hashes) != len(saved_hashes):
        return body+'<p class="warning">对照响应路径存在重复别名，暂不展示结果。</p>'
    root = (e.root/folder).resolve()
    for relative,digest in hashes.items():
        path = (root/relative).resolve()
        if relative not in expected or not path.is_relative_to(root) or not e.register_text(path) or e.inputs[path] != digest:
            return body+'<p class="warning">对照响应证据已变更或缺失，暂不展示结果。</p>'
    complete = bool(result.get('complete')) and bool(expected) and set(hashes)==expected
    body += '<p>分析状态：'+('全部对照已完成' if complete else '对照尚未完成')+'。原始选项是概率最大类别；最终判决保留冻结的 0.7 门槛。</p>'
    rows = []
    for row in result.get('rows',[]):
        paired = row.get('paired_with_full',{})
        proxy = row.get('verdict_geometry_proxy',{})
        geometry = '不适用（图像已替换）' if row['arm']=='substituted_images' else '/'.join(str(proxy.get(k,'—')) for k in ('TP','FN','FP','TN'))
        rows.append([esc(SPLITS.get(row['split'],row['split'])),esc(row['arm']),
            str(row['ok'])+'/'+str(row['planned']),esc(json.dumps(row.get('choices',{}),ensure_ascii=False)),
            esc(json.dumps(row.get('verdicts',{}),ensure_ascii=False)),geometry,paired.get('n'),
            paired.get('changed_choices'),paired.get('changed_verdicts'),num(paired.get('mean_max_probability_delta'),5)])
    body += table(['开发集合','对照组','有效/计划','原始选项','最终判决','最终几何TP/FN/FP/TN',
                   '与完整组配对数','原始选项变化','最终判决变化','平均最大类别概率差'],rows)
    if complete:
        compact_rows = [r for r in result.get('rows',[]) if r.get('arm') == 'compact']
        if compact_rows and not any(r.get('verdicts',{}).get('confirm',0) for r in compact_rows):
            body += '<p><strong>本次精简上下文没有恢复确认输出。</strong>这否定了“仅删除这些规划字段就足以解决当前确认瓶颈”的直接预期；不能据此断言所有输入组织方式都无效，也不能从开发候选推导新的导航成功率。</p>'
    body += '<p>概率差为每对输入三个类别概率差绝对值的最大值，再取平均。图像替换后的原标签不再有效，不计算其准确率；概率变化只说明这些输入下对图像有响应，不能证明视觉理解正确。对照案例共享开发轨迹，不能作为独立样本，也不用于调整在线实验阈值。完整混淆计数（含 uncertain）见 '+e.link(folder+'/analysis.json','对照分析')+'；'+e.link(folder+'/plan.json','冻结设计')+'。</p>'
    return body


def build(root=ROOT):
    e = Evidence(root)
    stages = [fixed_stage(e, folder, title) for folder, title in STAGES]
    known = {folder for folder, _ in STAGES}
    for path in sorted(e.root.glob('fixed_review_4b*')):
        if path.is_dir() and path.name not in known:
            stages.append(fixed_stage(e, path.name, path.name.replace('_', ' ')))
    online = online_section(e)
    research, runtime = research_section(e), runtime_section(e)
    workload = workload_section(e)
    stage_diagnostics = stage_section(e)
    usage = usage_section(e)
    ablation = usage_ablation_section(e)
    headline = '完整在线结果已生成，解释仍限于所测样本。' if online['complete'] else '候选诊断与在线导航分阶段呈现；当前不足以判定替换成功。'
    cards = ''.join('<div class="card"><span>'+esc(s['title'])+'</span><b>'+str(s['terminal'])+' / '+str(s['expected'] or '—')+'</b><span class="muted">已结束候选 · 错误 '+str(s['errors'])+'</span></div>' for s in stages)
    cards += '<div class="card"><span>完整导航</span><b>'+str(online['terminal'])+' / '+str(online['planned'] or '—')+'</b><span class="muted">已结束轨迹 · '+('最新分析齐备' if online['complete'] else '未完成证据链')+'</span></div>'
    timestamp = datetime.now().astimezone().isoformat(timespec='seconds')
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Jev 多模态导航实验报告</title><style>'+STYLE+'</style><main>'
    body += '<header><p>AVDN × CityNav · 可审计实验快照</p><h1>Jev 能否承担候选视觉复核？</h1><p>'+headline+'</p><p class="small">生成时间 '+timestamp+'。所有完成数与指标读取本地证据；再次运行脚本即可更新。报告不发起 API 调用。</p></header>'
    body += '<nav><a href="#usage">当前用法</a><a href="#fixed">候选诊断</a><a href="#online">全轨迹对照</a><a href="#stages">轨迹阶段</a><a href="#workload">调用与耗时</a><a href="#runtime">环境与验证</a><a href="#research">模型来源</a><a href="#limits">解释边界</a></nav><div class="cards">'+cards+'</div>'
    body += '<section>'+usage+'</section>'
    policy = e.load('model_cost_policy.json')
    if policy:
        body += '<section><h2>后续模型与费用约束</h2><p>按用户要求，本轮 240 条轨迹结束后不再新增 3.8 Max 调用。后续主控模型仅用 3.8 Flash 或 3.5 VL；本地 OneJev 诊断继续使用本地服务。</p><p>本轮 Max 成功与失败结果全部保留，作为后续历史基线。直接配对比较需核对任务输入、初始化、评估与运行协议；条件变化必须说明，不能视作同条件对照。</p><p>'+e.link('model_cost_policy.json', '费用约束记录')+' · '+e.link('max_baseline_index.json', 'Max 基线及文件校验索引')+' · '+e.link('max_baseline_index.sha256', '索引校验值')+'</p></section>'
    if ablation:
        body += '<section>'+ablation+'</section>'
    body += '<section><h2 id="fixed">固定候选：复核判决诊断</h2><p>TP：几何合格且 confirm；FN：几何合格但明确 reject；FP：几何不合格却 confirm；TN：几何不合格且明确 reject。uncertain 与错误单列，不算作明确拒绝。全 uncertain 时即使 FP/FN 为零，也没有确认能力证据。</p>'
    body += ''.join(s['body'] for s in stages)+'</section><section>'+online['body']+'</section><section>'+stage_diagnostics+'</section><section>'+workload+'</section><section>'+runtime+'</section><section>'+research+'</section>'
    body += '<section><h2 id="limits">解释边界</h2><ul><li>GT 只提供几何资格：近中心或框重叠不等于精确视觉身份，不能当成人工语义真值。</li><li>固定候选 AVDN 按几何结果分层选样、CityNav 含开发轨迹；14 个主诊断案例与 1 个事后案例独列。重复视角/同轨迹候选相关，不能作为独立随机样本。</li><li>在线每数据集仅 20 条；无预注册等效界、功效保证与正式等效检验，CI 跨零不等于两者等效。地图内相关性未进入轨迹 bootstrap。</li><li>失败保留计数；工具 unavailable 与调用错误分列。相同端口的 SAM3/DINO 后端身份若未冻结，仍存在服务变化的混杂风险。</li><li>性能、时延和 probability 均限于已记录设置；未运行或尚未完成的阶段不外推。</li></ul></section>'
    if e.warnings:
        body += '<p class="warning">'+esc('；'.join(e.warnings))+'</p>'
    body += '<p>'+e.link('completion_audit.json', '逐项完成核对与证据校验')+' · '+e.link('completion_validation.json', '本轮接入、回放和报告验证记录')+'</p>'
    body += '<footer><p>更新命令：<code>python tests/jev/build_report.py</code>。此文件为静态快照，不自动轮询。原始文件链接保留证据，秘密凭据不进入报告。</p><details><summary>输入证据 SHA-256（'+str(len(e.inputs))+' 个文件）</summary>'+table(['文件', 'SHA-256'], [[e.link(p), '<code>'+v+'</code>'] for p,v in sorted(e.inputs.items())])+'</details></footer></main></html>'
    output = e.root/'index.html'
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.html.writing')
    temporary.write_text(body, encoding='utf-8')
    temporary.replace(output)
    return dict(path=str(output), stages=[{k:s[k] for k in ('folder','terminal','expected','errors')} for s in stages],
                online={k:v for k,v in online.items() if k!='body'}, warnings=e.warnings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(build(args.root), ensure_ascii=False))
