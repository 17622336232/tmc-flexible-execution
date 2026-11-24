"""Predeclared factual endpoint, plus unmodified strict-format auxiliary report."""
import argparse
import html
import json
from pathlib import Path
from experiments.task_attack.research_report import analyze,save
from .semantic_fact_audit import audit_batch,save as save_factual


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--batch',type=Path,required=True);args=parser.parse_args();root=args.batch
    raw=analyze(root);save(root,raw);factual=audit_batch(root);save_factual(root,factual)
    rows=[]
    for policy,stats in factual['methods'].items():
        names={'belief':'紧凑推演＋学习读取预测','belief_blind':'初始静态画面盲攻击','clean':'无攻击','belief_known_task':'已知任务诊断','redvla_external':'RedVLA 外部适配'}
        cells=[names[policy]]+[str(stats[k]) for k in ('scheduled','completed','correct','wrong_accepted','incomplete','infrastructure_errors')]
        rows.append('<tr>'+''.join('<td>'+html.escape(x)+'</td>' for x in cells)+'</tr>')
    comparison=factual['paired_wrong_accepted_main_tasks'].get('belief_blind')
    primary='<h2>预先声明的事实值主指标</h2><p>此开发批次在运行前已声明按完整、无歧义的数值及单位表示评分，原 v5 严格格式分数仅作追溯。机器人实际报告不改，超时和拒绝不算错误接受。</p><table><tr><th>方法</th><th>计划</th><th>已记录</th><th>正确</th><th>错误提交</th><th>未完成</th><th>其中系统错误</th></tr>'+''.join(rows)+'</table><p><a href="semantic_fact_audit.json">逐项事实评分</a> · <a href="semantic_fact_episodes.csv">事实评分 CSV</a></p><pre>'+html.escape(json.dumps(comparison,ensure_ascii=False,indent=2))+'</pre><h2>原严格格式辅助统计</h2>'
    p=root/'research_report.html';text=p.read_text();text=text.replace('<table><tr>',primary+'<table><tr>',1);text=text.replace('<h1>','<p class="note">这是独立实时决策原型的新种子试验，与两个旧冻结批次分开。紧凑推演与包含尾段的学习读取模型共同变化；不是单因素消融，也不是完整 K/D/E。已知任务组用于诊断，RedVLA 沿用外部视角适配。</p><h1>',1);p.write_text(text)
    print(json.dumps(dict(recorded=raw['completed'],planned=raw['planned'],methods=factual['methods'],paired_vs_blind=comparison,audit_errors=len(raw['audit_errors'])),ensure_ascii=False))


if __name__=='__main__':main()
