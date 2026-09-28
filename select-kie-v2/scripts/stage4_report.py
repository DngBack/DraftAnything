"""Export a figure and archive fingerprints after the full stage4 evaluation."""
from pathlib import Path
import hashlib,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'results/stage4'
def main():
    metrics=json.loads((out/'metrics.json').read_text());fig,ax=plt.subplots(figsize=(9,4))
    labels=['Baseline','Length control','Clarified'];xs=range(3)
    for offset,tag,color in [(-.18,'qwen2b','#377eb8'),(.18,'qwen9b','#e87524')]:
        vals=[100*metrics['measures'][tag+'-'+s]['accuracy'] for s in ['baseline','length_control','clarified']]
        ax.bar([x+offset for x in xs],vals,width=.35,label=tag,color=color)
        for x,v in zip(xs,vals):ax.text(x+offset,v+1,f'{v:.1f}',ha='center',fontsize=9)
    ax.set_xticks(list(xs),labels);ax.set_ylim(0,105);ax.set_ylabel('Official field accuracy (%)');ax.legend();ax.set_title('Fresh100-document holdout; six fields per document')
    fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=180);plt.close(fig)
    paths=list(out.glob('*.json'))+list(out.glob('*.jsonl'))+list((ROOT/'scripts').glob('*stage4*.py'))+[ROOT/'docs/stage4-protocol.md',ROOT/'scripts/stage4_extract.py',ROOT/'scripts/stage4_evaluate.py',ROOT/'docs/stage4-results.md']
    archive={'sha256_at_archive':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name!='archive-manifest.json'},'human_labels':'pending','source_model_card':'https://huggingface.co/Qwen/Qwen3.5-9B'}
    (out/'archive-manifest.json').write_text(json.dumps(archive,indent=2))
if __name__=='__main__':main()
