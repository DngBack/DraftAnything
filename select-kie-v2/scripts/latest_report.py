"""Archive the full outcome, including stronger-baseline and sensitivity findings."""
from pathlib import Path
import csv,hashlib,json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT.parent/'selective-kie'))
from common import rc_curve

def main():
    s5=json.loads((ROOT/'results/stage5/metrics.json').read_text());s6=json.loads((ROOT/'results/stage6/metrics.json').read_text());sens=json.loads((ROOT/'results/stage6/sensitivity.json').read_text());tree=json.loads((ROOT/'results/stage5/semantic-hgb-metrics.json').read_text());field=json.loads((ROOT/'results/stage6/field-threshold-diagnostic.json').read_text())
    lines=['# Stage5: learned relation pilot','', 'Exploratory after viewing stage4; five fixed seeds per architecture. Generic38353 parameters versus relation37601 (generic2% larger). Frozen MiniLM encoder and identical raw information; BCE correctness only, no binding labels or auxiliary binding loss.','', '| Dataset / extractor | Generic AURC | Relation AURC | Generic coverage/risk | Relation coverage/risk |','|---|---:|---:|---:|---:|']
    for key,label in [('test','Stage2 OCR2B test300'),('stage4-qwen2b','Stage4 image2B clarified100'),('stage4-qwen9b','Stage4 image9B clarified100')]:
        a=s5['measures']['generic'][key];b=s5['measures']['relation'][key];lines.append(f"| {label} | {a['aurc']:.5f} | {b['aurc']:.5f} | {a['coverage']:.2%}/{a['risk']:.2%} | {b['coverage']:.2%}/{b['risk']:.2%} |")
    lines+=['','These thresholds are selected empirically at alpha5% on50 old calibration documents. Risk exceeds5% on the old OCR test; no risk-control guarantee. All seed AURCs and development stopping traces are retained in metrics/training-traces. Test300 has already been used, and stage4 was inspected before designing this head.','', 'Equal-information semantic HGB is an additional post-confirmation diagnostic, fixed prior HGB config without target fitting or tuning. The architecture result must be read with [fresh confirmation](stage6-results.md), not as proof of a novel binding method.','', '[Protocol](stage5-protocol.md). Encoder: [official MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).']
    (ROOT/'docs/stage5-results.md').write_text('\n'.join(lines)+'\n')
    lines=['# Stage6: xác nhận head đã khóa trên100 tài liệu mới','', '**Đã xác nhận gain so với MLP generic; chưa chứng minh superiority so với HGB hoặc gain riêng cho binding.**','', '100 tài liệu không trùng filename/OCR với800 tài liệu cũ. Qwen3.5-9B, schema rõ nghĩa, toàn bộ trang1280px;100 predictions/600 quyết định. Không fit/tune/calibrate lại head trên100 này. Head và scaler kiểm tra SHA256 trước khi chấm. Năm seed cố định lấy trung bình score. Primary official matcher, bootstrap2000 theo document; CI conditional trên head và threshold đã fit.','',f"Extractor accuracy: **{s6['extractor_accuracy']:.2%}**; normalized secondary: {s6['normalized_extractor_accuracy']:.2%}. Confidence extractor>0.99 vẫn sai{s6['high_confidence_errors']}/{s6['high_confidence_decisions']}. Không output cap/missing key. Confidence pipeline dùng public annotation OCR; chưa phải image-only pipeline và chưa kiểm tra quality OCR thực tế.",'', '| Head | AURC ↓ | Coverage | Lỗi/accept | Risk quan sát |','|---|---:|---:|---:|---:|']
    for name,label in [('generic','MLP generic, cùng thông tin'),('relation','Learned relation')]:
        m=s6['measures'][name];lines.append(f"| {label} | {m['aurc']:.5f} | {m['coverage']:.2%} | {m['errors']}/{m['accepted']} | {m['risk']:.2%} |")
    for name,m in s6['scalar_only_references'].items():lines.append(f"| {name}, scalar-only reference | {m['aurc']:.5f} | {m['coverage']:.2%} | {m['errors']}/{m['accepted']} | {m['risk']:.2%} |")
    m=tree['measures']['stage6'];lines.append(f"| HGB semantic, cùng thông tin; posthoc | {m['aurc']:.5f} | {m['coverage']:.2%} | {m['errors']}/{m['accepted']} | {m['risk']:.2%} |")
    b=s6['paired_bootstrap'];lines+=['',f"Primary ΔAURC relation−generic: {b['relation_minus_generic_aurc']:+.5f}, CI95% [{b['aurc_ci95'][0]:+.5f}, {b['aurc_ci95'][1]:+.5f}]. Coverage Δ{100*b['coverage_difference']:+.2f} điểm %, CI [{100*b['coverage_ci95'][0]:+.2f}, {100*b['coverage_ci95'][1]:+.2f}]. Hai risk là point estimate; không khẳng định chênh risk có ý nghĩa hay certificate5%.",'', '## Những đối chứng làm hẹp kết luận','', '1. **Coverage tăng phần lớn nhờ ABSENT.** Relation chấp nhận94/150 field absent (1 lỗi), generic0. Trên450 field có giá trị: relation244/450 (54.22%,8 lỗi), generic205/450 (45.56%,7 lỗi).94/133 quyết định tăng thêm là ABSENT. Không quy toàn bộ gain thành phát hiện binding.']
    n=sens['normalized_secondary'];p=sens['present_only_posthoc'];h=tree['stage6_relation_minus_semantic_hgb_aurc']
    lines += [f"2. **Normalized sensitivity chưa xác nhận ranking gain:** Δ{n['difference']:+.5f}, CI [{n['ci95'][0]:+.5f}, {n['ci95'][1]:+.5f}]. Present-only AURC diagnostic cũng có CI chứa0: Δ{p['difference']:+.5f}, CI [{p['ci95'][0]:+.5f}, {p['ci95'][1]:+.5f}].",f"3. **HGB cùng raw embeddings chưa bị vượt rõ ràng:** ΔAURC relation−HGB {h['difference']:+.5f}, CI [{h['ci95'][0]:+.5f}, {h['ci95'][1]:+.5f}]. HGB có Brier tốt hơn; baseline này thêm sau primary confirmation, nên là diagnostic. HGB fit train200 với cấu hình cố định, còn hai neural head dùng thêm dev50 cho early stopping; không quy phép so này thành tuning/supervision budget hoàn toàn bằng nhau.",'4. **Calibration là confounder lớn.** Với threshold riêng từng field, chọn chỉ trên calibration50 cũ: semantic-HGB coverage47.50%, risk12/285=4.21%; relation46.50%,10/279=3.58%; generic34.17%,9/205=4.39%. Đây là posthoc, không thay primary ngưỡng đã khóa.','5. **Không có per-field risk guarantee.** Ở primary pooled threshold, registrant_name của relation sai3/57=5.26%, signer_name1/12=8.33%, foreign principal chỉ accept1/100. Overall2.66% không chứng minh mọi field dưới5%.','6. **Chưa đủ novelty cho claim CVPR9/10.** Một shared projection và interaction MLP không tự xác lập đóng góp mới. Gold/span scope, reviewer audit, model family khác và dataset khác còn thiếu.','', '## Quyết định nghiên cứu tiếp','', 'Giữ head này làm prototype. Ưu tiên audit200 quyết định đã chuẩn bị và thống nhất atomic-value/ABSENT/role ontology trước khi dùng binding supervision. Sau đó kiểm tra calibration theo field/ABSENT với cùng thông tin và nhãn cho mọi baseline; cần nguồn khác và model family khác.89 Short-Form còn chưa dùng, nhưng không đủ thay thế generalization ngoài dataset. Không tune thêm trên stage6 rồi tiếp tục gọi nó là confirmation mới.','', '[Protocol](stage6-protocol.md), [audit hướng dẫn](stage4-audit-guide.md), [PDF qualitative checks by AI](stage4-qualitative.md).']
    (ROOT/'docs/stage6-results.md').write_text('\n'.join(lines)+'\n')
    rows=list(csv.DictReader((ROOT/'results/stage6/scores.csv').open()));err=np.array([1-int(r['correct']) for r in rows]);fig,ax=plt.subplots(figsize=(8,4.5))
    for name,label,color in [('generic','Generic MLP','#777777'),('relation','Relation head','#139b83')]:
        score=np.array([float(r[name+'_score']) for r in rows]);cov,risk,_=rc_curve(score,err);ax.plot(cov,risk,label=label,color=color);m=s6['measures'][name];ax.scatter(m['coverage'],m['risk'],color=color,s=50)
    sem=np.load(ROOT/'results/stage5/semantic-hgb-scores.npz')['stage6'];cov,risk,_=rc_curve(sem,err);ax.plot(cov,risk,label='Semantic HGB (posthoc)',color='#bc802d',alpha=.85)
    ax.axhline(.05,ls='--',color='#aaaaaa',label='5% reference, no guarantee');ax.set_xlabel('Coverage');ax.set_ylabel('Observed field error risk');ax.set_ylim(0,.25);ax.legend();ax.set_title('Frozen neural heads; fresh100-document holdout');fig.tight_layout();fig.savefig(ROOT/'results/stage6/risk-coverage.png',dpi=180);plt.close(fig)
    master=ROOT/'docs/research-progress.md';text=master.read_text();heading='## Cập nhật stage4–6: learned head và xác nhận mới'
    if heading not in text:
        update='''## Cập nhật stage4–6: learned head và xác nhận mới

Đã chạy thêm700 extraction (stage4:600; stage6:100), tạo4.800 training/evaluation decisions cho prototype và huấn luyện10 head neural (5 seed/architecture). Cùng thông tin, generic38353 parameters và relation37601. Không fine-tune extractor/encoder; không có auxiliary binding labels.

Stage4:9B với schema rõ nghĩa đạt76.00% accuracy, hơn prompt dài tương đương26.50 điểm % (CI[23.33,29.50]);2B đạt70.00%. AI PDF spot checks xác nhận ví dụ lấy cá nhân thay primary registrant và phát hiện gold/span ambiguity. Human audit200 decision vẫn trống,20 overlap sẵn cho reviewer2.

Stage6 giữ nguyên head/calibration trên100 tài liệu mới: relation AURC0.05715, coverage56.33%,9/338 lỗi (2.66%); generic MLP0.06588,34.17%,7/205 (3.41%). ΔAURC CI[−0.01715,−0.00049]. Đây là gain đã xác nhận theo document so với MLP, không phải risk certificate hay binding-specific result.

**Giới hạn quyết định:**94/133 accept tăng thêm là ABSENT; normalized/present-only AURC CI chứa0. HGB nhận cùng embedding chưa bị vượt rõ ràng (ΔCI chứa0), và threshold theo field làm coverage HGB47.50% gần relation46.50%. Prototype có ích nhưng chưa đủ claim novelty hoặc9/10CVPR. Cần audit/ontology và nguồn/model family khác.

[Stage4](stage4-results.md), [Stage5 pilot](stage5-results.md), [Stage6 confirmation và đối chứng](stage6-results.md), [audit viewer](../results/stage4/audit-viewer.html), [risk–coverage figure](../results/stage6/risk-coverage.png).

'''
        text=text.replace('## Những gì đã thực hiện trong lượt nghiên cứu tiếp',update+'## Những gì đã thực hiện trong lượt nghiên cứu tiếp');master.write_text(text)
    for stage in [4,5,6]:
        out=ROOT/f'results/stage{stage}';paths=list((ROOT/'scripts').glob('*.py'))+list((ROOT/'docs').glob(f'stage{stage}*.md'))+list(out.glob('*.json'))+list(out.glob('*.jsonl'))+list(out.glob('*.pt'))+list(out.glob('*.npz'))+list(out.glob('*.joblib'))+list(out.glob('*.csv'))+[ROOT/f'docs/stage{stage}-protocol.md',ROOT/f'docs/stage{stage}-results.md']
        (out/'archive-manifest.json').write_text(json.dumps({'sha256_at_archive':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.name!='archive-manifest.json'},'human_audit_status':'pending'},indent=2))
    print('Reports, plot and archive manifests written')
if __name__=='__main__':main()
