# Góc nhìn nghiên cứu mới cho cell ancestry từ ảnh thưa (CVPR 2028)

Ngày rà soát: 30/09/2026. Phạm vi: **chỉ dùng năm bộ dữ liệu đã có trong repo**, không thu thập thêm. Tài liệu gồm định hướng nghiên cứu và kết quả pilot đầu tiên; các pilot chưa đủ để đưa ra claim paper.

## 1. Đọc lại bằng chứng hiện có

- Bài toán hiện tại là gán mỗi tế bào ở frame cuối `b` về một tế bào ở frame đầu `a`, với lịch quan sát thưa. OIAM có 5 movie thuộc **cùng một thí nghiệm**, khoảng 1,4 triệu quan sát và 14.836 lần phân chia; bốn bộ CTC (HSC, MuSC, HeLa, SIM+) là kiểm tra phụ, gồm sanity và khác miền ([báo cáo v1](output_report/260924_cell_ancestry_baseline_study.md), [nguồn OIAM](https://publikationen.bibliothek.kit.edu/1000183586)). Không thể dùng 5 movie này để khẳng định tổng quát hóa sang thí nghiệm độc lập.
- Ở OIAM, H = 132 phút, phép nối gần nhất qua tất cả frame đạt khoảng 0,70–0,71 ancestry accuracy; hai ảnh đầu–cuối chỉ khoảng 0,26–0,36 với các heuristic v1. Từ stride 1 đến 32, chưa có track trung gian nào bị ẩn hoàn toàn, nhưng accuracy vẫn giảm đến khoảng 0,31. Vậy **mất tương ứng do dịch chuyển/mật độ** xuất hiện trước bài toán suy phân chia ẩn ([báo cáo v1](output_report/260924_cell_ancestry_baseline_study.md)).
- Các phép thử mới hơn, chỉ có ở `outputs/results`, cho thấy học quan hệ cùng họ từ hình học frame đích rồi gán tập thể (`family_set_raw`) trên M4, `ends`, đạt macro accuracy **0,412**, mean set-F1 **0,399**; `chained_lap_c4` là **0,384/0,367** trên đúng 16 window M4. Đây là pilot oracle-mask, chọn tham số trên M3, cần lặp lại theo protocol khóa trước khi dùng trong paper ([mã](../experiments/family_set_raw.py), [kết quả](../outputs/results/family_set_h132_raw_windows.parquet)). Nó gợi ý quan hệ **giữa các target trong cùng frame** có ích hơn một cost độc lập cho từng cặp anchor–target.
- **Không dùng điểm embedding `v0–v3` cũ chưa xác minh provenance để tuyên bố tiến bộ.** Mã đã ghi rõ kênh velocity trong các phép thử legacy được tính bằng khớp GT track ID xuyên thời gian; đây là rò rỉ đáp án ([mã feature](../experiments/train_embed.py), [mã eval](../experiments/eval_embed.py)). Khi đánh giá checkpoint `v3` với velocity bằng 0, M4 `s16` chỉ còn khoảng **0,073** trong artifact hiện có; checkpoint `clean_v1` đạt khoảng **0,297** ở `s16`, **0,336** ở `ends` ([kết quả sạch](../outputs/results/embed_h132_clean_v1.parquet)). Những số này là chẩn đoán; chưa phải so sánh phương pháp có protocol hoàn chỉnh. Cần lưu metadata checkpoint và tách mọi kết quả `oracle_velocity` khỏi bảng paper.
- Mask OIAM có thể dùng ở điều kiện oracle **chỉ sau khi tách instance và đổi ID theo từng frame**. `SEG`/`TRA` gốc chứa ID xuyên thời gian. CTC còn lại chỉ có gold mask thưa, do đó đánh giá chéo miền chủ yếu là oracle-location ([audit](260924_cell_ancestry_v0_report.md)).

## 2. Đối chiếu với các cộng đồng nghiên cứu khác

| Góc nhìn / keywords để tìm tiếp | Liên hệ trực tiếp với dữ liệu | Tiền lệ cần phân biệt | Đánh giá |
|---|---|---|---|
| **Vật liệu biến dạng**: `non-rigid registration`, `deformation graph`, `Lagrangian flow`, `collective motion`, `growth-induced flow` | Khuẩn lạc nở và trượt cục bộ; bù giãn nở toàn cục đã tăng 8–12 điểm ở OIAM nhưng không sửa xáo trộn lân cận. | [Partial Transport for Point-Cloud Registration](https://arxiv.org/abs/2309.15787), [GraphSCNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Qin_Deep_Graph-Based_Spatial_Consistency_for_Robust_Non-Rigid_Point_Cloud_Registration_CVPR_2023_paper.pdf), [DV-Matcher](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_DV-Matcher_Deformation-based_Non-rigid_Point_Cloud_Matching_Guided_by_Pre-trained_Visual_CVPR_2025_paper.html). Nghiên cứu thực nghiệm cũng cho thấy dòng chảy tập thể và pha trộn lineage trong khuẩn lạc; **không được giả sử mọi lân cận giữ nguyên** ([bằng chứng sinh học](https://pmc.ncbi.nlm.nih.gov/articles/PMC6170782/)). | **Ưu tiên 1**, nếu ghép được biến dạng và phân nhánh trong cùng mô hình. Chỉ thêm một bước registration là chưa mới. |
| **Vùng/tập hậu duệ**: `clone territory`, `set prediction`, `partition inference`, `higher-order assignment`, `lineage hypergraph` | Mục tiêu thực là một partition của target theo anchor, thường mỗi anchor có nhiều hậu duệ; pilot `family_set_raw` đã có tín hiệu. | [Moral Lineage Tracing](https://openaccess.thecvf.com/content_cvpr_2016/html/Jug_Moral_Lineage_Tracing_CVPR_2016_paper.html) tối ưu cả lineage; [Trackastra](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09819.pdf) học liên kết pairwise trong cửa sổ; [GNN cell tracker](https://arxiv.org/abs/2202.04731) dùng graph trên detections. Vì vậy novelty phải nằm ở **suy trực tiếp partition nhiều thế hệ giữa hai snapshot**, kết hợp hình học và sự biến dạng, chứ không ở chữ “graph”. | **Ưu tiên 1**, kết hợp với hàng trên để có một câu chuyện phương pháp thống nhất. |
| **Khớp cấu trúc tương đối**: `Delaunay subgraph matching`, `Gromov-Wasserstein`, `topology-aware correspondence`, `local constellation` | Khi dịch chuyển lớn hơn khoảng cách láng giềng, đặc trưng tương đối của một cụm có thể bền hơn tọa độ từng cell. | [Low-frame-rate Delaunay graph matching (ISBI 2011)](https://doi.org/10.1109/ISBI.2011.5872573) đã đề xuất ý này; [RAFT-UP](https://arxiv.org/abs/2603.18249) dùng ràng buộc hình học khi khớp tập không cùng kích thước. | **Thử như ablation/baseline mạnh**. Một graph matcher đơn lẻ khó là novelty đủ cho CVPR 2028. |
| **Vận chuyển khối lượng có sinh trưởng**: `branching transport`, `unbalanced OT`, `growth-aware mass`, `partial OT` | Số target tăng theo phân chia và tế bào rời biên; OT cân bằng khối lượng thường không đúng. | [Branching SDE trajectory inference](https://arxiv.org/abs/2307.07687) và [CellStream](https://arxiv.org/abs/2511.13786) xử lý snapshot ở dữ liệu omics; đó không phải pixel-level ancestry. Repo đã thử OT không học: trên M4 `ends`, uniform OT khoảng **0,364**, thua `chained_lap_c4` **0,384** ([kết quả](../outputs/results/ot_transfer_h132.parquet)). | **Thành phần mô hình**, không tự nó là phương án chính. Cần chứng minh học cost/flow và ràng buộc gia đình mang lại lợi ích thật. |
| **Bất định/khả định danh**: `posterior over lineages`, `set-valued prediction`, `selective risk`, `calibration`, `ambiguity certificate` | Hai snapshot có thể cho nhiều partition gần tương đương; một đáp án cứng che giấu điều đó. | [UAT 2018](https://academic.oup.com/bioinformatics/article/35/7/1221/5089229), [PyUAT](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0337110), và [ICCV 2025 “I dunno!”](https://openaccess.thecvf.com/content/ICCV2025/html/Paul_How_To_Make_Your_Cell_Tracker_Say_I_dunno_ICCV_2025_paper.html) đã nghiên cứu uncertainty của tracker/assignment. | **Đóng góp bổ trợ**: uncertainty ở **tập hậu duệ sau nhiều thế hệ** và tác động lên coverage/F1, không claim chung rằng “tracker biết từ chối”. |
| **Chọn thời điểm quan sát**: `active sensing`, `budgeted frame selection`, `value of information`, `online acquisition policy` | Movie dày sẵn có cho phép mô phỏng chỉ giữ B frame, so với lịch đều/ngẫu nhiên đã có. | [Nghiên cứu chọn khoảng chụp và subsampling cell tracking](https://pmc.ncbi.nlm.nih.gov/articles/PMC8321081/) đã tồn tại; CVPRW 2025 đã nêu các vấn đề low-frame-rate ([paper](https://openaccess.thecvf.com/content/CVPR2025W/CVMI/papers/Gachloo_Low-Frame-Rate_Cell_Tracking_Unmet_Needs_and_Future_Directions_CVPRW_2025_paper.pdf)). | **Hướng phụ có tiềm năng**, nếu policy chỉ dùng ảnh quá khứ/hiện tại và tăng accuracy ở cùng ngân sách. Dữ liệu retrospective không chứng minh giảm quang độc tính thực tế. |

## 3. Đề xuất paper chính: “growth-aware clonal transport”

**Câu hỏi:** Có thể suy trực tiếp một tập hậu duệ cho mỗi anchor từ 2–5 ảnh quan sát, bằng cách mô hình hóa khuẩn lạc như một trường biến dạng có sinh trưởng, thay vì ghép từng cặp tế bào độc lập không?

**Đầu vào:** ảnh, thời gian thực, detections/masks ở đúng các frame được giữ; không có GT ID, hidden-frame feature hay velocity suy bằng GT. **Đầu ra:** một partition của target ở `b` theo anchor ở `a`, cùng xác suất/độ bất định cho mỗi tập, và trạng thái unknown khi không đủ cơ sở. Không bắt mô hình dựng cây division chi tiết ở các frame chưa quan sát.

**Mô hình tối thiểu để kiểm tra giả thuyết:**

1. Từ hai tập detections và ảnh quan sát, ước lượng trường biến dạng **cục bộ, trơn có điều kiện**, cho phép co giãn khác nhau theo vùng; dùng ảnh nền/cạnh mask hoặc khớp cấu trúc point cloud. Không dùng GT lineage để ước lượng trường lúc test.
2. Tạo cost anchor→target sau biến dạng và affinity target↔target từ lân cận, hướng, mask shape. Dùng prior số hậu duệ học ở train và cơ chế partial mass tại biên; không áp cố định “mẹ chỉ có hai target” khi khoảng chụp dài hai chu kỳ.
3. Suy partition toàn cục (structured assignment/OT hoặc decoder có ràng buộc), rồi suy uncertainty trên **partition hoặc từng tập hậu duệ**. Báo cả prediction cứng và accuracy–coverage / set-F1–coverage.

Điểm khác cần chứng minh bằng ablation: `registration only`, `family affinity only`, `OT only`, `registration + affinity`, và mô hình đầy đủ. Nếu phần kết hợp không vượt các thành phần riêng, không có claim phương pháp thống nhất.

## 4. Các phép thử quyết định trước khi đầu tư mô hình lớn

1. **Khả năng giữ cấu trúc cục bộ:** trên OIAM train, đo tỷ lệ cạnh kNN/Delaunay hoặc thứ tự góc lân cận còn đúng sau 16/32/64/132 phút, theo mật độ và vị trí trong khuẩn lạc. So với null hoán vị và bù giãn nở toàn cục. Nếu cấu trúc sụp ở 132 phút, dùng graph chỉ cho bước trung gian, không làm trụ cột ở `ends`.
2. **Trần thực dụng của biến dạng:** ước lượng flow/registration chỉ từ frame quan sát, gán bằng **cùng** decoder `LAP c4`, so với `rc` và cost thô. Đánh giá trên M3; tuyệt đối không chọn flow bằng GT correspondence ở test. Nếu flow không tăng accuracy lẫn set-F1 ở vùng đông, dừng nhánh mechanics.
3. **Thông tin ảnh/mask có thật không:** so `location only`, `location + binary mask shape`, `location + raw image patch`, và patch bị hoán vị giữa tế bào lân cận. Trên OIAM, mask phải được relabel cục bộ trước khi vào model. Nếu hoán vị patch không làm giảm điểm, không claim học visual identity.
4. **Tập hậu duệ so với liên kết pairwise:** giữ nguyên cost và cùng ngân sách nhãn, so decoder độc lập/LAP/OT với decoder partition. Báo macro accuracy, pooled accuracy, mean set-F1, lỗi theo mật độ/`hidden_intermediates`/displacement. Count-MAE đứng riêng với prior 4 hậu duệ; nó không đủ làm metric chính.
5. **Quy trình đánh giá có kiểm soát:** dùng M0–M2 train và M3 validation cho các phép thử nhanh. M4 đã xuất hiện trong nhiều artifact/pilot, nên **không còn là test hoàn toàn chưa từng xem** ở cấp lựa chọn ý tưởng. Khi chốt mô hình, công bố lịch sử này và dùng leave-one-movie-out với cùng quy tắc fit/tune đã khóa để đo độ nhạy; không đổi phương pháp dựa trên từng fold rồi báo các fold ấy như held-out độc lập. Unit thống kê là movie/window, không phải từng cell độc lập. Thử oracle-location trên cả năm bộ; oracle-mask chỉ nơi có mask đầy đủ. Cuối cùng train detector/segmenter bằng ảnh/nhãn đang có và báo chất lượng hai tầng detection→ancestry.

**Ngưỡng go/no-go đề xuất, chưa phải kết quả:** ở `ends`, H=132, mô hình mới nên vượt pilot `family_set_raw` **0,412 accuracy và 0,399 set-F1** trên M4 với biên thực chất (gợi ý ≥ 5 điểm phần trăm trên cả hai, đồng thời không giảm pooled accuracy); có cải thiện ở window đông và ở `u3/u5` hoặc `s16/s32` mà không dùng velocity GT. Đây chỉ là mốc hiệu quả trên dữ liệu phát triển đã được xem, **không phải kiểm định held-out**. Cần báo từng movie/window và độ nhạy các fold; không diễn giải 5 điểm như kiểm định thống kê chắc chắn. Nếu không vượt được, đóng góp hợp lý hơn là benchmark/analysis về giới hạn thông tin và ambiguity, nhưng phải có thí nghiệm xác nhận riêng.

## 5. Ranh giới novelty và lộ trình đến CVPR 2028

- **Không bán ý tưởng chung** “low-frame-rate cell tracking”, “graph matching”, “uncertainty”, “OT” hoặc “adaptive intervals”: các chủ đề này đã có paper. Đóng góp khả dĩ là **joint non-rigid growth field + multi-generation descendant partition + calibrated family-level uncertainty**, cùng một benchmark cho phép cô lập các nguồn khó.
- **Ưu tiên thực hiện:** (i) kiểm toán và khóa kết quả sạch, (ii) phép thử topology/flow/patch ở mục 4, (iii) mô hình partition chung và ablation, (iv) detector thật + transfer location-only, (v) adaptive sampling chỉ nếu mô hình uncertainty được hiệu chuẩn tốt. Tất cả đều dùng dữ liệu hiện có.
- **Claim sinh học cần tránh:** không suy ra đã giảm phototoxicity từ việc bỏ frame sau khi thu thập; không gọi nhãn root xuất hiện muộn là cell mới đi vào; không dùng CTC ít window làm bằng chứng tổng quát hóa rộng. [CVPRW 2025](https://openaccess.thecvf.com/content/CVPR2025W/CVMI/papers/Gachloo_Low-Frame-Rate_Cell_Tracking_Unmet_Needs_and_Future_Directions_CVPRW_2025_paper.pdf) cũng chỉ ra các mơ hồ đánh giá khi tế bào ở biên ảnh.

**Kết luận lựa chọn:** Với đúng dữ liệu đang có, đường có cơ hội nhất là **suy vùng/tập hậu duệ trên khuẩn lạc đang biến dạng**. Nó bám sát failure mode đã đo, có pilot dương nhỏ nhưng thật, và có cầu nối kỹ thuật sang registration/structured prediction của thị giác máy tính. Phần bất định và chọn lịch chụp nên dùng để kiểm định giá trị của phương pháp chính sau khi kiểm tra sạch về leakage và mask provenance.

## 6. Kết quả vòng triển khai đầu tiên (30/09/2026)

Đã thêm bốn script, dùng đúng dữ liệu hiện có: [chẩn đoán topology và flow](../experiments/structure_flow_pilot.py), [ablation hình dạng](../experiments/family_shape_ablation.py), [prior lân cận](../experiments/family_topology_pilot.py), [kết hợp flow với gán tập hậu duệ](../experiments/family_flow_joint_pilot.py). Chúng tạo các file Parquet cùng tên tương ứng trong `outputs/results/` và không sửa nhãn nguồn. M4 là movie phát triển **đã được xem trước**, nên mọi số M4 dưới đây là mô tả, không phải xác nhận held-out mới.

### Cấu trúc còn lại và giới hạn của flow đơn lẻ

Với mỗi anchor còn hậu duệ ở `b`, lấy tâm của tập hậu duệ bằng **GT chỉ cho chẩn đoán**, rồi so đồ thị 6 láng giềng ở `a` với đồ thị các tâm đó ở `b`. Ở H=132 phút, edge recall là **0,793 / 0,803 / 0,777** trên M0/M1/M2 (mức hoán vị ngẫu nhiên **0,240 / 0,265 / 0,242**); M3 là **0,785** so với null **0,218**. Ở các window có >200 anchor, recall vẫn **0,648** so với null **0,023** (23 window gộp M0–M3). Đây là *khả năng tồn tại của cấu trúc họ*, không phải accuracy của thuật toán khớp đồ thị; chỉ tính anchor có hậu duệ quan sát được ([artifact](../outputs/results/topology_k6.parquet)).

Thử dùng OT không nhãn để tạo residual displacement, làm trơn bằng Gaussian rồi áp **cùng LAP c4 decoder** trên M3 `ends`: raw LAP **0,363 accuracy / 0,337 set-F1**, flow khởi tạo từ tọa độ thô tốt nhất **0,367/0,341**. Khởi tạo từ bù giãn nở toàn cục chỉ **0,342/0,335** ở lựa chọn tốt nhất. Trên window đông, bù giãn nở toàn cục thường làm hại; flow dựa vào nó không sửa được lỗi này. Các số này chỉ cho thấy **registration đơn giản chưa đủ**, không phủ định mọi mô hình biến dạng ([artifact](../outputs/results/anonymous_flow_h132_M3.parquet)).

### Hình dạng, topology prior và mô hình kết hợp

| Phép thử H=132, `ends` | M3 accuracy / set-F1 | M4 accuracy / set-F1 | Ghi chú |
|---|---:|---:|---|
| `family_set_raw` | **0,392 / 0,369** | **0,412 / 0,399** | Pilot trước, oracle mask và tâm oracle |
| Chỉ vị trí/lân cận, học lại network cùng kiến trúc | 0,369 / 0,350 | 0,385 / 0,374 | Cùng decoder đã khóa cho mô hình đầy đủ; chưa tune decoder riêng |
| Hoán vị hình dạng mask trong frame (3 seed) | 0,356–0,359 / 0,336–0,340 | 0,369–0,379 / 0,354–0,367 | Negative control cho đặc trưng hình dạng |
| Thêm thông điệp lân cận anchor, chọn gamma trên M3 | 0,397 / 0,375 | 0,408 / 0,394 | **Không tái hiện lợi ích trên M4** |
| **Flow không nhãn + decoder tập hậu duệ**, chọn bandwidth=2, strength=0,5 trên M3 | **0,401 / 0,378** | **0,427 / 0,413** | Kết hợp hai thành phần; chỉ dùng hai frame quan sát lúc inference |

Ở bài toán phân loại cặp target cùng họ, average precision trên M3 là **0,472** với hình dạng so với **0,194** khi chỉ có vị trí/lân cận; hoán vị hình dạng còn **0,165–0,168**. Tuy nhiên pooled accuracy M4 của mô hình đầy đủ là **0,188**, còn mô hình chỉ vị trí là **0,191**: ích lợi hình dạng hiện tập trung ở metric macro, chưa giải được vùng đông ([artifact ablation](../outputs/results/family_shape_ablation_h132.parquet)).

Mô hình kết hợp flow nâng pooled accuracy từ **0,176→0,183** trên M3 và **0,188→0,202** trên M4. Trong các window M4 có >200 anchor, macro accuracy tăng **0,095→0,112** và set-F1 **0,070→0,084**; mức tuyệt đối vẫn thấp. Nó cải thiện 11/17 window M3 và 12/16 window M4 về accuracy; gamma topology prior đơn giản lại giảm trên M4 ([artifact kết hợp](../outputs/results/family_flow_joint_h132.parquet), [artifact topology decoder](../outputs/results/family_topology_h132.parquet)).

**Quyết định sau pilot:** giữ **flow + suy tập hậu duệ** làm đường nghiên cứu chính; không mở rộng prior đồ thị lân cận dạng message passing hiện tại. Mức tăng 0,9–1,5 điểm phần trăm macro còn thấp hơn ngưỡng go/no-go 5 điểm đã đề xuất ở mục 4. Bước kế tiếp là học/ước lượng biến dạng ở cấp **cụm gia đình** thay vì barycenter OT từng target, kiểm tra visual patch trên ảnh thật, rồi đánh giá bằng detector/segmenter train từ nhãn sẵn có. Không tuyên bố phương pháp sẵn sàng cho CVPR lúc này.

## 7. Pilot hướng A: nén genealogy theo ancestry (01/10/2026)

Script: [coarse_grain_pilot.py](../experiments/coarse_grain_pilot.py). M3 (validation), H=132, 17 window mỗi schedule, chỉ dùng vị trí trừ `lumped_area`. Mọi phương pháp dùng chung một mô hình link (OT bán nới lỏng, `eps=0,1`, `tau=3` từ lần tune trước); tham số tăng trưởng theo tuổi fit trên M0–M2. M4 chưa chạy. Trackastra chưa cài nên không có; `pm_lap` là perturb-and-MAP trên LAP capacity 4, chỉ là proxy cho assignment sampling của PyUAT, và nhiệt độ của nó được chọn trên chính M3 (có lợi cho đối chứng).

| Phương pháp | `s16` acc / set-F1 | `s32` | `u5` | Trạng thái lưu (số phần tử, window lớn nhất) |
|---|---:|---:|---:|---:|
| `direct_ends` (chỉ hai frame đầu–cuối) | 0,353 / 0,348 | 0,353 / 0,348 | 0,353 / 0,348 | 1,24 triệu |
| `map_lap_c4` (một history) | 0,393 / 0,366 | 0,303 / 0,258 | 0,307 / 0,259 | 1,5 nghìn |
| `pm_lap` T=0,3, 16 mẫu | 0,366 / 0,334 | 0,307 / 0,257 | 0,303 / 0,257 | 24 nghìn |
| `sampled_16` | 0,427 / 0,419 | 0,366 / 0,359 | 0,361 / 0,355 | 24 nghìn |
| `sampled_256` | 0,428 / 0,422 | 0,378 / 0,374 | 0,359 / 0,354 | 380 nghìn |
| `lumped` (nén, không trạng thái) | 0,428 / 0,422 | 0,376 / 0,371 | 0,360 / 0,355 | 1,24 triệu |
| `sampled_age_16` | 0,423 / 0,414 | 0,381 / 0,375 | 0,374 / 0,365 | 48 nghìn |
| `lumped_age` (nén + trạng thái tuổi) | 0,449 / 0,443 | 0,383 / 0,378 | 0,383 / 0,378 | 1,25 triệu |
| `oracle_age` (tuổi từ GT, trần) | 0,465 / 0,460 | 0,392 / 0,387 | 0,390 / 0,386 | 1,24 triệu |
| `lumped_area` (diện tích mask oracle) | 0,594 / 0,586 | 0,453 / 0,446 | 0,433 / 0,428 | 1,24 triệu |

- **Không có trạng thái, nén không hơn sampling về accuracy.** `lumped` chính là giới hạn S→∞ của `sampled_S` và đã có sẵn trong repo (`ot_ancestry(soft=True)`); 16–64 mẫu đạt cùng accuracy và set-F1 (chênh ≤ 0,01, số window hơn/kém xấp xỉ nhau) với trạng thái nhỏ hơn 13–52 lần. `lumped` chỉ hơn ở NLL (`s16`: 7,29 so với 8,52 ở 16 mẫu, 7,72 ở 256 mẫu).
- **Posterior tự nó kém.** Ở `s16`, NLL 7,29 của `lumped` còn tệ hơn đoán đều trên anchor (5,77), ECE 0,26; tăng `eps` lên 1,0 hạ ECE còn 0,04 nhưng accuracy giảm 0,428→0,329. Ở window >200 anchor mọi phương pháp chỉ đạt 0,06–0,09.
- **Trạng thái tuổi có ích nhưng trần thấp.** `oracle_age` hơn `lumped` 3,7 / 1,6 / 3,0 điểm (`s16`/`s32`/`u5`). Bản nén mean-field `lumped_age` giữ được 2,1 / 0,7 / 2,3 điểm và không kém `sampled_age_16` (hơn 2,6 điểm ở `s16`, ngang nhau ở `s32`/`u5`) với khoảng một phần ba thời gian (65 s so với 203 s ở `s16`). Mức tăng ở `s32` nằm trong nhiễu window (10 hơn / 6 kém).
- **Đặc trưng quan sát được lấn át trạng thái lịch sử.** Diện tích mask hơn `oracle_age` 12,8 / 6,1 / 4,3 điểm; PyUAT với mô hình tăng trưởng (`FO+G+O+DD`, kết quả cũ) đạt 0,790 ở `s16` và 0,587 ở `s32`.

**Quyết định:** theo đúng tiêu chí “bỏ hướng nếu” của đề xuất, hướng A không tạo đóng góp phương pháp độc lập trên dữ liệu này: đối chứng sampling đạt cùng trade-off khi không có trạng thái, và phần trạng thái cần giữ chỉ đáng 2–4 điểm. Giữ `lumped_age` như một thành phần suy luận rẻ nếu mô hình chính cần posterior; chất lượng mô hình link (tăng trưởng, hình dạng, biến dạng) mới là nút thắt. Chưa kiểm tra: PyUAT sampler thật, Trackastra, classifier ancestry có học, trạng thái giàu hơn một biến tuổi, và M4.
