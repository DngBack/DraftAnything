# VietCompose: học chữ hoa tiếng Việt hiếm bằng chuyển giao thành phần

Cập nhật triển khai: dữ liệu HDF5 đã được cấp quyền đọc; hướng chính theo yêu cầu người dùng là fine-tune pretrained best 175.500. Xem `../acl_longtail/STATUS.md`. Nhánh scratch là exploratory riêng, không phải model đích.

Ngày: 2026-11-14. Trạng thái: thiết kế nghiên cứu tiếp nối; chưa triển khai các thuật toán mới bên dưới, chưa chạy training mới. Không xem pilot adapter cũ là bằng chứng thành công.

## 1. Câu hỏi và bằng chứng dữ liệu

Có thể học `Ễ` trong phong cách một writer chưa từng viết `Ễ` trong support, nhờ bằng chứng về thân chữ hoa, dấu mũ và dấu ngã từ những tổ hợp khác không? Cần đo riêng khả năng tái tạo hình chữ, bố trí dấu và phong cách cá nhân.

Đếm lại từ artifact `protocol/coverage.json`, trên split gốc trước lọc tổ hợp:

| Ký tự | Train writer thật | HAT OCR | Development |
|---|---:|---:|---:|
| E | 366 | 2029 | 19 |
| Ê | 0 | 1004 | 0 |
| Ẽ | 0 | 73 | 0 |
| Ễ | 0 | 100 | 0 |
| ê | 2730 | 4206 | 313 |
| ẽ | 468 | 149 | 71 |
| ễ | 166 | 188 | 25 |
| Ấ | 39 | 303 | 5 |

Đây là số lần ký tự xuất hiện, không phải số ảnh độc lập hay số writer. Nguồn: `LONG_TAIL_COVERAGE.json`. Raw HDF5 ở đường dẫn nguồn hiện không truy cập được trong môi trường này; chưa audit lại writer diversity hay pixel.

Như vậy `Ễ` không chỉ là lớp ít mẫu: nó là **zero-example trong miền writer thật**, nhưng có ảnh HAT và giám sát OCR. Có bằng chứng về chữ thường `ễ`, chữ hoa E và một số chữ hoa có mũ như Ấ. Không được giả định đã có Ê thật để làm bước trung gian.

Bốn dạng thiếu dữ liệu:

1. Thiếu toàn bộ glyph `(base, shape, tone, case)`.
2. Có từng thành phần nhưng thiếu tương tác: ngã + mũ + uppercase.
3. Có tổ hợp ở writer khác, thiếu tại writer mục tiêu.
4. Có ở HAT/font, thiếu trong miền ảnh viết tay writer thật.

Một hướng mạnh cần chứng minh chuyển giao qua các trục này, thay vì chỉ tăng tần suất prompt NGUYỄN.

## 2. Bài học từ nghiên cứu trước và pilot hiện tại

DM-Font dùng memory cho thành phần; LF-Font học phong cách cục bộ và factorization; MX-Font dùng nhiều chuyên gia cục bộ. Vì vậy tách chữ thành bộ phận hoặc thêm memory đơn thuần không phải đóng góp mới.

Class-balanced loss dựa trên effective number xử lý mất cân bằng khi có mẫu; n=0 không cung cấp bằng chứng hình ảnh. Logit adjustment là kỹ thuật cho phân loại, không được áp nguyên xi vào GAN/CTC rồi kỳ vọng generator học dấu. Các kỹ thuật này là đối chứng, không phải lời giải đầy đủ.

Pilot query adapter cũ: CER nguồn 13,63%, factorized 15,38%, whole-glyph 16,32%. Renderer lớp mực mới chỉ chạy font tổng hợp. Vì thế hướng tiếp theo phải kiểm tra giám sát và dữ liệu trước khi mở rộng mạng.

## 3. Giả thuyết có thể bác bỏ

H1: Chia sẻ dấu giữa chữ thường và hoa có điều kiện hình học tốt hơn bảng embedding glyph độc lập ở cùng ngân sách dữ liệu.

H2: Học tương tác `(modifier, tone, case)` tốt hơn chỉ bảo đảm từng thành phần xuất hiện nhiều. Nếu balanced glyph sampling bằng mô hình đề xuất thì lợi ích có thể do sampling.

H3: Memory thành phần của writer cải thiện độ giống writer khi support có bằng chứng phù hợp. Khi support không có ngã, model chỉ có thể dự đoán từ prior và nét chung; không có căn cứ bảo đảm đúng cách writer đó viết ngã.

H4: Thu thập ít ảnh thật theo độ thiếu thành phần mang lại hiệu quả tốt hơn cùng số ảnh chọn ngẫu nhiên. Kết quả phải đo bằng đường cong ngân sách, không chỉ một mốc.

## 4. Kiến trúc đề xuất

### 4.1. Giữ cơ chế thân/mũ/thanh riêng và thêm chuyển giao hoa/thường

Dùng renderer v2 làm một ứng viên, cùng baseline VATr và query adapter. Chữ được biểu diễn `(base, modifier, tone, case)`; hoa/thường là điều kiện cấu trúc, không phải resize chữ thường thành chữ hoa.

- Body branch: base + case + ngữ cảnh chữ + style. Không nhận tone/modifier để ngăn nhánh thân vẽ giấu dấu.
- Mark appearance: template mũ/ngã chia sẻ giữa các base, kết hợp residual style.
- Geometry: dự đoán anchor, scale, offset và quan hệ mũ–ngã từ body bounding box, case, modifier và lân cận trong từ.
- Presence gate: không có dấu thì lớp mực dấu bằng 0.
- Word refiner: giữ nối nét và khoảng cách; kiểm tra nó có tự xóa/vẽ lại dấu để vượt ràng buộc hay không.

Ví dụ mục tiêu: E thật hỗ trợ thân chữ hoa; ê/ễ hỗ trợ mũ và ngã; Ấ hỗ trợ hình học dấu trên chữ hoa có mũ; HAT cung cấp prior hình dạng Ê/Ễ. Đây là giả thuyết nguồn chuyển giao, không phải phép cắt dán được bảo đảm đúng.

### 4.2. Memory phong cách có ghi nhận bằng chứng

Memory chứa feature thành phần, writer, case, nguồn, độ tin cậy định vị và số ảnh độc lập. Không dùng ID ảnh HAT làm writer.

Feature theo thành phần cần mask/box thật hoặc alignment đã kiểm tra. Transcript cấp từ không tự cung cấp crop dấu đúng; dấu ngã có thể nhập nét mũ hoặc thân. Trước tiên gán nhãn tập nhỏ cho anchor/box và đánh giá bộ định vị trên writer giữ lại. Không gọi weak-attention map là ground truth.

Khi có thành phần trong support: attention lấy bằng chứng cục bộ. Khi thiếu: trộn prior quần thể với style toàn cục theo gate được học/calibrate bằng các episode che thành phần. Gate là độ tin cậy dự đoán cần kiểm định, không được báo như xác suất đúng nếu chưa calibrate.

### 4.3. Episodic training để thực sự học chuyển giao

Trong mỗi episode, chọn writer, target và support khác ảnh; có ba chế độ:

- support có thành phần cần thiết nhưng không có toàn bộ glyph target;
- support chỉ có bằng chứng hoa hoặc thường một phía;
- support thiếu một thành phần, model phải dùng fallback.

Train bằng target thật có sẵn; che bằng chứng trong support để mô phỏng thiếu dữ liệu. Không tự gán target ảnh giả cho glyph chưa có như ground truth thật. Các episode glyph phổ biến phải dùng cùng cơ chế để có đủ bài học trước khi đánh giá glyph hiếm.

## 5. Sampling theo tương tác và loss

Audit số ảnh độc lập, số writer và coverage cho glyph, component×case, shape×tone×case và base×case, tách real/HAT/synthetic. Số ký tự lặp trong một ảnh không được coi là nhiều bằng chứng phong cách.

Sampler đề xuất là mixture của sampling tự nhiên và sampling phủ nhóm còn thiếu; trọng số nghịch effective number có cap, chỉ định nghĩa trên nhóm có mẫu. Nhóm zero-count chuyển sang danh sách cần transfer/thu thập, không gán trọng số vô hạn. Không nhân độc lập mọi trọng số hiếm vì có thể tạo cực trị.

Giữ real/HAT/font thành các nhánh có source tag và ngân sách riêng. Bổ sung counterfactual hợp lệ như Ê/Ế/Ề/Ể/Ễ/Ệ để kiểm tra sửa đúng dấu nhưng giữ thân. Chỉ dùng loss bất biến trên body branch; không ép hai ảnh từ viết thật khác nhau phải trùng pixel toàn ảnh.

Loss ứng viên:

`L = L_image + λf L_feature + λc L_char + λm L_mark + λg L_geometry + λs L_style`.

- Pixel/feature: paired target thật; luôn có đối chứng feature-only để phát hiện nhòe.
- Character: OCR nội dung; không phải evaluator độc lập.
- Mark/geometry: mask/anchor được gán nhãn hoặc font có layer thật; pseudo-label phải có nhánh ablation và lọc chất lượng.
- Style: evaluator/encoder được kiểm định, tránh học nền giấy/nguồn thay writer.
- Mỗi loss được thêm riêng; hệ số chọn trên development, không tune trên benchmark cuối.

Không dùng chỉ global pixel L1 cho dấu nhỏ. Chuẩn hóa vùng mực dấu theo diện tích, có penalty false-positive khi dấu vắng và giữ metric toàn ảnh; nếu không model có thể giảm lỗi bằng xóa/nhòe dấu.

## 6. Nghiên cứu dữ liệu: zero-shot và few-shot là hai track riêng

Track A, transfer không có target handwriting: loại mọi ảnh/nhãn chứa tổ hợp giữ lại khỏi target, support, corpus sinh, OCR gradient và memory. Dùng training sạch mới; checkpoint legacy chỉ cho phép track thích nghi có exposure. Công bố canonical glyph target có được xem hay không và khớp quyền truy cập giữa các baseline. Kiểm kê cả font pretraining nếu tuyên bố không có hình target từ mọi nguồn.

Track B, long-tail thực tế: cho phép HAT và một lượng target thật kiểm soát; công bố exposure từng nguồn. Nếu HAT đã chứa Ễ thì gọi là thiếu dữ liệu miền writer thật, không gọi chưa thấy Ễ toàn cục.

Ngân sách 0/1/5/20 ảnh target thật phải là các tập train lồng nhau, tách writer evaluation. Đếm số writer cùng số ảnh. Thử một writer nhiều ảnh so với nhiều writer ít ảnh ở ngân sách bằng nhau. Với cá nhân hóa bằng K ảnh chính writer test, báo thành track adaptation riêng và giữ target test khác ảnh.

Thu thập hai nhóm:

- Training acquisition: chọn prompt tối đa hóa coverage còn thiếu của glyph và cặp tương tác; cap số prompt mỗi writer, có chữ riêng và trong từ.
- Evaluation: writer hoàn toàn riêng; glyph Ễ/Ắ/Ỡ/... và nhiều ngữ cảnh, gồm NGUYỄN nhưng không chỉ NGUYỄN. Đặt tên riêng và từ ít gặp để giảm học thuộc một từ.

Khởi đầu có thể đề xuất 20 writer mới và vài chục prompt mỗi người, nhưng đây là ngân sách pilot cần thỏa thuận khi thu thập, chưa phải power calculation. Không tự thu thập hoặc liên hệ người khác trong đợt này.

Giữ ảnh nguồn độ phân giải cao. So 32px và 64px riêng với ngân sách khớp; nếu nguồn chỉ còn 32px thì phóng to không khôi phục chi tiết dấu.

## 7. Thứ tự thí nghiệm và tiêu chí dừng

E0: audit coverage có case/tương tác/writer; hoàn thiện nhãn geometry và tập đánh giá hoa hiếm. Không bắt đầu claim zero-shot khi trainer chưa thực thi bộ lọc tất cả nhánh.

E1: VATr glyph với sampling tự nhiên vs balanced glyph vs balanced interaction, cùng source/budget. Mục tiêu tách tác động dữ liệu khỏi kiến trúc.

E2: whole-glyph vs query factorized vs renderer; thêm body/case geometry rồi memory từng bước. Khớp active parameter khi có thể, công bố FLOPs, updates và runtime. Không so GAN outer step với adapter update như cùng đơn vị.

E3: support đủ/thếu ngã, có/không có chữ hoa; ablation sharing giữa hoa–thường, geometry, memory và episodic hiding. Thêm arbitrary-factor control để kiểm tra lợi ích đến từ phân rã đúng.

E4: đường cong 0/1/5/20 ảnh thật; acquisition theo coverage vs ngẫu nhiên trên cùng pool và cùng budget. Ít nhất 3 seed cho nhánh quyết định; bootstrap theo writer, không coi nhiều prompt cùng writer là độc lập.

Primary metrics: macro exact accuracy trên glyph hoa đuôi dài do người đọc/nhãn độc lập xác nhận, tone/modifier/case errors có alignment, writer fidelity và paired blind preference. Báo micro CER để kiểm soát suy giảm phần head, cùng mark placement/absence trên subset gán nhãn. Tách standalone/context, genuine/HAT, seen/unseen writer và confidence interval.

Chọn head/tail bằng tần suất train trước khi xem test; zero là bucket riêng, không trộn với 1–20 hay >20. Ngưỡng bucket có thể hiệu chỉnh theo histogram trước khi khóa protocol.

Thành công: cải thiện tail qua nhiều seed và writer, không làm head/style suy giảm vượt biên đã đăng ký trên development. Không đặt biên số tùy tiện sau khi thấy test. Dừng mở rộng nếu chỉ OCR tốt hơn, dấu bị nhòe, hoặc balanced sampler baseline đã giải thích toàn bộ lợi ích. Kết quả âm vẫn trả lời được dữ liệu hay kiến trúc đang là nút thắt.

## 8. Đóng góp có thể bảo vệ và bước triển khai

Đóng góp ứng viên: benchmark thiếu dữ liệu chữ hoa theo tổ hợp×case×writer, cơ chế chuyển giao thành phần có xét bằng chứng và hình học, và định lượng lợi ích thu thập dữ liệu theo coverage. Chưa khẳng định mới toàn cục hay SOTA.

Bước tiếp theo trong code: audit metadata có writer và pair counts → split guard dùng chung cho mọi loader/corpus/OCR/memory → baseline sampler → paired real renderer training → memory có định vị đã kiểm tra. Chưa cần sửa baseline mặc định hay khởi chạy tiếp run dài hiện có.

## Tài liệu nguồn đã đối chiếu

- DM-Font: https://arxiv.org/abs/2005.10510
- LF-Font: https://arxiv.org/abs/2009.11042
- MX-Font: https://arxiv.org/abs/2104.00887
- Class-Balanced Loss: https://arxiv.org/abs/1901.05555
- Logit adjustment: https://arxiv.org/abs/2007.07314
- Repo: RESEARCH_PROPOSAL.md, 261114_report_02_Ket_qua_pilot_VietCompose.md, orthography.py, renderer.py, prepare_protocol.py, train_vietnamese.py.
