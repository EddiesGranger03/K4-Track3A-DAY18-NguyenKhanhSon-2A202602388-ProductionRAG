# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** NguyenKhanhSon
**Khóa:** K4 - Track 3A

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.0000 | 0.0000 | +0.0000 |
| Answer Relevancy | 0.0000 | 0.0000 | +0.0000 |
| Context Precision | 0.0000 | 0.0000 | +0.0000 |
| Context Recall | 0.0000 | 0.0000 | +0.0000 |

> Lưu ý quan trọng: cả 2 pipeline đều retrieval thành công (37/37 unit tests pass,
> Qdrant index 57 basic chunks / hierarchical children, BM25 + RRF + rerank hoạt động).
> Scores 0.0 là do **RAGAS không chấm được**: `OPENAI_API_KEY` trong `.env` là key
> dạng `sk-or-v1...` (OpenRouter), gọi thẳng `api.openai.com` nên RAGAS báo
> `AuthenticationError 401 Incorrect API key` cho cả 80 jobs (20 câu × 4 metrics).
> Answer cũng fallback về `contexts[0]` thay vì LLM generation vì cùng lỗi 401.
> Do đó bảng điểm không phản ánh chất lượng retrieval.

## Bottom-5 Failures

Lấy từ `reports/ragas_report.json` (tất cả score 0.0, sắp xếp theo thứ tự test set).
Phân loại theo 6 dạng câu hỏi trong `test_set.json`:

### #1 — Version conflict (phép năm v2023 vs v2024)
- **Question:** Nhân viên được nghỉ bao nhiêu ngày phép năm?
- **Expected:** 15 ngày (v2024 hiện hành), v2023 là 12 ngày đã bị thay thế.
- **Got:** context top-1 (fallback, không qua LLM do 401).
- **Worst metric:** faithfulness (0.0)
- **Error Tree:** Output sai → Context đủ? Có (cả 2 versions đều retrieve được) → Query OK? Có → Lỗi ở generation/rerank version.
- **Root cause:** Thiếu version filtering (metadata `version: v2024/v2023` chưa có trong index); rerank chưa ưu tiên doc hiện hành.
- **Suggested fix:** Enrichment `extract_metadata` gắn `version + effective_date`; retriever filter `version=current`; prompt yêu cầu cite version.

### #2 — Version conflict (mật khẩu v1 vs v2)
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** 120 ngày + MFA (v2.0); v1.0 là 90 ngày đã cũ.
- **Got:** context top-1 (fallback).
- **Worst metric:** faithfulness (0.0)
- **Error Tree:** Output sai → Context lẫn 2 versions → Query OK.
- **Root cause:** Giống #1: chunk v1/v2 đều match BM25 ("90 ngày"/"120 ngày"), RRF không phân biệt recency.
- **Suggested fix:** Thêm `recency boost` trong RRF hoặc metadata filter `policy_version=current`.

### #3 — Negation (thử việc KHÔNG được nghỉ phép)
- **Question:** Nhân viên thử việc có được nghỉ phép năm không?
- **Expected:** KHÔNG — phải xin nghỉ không lương + trưởng phòng duyệt.
- **Got:** context top-1 (fallback).
- **Worst metric:** context_precision (0.0, RAGAS NaN→0.0)
- **Error Tree:** Output sai → Context đúng nhưng chứa cả câu khẳng định ("được nghỉ 15 ngày") gây nhiễu → Query OK.
- **Root cause:** Dense/BM25 match từ khóa "nghỉ phép năm" nhưng không hiểu phủ định; thiếu negation handling.
- **Suggested fix:** Reranker cross-encoder đã có (ưu tiên câu có "KHÔNG/thử việc"); thêm HyQA questions dạng phủ định trong M5.

### #4 — Negation + safety (malware KHÔNG tự xử lý)
- **Question:** Khi phát hiện malware trên máy, nhân viên có nên tự xử lý không?
- **Expected:** KHÔNG — báo helpdesk trong 1 giờ, tự xử lý là vi phạm nghiêm trọng.
- **Got:** context top-1 (fallback).
- **Worst metric:** answer_relevancy (0.0)
- **Error Tree:** Output sai → Context có thể thiếu chunk IT-security nếu chunking cắt nhỏ → Query OK.
- **Root cause:** Câu hỏi dài, ít từ khóa overlap ("malware", "helpdesk"); BM25 tốt nhưng dense cần contextual prepend.
- **Suggested fix:** M5 `contextual_prepend` ("Trích từ quy định an toàn thông tin...") + HyQA đã implement; bật `methods=["combined"]` khi có key thật.

### #5 — Multi-hop numeric (Senior 9 năm thâm niên + lương)
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** 15 + 3 = 18 ngày; lương Senior (P3-P4) 20-35 triệu.
- **Got:** context top-1 (fallback, chỉ 1 chunk nên thiếu vế còn lại).
- **Worst metric:** context_recall (0.0)
- **Error Tree:** Output sai → Context thiếu (cần 2 chunks: leave + salary) → Query OK nhưng top-3 chưa đủ.
- **Root cause:** Single-hop retrieve top-3 sau rerank top-3, công thức tính (9÷3=3) cần reasoning; hierarchical child→parent chưa dùng parent ở bước answer.
- **Suggested fix:** Tăng `HYBRID_TOP_K` cho multi-hop, retrieve child → return parent (M1 hierarchical đã sẵn), prompt yêu cầu show calculation.

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên được nghỉ bao nhiêu ngày phép năm?" (version conflict điển hình).

**Error Tree walkthrough:**
1. Output đúng? Không — đáp án phải là 15 ngày (v2024), nhưng corpus chứa cả 12 ngày (v2023).
2. Context đúng? Một phần — cả 2 chunks đều được retrieve (BM25 match "nghỉ phép năm", dense match semantic), nhưng không có tín hiệu version.
3. Query rewrite OK? Có — query rõ ràng, không ambiguous.
4. Fix đề xuất: gắn `metadata.version` lúc enrichment (M5 `_enrich_single_call` đã trả `metadata`), index version vào Qdrant payload, filter `version=current` trước RRF; rerank ưu tiên doc có "v2024/hiện hành".

**Nếu có thêm 1 giờ, sẽ optimize:**
- Thay `OPENAI_API_KEY` thật (OpenAI, không phải OpenRouter) rồi chạy lại `main.py` để có RAGAS scores thật và so sánh Δ.
- Bật M5 combined mode trên toàn corpus (hiện fallback extractive do 401) để đo lại context_recall cho multi-hop.
- Thêm latency breakdown (chunk/embed/search/rerank/LLM) cho bonus rubric.
