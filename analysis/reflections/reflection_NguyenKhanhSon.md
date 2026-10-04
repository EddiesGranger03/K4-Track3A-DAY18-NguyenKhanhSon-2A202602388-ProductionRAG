# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** NguyenKhanhSon
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 2026-10-04

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Dùng MiniLM-L6-v2 + cosine, `threshold=0.85` (test dùng 0.5); có fallback nhóm câu khi không tải được model. Test `semantic <= basic+2 chunks` pass — nhóm câu cùng chủ đề giúp ít chunk hơn basic (57 basic paragraph chunks trên corpus). |
| Hierarchical chunking | M1 | `chunk_hierarchical()` | Parent 2048 chars + child 256 chars, child giữ `parent_id` link đúng parent; `avg_child < avg_parent` pass. Đây là strategy production: retrieve child (precision) → return parent (context), phù hợp multi-hop. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Split theo `#{1,3}` headers, giữ `section` trong metadata; test giữ được "Nghỉ phép năm" pass. Giữ nguyên tables/lists, không cắt giữa section. |
| Vietnamese segmentation + BM25 | M2 | `segment_vietnamese()` / `BM25Search` | underthesea `word_tokenize` + `replace("_"," ")` (fix lỗi `nghỉ_phép` vs `nghỉ phép`); BM25Okapi trên text đã segment, lọc `score>0`, `method="bm25"`. Query "nghỉ phép" trả về doc 12 ngày đúng. |
| Dense + Hybrid fusion | M2 | `DenseSearch` / `reciprocal_rank_fusion()` | bge-m3 (1024 dim) + Qdrant `query_points()`; RRF `1/(k+rank+1)`, k=60, merge ra `method="hybrid"`. Fix thêm: timeout 2→30s, upsert batch 32, fallback `:memory:` khi Docker rớt. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `BAAI/bge-reranker-v2-m3` qua `sentence_transformers.CrossEncoder` (không dùng FlagEmbedding vì crash transformers>=5); top-20→top-3, sort desc. Có fallback lexical-overlap nên offline vẫn pass test "nghỉ phép rank cao hơn VPN". |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | Wrap try/except; `Dataset.from_dict` + 4 metrics; sanitize NaN→0.0 (phát hiện khi RAGAS trả NaN do 401). `failure_analysis()` dùng Diagnostic Tree: faithfulness→hallucination, recall→thiếu chunk, precision→nhiễu, relevancy→prompt. |
| Contextual enrichment | M5 | `contextual_prepend()` / `_enrich_single_call()` | Combined mode 1 call/chunk trả `{summary, questions, context, metadata}`; fallback extractive khi không có key (test vẫn pass: contextual giữ nguyên text gốc, HyQA sinh câu hỏi `?`). `enrich_chunks()` pipeline đã nối vào `src/pipeline.py`. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Lỗi kỹ thuật gặp phải (Exact error message):**
  - `OverflowError: cannot convert longdouble infinity to integer` tại `numpy/core/getlimits.py:52` khi `from qdrant_client import QdrantClient` (Python 3.14 + numpy 1.26.4 build từ source).
  - `qdrant_client.http.exceptions.ResponseHandlingException: timed out` tại `client.upsert` (Qdrant, `timeout=2` quá ngắn + Docker Desktop bị tắt giữa chừng: `open //./pipe/dockerDesktopLinuxEngine: not found`).
  - `AuthenticationError 401 Incorrect API key provided: sk-or-v1...` cho cả 80 RAGAS jobs (key OpenRouter dùng nhầm cho `api.openai.com`).
- **Nguyên nhân gốc rễ & Cách debug:**
  - Check `py -0p` thấy máy có Python 3.12 → tạo `.venv` bằng 3.12 thay vì 3.14 theo đúng `.python-version`/README; cài lại sạch hết lỗi numpy.
  - Check `docker ps` + `curl localhost:6333` phát hiện daemon down → start lại Docker + `compose up -d`; đồng thời sửa code tăng timeout 30s, batch upsert, fallback in-memory.
  - Đọc kỹ message 401 thấy prefix `sk-or-v1` → xác định key sai loại; vẫn cho pipeline chạy fallback (answers = top context, RAGAS NaN→0.0) để đủ deliverables, ghi rõ trong failure analysis thay vì giấu.
- **Kiến thức còn thiếu & Cách khắc phục:**
  - Thiếu: phân biệt OpenAI key vs OpenRouter key + `OPENAI_BASE_URL`; RAGAS bản cũ (0.1.x) trả NaN thay vì throw khi 401.
  - Cách bổ sung: đọc doc OpenRouter/lanchain-openai về base_url; thêm hàm `_safe()` sanitize NaN; sau này xin key OpenAI thật rồi chạy lại `main.py` để lấy Δ điểm.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Chatbot tra cứu quy chế HR nội bộ (tiếng Việt)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Markdown policies → basic paragraph chunking → dense-only (bge-m3) → top-3 → GPT trả lời.
- **Vấn đề / Bottlenecks đang gặp:** Dính version cũ (v2023/v1.0 lẫn với hiện hành); câu phủ định ("KHÔNG được...") hay bị trả lời ngược; câu multi-hop số học (phép + lương) thiếu 1 vế context; chưa có eval chuẩn, chỉ đọc tay.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Hierarchical (parent 2048/child 256) làm default — child để retrieve chính xác, parent để giữ context bảng lương; structure-aware cho file markdown có headers.
2. **Search retrieval:** Hybrid BM25 (tiếng Việt, đã fix `_`) + Dense + RRF — BM25 bắt số/ngày chính xác ("120 ngày", "15 ngày"), dense bắt paraphrase.
3. **Reranking:** Có — `bge-reranker-v2-m3` top-20→top-3; đo latency, nếu >500ms thì cân nhắc FlashRank cho realtime.
4. **Evaluation:** RAGAS 4 metrics trên 20 câu theo 6 dạng (lookup/version/negation/multi-hop/numeric/ambiguous); track faithfulness ≥0.85 làm gate trước khi release.
5. **Enrichment:** Combined single-call (summary + HyQA + context + metadata gồm `version/effective_date/category`); filter version=current lúc retrieve để hết lỗi version conflict.

#### 3. Timeline triển khai
- **Tuần 1:** Gắn metadata version + filter; xin OpenAI key thật; chạy lại baseline vs production lấy Δ.
- **Tuần 2:** Bật enrichment combined toàn corpus; thêm negation HyQA; benchmark rerank latency; viết failure analysis định kỳ.
