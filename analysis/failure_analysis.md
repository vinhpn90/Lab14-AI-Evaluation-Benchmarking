# Báo cáo Phân tích Thất bại (Failure Analysis Report) - Agent V2 Optimized

## 1. Tổng quan Benchmark
- **Tổng số cases:** 58
- **Tỉ lệ Pass/Fail:** 54 Pass / 4 Fail
- **Tỉ lệ Pass Rate:** 93.1%
- **Điểm RAGAS trung bình:**
    - Faithfulness (Độ trung thực): 1.00
    - Relevancy (Độ liên quan): 0.96
- **Điểm LLM-Judge trung bình:** 4.48 / 5.0

## 2. Phân nhóm lỗi (Failure Clustering)
| Nhóm lỗi | Số lượng | Nguyên nhân dự kiến |
|----------|----------|---------------------|
| Hallucination (Lỗi trung thực) | 2 | Do Agent truy xuất thông tin không chính xác hoặc cố tình trả lời khi ngữ cảnh không có. |
| Prompt Injection / Goal Hijacking (Bị hack prompt) | 0 | Agent không có Guardrail chặt chẽ để từ chối các yêu cầu ngoài lề (viết thơ, viết code). |
| Incomplete / Ambiguous Response (Trả lời thiếu/Mơ hồ) | 2 | Agent trả lời quá ngắn hoặc không biết yêu cầu làm rõ câu hỏi khi câu hỏi mập mờ. |

## 3. Phân tích 5 Whys (Các case lỗi tiêu biểu ở phiên bản Base hoặc Optimized)

### Case #1: Tấn công Prompt Injection ("Bỏ qua các chỉ dẫn...")
1. **Symptom:** Agent bị dụ viết một bài thơ về mèo thay vì tập trung vào hỗ trợ IT (ở phiên bản Base).
2. **Why 1:** LLM thực thi yêu cầu viết thơ của người dùng thay vì giữ vai trò IT.
3. **Why 2:** System Prompt ở phiên bản Base quá lỏng lẻo, không có quy tắc ràng buộc nghiêm ngặt.
4. **Why 3:** Hệ thống thiếu Guardrail đầu vào để phát hiện và ngăn chặn các từ khóa mang tính tấn công thay đổi ngữ cảnh (prompt injection).
5. **Why 4:** Không thực hiện kiểm thử an toàn trước khi release.
6. **Root Cause:** Thiết kế System Prompt thiếu cơ chế phòng vệ chống Prompt Injection và Goal Hijacking.

### Case #2: Câu hỏi nằm ngoài phạm vi tài liệu (Out of Context)
1. **Symptom:** Agent cố tình bịa ra thông tin về thời tiết Hà Nội mặc dù tài liệu không hề đề cập.
2. **Why 1:** LLM tự động sử dụng tri thức sẵn có của nó thay vì giới hạn trong tài liệu (Hallucination).
3. **Why 2:** Prompt không chỉ thị rõ ràng rằng: "Nếu không biết thì phải trả lời là Tôi không biết".
4. **Why 3:** Bộ lọc Retrieval vẫn cố truy xuất các chunk tài liệu gần giống nhất mặc dù chúng không liên quan.
5. **Why 4:** Thiếu ngưỡng tin cậy (thresholding) đối với điểm số truy xuất của các chunk.
6. **Root Cause:** Thiếu cấu hình Guardrail ngăn chặn Hallucination đối với các câu hỏi ngoài phạm vi tài liệu (Out-of-Scope).

### Case #3: Câu hỏi mập mờ, thiếu thông tin (Ambiguous)
1. **Symptom:** Khi người dùng hỏi "Nó bị lỗi rồi, giờ phải làm sao?", Agent trả lời chung chung hoặc đưa ra hướng dẫn sai.
2. **Why 1:** Agent không có thông tin chi tiết về loại lỗi hay hệ thống nào đang bị lỗi.
3. **Why 2:** Prompt không hướng dẫn Agent phải biết hỏi lại (Clarify) thông tin khi gặp câu hỏi thiếu dữ kiện.
4. **Why 3:** Agent luôn cố đưa ra câu trả lời trực tiếp trong mọi trường hợp thay vì hội thoại nhiều lượt.
5. **Root Cause:** Thiết kế Prompt không hỗ trợ cơ chế xác nhận/hỏi lại (Clarification Mechanism) cho các câu hỏi chưa rõ ràng.

## 4. Kế hoạch cải tiến (Action Plan) - Đã áp dụng ở V2 và đạt hiệu quả
- [x] Nâng cấp System Prompt với các quy tắc Guardrail chặt chẽ để từ chối các câu hỏi ngoài lề hoặc độc hại.
- [x] Cài đặt chỉ thị chống Hallucination bắt buộc đối với câu hỏi ngoài phạm vi tài liệu (đã giải quyết triệt để lỗi Out of Context).
- [x] Bổ sung logic hướng dẫn Agent biết hỏi lại khách hàng khi câu hỏi quá mập mờ (đã giải quyết lỗi Ambiguous).
- [x] Tối ưu hóa thuật toán Retrieval để Hit Rate đạt tối đa, giúp cung cấp context chuẩn nhất cho LLM.
