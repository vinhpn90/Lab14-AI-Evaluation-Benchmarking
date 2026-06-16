# BÁO CÁO PHẢN HỒI CÁ NHÂN (INDIVIDUAL REFLECTION REPORT)
**Họ và tên:** Phạm Ngọc Vinh  
**MSSV:** 2A202600563  
**Học phần:** Lab Day 14: AI Evaluation Factory (Team Edition)  
**Vai trò trong nhóm:** AI Engineer & Tech Lead  

---

## 👤 1. Đóng góp Kỹ thuật (Engineering Contribution)

Trong dự án **AI Evaluation Factory**, tôi chịu trách nhiệm chính trong việc phát triển và tối ưu hóa các module cốt lõi của Eval Engine, cụ thể bao gồm các hạng mục sau:

### 1.1. Triển khai Async Runner với cơ chế Semaphore (`engine/runner.py`)
- **Vấn đề:** Ban đầu hệ thống sử dụng cơ chế chia nhỏ lô (naive batching) tuần tự, làm giảm đáng kể hiệu năng khi chạy 50+ test cases (mất hơn 10 phút) và dễ gây ra hiện tượng nghẽn cổ chai hoặc bị khóa do quá tải API.
- **Giải pháp:** Tôi đã tối ưu hóa mã nguồn bằng cách sử dụng `asyncio.Semaphore(concurrency_limit=5)` kết hợp với `asyncio.gather`. Phương pháp này giúp duy trì hàng đợi song song liên tục, các tác vụ hoàn thành sẽ nhường chỗ cho tác vụ mới ngay lập tức mà không phải chờ đợi toàn bộ lô kết thúc. Hệ thống hoàn thành benchmark 58 cases chỉ trong khoảng 1-2 phút.

### 1.2. Phát triển kiến trúc Multi-Judge Consensus Engine (`engine/llm_judge.py`)
- **Thiết kế:** Xây dựng hệ thống đồng thuận Multi-Judge sử dụng mô hình VLLM (`gpt-4o-mini`) với hai cấu hình hệ thống (System Prompts) khác nhau:
  - **Judge A (Accuracy & Completeness):** Chuyên đánh giá độ chính xác thông tin và độ hoàn thiện của nội dung so với Ground Truth.
  - **Judge B (Tone & Professionalism):** Chuyên đánh giá sự lịch sự, tính an toàn và khả năng từ chối các câu hỏi phá hoại hệ thống (Guardrails).
- **Consensus Logic:** Triển khai tính toán **Hệ số đồng thuận (Agreement Rate)** tự động. Trong trường hợp hai Judge bất đồng ý kiến (điểm lệch nhau lớn hơn 1.0 điểm), hệ thống sẽ tự động kích hoạt **Judge C (Mediator - Trọng tài)** làm nhiệm vụ phân xử và đưa ra điểm số quyết định cuối cùng.

### 1.3. Triển khai chỉ số đánh giá tầng Retrieval (`engine/retrieval_eval.py`)
- Trực tiếp viết thuật toán tính toán **Hit Rate** (ở mức Top-3) và **Mean Reciprocal Rank (MRR)** dựa trên việc so sánh các chunk ID kỳ vọng (`expected_retrieval_ids`) và các chunk ID thực tế được Agent truy xuất (`retrieved_ids`).

---

## 📚 2. Độ sâu Kỹ thuật (Technical Depth)

Dưới đây là giải thích chi tiết các khái niệm học thuật và các bài toán đánh giá tối ưu hóa AI tôi đã nghiên cứu và áp dụng trong bài lab:

### 2.1. Mean Reciprocal Rank (MRR)
MRR là chỉ số đo lường hiệu năng của hệ thống tìm kiếm thông tin (Retrieval). Công thức tính MRR cho một tập hợp các truy vấn $Q$ là:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
Trong đó $\text{rank}_i$ là vị trí xuất hiện đầu tiên của tài liệu liên quan (ground truth chunk) trong danh sách kết quả được truy xuất của câu hỏi thứ $i$. Nếu không tìm thấy, $\frac{1}{\text{rank}_i} = 0$.
- **Ý nghĩa thực tế:** Khác với Hit Rate (chỉ quan tâm có tìm thấy hay không), MRR đánh giá mức độ tối ưu của việc xếp hạng. Một hệ thống đưa tài liệu đúng lên vị trí số 1 (MRR = 1.0) sẽ tốt hơn rất nhiều so với hệ thống đưa lên vị trí số 3 (MRR = 0.33) vì nó giúp LLM tập trung vào ngữ cảnh quan trọng nhất nằm ở đầu prompt.

### 2.2. Cohen's Kappa
Cohen's Kappa ($\kappa$) là một chỉ số thống kê dùng để đo lường độ tin cậy đồng thuận giữa hai giám khảo (inter-rater reliability) khi phân loại các biến định danh. Công thức:
$$\kappa = \frac{p_o - p_e}{1 - p_e}$$
Trong đó:
- $p_o$ (observed agreement): Tỷ lệ đồng thuận thực tế được quan sát giữa hai Judge.
- $p_e$ (expected agreement): Tỷ lệ đồng thuận ngẫu nhiên dự kiến (nếu cả hai Judge chấm điểm hoàn toàn ngẫu nhiên dựa trên phân phối điểm).
- **Ứng dụng:** Đo lường $\kappa$ giúp chúng tôi xác định xem sự đồng thuận của hai LLM Judge là do chất lượng câu trả lời rõ ràng hay chỉ là sự trùng hợp ngẫu nhiên, từ đó hiệu chỉnh các tiêu chí chấm điểm (rubrics) để đạt độ khách quan cao hơn.

### 2.3. Position Bias (Thiên vị Vị trí)
Position Bias là hiện tượng LLM (khi đóng vai trò làm Judge so sánh pairwise giữa hai câu trả lời A và B) có xu hướng ưu ái lựa chọn câu trả lời đứng ở vị trí đầu tiên (hoặc đôi khi là cuối cùng) bất kể chất lượng thực tế của chúng.
- **Giải pháp:** Để phát hiện và loại bỏ bias này, tôi đã thiết kế hàm `check_position_bias`. Hàm này thực hiện đánh giá hai lần: lần 1 đưa câu trả lời của Agent V1 vào vị trí A và V2 vào vị trí B; lần 2 tráo đổi vị trí (V2 vào vị trí A và V1 vào vị trí B). Nếu Judge thay đổi quyết định chọn lựa nhãn vị trí nhưng thực chất chọn cùng một câu trả lời thì hệ thống khách quan; nếu Judge luôn chọn nhãn "A" trong cả hai lần chạy, hệ thống đã bị Position Bias.

### 2.4. Trade-off giữa Chi phí và Chất lượng (Cost vs Quality)
Trong các dự án thực tế, việc sử dụng gpt-4o-mini để chấm điểm hàng triệu lượt hội thoại mỗi ngày là bất khả thi về mặt tài chính. Chúng tôi đã phân tích và áp dụng các giải pháp giảm thiểu chi phí:
- **Tối ưu hóa Token:** Giới hạn `max_tokens` của Judge vừa đủ để sinh điểm số và lý do ngắn gọn (dưới 256 tokens) thay vì để mô hình suy nghĩ lan man.
- **Cơ chế Hybrid Eval (Đề xuất giảm 30% chi phí):** Chạy các bộ lọc regex/rule-based hoặc mô hình nhỏ (như Llama-3-8B-Instruct) để chấm điểm trước. Chỉ những trường hợp điểm số không chắc chắn (lệch nhiều) hoặc các câu hỏi thuộc nhóm lỗi phức tạp mới đẩy lên mô hình lớn (gpt-4o-mini/Claude) để chấm điểm Consensus.

---

## 🛠️ 3. Giải quyết Vấn đề (Problem Solving)

Trong quá trình phát triển hệ thống đánh giá phức tạp này, tôi đã đối mặt và giải quyết các bài toán thực tế sau:

1. **Xử lý đặc thù của Reasoning Model trên VLLM (`gpt-4o-mini`):**
   - *Vấn đề:* Khi gọi API, mô hình suy nghĩ thường trả về trường `content` bằng `None` do dồn toàn bộ tokens vào phân tích suy nghĩ (`reasoning_content`) hoặc bị lỗi cắt ngắn do `max_tokens` quá thấp.
   - *Giải pháp:* Tôi đã bổ sung cơ chế kiểm tra và trích xuất dữ liệu thông minh: Nếu `content` bị `None`, hệ thống sẽ tự động đọc từ các thuộc tính bổ sung như `reasoning_content` hoặc `reasoning`. Ngoài ra, tôi tăng `max_tokens` lên 512 đối với các tác vụ sinh câu trả lời phức tạp để đảm bảo mô hình có đủ không gian suy nghĩ và đưa ra kết quả cuối cùng.
