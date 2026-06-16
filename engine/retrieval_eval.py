import os
import json
import asyncio
import re
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán Hit Rate: Có ít nhất một expected_id nằm trong top_k của retrieved_ids không.
        Nếu expected_ids rỗng (case out-of-context hoặc adversarial), mặc định trả về 1.0.
        """
        if not expected_ids:
            return 1.0
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank (MRR): 
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (1-indexed). Nếu không tìm thấy, trả về 0.0.
        Nếu expected_ids rỗng, mặc định trả về 1.0.
        """
        if not expected_ids:
            return 1.0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

class ExpertEvaluator:
    """
    Evaluator nâng cao dùng để đánh giá RAG metrics:
    - Faithfulness (Độ trung thực): Câu trả lời có dựa trên context không?
    - Relevancy (Độ liên quan): Câu trả lời có trả lời đúng trọng tâm câu hỏi không?
    - Retrieval metrics (Hit Rate & MRR).
    """
    def __init__(self):
        self.retrieval_eval = RetrievalEvaluator()

    async def score(self, case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("metadata", {}).get("retrieved_ids", [])
        
        hit_rate = self.retrieval_eval.calculate_hit_rate(expected_ids, retrieved_ids, top_k=3)
        mrr = self.retrieval_eval.calculate_mrr(expected_ids, retrieved_ids)
        
        question = case["question"]
        answer = response["answer"]
        contexts = response["contexts"]
        
        # Đánh giá Faithfulness & Relevancy qua VLLM
        faithfulness = 1.0
        relevancy = 1.0
        
        # Nếu câu trả lời là từ chối (do adversarial hoặc out-of-context) thì độ trung thực và độ liên quan mặc định là 1.0
        refusal_phrases = ["tôi xin lỗi", "không có trong các tài liệu", "không được phép", "vui lòng cung cấp thêm"]
        is_refusal = any(p in answer.lower() for p in refusal_phrases)
        
        if is_refusal:
            faithfulness = 1.0
            relevancy = 1.0
        else:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
            
            base_url = os.environ.get("OPENAI_BASE_URL", None)
            if base_url:
                client = OpenAI(base_url=base_url, api_key=api_key or None)
            else:
                client = OpenAI(api_key=api_key or None)
            
            # Formulate validation prompt
            prompt = f"""Hãy đánh giá câu trả lời của AI Agent dựa trên câu hỏi và ngữ cảnh dưới đây.

Câu hỏi: "{question}"
Ngữ cảnh (Contexts):
{" - ".join(contexts)}
Câu trả lời (Answer): "{answer}"

Yêu cầu chấm điểm 2 tiêu chí trên thang điểm từ 0.0 đến 1.0:
1. Faithfulness (Độ trung thực): Điểm 1.0 nếu tất cả thông tin trong câu trả lời ĐỀU được hỗ trợ bởi ngữ cảnh. Điểm 0.0 nếu câu trả lời tự bịa thông tin không có trong ngữ cảnh.
2. Relevancy (Độ liên quan): Điểm 1.0 nếu câu trả lời trả lời đúng trọng tâm câu hỏi. Điểm 0.0 nếu câu trả lời lạc đề.

Đầu ra bắt buộc là một JSON Object duy nhất có định dạng:
{{
  "faithfulness": <score_float>,
  "relevancy": <score_float>
}}
Chỉ trả về JSON, không giải thích gì thêm."""

            try:
                loop = asyncio.get_event_loop()
                completion = await loop.run_in_executor(
                    None,
                    lambda: client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": "You are a precise evaluation judge assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=256,
                        temperature=0.01
                    )
                )
                
                raw_content = completion.choices[0].message.content
                if not raw_content:
                    # If content is None (reasoning model), try to get reasoning_content
                    raw_content = getattr(completion.choices[0].message, "reasoning_content", None) or getattr(completion.choices[0].message, "reasoning", "") or ""
                
                content = raw_content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                
                # Try to extract JSON if there's surrounding text
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)
                
                scores = json.loads(content)
                faithfulness = float(scores.get("faithfulness", 1.0))
                relevancy = float(scores.get("relevancy", 1.0))
            except Exception as e:
                # Fallback heuristics if API call fails
                # Faithfulness heuristic: word overlap between answer and contexts
                # Relevancy heuristic: word overlap between answer and question
                print(f"⚠️ Ragas Eval Fallback: {e}")
                faithfulness = self._heuristic_faithfulness(answer, contexts)
                relevancy = self._heuristic_relevancy(answer, question)
                
        return {
            "faithfulness": faithfulness,
            "relevancy": relevancy,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr
            }
        }
        
    def _heuristic_faithfulness(self, answer: str, contexts: List[str]) -> float:
        if not contexts:
            return 0.0
        ans_words = set(re.findall(r'\w+', answer.lower()))
        ctx_words = set(re.findall(r'\w+', " ".join(contexts).lower()))
        if not ans_words:
            return 1.0
        intersect = ans_words.intersection(ctx_words)
        return min(1.0, len(intersect) / len(ans_words) * 1.5)  # Scale factor

    def _heuristic_relevancy(self, answer: str, question: str) -> float:
        ans_words = set(re.findall(r'\w+', answer.lower()))
        q_words = set(re.findall(r'\w+', question.lower()))
        stopwords = {"và", "hoặc", "nhưng", "là", "thì", "mà", "của", "cho", "để", "ở", "trong", "có", "tôi", "bạn", "làm", "sao", "thế", "nào"}
        q_keywords = q_words - stopwords
        if not q_keywords:
            return 1.0
        intersect = ans_words.intersection(q_keywords)
        return min(1.0, len(intersect) / len(q_keywords) * 2.0)
