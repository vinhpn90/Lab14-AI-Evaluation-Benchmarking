import asyncio
import time
from typing import List, Dict

class BenchmarkRunner:
    def __init__(self, agent, evaluator, judge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def run_single_test(self, test_case: Dict) -> Dict:
        start_time = time.perf_counter()
        
        # 1. Gọi Agent
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start_time
        
        # 2. Chạy RAGAS/Retrieval metrics
        ragas_scores = await self.evaluator.score(test_case, response)
        
        # 3. Chạy Multi-Judge
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"], 
            response["answer"], 
            test_case["expected_answer"]
        )
        
        # Tính chi phí mô phỏng dựa trên số tokens và thời gian
        # Giả lập giá vLLM: 0.15 USD / 1M tokens
        tokens_used = response["metadata"].get("tokens_used", 0)
        cost = (tokens_used / 1_000_000) * 0.15
        
        return {
            "test_case": test_case["question"],
            "agent_response": response["answer"],
            "latency": latency,
            "tokens_used": tokens_used,
            "cost_usd": cost,
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": "fail" if judge_result["final_score"] < 3.0 else "pass"
        }

    async def run_all(self, dataset: List[Dict], concurrency_limit: int = 5) -> List[Dict]:
        """
        Chạy song song sử dụng asyncio.Semaphore để kiểm soát mức độ đồng thời (concurrency_limit).
        Giúp tối ưu hóa tốc độ chạy mà không bị nghẽn mạng hay Rate Limit của LLM.
        """
        sem = asyncio.Semaphore(concurrency_limit)
        
        async def worker(test_case: Dict) -> Dict:
            async with sem:
                return await self.run_single_test(test_case)
                
        tasks = [worker(case) for case in dataset]
        results = await asyncio.gather(*tasks)
        return results
