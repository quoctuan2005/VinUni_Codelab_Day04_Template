"""
Lab #4: System Prompt Engineering & Tool Calling Engine
Học viên hoàn thiện các mục TODO để hoàn thành bài lab.

Kiến trúc:
  - ChatbotBaseline: LLM thuần, không dùng tool → quan sát hallucination.
  - ToolCallingAgent: Agent dùng System Prompt + 2 Tool Schemas.
"""

import json
import re
from typing import Dict, Any, List
from tools import TOOL_DEFINITIONS, TOOL_MAP, search_product_catalog, submit_support_ticket

# ═══════════════════════════════════════════════════════════════════════════
# TODO 1: Thiết kế SYSTEM PROMPT cấp sản xuất
# Yêu cầu: Phải chứa Persona, Core Rules, Operational Boundaries, Output Contract.
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """
Bạn là VinAssistant — trợ lý AI chính thức của hệ sinh thái Vingroup.

## 1. PERSONA
- Tên: VinAssistant
- Vai trò: Chuyên viên tư vấn sản phẩm, dịch vụ và hỗ trợ khách hàng của hệ sinh thái Vingroup (VinFast, Vinpearl).
- Phong cách giao tiếp: Chuyên nghiệp, tận tâm, lịch sự, chính xác và trung thực.

## 2. AVAILABLE TOOLS
Bạn có quyền truy cập vào 2 công cụ ngoại vi:
1. `search_product_catalog(category: str, max_price: int)`:
   - Tra cứu danh mục sản phẩm xe điện VinFast ('xe_dien') hoặc dịch vụ du lịch Vinpearl ('du_lich') theo mức giá tối đa (VNĐ).
2. `submit_support_ticket(customer_name: str, issue_description: str, priority: str)`:
   - Ghi nhận yêu cầu hỗ trợ, phản hồi, hoặc báo cáo sự cố của khách hàng vào hệ thống tiếp nhận sự cố.

## 3. CORE RULES
1. TUYỆT ĐỐI KHÔNG BỊA ĐẶT THÔNG TIN: Không tự ý sáng tạo thông tin sản phẩm, giá bán, hay mã hỗ trợ. Phải dựa trên dữ liệu từ công cụ.
2. PHẢI GỌI TOOL KHI CẦN THIẾT:
   - Khi khách hàng hỏi về thông tin sản phẩm hoặc giá cả -> BẮT BUỘC gọi `search_product_catalog`.
   - Khi khách hàng báo lỗi, sự cố hoặc khiếu nại -> BẮT BUỘC gọi `submit_support_ticket`.
3. FALLBACK AN TOÀN: Khi công cụ không tìm thấy dữ liệu, hãy phản hồi trung thực: "Rất tiếc, không tìm thấy sản phẩm phù hợp".
4. TRẢ LỜI FAQ TRỰC TIẾP: Các câu hỏi chính sách chung (như chính sách bảo hành pin 10 năm của VinFast) có thể trả lời trực tiếp mà không cần gọi tool.

## 4. OPERATIONAL BOUNDARIES
- Chỉ trả lời và hỗ trợ các sản phẩm, dịch vụ nằm trong hệ sinh thái Vingroup.
- Từ chối lịch sự các chủ đề không liên quan ngoài phạm vi hoạt động.

## 5. OUTPUT CONTRACT
Mỗi bước xử lý tuân thủ quy trình ReAct:
- Thought: Suy nghĩ và phân tích ý định của người dùng.
- Action: Tên tool cần thực thi (nếu cần).
- Action Input: Tham số đầu vào cho tool.
- Observation: Kết quả nhận được từ tool.
- Final Answer: Câu trả lời hoàn chỉnh, chính xác, thân thiện gửi tới người dùng.
"""


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ChatbotBaseline
# ═══════════════════════════════════════════════════════════════════════════

class ChatbotBaseline:
    """Baseline LLM Chatbot — Không sử dụng Tool Calling hay ReAct Loop."""

    def query(self, user_input: str) -> Dict[str, Any]:
        """
        Baseline trả lời tĩnh hoặc gọi LLM thuần mà không có Tool.
        Dễ gặp hiện tượng hallucination đối với dữ liệu nội bộ.
        """
        return {
            "answer": f"[Chatbot Baseline] Trả lời cho: {user_input}",
            "tool_calls": [],
            "status": "success",
            "mode": "mock_baseline"
        }


# ═══════════════════════════════════════════════════════════════════════════
# CLASS: ToolCallingAgent
# ═══════════════════════════════════════════════════════════════════════════

class ToolCallingAgent:
    """Agent với System Prompt Engineering & Tool Calling."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace: List[Dict[str, Any]] = []

    def _detect_intent(self, user_input: str) -> Dict[str, Any]:
        """
        Phân tích ý định từ user_input bằng keyword matching & regex:
          - is_faq: Câu hỏi chính sách chung (ví dụ: bảo hành pin)
          - needs_catalog: Cần tra cứu catalog sản phẩm/dịch vụ
          - needs_ticket: Cần tạo ticket hỗ trợ
        """
        text = user_input.lower()

        # 1. Kiểm tra FAQ (Chính sách bảo hành pin VinFast)
        is_faq = False
        if any(w in text for w in ["bao lâu", "thời hạn", "chính sách"]) and any(w in text for w in ["bảo hành", "pin"]):
            is_faq = True

        # 2. Kiểm tra nhu cầu tra cứu Catalog
        needs_catalog = False
        catalog_args = {}
        catalog_keywords = ["xem", "tìm", "mua", "giá", "dưới", "tra cứu", "có xe", "có resort", "danh sách", "nghỉ", "đặt phòng"]
        if not is_faq and any(kw in text for kw in catalog_keywords):
            # Xác định category
            if any(w in text for w in ["resort", "du lịch", "du lich", "khách sạn", "khach san", "phú quốc", "nha trang", "landmark"]):
                category = "du_lich"
            else:
                category = "xe_dien"

            # Trích xuất mức giá tối đa (nếu có)
            price_match = re.search(r'(?:dưới|tầm|khoảng|mức giá|giá)\s*([0-9]+(?:\.[0-9]+)?)\s*(triệu|tr|tỷ|ty)', text)
            if price_match:
                val = float(price_match.group(1))
                unit = price_match.group(2)
                if "tỷ" in unit or "ty" in unit:
                    max_price = int(val * 1_000_000_000)
                else:
                    max_price = int(val * 1_000_000)
            else:
                max_price = 999999999999

            catalog_args = {
                "category": category,
                "max_price": max_price
            }
            needs_catalog = True

        # 3. Kiểm tra nhu cầu gửi Support Ticket
        needs_ticket = False
        ticket_args = {}
        ticket_keywords = ["lỗi", "hỏng", "sự cố", "phản ánh", "phản hồi", "khiếu nại", "yêu cầu hỗ trợ", "gấp", "nghiêm trọng", "ẩm mốc"]
        if any(kw in text for kw in ticket_keywords):
            # Trích xuất tên khách hàng
            name_match = re.search(r'(?:tôi tên là|tôi tên|tên tôi là|tên là)\s+([A-ZÀ-Ỹa-zà-ỹ\s]+?)(?:,|\.|$| xe| phòng)', user_input, re.IGNORECASE)
            customer_name = name_match.group(1).strip() if name_match else "Khách hàng"

            # Trích xuất độ ưu tiên
            if any(w in text for w in ["gấp", "nghiêm trọng", "khẩn cấp", "high"]):
                priority = "high"
            elif any(w in text for w in ["thấp", "low"]):
                priority = "low"
            else:
                priority = "medium"

            # Trích xuất mô tả vấn đề
            desc_match = re.search(r'((?:xe [^,\.]+(?:bị|lỗi)[^,\.]*)|(?:phòng [^,\.]+(?:bị|ẩm mốc)[^,\.]*)|(?:[^,\.]*bị lỗi[^,\.]*)|(?:[^,\.]*ẩm mốc[^,\.]*))', user_input, re.IGNORECASE)
            if desc_match:
                issue_description = desc_match.group(1).strip()
            else:
                issue_description = user_input.strip()

            ticket_args = {
                "customer_name": customer_name,
                "issue_description": issue_description,
                "priority": priority
            }
            needs_ticket = True

        return {
            "is_faq": is_faq,
            "needs_catalog": needs_catalog,
            "catalog_args": catalog_args,
            "needs_ticket": needs_ticket,
            "ticket_args": ticket_args
        }

    def run(self, user_input: str) -> Dict[str, Any]:
        """Điểm vào chính — chạy Agent Loop."""
        self.trace = []
        intents = self._detect_intent(user_input)

        # Trường hợp 1: FAQ không cần gọi Tool
        if intents["is_faq"]:
            answer = "Chính sách bảo hành pin xe điện VinFast kéo dài 10 năm (hoặc 200.000 km tuỳ điều kiện nào đến trước), áp dụng chính hãng trên toàn quốc."
            self.trace.append({
                "step": "faq_response",
                "user_input": user_input,
                "thought": "Câu hỏi thuộc danh mục FAQ về chính sách bảo hành pin xe điện, trả lời trực tiếp mà không cần gọi tool.",
                "answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # Trường hợp câu hỏi chung chung
        if not intents["needs_catalog"] and not intents["needs_ticket"]:
            answer = "Xin chào! Tôi là VinAssistant. Tôi có thể hỗ trợ bạn tra cứu xe điện VinFast, du lịch Vinpearl hoặc tiếp nhận yêu cầu hỗ trợ kỹ thuật."
            self.trace.append({
                "step": "general_response",
                "user_input": user_input,
                "answer": answer
            })
            return {
                "answer": answer,
                "trace": self.trace,
                "iterations": 1,
                "status": "completed"
            }

        # Trường hợp cần gọi Tool (Single hoặc Multi-tool)
        actions_to_execute = []
        if intents["needs_catalog"]:
            actions_to_execute.append(("search_product_catalog", intents["catalog_args"]))
        if intents["needs_ticket"]:
            actions_to_execute.append(("submit_support_ticket", intents["ticket_args"]))

        iteration = 0
        step_observations = {}

        for tool_name, tool_args in actions_to_execute:
            iteration += 1
            if iteration > self.max_iterations:
                return {
                    "answer": "Lỗi: Vượt quá số bước tối đa cho phép (max_iterations_reached).",
                    "trace": self.trace,
                    "iterations": iteration,
                    "status": "max_iterations_reached"
                }

            tool_fn = TOOL_MAP[tool_name]
            obs = tool_fn(**tool_args)

            self.trace.append({
                "iteration": iteration,
                "thought": f"Gọi tool {tool_name} với tham số: {tool_args}",
                "action": tool_name,
                "action_input": tool_args,
                "observation": obs
            })
            step_observations[tool_name] = obs

        # Tổng hợp câu trả lời cuối cùng (Final Answer Synthesis)
        answer_parts = []
        if "search_product_catalog" in step_observations:
            catalog_res = step_observations["search_product_catalog"]
            if not catalog_res or len(catalog_res) == 0:
                answer_parts.append("Rất tiếc, không tìm thấy sản phẩm hoặc gói dịch vụ nào phù hợp với mức giá bạn yêu cầu.")
            else:
                items_desc = []
                for p in catalog_res:
                    price_str = f"{p['price_vnd']:,} VNĐ"
                    items_desc.append(f"- {p['name']} ({p.get('brand', 'Vingroup')}): {price_str}. {p.get('description', '')}")
                answer_parts.append("Dưới đây là các sản phẩm/dịch vụ phù hợp tìm được:\n" + "\n".join(items_desc))

        if "submit_support_ticket" in step_observations:
            ticket_res = step_observations["submit_support_ticket"]
            answer_parts.append(
                f"Yêu cầu hỗ trợ của quý khách {ticket_res['customer_name']} đã được tiếp nhận thành công. "
                f"Mã ticket: {ticket_res['ticket_id']} (Mức độ ưu tiên: {ticket_res['priority']}, Trạng thái: {ticket_res['status']}). "
                f"Đội ngũ kỹ thuật sẽ xử lý sự cố trong thời gian sớm nhất."
            )

        final_answer = "\n\n".join(answer_parts)
        self.trace.append({
            "step": "final_answer",
            "answer": final_answer
        })

        return {
            "answer": final_answer,
            "trace": self.trace,
            "iterations": iteration,
            "status": "completed"
        }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN — Chạy thử nhanh
# ═══════════════════════════════════════════════════════════════════════════

def main():
    user_query = "Tôi muốn xem xe điện VinFast giá dưới 600 triệu."

    print("=== RUNNING CHATBOT BASELINE ===")
    chatbot = ChatbotBaseline()
    print(chatbot.query(user_query))

    print("\n=== RUNNING TOOL CALLING AGENT ===")
    agent = ToolCallingAgent(max_iterations=5)
    result = agent.run(user_query)
    print("Result:\n", result["answer"])
    print("\nTrace Log:\n", json.dumps(agent.trace, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
