# RAG_TOEIC Root Agent - LangChain + Gemini (bắt buộc)

## Cấu hình môi trường (bắt buộc)

Thiết lập biến môi trường để dùng Gemini thật (qua LangChain):

```bash
export GOOGLE_API_KEY="<your_google_api_key>"
export GEMINI_MODEL="gemini-1.5-flash" # hoặc gemini-2.0-flash-exp, gemini-2.5-flash nếu có
```

Nếu không đặt `GOOGLE_API_KEY`, hệ thống sẽ lỗi ngay khi khởi tạo. MockLLM đã được gỡ bỏ.

## Chạy ví dụ nhanh

```bash
python -m src.core.langchain_root_agent
```

Chương trình sẽ:
- Khởi tạo Root Agent, chạy 1 request test
- In kết quả trả về
- In trạng thái agent (`print_status`)
- In toàn bộ trace gồm prompts/outputs của LLM, actions và tool traces (`print_full_trace`)

## Test nhanh (yêu cầu GOOGLE_API_KEY)

```bash
pytest -q tests/test_root_agent_trace.py
```

## Checklist

- [x] Tích hợp LLM thật (Gemini) qua env, không còn `MockLLM`
- [x] Ghi nhận đầy đủ prompts/outputs LLM, tool traces
- [x] Hàm in toàn bộ trace (`print_full_trace`)
- [x] Test harness gọi agent và in trace
 - [x] Consolidate: chỉ giữ `src/core/langchain_root_agent.py` và test Gemini

# RAG_Project
weight : https://drive.google.com/drive/folders/1njve-dILpn-wqR32L7Yqk1wwMN2ADfl8?usp=sharing

# Run Docker Compose
docker compose -f infrastructure/docker/docker-compose.yml up

# Chạy minikube
minikube start --driver=docker --gpus=all

# Tạo namespace mới
kubectl create namespace rag-app

# Tạo secret
kubectl create secret generic rag-app-secrets --from-env-file=config/.env -n rag-app

# Kiểm tra secret
kubectl get secrets -n rag-app

# Khởi tạo
kubectl apply -f deployment.yml
kubectl get pods -n rag-app
kubectl describe pod ai-server-deployment-784fcfccd4-fc5q6 -n rag-app

# Chạy Streamlit
streamlit run Front_end/chatbot_app.py