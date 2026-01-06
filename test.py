# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# from unsloth import FastLanguageModel

# # Load the model and tokenizer once at startup
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name="unsloth/Qwen3-0.6B",
#     max_seq_length=32768,
#     dtype=None,
#     load_in_4bit=True,
# )

# app = FastAPI(title="Qwen3 API", description="API for text generation using Qwen3-0.6B model")

# class GenerateRequest(BaseModel):
#     prompt: str
#     max_new_tokens: int = 32768
#     enable_thinking: bool = False

# @app.post("/generate")
# async def generate_text(request: GenerateRequest):
#     try:
#         messages = [{"role": "user", "content": request.prompt}]
#         text = tokenizer.apply_chat_template(
#             messages,
#             tokenize=False,
#             add_generation_prompt=True,
#             enable_thinking=request.enable_thinking
#         )
#         model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
#         generated_ids = model.generate(
#             **model_inputs,
#             max_new_tokens=request.max_new_tokens
#         )
#         output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        
#         # Parsing thinking content
#         try:
#             index = len(output_ids) - output_ids[::-1].index(151668)
#         except ValueError:
#             index = 0
        
#         content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
#         return {"content": content}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)


from sklearn.feature_extraction.text import TfidfVectorizer
import networkx as nx
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

TEXT = """Artificial intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. These intelligent machines can perform tasks that typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation. AI can be categorized into two main types: narrow AI, which is designed for specific tasks, and general AI, which possesses the ability to perform any intellectual task that a human can do."""
def summarize(text, k=3):
    sents = nltk.sent_tokenize(text)
    tfidf = TfidfVectorizer().fit_transform(sents)
    sim = (tfidf * tfidf.T).toarray()
    scores = nx.pagerank(nx.from_numpy_array(sim))
    idx = sorted(scores, key=scores.get, reverse=True)[:k]
    return " ".join([sents[i] for i in sorted(idx)])

print(summarize(TEXT, k=3))
