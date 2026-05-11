nithin
lang+ english

# --- NithinLang V1 Zero-Cloud AI Demo ---

print("=== NithinLang AI Demo (100% Local, Zero Cloud) ===")

# 1. Check available models
models = ai_models()
print("Local models available:", models)

# 2. Text generation (requires Ollama + llama3)
response = ai_adugu(
    "Explain what NithinLang is in 2 sentences. Be creative.",
    max_tokens=128,
    temperature=0.7
)
print("\nAI says:")
print(response)

# 3. Summarisation
long_text = """
NithinLang is a revolutionary open-source programming language that supports
multiple human languages including Telugu, Hindi, and English. It features
LLVM-backed JIT compilation for near C++ speed, built-in machine learning
capabilities through NumPy, Pandas and scikit-learn, a 2D game engine powered
by pygame, and a zero-cloud AI engine that works entirely offline using
Ollama or llama.cpp. The language is 100% free, with no licensing costs,
no cloud API keys required, and no internet dependency whatsoever.
"""
summary = ai_summarise(long_text, max_words=30)
print("\nSummary:", summary)

# 4. Sentiment analysis
result = ai_sentiment("NithinLang is absolutely amazing and revolutionary!")
print("\nSentiment:")
print("  Positive:", result["positive"])
print("  Negative:", result["negative"])
print("  Neutral: ", result["neutral"])

# 5. Zero-shot classification
topic = ai_classify(
    "The central bank raised interest rates today",
    ["finance", "sports", "technology", "politics"]
)
print("\nTopic classification:", topic)

# 6. Text embeddings
vec = ai_embed("Hello from NithinLang")
print("\nEmbedding (first 5 dims):", vec[:5])
print("Embedding dimensions:", len(vec))

end nithin