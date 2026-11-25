import json
import time
from pathlib import Path

import pymupdf
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import chat


# ==========================================
# SETTINGS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESULT_FILE = BASE_DIR / "results" / "experiments_v2.json"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "qwen3:4b"

QUESTIONS = [
    "Що таке штучний інтелект?",
    "Для чого потрібен Chunk Overlap у RAG?",
    "Що таке embeddings і для чого вони використовуються?",
    "Як параметр Top-K впливає на семантичний пошук?",
    "Яка вартість навчання в університеті у 2026 році?"
]

CONFIGS = [
    (500, 100, 3),
    (500, 100, 1),
    (500, 100, 10),
    (300, 100, 3),
    (1000, 100, 3),
    (500, 0, 3),
    (500, 200, 3)
]


# ==========================================
# 1. LOAD PDF DOCUMENTS
# ==========================================

documents = []
pdf_files = sorted(DATA_DIR.glob("*.pdf"))

for pdf_path in pdf_files:
    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text")

            if text.strip():
                documents.append({
                    "source": pdf_path.name,
                    "page": page_number,
                    "text": text
                })

if not documents:
    raise ValueError("PDF-документи не знайдено.")

print(f"Завантажено PDF: {len(pdf_files)}")
print(f"Зчитано сторінок: {len(documents)}")


# ==========================================
# 2. LOAD EMBEDDING MODEL
# ==========================================

model = SentenceTransformer(EMBEDDING_MODEL)

# Зберігаємо підготовлені дані для повторного використання
cache = {}


# ==========================================
# 3. RUN EXPERIMENTS
# ==========================================

results = []
RESULT_FILE.parent.mkdir(exist_ok=True)

total = len(CONFIGS) * len(QUESTIONS)
current = 0

for chunk_size, chunk_overlap, top_k in CONFIGS:

    print(
        f"\nКонфігурація: size={chunk_size}, "
        f"overlap={chunk_overlap}, top_k={top_k}"
    )

    key = (chunk_size, chunk_overlap)

    # Перестворюємо чанки лише при зміні Size або Overlap
    if key not in cache:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = []

        for doc in documents:
            for text in splitter.split_text(doc["text"]):
                chunks.append({
                    "source": doc["source"],
                    "page": doc["page"],
                    "text": text
                })

        embeddings = model.encode(
            [chunk["text"] for chunk in chunks],
            normalize_embeddings=True,
            show_progress_bar=False
        )

        cache[key] = (chunks, embeddings)

    chunks, embeddings = cache[key]

    # Тестуємо всі запитання
    for question in QUESTIONS:
        current += 1
        start = time.perf_counter()

        question_embedding = model.encode(
            question,
            normalize_embeddings=True
        )

        similarities = np.dot(
            embeddings,
            question_embedding
        )

        top_indices = np.argsort(similarities)[::-1][:top_k]

        retrieved = []

        for index in top_indices:
            chunk = chunks[index]

            retrieved.append({
                "source": chunk["source"],
                "page": chunk["page"],
                "similarity": round(float(similarities[index]), 4),
                "text": chunk["text"]
            })

        context = "\n\n".join(
            item["text"] for item in retrieved
        )

        prompt = f"""
Відповідай українською мовою.
Використовуй тільки інформацію з контексту.
Якщо відповіді немає в контексті, повідом про це.
Відповідай коротко та зрозуміло.

Контекст:
{context}

Запитання: {question}
"""

        response = chat(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ],
            think=False,
            options={"temperature": 0}
        )

        elapsed = round(time.perf_counter() - start, 2)

        results.append({
            "question": question,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "top_k": top_k,
            "chunk_count": len(chunks),
            "retrieved": retrieved,
            "answer": response.message.content,
            "time_seconds": elapsed
        })

        # Зберігаємо результат після кожного запитання
        with open(RESULT_FILE, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "embedding_model": EMBEDDING_MODEL,
                    "llm_model": LLM_MODEL,
                    "pdf_count": len(pdf_files),
                    "results": results
                },
                file,
                ensure_ascii=False,
                indent=2
            )

        print(
            f"[{current}/{total}] "
            f"{question[:45]}... "
            f"({elapsed} с)"
        )


# ==========================================
# 4. SUMMARY
# ==========================================

print("\nТЕСТУВАННЯ ЗАВЕРШЕНО")
print(f"Конфігурацій: {len(CONFIGS)}")
print(f"Запитань: {len(QUESTIONS)}")
print(f"Результатів: {len(results)}")
print(f"Файл: {RESULT_FILE}")
