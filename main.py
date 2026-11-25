from pathlib import Path
import pymupdf
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ollama import chat


# ==========================================
# SETUP
# ==========================================

DATA_DIR = Path(__file__).resolve().parent / "data"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 3

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

QUESTION = "Що таке штучний інтелект?"


# ==========================================
# STEP 1. READING THE PDF
# ==========================================

documents = []

for pdf_path in sorted(DATA_DIR.glob("*.pdf")):
    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            documents.append({
                "source": pdf_path.name,
                "page": page_number,
                "text": page.get_text("text")
            })

print("\n[1] ЗЧИТУВАННЯ PDF")
print(f"Зчитано сторінок: {len(documents)}")
print(f"Загальна кількість символів: {sum(len(doc['text']) for doc in documents)}")


# ==========================================
# STEP 2. CUTTING INTO PIECES
# ==========================================

# def split_text(text, chunk_size, chunk_overlap):
#     chunks = []
#     step = chunk_size - chunk_overlap

#     for start in range(0, len(text), step):
#         chunk = text[start:start + chunk_size]
#         chunks.append(chunk)

#         if start + chunk_size >= len(text):
#             break

#     return chunks

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

all_chunks = []

for doc in documents:
    # chunks = split_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)
    chunks = splitter.split_text(doc["text"])

    for chunk in chunks:
        all_chunks.append({
            "source": doc["source"],
            "page": doc["page"],
            "text": chunk
        })

print("\n[2] CHUNKING")
print(f"Chunk Size: {CHUNK_SIZE}")
print(f"Chunk Overlap: {CHUNK_OVERLAP}")
print(f"Створено чанків: {len(all_chunks)}")
print(f"Приклад чанка: {all_chunks[0]['text'][:150]}...")


# ==========================================
# STEP 3. CREATING EMBEDDINGS
# ==========================================

model = SentenceTransformer(MODEL_NAME)

chunk_texts = [item["text"] for item in all_chunks]

embeddings = model.encode(
    chunk_texts,
    normalize_embeddings=True,
    show_progress_bar=False
)

print("\n[3] EMBEDDINGS")
print(f"Створено векторів: {len(embeddings)}")
print(f"Розмір вектора: {len(embeddings[0])}")
print(f"Приклад вектора: {embeddings[0][:5]}")


# ==========================================
# STEP 4. SEMANTIC SEARCH
# ==========================================

question_embedding = model.encode(
    QUESTION,
    normalize_embeddings=True
)

similarities = np.dot(embeddings, question_embedding)

top_indices = np.argsort(similarities)[::-1][:TOP_K]

print("\n[4] SEMANTIC SEARCH")
print(f"Top-K: {TOP_K}")
print(f"Запитання: {QUESTION}")

for rank, index in enumerate(top_indices, start=1):
    chunk = all_chunks[index]

    print(f"\nРезультат {rank} | Подібність: {similarities[index]:.4f}")
    print(f"Джерело: {chunk['source']} | Сторінка: {chunk['page']}")
    print(f"Текст: {chunk['text'][:250]}...")


# ==========================================
# STEP 5. GENERATING A RESPONSE
# ==========================================

context = "\n\n".join(
    f"Джерело: {all_chunks[index]['source']}\n"
    f"Сторінка: {all_chunks[index]['page']}\n"
    f"Текст: {all_chunks[index]['text']}"
    for index in top_indices
)

prompt = f"""
Відповідай українською мовою.
Використовуй тільки інформацію з контексту.
Якщо відповіді немає в контексті, повідом про це.
Не вигадуй факти.
Відповідай коротко та зрозуміло.

Контекст:
{context}

Запитання: {QUESTION}
"""

response = chat(
    model="qwen3:4b",
    messages=[
        {"role": "user", "content": prompt}
    ],
    think=False,
    options={"temperature": 0}
)

print("\n[5] GENERATION")
print(response.message.content)
