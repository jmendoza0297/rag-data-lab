import os
import django
import time

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rag_project.settings')
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"
django.setup()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

print("\n=======================================================")
print("🧠 LABORATORIO DE VECTORES EN TIEMPO REAL - INICIADO 🧠")
print("=======================================================\n")

pdf_path = input("📍 PASO 1: Arrastra aquí un archivo PDF (o escribe su ruta) y presiona Enter:\n> ").strip()
pdf_path = pdf_path.replace("& ", "").replace("'", "").replace('"', "").strip()

if not os.path.exists(pdf_path):
    print("❌ El archivo no existe. Asegúrate de poner la ruta correcta.")
    exit()

input("\n▶️ Presiona [Enter] para EXTRAER EL TEXTO CRUDO...")
loader = PyPDFLoader(pdf_path)
docs = loader.load()
texto_limpio = "".join([d.page_content for d in docs]).strip()

if not texto_limpio:
    print("⚠️ El PDF parece ser una imagen escaneada. No se pudo extraer texto.")
    exit()

print(f"\n📄 --- TEXTO EXTRAÍDO (Muestra de 300 caracteres) ---")
print(f"{texto_limpio[:]}...\n---------------------------------------------------")

input("\n▶️ Presiona [Enter] para APLICAR LA FRAGMENTACIÓN (Chunking)...")
splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
chunks = splitter.split_documents(docs)
print(f"\n✂️ Se generaron {len(chunks)} fragmentos exactos.")
for i in range(min(3, len(chunks))):
    print(f"\n   [CHUNK {i+1}]: {chunks[i].page_content}")

input("\n▶️ Presiona [Enter] para CARGAR EL MOTOR MATEMÁTICO (sentence-transformers)...")
print("⏳ Cargando red neuronal en memoria...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("✅ Motor matemático cargado y listo.")

input("\n▶️ Presiona [Enter] para CONVERTIR EL PRIMER CHUNK EN VECTORES (Magia Pura)...")
vector = embeddings.embed_query(chunks[0].page_content)
print(f"\n🌌 ¡Transformación Exitosa! El texto se convirtió en una matriz de {len(vector)} dimensiones (números flotantes).")
print(f"🔢 Muestra matemática del Vector:\n[ {', '.join(map(lambda x: str(round(x, 4)), vector[:10]))}, ... y {len(vector)-10} números más ]")

input("\n▶️ Presiona [Enter] para VER CÓMO SE GUARDA EN CHROMADB...")
print("\n🗄️ === RADIOGRAFÍA DE LA BASE DE DATOS VECTORIAL (CHROMADB) ===")
print("-" * 100)
print(f"{'ID_DOCUMENTO':<20} | {'TEXTO (METADATO)':<30} | {'EMBEDDING (BLOB)':<30} | {'METADATOS'}")
print("-" * 100)
for i in range(min(3, len(chunks))):
    resumen_texto = chunks[i].page_content[:25].replace('\n', ' ') + "..."
    print(f"doc_chunk_{i:<10} | {resumen_texto:<30} | [0.0123, -0.9841, ...]        | {{'page': {chunks[i].metadata.get('page', 0)}}}")
print("-" * 100)

print("\n🎯 ¡Laboratorio completado! Ahora has visto exactamente cómo funciona el corazón de la IA.")
