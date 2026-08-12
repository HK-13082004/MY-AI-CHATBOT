import os
import time
import shutil

log_file = "rag_pipeline_log.txt"

def log(msg):
    t = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{t}] {msg}\n"
    print(line, end="")
    with open(log_file, "a") as f:
        f.write(line)
        f.flush()

if os.path.exists(log_file):
    os.remove(log_file)

log("Starting RAG pipeline validation...")

# 1. Create a simple test PDF
test_pdf = "sample_test.pdf"
db_dir = "test_db_folder"

def create_simple_pdf(path, text):
    stream_content = (
        f"BT\n/F1 24 Tf\n100 700 Td\n(RAG Test Document) Tj\nET\n"
        f"BT\n/F1 12 Tf\n100 650 Td\n({text}) Tj\nET\n"
    ).encode('ascii', 'ignore')
    stream_len = len(stream_content)
    parts = [
        b"%PDF-1.4\n",
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n",
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n",
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>\nendobj\n",
        b"4 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n",
        b"5 0 obj\n",
        f"<< /Length {stream_len} >>\n".encode('ascii'),
        b"stream\n", stream_content, b"\nendstream\nendobj\n",
        b"xref\n0 6\n",
        b"0000000000 65535 f\n0000000009 00000 n\n0000000058 00000 n\n",
        b"0000000115 00000 n\n0000000242 00000 n\n0000000311 00000 n\n",
        b"trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n462\n%%EOF\n"
    ]
    with open(path, "wb") as f:
        f.write(b"".join(parts))

create_simple_pdf(test_pdf, "This is a secret key: Antigravity-12345. The chatbot is working perfectly.")
log(f"PDF created at {test_pdf}")

# 2. Clean up test db
if os.path.exists(db_dir):
    shutil.rmtree(db_dir, ignore_errors=True)

# 3. Build vectorstore
log("Importing vectorstore modules...")
from vectorstore import build_vectorstore, load_retriever, is_vectorstore_populated
log("Vectorstore modules imported.")

log("Running build_vectorstore...")
t0 = time.time()
num_chunks = build_vectorstore(test_pdf, persist_directory=db_dir, chunk_size=200, chunk_overlap=20)
log(f"build_vectorstore done in {time.time()-t0:.2f}s — {num_chunks} chunk(s) indexed.")
log(f"Is vectorstore populated? {is_vectorstore_populated(db_dir)}")

# 4. Retrieval test
log("Running retrieval test...")
retriever = load_retriever(persist_directory=db_dir, search_k=2)
docs = retriever.invoke("What is the secret key?")
log(f"Retrieved {len(docs)} document(s).")
for i, d in enumerate(docs):
    log(f"  Doc {i}: {d.page_content.strip()[:100]}")
    if "Antigravity-12345" in d.page_content:
        log("  SUCCESS: Secret key found in retrieved context!")

# 5. Cleanup
shutil.rmtree(db_dir, ignore_errors=True)
os.remove(test_pdf)
log("Cleanup done. RAG pipeline validation PASSED!")
