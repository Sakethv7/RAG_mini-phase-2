# ingest.py
from rag_simple import SimpleRAG
import sys

if __name__ == "__main__":
    # Uses the local NumPy index at ./lite_index (default). Change via index_dir if you want.
    rag = SimpleRAG(index_dir="./lite_index")

    # Optional: wipe the index first with: python ingest.py --reset
    if "--reset" in sys.argv:
        rag.recreate_collection()
        print("Index reset.")

    result = rag.ingest_folder("documents")
    print("Ingestion completed!")
    print(result)
