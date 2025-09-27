from rag_simple import SimpleRAG

if __name__ == "__main__":
    rag = SimpleRAG()  # local store; no docker needed
    while True:
        q = input("Q: ").strip()
        if not q:
            break
        out = rag.answer(q, k=1)  # keep k=2 for short, focused prompts
        print("\n-- Answer --\n", out["response"])
        print("\n-- Matches --")
        for m in out["matches"]:
            print(f"score={m['score']:.3f}  {m['source']}  chunk={m['chunk_id']}")
        print()
