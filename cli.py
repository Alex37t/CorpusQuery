"""
Command-Line Interface (CLI) for interacting with the application.
"""

import argparse
import os
import rag_backend as backend


def handle_query(chain, query):
    """Executes a query and prints the result to the console."""
    print("\nThinking...")
    result = chain.invoke(query)

    print("\n--- Answer ---")
    print(result["answer"])

    print("\n--- Sources ---")
    for doc in result["context"]:
        source_file = doc.metadata.get('source', 'Unknown')
        page_number = doc.metadata.get('page')
        epub_section = doc.metadata.get('category')

        location_info = ""
        if page_number is not None:
            location_info = f"Page: {page_number + 1}"
        elif epub_section:
            location_info = f"Section: {epub_section}"

        # os.path.basename is used to show only the filename, not the full path
        if location_info:
            print(f"- Source: {os.path.basename(source_file)}, {location_info}")
        else:
            print(f"- Source: {os.path.basename(source_file)}")


def main():
    parser = argparse.ArgumentParser(description="Query and explore your personal corpus of PDF and EPUB documents.")

    parser.add_argument("--question", type=str, help="A specific question to ask the documents.")
    parser.add_argument("--rebuild-db", action="store_true", help="Force rebuild of the vector database.")

    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="The number of text chunks to retrieve for context (default: 8)."
    )
    parser.add_argument(
        "--lambda-mult",
        type=float,
        default=0.5,
        help="The diversity factor for MMR search (0.0 for max diversity, 1.0 for max relevance; default: 0.5)."
    )

    args = parser.parse_args()

    # --- Pass the new arguments to the backend ---
    vector_store = backend.get_or_create_vector_store(args.rebuild_db)
    qa_chain = backend.create_rag_chain(
        vector_store,
        k=args.k,
        lambda_mult=args.lambda_mult
    )

    if args.question:
        handle_query(qa_chain, args.question)
    else:
        print("\n--- CorpusQuery (Interactive Mode) ---")
        print("Type 'exit' to quit.")
        while True:
            user_question = input("\nYour Question: ")
            if user_question.lower().strip() == 'exit':
                break
            handle_query(qa_chain, user_question)


if __name__ == '__main__':
    main()
