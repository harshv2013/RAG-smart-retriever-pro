"""
Interactive Test Script for SmartRetriever Pro

Test all features:
- Document loading
- Querying
- System statistics
- Cache performance
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger
from colorama import init, Fore, Style
from rag_system import rag
import json

# Initialize colorama for colored output
init(autoreset=True)


def print_header(text: str):
    """Print colored header"""
    print(f"\n{Fore.CYAN}{'='*70}")
    print(f"{Fore.CYAN}{text}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Fore.GREEN}✅ {text}{Style.RESET_ALL}")


def print_error(text: str):
    """Print error message"""
    print(f"{Fore.RED}❌ {text}{Style.RESET_ALL}")


def print_info(text: str):
    """Print info message"""
    print(f"{Fore.YELLOW}ℹ️  {text}{Style.RESET_ALL}")


def show_stats():
    """Display system statistics"""
    print_header("📊 System Statistics")
    
    stats = rag.get_stats()
    
    print(f"{Fore.CYAN}Database:{Style.RESET_ALL}")
    print(f"  Documents: {stats['database']['total_documents']}")
    print(f"  Chunks: {stats['database']['total_chunks']}")
    print(f"  Embeddings: {stats['database']['total_embeddings']}")
    print(f"  Avg chunks/doc: {stats['database']['avg_chunks_per_doc']:.1f}")
    
    print(f"\n{Fore.CYAN}FAISS Index:{Style.RESET_ALL}")
    print(f"  Vectors: {stats['faiss']['total_vectors']}")
    print(f"  Dimension: {stats['faiss']['dimension']}")
    print(f"  Type: {stats['faiss']['index_type']}")
    print(f"  Trained: {stats['faiss']['is_trained']}")
    
    print(f"\n{Fore.CYAN}Cache:{Style.RESET_ALL}")
    cache_stats = stats['cache']
    if cache_stats.get('status') == 'available':
        print(f"  Status: {Fore.GREEN}Available{Style.RESET_ALL}")
        print(f"  Total keys: {cache_stats.get('total_keys', 0)}")
        print(f"  Embedding cache: {cache_stats.get('embedding_keys', 0)} keys")
        print(f"  Retrieval cache: {cache_stats.get('retrieval_keys', 0)} keys")
        print(f"  Response cache: {cache_stats.get('response_keys', 0)} keys")
        print(f"  Memory used: {cache_stats.get('memory_used_mb', 0):.2f} MB")
        print(f"  Hit rate: {cache_stats.get('hit_rate', 0):.1%}")
    else:
        print(f"  Status: {Fore.YELLOW}Unavailable{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}Files:{Style.RESET_ALL}")
    print(f"  Stored: {stats['files']['storage_files']}")
    print(f"  Processed: {stats['files']['processed_files']}")
    print(f"  Total size: {stats['files']['total_size_mb']:.2f} MB")
    
    print(f"\n{Fore.CYAN}Query History:{Style.RESET_ALL}")
    query_stats = stats['queries']
    print(f"  Total queries: {query_stats['total_queries']}")
    print(f"  Cache hits: {query_stats['cache_hits']}")
    print(f"  Cache hit rate: {query_stats['cache_hit_rate']:.1%}")
    print(f"  Avg response time: {query_stats['avg_response_time_ms']:.0f}ms")


def test_document_loading():
    """Test document loading"""
    print_header("📚 Document Loading Test")
    
    # Check if data/documents exists
    docs_dir = Path("data/documents")
    if not docs_dir.exists():
        print_error(f"Documents directory not found: {docs_dir}")
        print_info("Creating sample documents...")
        
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample documents
        sample_docs = {
            "python_intro.txt": """Python is a high-level, interpreted programming language known for its simplicity and readability. 
Created by Guido van Rossum and first released in 1991, Python emphasizes code readability with its notable use of significant whitespace.

Python supports multiple programming paradigms, including object-oriented, functional, and procedural programming. 
It has a comprehensive standard library and a large ecosystem of third-party packages.""",
            
            "machine_learning.txt": """Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

There are three main types of machine learning:
1. Supervised Learning - Learning from labeled data
2. Unsupervised Learning - Finding patterns in unlabeled data
3. Reinforcement Learning - Learning through trial and error with rewards

Popular machine learning frameworks include scikit-learn, TensorFlow, and PyTorch."""
        }
        
        for filename, content in sample_docs.items():
            filepath = docs_dir / filename
            with open(filepath, 'w') as f:
                f.write(content)
        
        print_success(f"Created {len(sample_docs)} sample documents")
    
    # Load documents
    print_info(f"Loading documents from {docs_dir}...")
    
    try:
        results = rag.add_documents_from_directory(str(docs_dir))
        
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        print_success(f"Loaded {len(successful)}/{len(results)} documents")
        
        if failed:
            for r in failed:
                print_error(f"{r['filename']}: {r.get('error', 'Unknown error')}")
        
    except Exception as e:
        print_error(f"Document loading failed: {e}")


def test_query():
    """Test querying"""
    print_header("❓ Query Test")
    
    test_queries = [
        "What is Python?",
        "Explain machine learning",
        "What are the types of machine learning?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{Fore.YELLOW}Query {i}/{len(test_queries)}:{Style.RESET_ALL} {query}")
        print("-" * 70)
        
        try:
            result = rag.query(query)
            
            print(f"\n{Fore.CYAN}Answer:{Style.RESET_ALL}")
            print(result['answer'])
            
            print(f"\n{Fore.CYAN}Metadata:{Style.RESET_ALL}")
            print(f"  Sources: {', '.join(result['sources'])}")
            print(f"  Chunks retrieved: {result['chunks_retrieved']}")
            print(f"  Response time: {result['response_time_ms']:.2f}ms")
            print(f"  Cached: {'Yes' if result.get('cached') else 'No'}")
            
            if result.get('chunks'):
                print(f"\n{Fore.CYAN}Top Retrieved Chunks:{Style.RESET_ALL}")
                for j, chunk in enumerate(result['chunks'][:2], 1):
                    print(f"  {j}. {chunk['document']['filename']} (similarity: {chunk['similarity']:.3f})")
                    print(f"     {chunk['content'][:100]}...")
            
        except Exception as e:
            print_error(f"Query failed: {e}")
        
        input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")


def test_cache_performance():
    """Test cache performance"""
    print_header("⚡ Cache Performance Test")
    
    query = "What is Python?"
    
    # First query (cache miss)
    print_info("First query (cache miss expected)...")
    result1 = rag.query(query, use_cache=True)
    time1 = result1['response_time_ms']
    print(f"  Response time: {time1:.2f}ms")
    print(f"  Cached: {result1.get('cached', False)}")
    
    # Second query (cache hit)
    print_info("\nSecond query (cache hit expected)...")
    result2 = rag.query(query, use_cache=True)
    time2 = result2['response_time_ms']
    print(f"  Response time: {time2:.2f}ms")
    print(f"  Cached: {result2.get('cached', False)}")
    
    # Performance gain
    if time2 < time1:
        speedup = time1 / time2
        print_success(f"\nCache speedup: {speedup:.1f}x faster!")
    
    # Third query (no cache)
    print_info("\nThird query (cache disabled)...")
    result3 = rag.query(query, use_cache=False)
    time3 = result3['response_time_ms']
    print(f"  Response time: {time3:.2f}ms")


def interactive_mode():
    """Interactive query mode"""
    print_header("💬 Interactive Query Mode")
    print_info("Type your questions (or 'quit' to exit)\n")
    
    while True:
        try:
            query = input(f"{Fore.GREEN}Your question: {Style.RESET_ALL}").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print_info("Exiting interactive mode...")
                break
            
            if not query:
                continue
            
            result = rag.query(query)
            
            print(f"\n{Fore.CYAN}Answer:{Style.RESET_ALL}")
            print(result['answer'])
            
            print(f"\n{Fore.YELLOW}Sources:{Style.RESET_ALL} {', '.join(result['sources'])}")
            print(f"{Fore.YELLOW}Response time:{Style.RESET_ALL} {result['response_time_ms']:.2f}ms")
            print()
            
        except KeyboardInterrupt:
            print_info("\n\nExiting...")
            break
        except Exception as e:
            print_error(f"Error: {e}\n")


def main_menu():
    """Main menu"""
    while True:
        print_header("🧠 SmartRetriever Pro - Interactive Test Suite")
        
        print("1. 📊 Show System Statistics")
        print("2. 📚 Test Document Loading")
        print("3. ❓ Test Queries (Automated)")
        print("4. ⚡ Test Cache Performance")
        print("5. 💬 Interactive Query Mode")
        print("6. 🗑️  Clear Cache")
        print("7. ❌ Exit")
        
        choice = input(f"\n{Fore.GREEN}Select option (1-7): {Style.RESET_ALL}").strip()
        
        if choice == '1':
            show_stats()
            input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")
            
        elif choice == '2':
            test_document_loading()
            input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")
            
        elif choice == '3':
            test_query()
            
        elif choice == '4':
            test_cache_performance()
            input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")
            
        elif choice == '5':
            interactive_mode()
            
        elif choice == '6':
            rag.clear_cache()
            print_success("Cache cleared!")
            input(f"\n{Fore.GREEN}Press Enter to continue...{Style.RESET_ALL}")
            
        elif choice == '7':
            print_info("Goodbye!")
            break
            
        else:
            print_error("Invalid option. Please try again.")


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print_info("\n\nInterrupted. Goodbye!")
        sys.exit(0)
