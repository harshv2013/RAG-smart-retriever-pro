"""
Document Loading Script

Bulk load documents into SmartRetriever Pro
"""
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger
from rag_system import rag
from config import config


def load_documents(directory: str, recursive: bool = True):
    """
    Load all documents from a directory
    
    Args:
        directory: Path to directory containing documents
        recursive: Whether to search recursively
    """
    print("="*70)
    print("📚 SmartRetriever Pro - Document Loader")
    print("="*70)
    
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.error(f"Directory not found: {directory}")
        return False
    
    logger.info(f"\nLoading documents from: {directory}")
    logger.info(f"Recursive: {recursive}")
    logger.info(f"Allowed extensions: {config.ALLOWED_EXTENSIONS}")
    
    try:
        # Load documents
        results = rag.add_documents_from_directory(
            directory_path=str(directory_path),
            recursive=recursive
        )
        
        # Summary
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        print("\n" + "="*70)
        print("📊 Loading Summary")
        print("="*70)
        print(f"\nTotal documents processed: {len(results)}")
        print(f"✅ Successful: {len(successful)}")
        print(f"❌ Failed: {len(failed)}")
        
        if successful:
            total_chunks = sum(r.get('chunks_created', 0) for r in successful)
            avg_time = sum(r.get('processing_time', 0) for r in successful) / len(successful)
            
            print(f"\nTotal chunks created: {total_chunks}")
            print(f"Average processing time: {avg_time:.2f}s per document")
        
        if failed:
            print("\nFailed documents:")
            for r in failed:
                print(f"  - {r['filename']}: {r.get('error', 'Unknown error')}")
        
        # System stats
        stats = rag.get_stats()
        print(f"\n📈 System Stats:")
        print(f"  Total documents in DB: {stats['database']['total_documents']}")
        print(f"  Total chunks in DB: {stats['database']['total_chunks']}")
        print(f"  FAISS index size: {stats['faiss']['total_vectors']}")
        
        print("\n" + "="*70)
        
        return len(failed) == 0
        
    except Exception as e:
        logger.error(f"❌ Document loading failed: {e}")
        return False


def load_single_document(filepath: str):
    """
    Load a single document
    
    Args:
        filepath: Path to document file
    """
    print("="*70)
    print("📄 SmartRetriever Pro - Single Document Loader")
    print("="*70)
    
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return False
    
    try:
        result = rag.add_document(filepath)
        
        print("\n" + "="*70)
        print("✅ Document loaded successfully!")
        print("="*70)
        print(f"\nDocument ID: {result['document_id']}")
        print(f"Filename: {result['filename']}")
        print(f"Chunks created: {result['chunks_created']}")
        print(f"Processing time: {result['processing_time']:.2f}s")
        print(f"File hash: {result['file_hash']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Document loading failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SmartRetriever Pro - Document Loader")
    parser.add_argument(
        'path',
        help='Path to document file or directory'
    )
    parser.add_argument(
        '--no-recursive',
        action='store_true',
        help='Don\'t search subdirectories (for directory input)'
    )
    
    args = parser.parse_args()
    
    path = Path(args.path)
    
    if path.is_file():
        success = load_single_document(str(path))
    elif path.is_dir():
        success = load_documents(str(path), recursive=not args.no_recursive)
    else:
        logger.error(f"Invalid path: {args.path}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)
