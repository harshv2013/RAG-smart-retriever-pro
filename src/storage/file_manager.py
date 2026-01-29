"""
File Storage Manager

Handles:
- File upload and storage
- File retrieval
- File type detection
- Storage organization
"""
import os
import shutil
import hashlib
from typing import Optional, Tuple
from pathlib import Path
from loguru import logger

from config import config


class FileManager:
    """
    Manages file storage and retrieval
    
    Features:
    - Organized storage structure
    - File deduplication
    - Supported format validation
    - Safe file operations
    """
    
    def __init__(self):
        """Initialize file manager and create directories"""
        self.storage_path = Path(config.STORAGE_PATH)
        self.upload_path = Path(config.UPLOAD_PATH)
        self.processed_path = Path(config.PROCESSED_PATH)
        
        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.upload_path.mkdir(parents=True, exist_ok=True)
        self.processed_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ File manager initialized (storage: {self.storage_path})")
    
    def save_uploaded_file(
        self,
        file_content: bytes,
        filename: str,
        validate: bool = True
    ) -> Tuple[str, str]:
        """
        Save uploaded file
        
        Args:
            file_content: File content as bytes
            filename: Original filename
            validate: Whether to validate file type
            
        Returns:
            Tuple of (filepath, file_hash)
        """

        if validate:
            if not self._is_allowed_file(filename):
                raise ValueError(
                    f"File type not allowed: {filename} "
                    f"(allowed: {config.ALLOWED_EXTENSIONS})"
                )
        
        # Calculate file hash
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Check if file already exists (deduplication)
        existing = self._find_by_hash(file_hash)
        if existing:
            logger.info(f"File already exists: {filename} -> {existing}")
            return existing, file_hash
        
        # Create storage path based on hash (for organization)
        # First 2 chars of hash as subdirectory
        subdir = self.storage_path / file_hash[:2]
        subdir.mkdir(exist_ok=True)
        
        # Save file
        filepath = subdir / filename
        with open(filepath, 'wb') as f:
            f.write(file_content)
        
        logger.info(f"✅ File saved: {filename} -> {filepath}")
        return str(filepath), file_hash
    
    def save_file_from_path(
        self,
        source_path: str,
        filename: str,
        validate: bool = True
    ) -> Tuple[str, str]:
        """
        Save file from existing path
        
        Args:
            source_path: Source file path
            validate: Whether to validate file type
            
        Returns:
            Tuple of (filepath, file_hash)
        """
        source_path = Path(source_path)
        
        if not source_path.exists():
            raise FileNotFoundError(f"File not found: {source_path}")
        
        # Read file
        with open(source_path, 'rb') as f:
            file_content = f.read()
        
        return self.save_uploaded_file(
            file_content,
            filename,
            validate=validate
        )
    
    def read_file(self, filepath: str) -> bytes:
        """
        Read file content
        
        Args:
            filepath: File path
            
        Returns:
            File content as bytes
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'rb') as f:
            return f.read()
    
    def read_text_file(self, filepath: str, encoding: str = 'utf-8') -> str:
        """
        Read text file content
        
        Args:
            filepath: File path
            encoding: Text encoding
            
        Returns:
            File content as string
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        with open(filepath, 'r', encoding=encoding, errors='ignore') as f:
            return f.read()
    
    def delete_file(self, filepath: str) -> bool:
        """
        Delete a file
        
        Args:
            filepath: File path
            
        Returns:
            True if deleted, False if not found
        """
        filepath = Path(filepath)
        
        if filepath.exists():
            filepath.unlink()
            logger.info(f"✅ File deleted: {filepath}")
            return True
        
        logger.warning(f"File not found for deletion: {filepath}")
        return False
    
    def move_to_processed(self, filepath: str) -> str:
        """
        Move file to processed directory
        
        Args:
            filepath: Source file path
            
        Returns:
            New filepath
        """
        source = Path(filepath)
        
        if not source.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # Create destination path
        dest = self.processed_path / source.name
        
        # Move file
        shutil.move(str(source), str(dest))
        
        logger.info(f"✅ File moved to processed: {source.name}")
        return str(dest)
    
    def _is_allowed_file(self, filename: str) -> bool:
        """
        Check if file type is allowed
        
        Args:
            filename: Filename to check
            
        Returns:
            True if allowed
        """
        ext = Path(filename).suffix.lower()
        allowed = config.ALLOWED_EXTENSIONS
        
        if ext not in allowed:
            logger.warning(f"File type not allowed: {ext} (allowed: {allowed})")
            return False
        
        return True

    # def _is_allowed_file(self, filename: str) -> bool:
    #     ext = Path(filename).suffix.lower()
    #     return ext in config.ALLOWED_EXTENSIONS
    
    def _find_by_hash(self, file_hash: str) -> Optional[str]:
        """
        Find file by hash (for deduplication)
        
        Args:
            file_hash: File hash
            
        Returns:
            Filepath if found, None otherwise
        """
        # Search in storage directory
        subdir = self.storage_path / file_hash[:2]
        
        if subdir.exists():
            for filepath in subdir.iterdir():
                # Verify hash
                with open(filepath, 'rb') as f:
                    if hashlib.sha256(f.read()).hexdigest() == file_hash:
                        return str(filepath)
        
        return None
    
    def get_file_info(self, filepath: str) -> dict:
        """
        Get file information
        
        Args:
            filepath: File path
            
        Returns:
            Dictionary with file info
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        stat = filepath.stat()
        
        return {
            "filename": filepath.name,
            "filepath": str(filepath),
            "size_bytes": stat.st_size,
            "size_mb": stat.st_size / 1024 / 1024,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
            "extension": filepath.suffix
        }
    
    def list_files(self, directory: str = None) -> list:
        """
        List files in directory
        
        Args:
            directory: Directory path (default: storage_path)
            
        Returns:
            List of file paths
        """
        directory = Path(directory) if directory else self.storage_path
        
        files = []
        for filepath in directory.rglob('*'):
            if filepath.is_file():
                files.append(str(filepath))
        
        return files
    
    def get_stats(self) -> dict:
        """Get storage statistics"""
        storage_files = self.list_files(self.storage_path)
        upload_files = self.list_files(self.upload_path)
        processed_files = self.list_files(self.processed_path)
        
        total_size = sum(
            Path(f).stat().st_size
            for f in storage_files + upload_files + processed_files
        )
        
        return {
            "storage_files": len(storage_files),
            "upload_files": len(upload_files),
            "processed_files": len(processed_files),
            "total_size_mb": total_size / 1024 / 1024,
            "storage_path": str(self.storage_path)
        }


# Singleton instance
file_manager = FileManager()


if __name__ == "__main__":
    # Test file manager
    print("Testing File Manager...")
    
    # Create test file
    test_content = b"This is a test file content"
    filepath, file_hash = file_manager.save_uploaded_file(
        test_content,
        "test.txt"
    )
    print(f"✅ File saved: {filepath}")
    print(f"   Hash: {file_hash}")
    
    # Read file
    content = file_manager.read_file(filepath)
    print(f"✅ File read: {len(content)} bytes")
    
    # Get stats
    stats = file_manager.get_stats()
    print(f"\nStorage Stats: {stats}")
    
    print("\n✅ File Manager test completed")
