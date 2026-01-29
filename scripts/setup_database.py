"""
Database Setup Script

Initializes PostgreSQL database with all tables
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from loguru import logger
from database.postgres import db_manager
# from config import config, validate_config
from config import config


def setup_database():
    """
    Setup database tables
    """
    print("="*70)
    print("🗄️  SmartRetriever Pro - Database Setup")
    print("="*70)
    
    # # Validate configuration
    # try:
    #     validate_config()
    # except Exception as e:
    #     logger.error(f"Configuration error: {e}")
    #     return False
    
    # Create tables
    try:
        logger.info("\nCreating database tables...")
        db_manager.create_tables()
        
        logger.info("\n✅ Database setup completed successfully!")
        logger.info("\nTables created:")
        logger.info("  - documents")
        logger.info("  - chunks")
        logger.info("  - embeddings")
        logger.info("  - query_logs")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Database setup failed: {e}")
        logger.error("\nPlease check:")
        logger.error("  1. PostgreSQL is running")
        logger.error("  2. Database credentials are correct in .env")
        logger.error("  3. Database exists or user has CREATE DATABASE permission")
        return False


def reset_database():
    """
    Drop and recreate all tables (USE WITH CAUTION!)
    """
    print("\n" + "="*70)
    print("⚠️  WARNING: This will delete ALL data!")
    print("="*70)
    
    response = input("\nAre you sure you want to reset the database? (yes/no): ")
    
    if response.lower() != 'yes':
        print("Operation cancelled.")
        return False
    
    try:
        logger.warning("Dropping all tables...")
        db_manager.drop_tables()
        
        logger.info("Creating fresh tables...")
        db_manager.create_tables()
        
        logger.info("✅ Database reset completed!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Database reset failed: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SmartRetriever Pro - Database Setup")
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset database (WARNING: deletes all data)'
    )
    
    args = parser.parse_args()
    
    if args.reset:
        success = reset_database()
    else:
        success = setup_database()
    
    sys.exit(0 if success else 1)
