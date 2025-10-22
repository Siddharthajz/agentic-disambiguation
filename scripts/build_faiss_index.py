#!/usr/bin/env python3
"""
Build FAISS index for dense retrieval.

This script reads pre-computed Wikipedia embeddings from a local NDJSON file,
builds a FAISS index, and saves both the index and metadata for use with DenseRetriever.

Usage:
    python scripts/build_faiss_index.py --input-file data/wiki_minilm.ndjson --output-dir ./data
"""

import argparse
import json
import os
import logging

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# --- Configuration ---
# The model used to create the embeddings. MUST be used for queries.
MODEL_NAME = 'all-MiniLM-L6-v2'
# Default input file
DEFAULT_INPUT_FILE = 'data/wiki_minilm.ndjson'


def main():
    parser = argparse.ArgumentParser(
        description='Build FAISS index from pre-computed Wikipedia embeddings in NDJSON format'
    )
    parser.add_argument(
        '--input-file',
        type=str,
        default=DEFAULT_INPUT_FILE,
        help=f'Path to input NDJSON file with embeddings (default: {DEFAULT_INPUT_FILE})'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data',
        help='Directory to save index and metadata files (default: ./data)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_NAME,
        help=f'Sentence-transformers model name (default: {MODEL_NAME})'
    )
    parser.add_argument(
        '--index-name',
        type=str,
        default='ambigqa_wiki.index',
        help='Output index filename (default: ambigqa_wiki.index)'
    )
    parser.add_argument(
        '--metadata-name',
        type=str,
        default='ambigqa_wiki_metadata.json',
        help='Output metadata filename (default: ambigqa_wiki_metadata.json)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the index by running a test query'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of articles to process (for testing)'
    )

    args = parser.parse_args()

    # Verify input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file not found: {args.input_file}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    index_path = os.path.join(args.output_dir, args.index_name)
    metadata_path = os.path.join(args.output_dir, args.metadata_name)

    logger.info("=" * 80)
    logger.info("Building FAISS index for dense retrieval")
    logger.info("=" * 80)

    # Step 1: Load pre-computed dataset from local NDJSON file
    logger.info(f"1. Loading pre-computed embeddings from local file: {args.input_file}")

    # Get file size for progress tracking
    file_size_mb = os.path.getsize(args.input_file) / (1024 * 1024)
    logger.info(f"   Input file size: {file_size_mb:.2f} MB")

    # Step 2: Extract embeddings and metadata
    logger.info("2. Extracting embeddings and metadata from NDJSON...")

    embeddings_list = []
    metadata = []
    processed_count = 0

    with open(args.input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            # Skip empty lines
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"   Skipping line {line_num}: JSON decode error - {e}")
                continue

            # Extract embedding (field name: 'all-MiniLM-L6-v2')
            embedding = item.get('all-MiniLM-L6-v2')
            if embedding is None:
                logger.warning(f"   Skipping line {line_num}: no embedding found")
                continue

            # Extract and parse body field
            body = item.get('body', '')

            # Parse title and content from body (format: "Title: X Content: Y")
            title = 'Unknown'
            text = body

            if body.startswith('Title: '):
                # Split by 'Content: ' to separate title and content
                parts = body.split(' Content: ', 1)
                if len(parts) == 2:
                    title = parts[0].replace('Title: ', '').strip()
                    text = parts[1].strip()
                else:
                    # No content part, whole body is title
                    title = body.replace('Title: ', '').strip()
                    text = body

            embeddings_list.append(embedding)
            metadata.append({
                'title': title,
                'text': text
            })

            processed_count += 1

            # Progress updates
            if processed_count % 100000 == 0:
                logger.info(f"   Processed {processed_count} articles...")

            # Apply limit if specified
            if args.limit and processed_count >= args.limit:
                logger.info(f"   Reached limit of {args.limit} articles")
                break

    logger.info(f"   ✓ Processed {processed_count} articles total")

    embeddings = np.array(embeddings_list, dtype=np.float32)
    logger.info(f"   ✓ Extracted {len(embeddings)} embeddings")
    logger.info(f"   ✓ Embedding dimension: {embeddings.shape[1]}")

    # Show sample metadata
    if len(metadata) > 0:
        logger.info("\n   Sample metadata (first 3 items):")
        for i, meta in enumerate(metadata[:3]):
            title_preview = meta['title'][:60] + "..." if len(meta['title']) > 60 else meta['title']
            text_preview = meta['text'][:100] + "..." if len(meta['text']) > 100 else meta['text']
            logger.info(f"   [{i}] Title: {title_preview}")
            logger.info(f"       Text: {text_preview}")
        logger.info("")

    # Step 3: Build FAISS index
    logger.info("3. Building FAISS index from pre-computed embeddings...")

    embedding_dim = embeddings.shape[1]

    # The all-MiniLM-L6-v2 model uses cosine similarity, which is equivalent to
    # dot product on normalized vectors. We'll normalize and use IndexFlatIP.
    # Normalize embeddings to unit length for cosine similarity
    faiss.normalize_L2(embeddings)

    # Use IndexFlatIP (Inner Product) for dot product similarity
    index = faiss.IndexFlatIP(embedding_dim)

    # Add embeddings to the index. This is fast for IndexFlatIP.
    index.add(embeddings)

    logger.info(f"   ✓ FAISS index created with {index.ntotal} vectors")
    logger.info(f"   ✓ Index type: {type(index).__name__}")

    # Step 4: Save index and metadata
    logger.info("4. Saving index and metadata to disk...")

    faiss.write_index(index, index_path)
    logger.info(f"   ✓ Index saved to: {index_path}")
    logger.info(f"   ✓ Index file size: {os.path.getsize(index_path) / 1e9:.2f} GB")

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)
    logger.info(f"   ✓ Metadata saved to: {metadata_path}")
    logger.info(f"   ✓ Metadata file size: {os.path.getsize(metadata_path) / 1e6:.2f} MB")

    # Step 5: Verify (optional)
    if args.verify:
        logger.info("5. Verifying index with test query...")
        verify_index(index_path, metadata_path, args.model)

    logger.info("=" * 80)
    logger.info("✓ Index building complete!")
    logger.info("=" * 80)
    logger.info(f"\nTo use this index with DenseRetriever:")
    logger.info(f"  - Set dense_index='{index_path}'")
    logger.info(f"  - Set dense_metadata='{metadata_path}'")
    logger.info(f"  - Set dense_encoder='{args.model}'")


def verify_index(index_path: str, metadata_path: str, model_name: str):
    """
    Verify the index by running a test query.

    Args:
        index_path: Path to FAISS index file
        metadata_path: Path to metadata JSON file
        model_name: Sentence-transformers model name
    """
    logger.info("   Loading index and encoder for verification...")

    # Load index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    # Load encoder
    encoder = SentenceTransformer(model_name)

    # Test query
    test_query = "When was the NBA 3-point line introduced?"
    logger.info(f"   Running test query: '{test_query}'")

    # Encode query and normalize for cosine similarity
    query_embedding = encoder.encode([test_query], convert_to_numpy=True, normalize_embeddings=True)

    # Search
    k = 5
    scores, indices = index.search(query_embedding, k)

    logger.info(f"   ✓ Retrieved top-{k} results:")
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), 1):
        doc = metadata[idx]
        title = doc['title'][:60]
        logger.info(f"      {rank}. [{score:.4f}] {title}")

    logger.info("   ✓ Verification successful!")


if __name__ == '__main__':
    main()
