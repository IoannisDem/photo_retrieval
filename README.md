# Photo retrieval

Photo retrieval pipeline for indexing and searching image collections.

## Overview

The ingestion workflow reads images from a local folder, uploads the originals
to S3, detects faces, creates face embeddings, creates CLIP image embeddings,
and stores the image and face records in PostgreSQL with pgvector.

The retrieval workflow accepts a query image and a text prompt. It creates
CLIP image and text embeddings, detects faces in the query image, searches the
stored vectors, and returns the highest-ranked matching images.

The project is organized around configurable local models, reusable storage
clients, composable pipeline services, and mocked unit tests.
