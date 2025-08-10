# Local LLM RAG System for Text 
# Requirements: uv, ollama, chromadb, sentence-transformers

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
import requests
import json
from datetime import datetime
import re


class LocalRAGSystem:
    def __init__(self, file_path: str, db_path: str = "./chroma_db", collection_name: str = "rag_data"):
        """
        Initialize the RAG system for text 
        
        Args:
            file_path: Path to the file containing data (CSV or TXT)
            db_path: Path to store ChromaDB database
            collection_name: Name of the ChromaDB collection
        """
        self.file_path = file_path
        self.db_path = db_path
        self.file_type = os.path.splitext(file_path)[1].lower()
        
        # Initialize embedding model (runs locally)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB (local vector database)
        print("Initializing ChromaDB...")
        self.chroma_client = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Ollama API endpoint (local)
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Load and process data
        self.load_text_data()
    
    
    
    def load_text_data(self):
        """Load and process text data from a TXT file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                self.text_content = file.read()
            
            print(f"Loaded text file: {self.file_path}")
            print(f"Text length: {len(self.text_content)} characters")
            
            # Create text chunks from the document
            self.create_text_chunks()
            
        except Exception as e:
            print(f"Error loading text file: {e}")
            raise
    
   
    def create_text_chunks(self):
        """Split text document into semantically meaningful chunks for embedding
        Uses paragraph boundaries as primary dividers and handles long paragraphs intelligently.
        Includes chunk overlap to maintain context between chunks.
        """
        print("Creating semantically meaningful text chunks from document...")
        
        self.text_chunks = []
        self.metadata_list = []
        
        # Split text into paragraphs first (primary division)
        paragraphs = re.split(r'\n\s*\n', self.text_content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Parameters for chunk creation
        max_chunk_size = 1500  # Characters per chunk
        min_chunk_size = 200   # Minimum characters to form a chunk
        overlap = 100          # Characters of overlap between chunks
        
        chunks = []
        chunk_summaries = []  # Store a brief summary of each chunk
        
        # Process paragraphs into chunks
        current_chunk = ""
        current_paragraphs = []
        
        for para in paragraphs:
            # If adding this paragraph would exceed max chunk size and we already have content
            if len(current_chunk) + len(para) > max_chunk_size and len(current_chunk) >= min_chunk_size:
                # Store the current chunk
                chunks.append(current_chunk)
                
                # Create a simple summary of the chunk content
                summary = self._summarize_chunk(current_paragraphs)
                chunk_summaries.append(summary)
                
                # Start a new chunk with overlap from the previous chunk
                # Take the last part of the previous chunk for context continuity
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + "\n\n" + para
                current_paragraphs = [para]
            else:
                # Add to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + para
                else:
                    current_chunk = para
                current_paragraphs.append(para)
        
        # Add the last chunk if it has content
        if current_chunk and len(current_chunk) >= min_chunk_size:
            chunks.append(current_chunk)
            summary = self._summarize_chunk(current_paragraphs)
            chunk_summaries.append(summary)
        
        # Handle very long paragraphs that exceed max_chunk_size on their own
        final_chunks = []
        final_summaries = []
        
        for i, chunk in enumerate(chunks):
            if len(chunk) > max_chunk_size * 1.5:  # If chunk is still too long
                # Split by sentences instead
                sentences = re.split(r'(?<=[.!?])\s+', chunk)
                temp_chunk = ""
                temp_paragraphs = []
                
                for sentence in sentences:
                    if len(temp_chunk) + len(sentence) > max_chunk_size and len(temp_chunk) >= min_chunk_size:
                        final_chunks.append(temp_chunk)
                        final_summaries.append(self._summarize_chunk(temp_paragraphs))
                        
                        # Start new chunk with overlap
                        overlap_text = temp_chunk[-overlap:] if len(temp_chunk) > overlap else ""
                        temp_chunk = overlap_text + " " + sentence
                        temp_paragraphs = [sentence]
                    else:
                        if temp_chunk:
                            temp_chunk += " " + sentence
                        else:
                            temp_chunk = sentence
                        temp_paragraphs.append(sentence)
                
                # Add the last sub-chunk
                if temp_chunk and len(temp_chunk) >= min_chunk_size:
                    final_chunks.append(temp_chunk)
                    final_summaries.append(self._summarize_chunk(temp_paragraphs))
            else:
                final_chunks.append(chunk)
                final_summaries.append(chunk_summaries[i])
        
        # Create text chunks and metadata
        for i, (chunk, summary) in enumerate(zip(final_chunks, final_summaries)):
            self.text_chunks.append(chunk)
            
            metadata = {
                "chunk_id": i,
                "position": i / len(final_chunks),
                "source": "text",
                "filename": os.path.basename(self.file_path),
                "summary": summary  # Include summary in metadata for better retrieval
            }
            
            self.metadata_list.append(metadata)
        
        print(f"Created {len(self.text_chunks)} semantically meaningful chunks from document")
    
   
    
    def generate_embeddings(self):
        """Generate embeddings for all text chunks"""
        print("Generating embeddings...")
        
        # Check if embeddings already exist
        try:
            existing_count = self.collection.count()
            if existing_count > 0:
                print(f"Found {existing_count} existing embeddings. Skipping generation.")
                return
        except:
            pass
        
        # Generate embeddings in batches
        batch_size = 100
        for i in range(0, len(self.text_chunks), batch_size):
            batch_texts = self.text_chunks[i:i+batch_size]
            batch_metadata = self.metadata_list[i:i+batch_size]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(batch_texts)
            
            # Add to ChromaDB
            ids = [f"chunk_{i+j}" for j in range(len(batch_texts))]
            
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=batch_texts,
                metadatas=batch_metadata,
                ids=ids
            )
            
            print(f"Processed batch {i//batch_size + 1}/{(len(self.text_chunks)-1)//batch_size + 1}")
        
        print("Embeddings generated and stored!")
    
    def search_similar(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for similar transactions based on query"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=n_results
        )
        
        return results
    
    def query_llm(self, prompt: str, model: str = "gemma3:latest") -> str:
        """Query the local Ollama LLM"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "max_tokens": 1000
                }
            }
            
            response = requests.post(self.ollama_url, json=payload, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running locally."
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def answer_question(self, question: str) -> str:
        """Answer a question using RAG on either text data"""
        print(f"\nProcessing question: {question}")
        
        # Step 1: Retrieve relevant documents
        similar_results = self.search_similar(question, n_results=5)
        
        if not similar_results['documents'][0]:
            return "I couldn't find any relevant information in the data."
        
        # Step 2: Prepare context from retrieved documents
        context_items = []
        for i, (doc, metadata) in enumerate(zip(similar_results['documents'][0], similar_results['metadatas'][0])):
            source_type = metadata.get('source', 'unknown')   
            # Text document formatting
            position = metadata.get('position', 0)
            position_text = f"(from {int(position * 100)}% into the document)"
            context_items.append(f"Excerpt {i+1} {position_text}: {doc}")
    
        context = "\n\n".join(context_items)
        
        # Step 3: Create prompt for LLM based on source type
        source_type = similar_results['metadatas'][0][0].get('source', 'unknown')
        
        # Text document prompt with enhanced instructions for better summarization
        filename = similar_results['metadatas'][0][0].get('filename', 'document')
        prompt =  f"""Based on the following excerpts from "{filename}", please answer the user's question accurately and concisely.

Document Excerpts:
{context}

User Question: {question}

IMPORTANT INSTRUCTIONS:
1. Provide to the point answer that covers all relevant aspects found in the excerpts.

Answer:"""

        """
        IMPORTANT INSTRUCTIONS:
1. Provide a comprehensive and detailed answer that covers all relevant aspects found in the excerpts.
2. When appropriate, structure your answer with main concepts and supporting details.
3. Include specific terminology, frameworks, or models mentioned in the excerpts.
4. If the content describes steps or a process, include all steps in your answer.
5. For concepts with multiple parts or categories, list and explain each one.
6. Present information in a cohesive narrative that connects related ideas from different excerpts.
7. Answer using ONLY information from the provided excerpts.
        """
        # Step 4: Query LLM with a high token limit for detailed responses
        payload = {
            "model": "gemma3:latest",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "top_p": 0.9,
                "max_tokens": 2000  # Increased token limit for detailed responses
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=120)  # Extended timeout
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "No response generated")
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Could not connect to Ollama. Make sure Ollama is running locally."
        except Exception as e:
            return f"Error querying LLM: {str(e)}"
    
    def chat_loop(self):
        """Interactive chat loop"""
        print("\n" + "="*50)
        print("Local RAG System Ready!")
        print("Ask questions about your data.")
        print("Type 'quit' to exit.")
        print("="*50)
        
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not question:
                continue
            
            answer = self.answer_question(question)
            print(f"\nAnswer: {answer}")

    def _summarize_chunk(self, paragraphs: List[str]) -> str:
        """Create a semantic summary for a chunk of text
        This helps in better retrieving chunks by their semantic content
        """
        if not paragraphs:
            return ""
            
        # Join the paragraphs with a space for processing
        text = " ".join(paragraphs)
        
        # Extract key phrases using simple NLP techniques
        # 1. Find key sentences (first sentence, sentences with important keywords)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if not sentences:
            return ""
            
        # Always include the first sentence as it often contains the main idea
        key_sentences = [sentences[0]]
        
        # Look for sentences with important signaling words that indicate main concepts
        important_signals = ['key', 'important', 'main', 'critical', 'focus', 'framework', 
                            'method', 'technique', 'strategy', 'steps', 'phases', 'stages',
                            'quadrant', 'concept', 'principle', 'fundamental', 'essential']
        
        for sentence in sentences[1:]:
            if any(signal in sentence.lower() for signal in important_signals):
                key_sentences.append(sentence)
                
            # Limit to 3 sentences for a concise summary
            if len(key_sentences) >= 3:
                break
        
        # Create a summary from key sentences
        summary = " ".join(key_sentences)
        
        # Clean up and truncate
        summary = re.sub(r'\s+', ' ', summary.strip())
        
        # Extract key terms using basic frequency analysis
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        
        for word in words:
            if word not in ['this', 'that', 'they', 'there', 'their', 'these', 'those', 'with', 'from', 'have', 'more', 'also', 'some', 'what', 'when', 'where', 'which', 'about']:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get the top 5 most frequent words as key terms
        key_terms = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        key_terms_str = ", ".join([term for term, _ in key_terms])
        
        # Add key terms to the summary
        final_summary = f"{summary} [Key terms: {key_terms_str}]"
        
        # Truncate if too long
        return final_summary if len(final_summary) <= 200 else final_summary[:197] + "..."

def main():
    print("Local RAG System")
    print("="*30)
    
    # Get CSV file path
    file_path = input("Enter the path to your TXT file: ").strip()
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        return
    
    try:
        # Initialize RAG system
        rag_system = LocalRAGSystem(file_path)
        
        # Generate embeddings
        rag_system.generate_embeddings()
        
        # Start chat loop
        rag_system.chat_loop()
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have installed all required packages and Ollama is running.")

if __name__ == "__main__":
    main()
