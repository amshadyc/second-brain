"""
Command-line interface for the RAG-based personal knowledge system.

This CLI allows users to query their notes, retrieve relevant chunks,
and get AI-powered analysis using Gemini.
"""

import sys
from pathlib import Path
from datetime import datetime
import re

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import PROMPTS_DIR, RESPONSES_DIR
from config.api_keys import GEMINI_API_KEY
from scripts.retrieve import Retriever
import google.generativeai as genai


class RAGSystem:
    """
    Main RAG system class that combines retrieval and generation.
    """
    
    def __init__(self):
        """Initialize RAG system with retriever and Gemini client."""
        print("Initializing RAG system...")
        
        # Initialize retriever
        self.retriever = Retriever()
        
        # Initialize Gemini
        genai.configure(api_key=GEMINI_API_KEY)
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        # Load prompts
        self.analysis_prompt_template = self._load_prompt('analysis.txt')
        self.summary_prompt_template = self._load_prompt('summary.txt')
        self.patterns_prompt_template = self._load_prompt('patterns.txt')
        
        print("RAG system ready!\n")
    
    def _load_prompt(self, filename):
        """Load prompt template from file."""
        prompt_path = PROMPTS_DIR / filename
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _timestamp_to_readable(self, timestamp):
        """
        Convert timestamp to readable date format.
        
        Args:
            timestamp: Timestamp (could be string or int, in microseconds)
            
        Returns:
            Readable date string like "May 2022" or None if conversion fails
        """
        try:
            # Handle both string and int timestamps
            ts = int(timestamp) if timestamp else None
            if not ts:
                return None
            
            # Convert from microseconds to seconds
            ts_seconds = ts / 1_000_000 if ts > 1e12 else ts
            
            # Convert to datetime
            dt = datetime.fromtimestamp(ts_seconds)
            # Return format: "Month YYYY"
            return dt.strftime("%B %Y")
        except (ValueError, OSError, OverflowError):
            return None
    
    def _format_retrieved_notes(self, chunks):
        """
        Format retrieved chunks into a readable string without note numbers.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            Formatted string of notes
        """
        formatted = []
        for chunk in chunks:
            note_text = chunk['text']
            # Add readable date if available (without timestamp)
            readable_date = None
            if chunk.get('created_at'):
                readable_date = self._timestamp_to_readable(chunk['created_at'])
            
            if readable_date:
                note_text += f"\n(Created: {readable_date})"
            formatted.append(note_text)
        return "\n\n".join(formatted)
    
    def _build_prompt(self, template, query, chunks):
        """
        Build prompt from template with query and retrieved chunks.
        
        Args:
            template: Prompt template string
            query: User query
            chunks: Retrieved chunks
            
        Returns:
            Formatted prompt string
        """
        retrieved_notes = self._format_retrieved_notes(chunks)
        return template.format(
            retrieved_notes=retrieved_notes,
            query=query
        )
    
    def _generate_filename(self, query, response_text):
        """
        Generate a filename based on the query and response.
        
        Args:
            query: User query string
            response_text: Response text from Gemini
            
        Returns:
            Filename string
        """
        # Extract first few meaningful words from query
        query_words = re.sub(r'[^\w\s]', '', query.lower())
        query_words = query_words.split()[:5]  # Take first 5 words
        query_slug = '_'.join(query_words)
        
        # If query is too short or empty, use a default
        if len(query_slug) < 3:
            query_slug = "query"
        
        # Limit length
        query_slug = query_slug[:50]
        
        # Add timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        return f"{query_slug}_{timestamp}.md"
    
    def _save_response(self, query, response_text, filename):
        """
        Save response to a markdown file with query at the top.
        
        Args:
            query: User query string
            response_text: Response text from Gemini
            filename: Filename to save to
            
        Returns:
            Path to saved file
        """
        filepath = RESPONSES_DIR / filename
        
        # Format content with query at the top
        content = f"# Query\n\n{query}\n\n---\n\n{response_text}"
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def query(self, user_query, mode='analysis', top_k=10):
        """
        Process a user query: retrieve relevant chunks and generate response.
        
        Args:
            user_query: User's question or query
            mode: 'analysis', 'summary', or 'patterns'
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (response_text, saved_filepath)
        """
        print(f"Query: {user_query}\n")
        print(f"Retrieving top {top_k} relevant chunks...")
        
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(user_query, top_k)
        
        if not chunks:
            return "No relevant notes found.", None
        
        print(f"Retrieved {len(chunks)} chunks\n")
        
        # Select prompt template based on mode
        if mode == 'analysis':
            template = self.analysis_prompt_template
        elif mode == 'summary':
            template = self.summary_prompt_template
        elif mode == 'patterns':
            template = self.patterns_prompt_template
        else:
            template = self.analysis_prompt_template
        
        # Build prompt
        prompt = self._build_prompt(template, user_query, chunks)
        
        # Generate response with Gemini
        print("Generating response with Gemini...\n")
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Save response
            filename = self._generate_filename(user_query, response_text)
            saved_path = self._save_response(user_query, response_text, filename)
            print(f"Response saved to: {saved_path}\n")
            
            return response_text, saved_path
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            return error_msg, None


def main():
    """Main CLI function."""
    try:
        # Initialize RAG system
        rag = RAGSystem()
        
        print("=" * 70)
        print("Personal Knowledge RAG System")
        print("=" * 70)
        print("\nModes:")
        print("  - analysis: Deep analysis with themes, patterns, insights")
        print("  - summary: Condensed narrative summary")
        print("  - patterns: Detect repeated beliefs and thought loops")
        print("\nCommands:")
        print("  - Type your query to search and analyze")
        print("  - Type 'mode:analysis', 'mode:summary', or 'mode:patterns' to change mode")
        print("  - Type 'quit' or 'exit' to exit")
        print("=" * 70)
        print()
        
        current_mode = 'analysis'
        
        while True:
            try:
                user_input = input("Query: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                # Check for mode change
                if user_input.startswith('mode:'):
                    mode_name = user_input.split(':', 1)[1].strip().lower()
                    if mode_name in ['analysis', 'summary', 'patterns']:
                        current_mode = mode_name
                        print(f"Mode changed to: {current_mode}\n")
                    else:
                        print(f"Invalid mode. Use: analysis, summary, or patterns\n")
                    continue
                
                # Process query
                print("\n" + "=" * 70)
                response, saved_path = rag.query(user_input, mode=current_mode)
                print(response)
                print("=" * 70 + "\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {str(e)}\n")
    
    except Exception as e:
        print(f"Failed to initialize RAG system: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

