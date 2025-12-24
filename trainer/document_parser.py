"""
Document Parser for Trainer Mode.

Parses minified JSON documents (Master Framework Agreements, etc.)
and extracts hierarchical clause structure.
"""
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class ParsedClause:
    """Represents a parsed clause from a document."""
    uid: str
    document_uid: str
    parent_uid: Optional[str]
    level: int
    position: int
    title_text: str
    body_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_clauses: List["ParsedClause"] = field(default_factory=list)
    
    def get_full_path(self) -> str:
        """Get the full hierarchical path of this clause."""
        parts = []
        if self.title_text:
            parts.append(self.title_text)
        return " > ".join(parts) if parts else f"Clause {self.uid}"
    
    def get_context(self, include_parent: bool = True) -> str:
        """Get contextual information about this clause."""
        context_parts = []
        if self.title_text:
            context_parts.append(f"Title: {self.title_text}")
        if self.metadata.get("type"):
            context_parts.append(f"Type: {self.metadata['type']}")
        context_parts.append(f"Level: {self.level}")
        return " | ".join(context_parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "uid": self.uid,
            "document_uid": self.document_uid,
            "parent_uid": self.parent_uid,
            "level": self.level,
            "position": self.position,
            "title_text": self.title_text,
            "body_text": self.body_text,
            "metadata": self.metadata,
            "sub_clauses": [sc.to_dict() for sc in self.sub_clauses]
        }


@dataclass
class ParsedDocument:
    """Represents a parsed securitization document."""
    document_uid: str
    document_type: str
    title: str
    clauses: List[ParsedClause]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_clauses_flat(self) -> List[ParsedClause]:
        """Get all clauses in a flat list (depth-first traversal)."""
        result = []
        
        def traverse(clause: ParsedClause):
            result.append(clause)
            for sub_clause in clause.sub_clauses:
                traverse(sub_clause)
        
        for clause in self.clauses:
            traverse(clause)
        
        return result
    
    def find_clause_by_uid(self, uid: str) -> Optional[ParsedClause]:
        """Find a clause by its UID."""
        all_clauses = self.get_all_clauses_flat()
        for clause in all_clauses:
            if clause.uid == uid:
                return clause
        return None
    
    def get_clauses_by_level(self, level: int) -> List[ParsedClause]:
        """Get all clauses at a specific level."""
        all_clauses = self.get_all_clauses_flat()
        return [c for c in all_clauses if c.level == level]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "document_uid": self.document_uid,
            "document_type": self.document_type,
            "title": self.title,
            "clauses": [c.to_dict() for c in self.clauses],
            "metadata": self.metadata
        }


class DocumentParser:
    """
    Parses minified JSON documents from securitization transactions.
    
    Handles Master Framework Agreements and similar structured legal documents.
    """
    
    def __init__(self):
        self.supported_formats = ["json"]
    
    def parse_json_file(self, file_path: str) -> ParsedDocument:
        """
        Parse a JSON file containing a securitization document.
        
        Args:
            file_path: Path to the JSON file
        
        Returns:
            ParsedDocument with hierarchical clause structure
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return self.parse_json_data(data)
    
    def parse_json_string(self, json_string: str) -> ParsedDocument:
        """
        Parse a JSON string containing a securitization document.
        
        Args:
            json_string: JSON string
        
        Returns:
            ParsedDocument with hierarchical clause structure
        """
        data = json.loads(json_string)
        return self.parse_json_data(data)
    
    def parse_json_data(self, data: Dict[str, Any]) -> ParsedDocument:
        """
        Parse JSON data containing a securitization document.
        
        Args:
            data: Dictionary containing document data
        
        Returns:
            ParsedDocument with hierarchical clause structure
        """
        # Handle nested "document" key structure
        if "document" in data and isinstance(data["document"], dict):
            # Extract from nested document structure
            doc_data = data["document"]
        else:
            # Use data directly if it's already the document structure
            doc_data = data
        
        # Extract document metadata
        document_uid = doc_data.get("uid", data.get("uid", "DOC_UID-UNKNOWN"))
        document_type = doc_data.get("document_type", data.get("document_type", "Master Framework Agreement"))
        
        # Extract title - prefer title_text, fallback to first line of content_text, then default
        title = doc_data.get("title_text") or data.get("title_text")
        if not title and "content_text" in doc_data:
            # Extract first meaningful line from content_text
            content_lines = doc_data["content_text"].split("\n")
            for line in content_lines:
                line = line.strip()
                if line and len(line) < 200 and not line.startswith("[[slot:"):
                    title = line
                    break
        if not title:
            title = "Untitled Document"
        
        # Parse root clauses
        root_clauses = []
        if "clauses" in doc_data:
            for clause_data in doc_data["clauses"]:
                parsed_clause = self._parse_clause(clause_data, document_uid)
                if parsed_clause:
                    root_clauses.append(parsed_clause)
        elif "clauses" in data:
            # Fallback to top-level clauses
            for clause_data in data["clauses"]:
                parsed_clause = self._parse_clause(clause_data, document_uid)
                if parsed_clause:
                    root_clauses.append(parsed_clause)
        
        # Extract document-level metadata
        metadata = {
            "jurisdiction": doc_data.get("jurisdiction", data.get("jurisdiction", "Unknown")),
            "parties": doc_data.get("parties", data.get("parties", [])),
            "effective_date": doc_data.get("effective_date", data.get("effective_date")),
            "governing_law": doc_data.get("governing_law", data.get("governing_law")),
        }
        
        return ParsedDocument(
            document_uid=document_uid,
            document_type=document_type,
            title=title,
            clauses=root_clauses,
            metadata=metadata
        )
    
    def _parse_clause(self, clause_data: Dict[str, Any], document_uid: str) -> Optional[ParsedClause]:
        """
        Recursively parse a clause and its sub-clauses.
        
        Args:
            clause_data: Dictionary containing clause data
            document_uid: UID of the parent document
        
        Returns:
            ParsedClause with sub-clauses
        """
        if not clause_data:
            return None
        
        # Extract clause fields
        uid = clause_data.get("uid", "")
        parent_uid = clause_data.get("parent_uid")
        level = clause_data.get("level", 0)
        position = clause_data.get("position", 0)
        title_text = clause_data.get("title_text", "")
        body_text = clause_data.get("body_text", "")
        metadata = clause_data.get("metadata", {})
        
        # Parse sub-clauses recursively
        sub_clauses = []
        if "clauses" in clause_data:
            for sub_clause_data in clause_data["clauses"]:
                parsed_sub_clause = self._parse_clause(sub_clause_data, document_uid)
                if parsed_sub_clause:
                    sub_clauses.append(parsed_sub_clause)
        
        return ParsedClause(
            uid=uid,
            document_uid=document_uid,
            parent_uid=parent_uid,
            level=level,
            position=position,
            title_text=title_text,
            body_text=body_text,
            metadata=metadata,
            sub_clauses=sub_clauses
        )
    
    def extract_text_content(self, clause: ParsedClause, include_sub_clauses: bool = False) -> str:
        """
        Extract text content from a clause.
        
        Args:
            clause: The clause to extract text from
            include_sub_clauses: Whether to include sub-clause text
        
        Returns:
            Extracted text content
        """
        parts = []
        
        if clause.title_text:
            parts.append(f"Title: {clause.title_text}")
        
        if clause.body_text:
            # Clean up body text (remove slot placeholders)
            body = clause.body_text.replace("[[slot:sub_clauses]]", "").strip()
            if body:
                parts.append(body)
        
        if include_sub_clauses:
            for sub_clause in clause.sub_clauses:
                sub_text = self.extract_text_content(sub_clause, include_sub_clauses=True)
                if sub_text:
                    parts.append(sub_text)
        
        return "\n\n".join(parts)
    
    def get_clause_hierarchy(self, document: ParsedDocument) -> str:
        """
        Get a human-readable representation of the clause hierarchy.
        
        Args:
            document: The parsed document
        
        Returns:
            String representation of the hierarchy
        """
        lines = [f"Document: {document.title}"]
        lines.append("=" * 80)
        
        def traverse(clause: ParsedClause, indent: int = 0):
            prefix = "  " * indent
            title = clause.title_text or f"[Clause {clause.uid}]"
            lines.append(f"{prefix}- {title} (Level {clause.level})")
            for sub_clause in clause.sub_clauses:
                traverse(sub_clause, indent + 1)
        
        for clause in document.clauses:
            traverse(clause)
        
        return "\n".join(lines)

