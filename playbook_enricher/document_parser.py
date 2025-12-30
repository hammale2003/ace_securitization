"""
Document Parser for Playbook Enricher.

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
    
    def get_full_text(self) -> str:
        """Get full text including title and body."""
        parts = []
        if self.title_text:
            parts.append(self.title_text)
        if self.body_text:
            body = self.body_text.replace("[[slot:sub_clauses]]", "").strip()
            if body:
                parts.append(body)
        return "\n".join(parts)
    
    def is_operative(self) -> bool:
        """Check if this is an operative or definition clause."""
        clause_type = self.metadata.get("type", "")
        return clause_type in ["operative_clause", "definition_clause"]


@dataclass
class ParsedDocument:
    """Represents a parsed securitization document."""
    document_uid: str
    document_type: str
    title: str
    clauses: List[ParsedClause]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_clauses_flat(self) -> List[ParsedClause]:
        """Get all clauses in a flat list."""
        result = []
        
        def traverse(clause: ParsedClause):
            result.append(clause)
            for sub_clause in clause.sub_clauses:
                traverse(sub_clause)
        
        for clause in self.clauses:
            traverse(clause)
        
        return result
    
    def get_operative_clauses(self) -> List[ParsedClause]:
        """Get only operative and definition clauses."""
        return [c for c in self.get_all_clauses_flat() if c.is_operative()]


class DocumentParser:
    """Parses minified JSON documents from securitization transactions."""
    
    def parse_json_file(self, file_path: str) -> ParsedDocument:
        """Parse a JSON file containing a securitization document."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return self.parse_json_data(data)
    
    def parse_json_string(self, json_string: str) -> ParsedDocument:
        """Parse a JSON string containing a securitization document."""
        data = json.loads(json_string)
        return self.parse_json_data(data)
    
    def parse_json_data(self, data: Dict[str, Any]) -> ParsedDocument:
        """Parse JSON data containing a securitization document."""
        if "document" in data and isinstance(data["document"], dict):
            doc_data = data["document"]
        else:
            doc_data = data
        
        document_uid = doc_data.get("uid", data.get("uid", "DOC_UID-UNKNOWN"))
        document_type = doc_data.get("document_type", data.get("document_type", "Master Framework Agreement"))
        
        title = doc_data.get("title_text") or data.get("title_text")
        if not title and "content_text" in doc_data:
            content_lines = doc_data["content_text"].split("\n")
            for line in content_lines:
                line = line.strip()
                if line and len(line) < 200 and not line.startswith("[[slot:"):
                    title = line
                    break
        if not title:
            title = "Untitled Document"
        
        root_clauses = []
        if "clauses" in doc_data:
            for clause_data in doc_data["clauses"]:
                parsed_clause = self._parse_clause(clause_data, document_uid)
                if parsed_clause:
                    root_clauses.append(parsed_clause)
        elif "clauses" in data:
            for clause_data in data["clauses"]:
                parsed_clause = self._parse_clause(clause_data, document_uid)
                if parsed_clause:
                    root_clauses.append(parsed_clause)
        
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
        """Recursively parse a clause and its sub-clauses."""
        if not clause_data:
            return None
        
        uid = clause_data.get("uid", "")
        parent_uid = clause_data.get("parent_uid")
        level = clause_data.get("level", 0)
        position = clause_data.get("position", 0)
        title_text = clause_data.get("title_text", "")
        body_text = clause_data.get("body_text", "")
        metadata = clause_data.get("metadata", {})
        
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

