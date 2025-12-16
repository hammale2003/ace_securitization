"""
ProseMirror JSON utilities for the ACE system.

Converts plain text and structured content to ProseMirror document format.
"""
import re
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum


class MarkType(Enum):
    """Available mark types in ProseMirror."""
    STRONG = "strong"
    EM = "em"
    UNDERLINE = "underline"
    CODE = "code"
    LINK = "link"


class NodeType(Enum):
    """Available node types in ProseMirror."""
    DOC = "doc"
    PARAGRAPH = "paragraph"
    TEXT = "text"
    HEADING = "heading"
    BULLET_LIST = "bullet_list"
    ORDERED_LIST = "ordered_list"
    LIST_ITEM = "list_item"
    BLOCKQUOTE = "blockquote"
    CODE_BLOCK = "code_block"
    HARD_BREAK = "hard_break"
    HORIZONTAL_RULE = "horizontal_rule"


@dataclass
class TextNode:
    """A text node in ProseMirror."""
    text: str
    marks: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "type": "text",
            "text": self.text
        }
        if self.marks:
            result["marks"] = self.marks
        return result


@dataclass
class ParagraphNode:
    """A paragraph node in ProseMirror."""
    content: List[Union[TextNode, Dict[str, Any]]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "paragraph",
            "content": [
                c.to_dict() if hasattr(c, 'to_dict') else c 
                for c in self.content
            ] if self.content else []
        }


@dataclass
class HeadingNode:
    """A heading node in ProseMirror."""
    level: int
    content: List[Union[TextNode, Dict[str, Any]]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "heading",
            "attrs": {"level": self.level},
            "content": [
                c.to_dict() if hasattr(c, 'to_dict') else c 
                for c in self.content
            ] if self.content else []
        }


@dataclass 
class ListItemNode:
    """A list item node in ProseMirror."""
    content: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "list_item",
            "content": self.content
        }


@dataclass
class BulletListNode:
    """A bullet list node in ProseMirror."""
    items: List[ListItemNode] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "bullet_list",
            "content": [item.to_dict() for item in self.items]
        }


@dataclass
class OrderedListNode:
    """An ordered list node in ProseMirror."""
    items: List[ListItemNode] = field(default_factory=list)
    start: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "ordered_list",
            "attrs": {"start": self.start},
            "content": [item.to_dict() for item in self.items]
        }


@dataclass
class DocNode:
    """The root document node in ProseMirror."""
    content: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "doc",
            "content": self.content
        }


class ProseMirrorBuilder:
    """
    Builder class for creating ProseMirror documents.
    
    Usage:
        builder = ProseMirrorBuilder()
        doc = builder.text_to_doc("Hello **world**!")
    """
    
    def __init__(self):
        pass
    
    def create_text(self, text: str, marks: List[str] = None) -> Dict[str, Any]:
        """Create a text node with optional marks."""
        node = {"type": "text", "text": text}
        if marks:
            node["marks"] = [{"type": m} for m in marks]
        return node
    
    def create_strong_text(self, text: str) -> Dict[str, Any]:
        """Create bold text."""
        return self.create_text(text, marks=["strong"])
    
    def create_em_text(self, text: str) -> Dict[str, Any]:
        """Create italic text."""
        return self.create_text(text, marks=["em"])
    
    def create_paragraph(self, content: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create a paragraph node."""
        return {
            "type": "paragraph",
            "content": content or []
        }
    
    def create_heading(self, text: str, level: int = 1) -> Dict[str, Any]:
        """Create a heading node."""
        return {
            "type": "heading",
            "attrs": {"level": level},
            "content": [self.create_text(text)]
        }
    
    def create_bullet_list(self, items: List[str]) -> Dict[str, Any]:
        """Create a bullet list from strings."""
        list_items = []
        for item in items:
            list_items.append({
                "type": "list_item",
                "content": [self.create_paragraph([self.create_text(item)])]
            })
        return {
            "type": "bullet_list",
            "content": list_items
        }
    
    def create_ordered_list(self, items: List[str], start: int = 1) -> Dict[str, Any]:
        """Create an ordered list from strings."""
        list_items = []
        for item in items:
            list_items.append({
                "type": "list_item",
                "content": [self.create_paragraph([self.create_text(item)])]
            })
        return {
            "type": "ordered_list",
            "attrs": {"start": start},
            "content": list_items
        }
    
    def create_blockquote(self, text: str) -> Dict[str, Any]:
        """Create a blockquote node."""
        return {
            "type": "blockquote",
            "content": [self.create_paragraph([self.create_text(text)])]
        }
    
    def create_doc(self, content: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a document node."""
        return {
            "type": "doc",
            "content": content
        }
    
    def parse_inline_formatting(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse inline formatting like **bold** and *italic*.
        
        Returns list of text nodes with appropriate marks.
        """
        nodes = []
        
        # Pattern for **bold** and *italic*
        pattern = r'(\*\*(.+?)\*\*|\*(.+?)\*|"([^"]+)"|([^*"]+))'
        
        for match in re.finditer(pattern, text):
            if match.group(2):  # **bold**
                nodes.append(self.create_strong_text(match.group(2)))
            elif match.group(3):  # *italic*
                nodes.append(self.create_em_text(match.group(3)))
            elif match.group(4):  # "quoted" - make bold for definitions
                nodes.append(self.create_strong_text(f'"{match.group(4)}"'))
            elif match.group(5):  # plain text
                if match.group(5).strip():
                    nodes.append(self.create_text(match.group(5)))
        
        # If no matches, return plain text
        if not nodes:
            nodes = [self.create_text(text)]
        
        return nodes
    
    def text_to_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """
        Convert plain text to paragraph nodes.
        
        Splits on double newlines for paragraphs.
        """
        paragraphs = []
        
        # Split on double newlines
        parts = re.split(r'\n\n+', text.strip())
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            # Check for list patterns
            lines = part.split('\n')
            
            # Check if it's a bullet list
            if all(line.strip().startswith(('- ', '• ', '* ')) for line in lines if line.strip()):
                items = [line.strip().lstrip('-•* ').strip() for line in lines if line.strip()]
                paragraphs.append(self.create_bullet_list(items))
            
            # Check if it's an ordered list
            elif all(re.match(r'^\d+[\.\)]\s', line.strip()) for line in lines if line.strip()):
                items = [re.sub(r'^\d+[\.\)]\s*', '', line.strip()) for line in lines if line.strip()]
                paragraphs.append(self.create_ordered_list(items))
            
            else:
                # Regular paragraph - join lines with spaces
                content = ' '.join(line.strip() for line in lines)
                inline_nodes = self.parse_inline_formatting(content)
                paragraphs.append(self.create_paragraph(inline_nodes))
        
        return paragraphs
    
    def text_to_doc(self, text: str) -> Dict[str, Any]:
        """
        Convert plain text to a full ProseMirror document.
        """
        paragraphs = self.text_to_paragraphs(text)
        return self.create_doc(paragraphs)
    
    def definition_to_doc(self, term: str, definition: str) -> Dict[str, Any]:
        """
        Create a definition-style document.
        
        Format: **"Term"** means [definition]
        """
        content = []
        
        # Create the definition paragraph
        para_content = [
            self.create_strong_text(f'"{term}"'),
            self.create_text(" "),
        ]
        
        # Parse the definition for any inline formatting
        para_content.extend(self.parse_inline_formatting(definition))
        
        content.append(self.create_paragraph(para_content))
        
        return self.create_doc(content)
    
    def clause_to_doc(self, clause_text: str, title: str = None) -> Dict[str, Any]:
        """
        Create a clause-style document.
        
        Optionally includes a title/heading.
        """
        content = []
        
        if title:
            content.append(self.create_heading(title, level=2))
        
        content.extend(self.text_to_paragraphs(clause_text))
        
        return self.create_doc(content)


def convert_to_prosemirror(
    text: str, 
    format_type: str = "paragraph",
    term: str = None,
    title: str = None
) -> Dict[str, Any]:
    """
    Convenience function to convert text to ProseMirror format.
    
    Args:
        text: The text to convert
        format_type: One of "paragraph", "definition", "clause", "list"
        term: For definitions, the term being defined
        title: For clauses, an optional title
    
    Returns:
        ProseMirror document JSON
    """
    builder = ProseMirrorBuilder()
    
    if format_type == "definition" and term:
        return builder.definition_to_doc(term, text)
    elif format_type == "clause":
        return builder.clause_to_doc(text, title=title)
    else:
        return builder.text_to_doc(text)


def simple_text_to_prosemirror(text: str) -> Dict[str, Any]:
    """
    Simple conversion of text to ProseMirror format.
    
    Creates a basic document with paragraphs.
    """
    builder = ProseMirrorBuilder()
    return builder.text_to_doc(text)


def create_definition_prosemirror(term: str, definition: str) -> Dict[str, Any]:
    """
    Create a ProseMirror document for a definition.
    
    Example output for term="Profit", definition="means the profit before tax...":
    {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "\"Profit\"", "marks": [{"type": "strong"}]},
                    {"type": "text", "text": " means the profit before tax..."}
                ]
            }
        ]
    }
    """
    builder = ProseMirrorBuilder()
    return builder.definition_to_doc(term, definition)


def create_structured_definition_with_slot(
    term: str,
    intro_text: str,
    sub_clauses: List[str],
    terms_list: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a structured definition with enumerated sub-clauses using slots.
    
    This format is used for legal definitions with (a), (b), (c) sub-clauses.
    
    Args:
        term: The term being defined (e.g., "Account Bank Event")
        intro_text: The introductory text before the enumeration (e.g., "means, in respect of...")
        sub_clauses: List of sub-clause texts (e.g., ["the occurrence of...", "it being or becoming..."])
        terms_list: Optional list of all terms being defined (defaults to [term])
    
    Returns:
        Structured content with main_clause and sub_clauses arrays
        
    Example:
        term = "Account Bank Event"
        intro_text = "means, in respect of an Issuer Account Bank or Collections Account Bank, any of:"
        sub_clauses = [
            "(a) the occurrence of an Insolvency Event;",
            "(b) it being or becoming subject to Insolvency Proceedings;"
        ]
        
        Returns structure suitable for reformulation output with slots.
    """
    if terms_list is None:
        terms_list = [term]
    
    # Build main clause body_doc with slot
    main_body = {
        "type": "doc",
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "\""},
                    {"type": "text", "text": term, "marks": [{"type": "strong"}]},
                    {"type": "text", "text": "\" " + intro_text}
                ]
            },
            {
                "type": "slot",
                "attrs": {"name": "sub_clauses"}
            }
        ]
    }
    
    # Build sub_clauses array (no metadata, just ProseMirror docs)
    sub_clause_objects = []
    for clause_text in sub_clauses:
        sub_clause_objects.append({
            "type": "doc",
            "content": [
                {
                    "type": "paragraph",
                    "content": [
                        {"type": "text", "text": clause_text}
                    ]
                }
            ]
        })
    
    return {
        "main_clause": main_body,
        "sub_clauses": sub_clause_objects
    }
