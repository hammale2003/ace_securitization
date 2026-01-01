"""
ACORD Dataset Loader.

Loads and processes the Atticus Clause Retrieval Dataset (ACORD),
derived from CUAD (Contract Understanding Atticus Dataset).
"""
import json
import csv
import re
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from config import ACORD_CLAUSE_TYPES


@dataclass
class ACORDSample:
    """A single ACORD evaluation sample."""
    contract_id: str
    contract_text: str
    clause_type: str
    query: str
    relevant_passages: List[str]
    has_clause: bool
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.relevant_passages, str):
            self.relevant_passages = [self.relevant_passages] if self.relevant_passages else []
        self.has_clause = len(self.relevant_passages) > 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'contract_id': self.contract_id,
            'contract_text': self.contract_text,
            'clause_type': self.clause_type,
            'query': self.query,
            'relevant_passages': self.relevant_passages,
            'has_clause': self.has_clause,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ACORDSample':
        return cls(
            contract_id=data.get('contract_id', ''),
            contract_text=data['contract_text'],
            clause_type=data['clause_type'],
            query=data.get('query', f"Extract the {data['clause_type']} clause"),
            relevant_passages=data.get('relevant_passages', []),
            has_clause=data.get('has_clause', bool(data.get('relevant_passages', []))),
            metadata=data.get('metadata', {})
        )


@dataclass
class ACORDDataset:
    """Container for ACORD dataset."""
    samples: List[ACORDSample]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> ACORDSample:
        return self.samples[idx]
    
    def __iter__(self):
        return iter(self.samples)
    
    def get_clause_types(self) -> List[str]:
        """Get unique clause types."""
        return list(set(s.clause_type for s in self.samples))
    
    def filter_by_clause_type(self, clause_type: str) -> 'ACORDDataset':
        """Filter by clause type."""
        filtered = [s for s in self.samples if s.clause_type == clause_type]
        return ACORDDataset(samples=filtered, metadata=self.metadata)
    
    def filter_has_clause(self, has_clause: bool = True) -> 'ACORDDataset':
        """Filter by clause presence."""
        filtered = [s for s in self.samples if s.has_clause == has_clause]
        return ACORDDataset(samples=filtered, metadata=self.metadata)
    
    def split(self, train_ratio: float = 0.7, val_ratio: float = 0.15,
              seed: int = 42, stratify: bool = True) -> Tuple['ACORDDataset', 'ACORDDataset', 'ACORDDataset']:
        """Split into train/val/test sets."""
        random.seed(seed)
        
        if stratify:
            by_type: Dict[str, List[ACORDSample]] = {}
            for sample in self.samples:
                if sample.clause_type not in by_type:
                    by_type[sample.clause_type] = []
                by_type[sample.clause_type].append(sample)
            
            train_samples, val_samples, test_samples = [], [], []
            
            for clause_type, type_samples in by_type.items():
                random.shuffle(type_samples)
                n = len(type_samples)
                train_end = int(n * train_ratio)
                val_end = int(n * (train_ratio + val_ratio))
                
                train_samples.extend(type_samples[:train_end])
                val_samples.extend(type_samples[train_end:val_end])
                test_samples.extend(type_samples[val_end:])
        else:
            samples = list(self.samples)
            random.shuffle(samples)
            n = len(samples)
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_samples = samples[:train_end]
            val_samples = samples[train_end:val_end]
            test_samples = samples[val_end:]
        
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        return (
            ACORDDataset(samples=train_samples, metadata={'split': 'train', **self.metadata}),
            ACORDDataset(samples=val_samples, metadata={'split': 'val', **self.metadata}),
            ACORDDataset(samples=test_samples, metadata={'split': 'test', **self.metadata})
        )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        clause_counts = {}
        positive_counts = {}
        
        for sample in self.samples:
            clause_type = sample.clause_type
            clause_counts[clause_type] = clause_counts.get(clause_type, 0) + 1
            if sample.has_clause:
                positive_counts[clause_type] = positive_counts.get(clause_type, 0) + 1
        
        total = len(self.samples)
        positive = sum(1 for s in self.samples if s.has_clause)
        
        return {
            'total_samples': total,
            'unique_contracts': len(set(s.contract_id for s in self.samples)),
            'clause_types': len(clause_counts),
            'samples_by_clause_type': clause_counts,
            'positive_samples_by_clause_type': positive_counts,
            'positive_rate': positive / total if total > 0 else 0
        }
    
    def to_json(self, path: str) -> None:
        """Save to JSON file."""
        data = {
            'samples': [s.to_dict() for s in self.samples],
            'metadata': self.metadata
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


class ACORDLoader:
    """Loader for ACORD dataset from various formats."""
    
    @staticmethod
    def from_json(path: str) -> ACORDDataset:
        """Load from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'samples' in data:
            samples = [ACORDSample.from_dict(s) for s in data['samples']]
            metadata = data.get('metadata', {})
        elif isinstance(data, list):
            samples = [ACORDSample.from_dict(s) for s in data]
            metadata = {}
        else:
            raise ValueError(f"Unexpected JSON structure in {path}")
        
        return ACORDDataset(samples=samples, metadata=metadata)
    
    @staticmethod
    def from_cuad_json(path: str) -> ACORDDataset:
        """Load from CUAD JSON format."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        
        for doc in data.get('data', []):
            contract_id = doc.get('title', '')
            
            for para in doc.get('paragraphs', []):
                contract_text = para.get('context', '')
                
                for qa in para.get('qas', []):
                    question = qa.get('question', '')
                    clause_type = ACORDLoader._extract_clause_type(question)
                    
                    answers = qa.get('answers', [])
                    relevant_passages = [a.get('text', '') for a in answers if a.get('text')]
                    is_impossible = qa.get('is_impossible', False)
                    
                    if clause_type:
                        samples.append(ACORDSample(
                            contract_id=contract_id,
                            contract_text=contract_text,
                            clause_type=clause_type,
                            query=question,
                            relevant_passages=relevant_passages,
                            has_clause=not is_impossible and len(relevant_passages) > 0
                        ))
        
        return ACORDDataset(
            samples=samples,
            metadata={'source': 'cuad', 'source_path': path}
        )
    
    @staticmethod
    def _extract_clause_type(question: str) -> Optional[str]:
        """Extract clause type from CUAD question."""
        question_lower = question.lower()
        
        for ct in ACORD_CLAUSE_TYPES:
            if ct.lower() in question_lower:
                return ct
        
        patterns = [
            r"related to (?:the )?(.+?)(?:\.|$)",
            r"that (?:describe|discuss|mention) (?:the )?(.+?)(?:\.|$)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip().title()
        
        return None
    
    @staticmethod
    def from_csv(path: str, text_col: str = 'contract_text',
                 clause_type_col: str = 'clause_type',
                 passage_col: str = 'relevant_passage',
                 id_col: str = 'contract_id') -> ACORDDataset:
        """Load from CSV file."""
        samples = []
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                samples.append(ACORDSample(
                    contract_id=row.get(id_col, ''),
                    contract_text=row[text_col],
                    clause_type=row[clause_type_col],
                    query=row.get('query', f"Extract the {row[clause_type_col]} clause"),
                    relevant_passages=[row[passage_col]] if row.get(passage_col) else [],
                    has_clause=bool(row.get(passage_col))
                ))
        
        return ACORDDataset(samples=samples, metadata={'source': 'csv'})


# ============================================================================
# Synthetic Data Generation
# ============================================================================

CONTRACT_TEMPLATES = {
    "NDA": '''CONFIDENTIALITY AND NON-DISCLOSURE AGREEMENT

This Confidentiality Agreement ("Agreement") is entered into as of {date} by and between {party1} ("Disclosing Party") and {party2} ("Receiving Party").

1. DEFINITION OF CONFIDENTIAL INFORMATION
"Confidential Information" means any and all non-public information disclosed by the Disclosing Party.

2. OBLIGATIONS
The Receiving Party agrees to hold all Confidential Information in strict confidence.

3. TERM AND TERMINATION
This Agreement shall remain in effect for two (2) years from the Effective Date.
{termination_clause}

4. GOVERNING LAW
{governing_law_clause}

5. LIMITATION OF LIABILITY
{liability_clause}

IN WITNESS WHEREOF, the parties have executed this Agreement.
''',

    "LICENSE": '''SOFTWARE LICENSE AGREEMENT

This License Agreement ("Agreement") is entered into as of {date} between {party1} ("Licensor") and {party2} ("Licensee").

1. GRANT OF LICENSE
{license_grant_clause}

2. RESTRICTIONS
Licensee shall not sublicense, reverse engineer, or modify the Software.

3. FEES
Licensee agrees to pay the license fees set forth in Schedule A.

4. TERMINATION
{termination_clause}

5. LIMITATION OF LIABILITY
{liability_clause}

6. GOVERNING LAW
{governing_law_clause}
''',

    "SERVICE": '''MASTER SERVICES AGREEMENT

This Services Agreement ("Agreement") is entered into as of {date} between {party1} ("Provider") and {party2} ("Client").

1. SERVICES
Provider agrees to perform the services described in each Statement of Work.

2. COMPENSATION
Client shall pay Provider the fees specified in each SOW.

3. INTELLECTUAL PROPERTY
{ip_clause}

4. TERM AND TERMINATION
{termination_clause}

5. LIMITATION OF LIABILITY
{liability_clause}

6. GOVERNING LAW
{governing_law_clause}

7. NON-SOLICITATION
{non_solicitation_clause}
'''
}

CLAUSE_VARIATIONS = {
    "Termination For Convenience": [
        "Either party may terminate this Agreement for any reason upon thirty (30) days prior written notice to the other party.",
        "Either party may terminate this Agreement for convenience upon sixty (60) days prior written notice.",
        "Client may terminate this Agreement at any time upon providing thirty (30) days advance written notice to Provider.",
        ""  # Empty = clause not present
    ],
    "Governing Law": [
        "This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of laws principles.",
        "This Agreement shall be governed by the laws of the State of California.",
        "This Agreement is governed by the laws of the State of New York, excluding its conflicts of law rules.",
        ""
    ],
    "Cap On Liability": [
        "In no event shall either party's total liability exceed the fees paid under this Agreement in the preceding twelve (12) months.",
        "Neither party's liability shall exceed the greater of (i) $100,000 or (ii) the amounts paid in the prior 12 months.",
        "The total liability of Licensor shall not exceed the license fees paid by Licensee.",
        ""
    ],
    "License Grant": [
        "Licensor hereby grants to Licensee a non-exclusive, non-transferable license to use the Software solely for Licensee's internal business purposes.",
        "Subject to the terms herein, Licensor grants Licensee a worldwide, non-exclusive license to access and use the Software.",
        ""
    ],
    "Ip Ownership Assignment": [
        "All Work Product created by Provider shall be owned exclusively by Client. Provider hereby assigns all rights in the Work Product to Client.",
        "Client shall own all intellectual property rights in any deliverables created under this Agreement.",
        ""
    ],
    "Non-Solicitation Of Employees": [
        "During the term and for one (1) year thereafter, neither party shall solicit for employment any employee of the other party.",
        "Client agrees not to solicit or hire any Provider personnel for a period of twelve (12) months following termination.",
        ""
    ]
}


def create_synthetic_sample(contract_type: str = None, clause_type: str = None) -> ACORDSample:
    """Create a synthetic ACORD sample."""
    if contract_type is None:
        contract_type = random.choice(list(CONTRACT_TEMPLATES.keys()))
    
    if clause_type is None:
        clause_type = random.choice(list(CLAUSE_VARIATIONS.keys()))
    
    template = CONTRACT_TEMPLATES.get(contract_type, CONTRACT_TEMPLATES["NDA"])
    
    # Generate clause variations
    clauses = {}
    for ct, variations in CLAUSE_VARIATIONS.items():
        clauses[ct.lower().replace(" ", "_") + "_clause"] = random.choice(variations)
    
    # Fill template
    contract_text = template.format(
        date=f"January {random.randint(1, 28)}, 202{random.randint(0, 4)}",
        party1=random.choice(["ABC Corporation", "XYZ Inc.", "Acme LLC", "TechCorp"]),
        party2=random.choice(["DEF Industries", "Beta Co.", "Client Corp", "Partner Inc."]),
        termination_clause=clauses.get("termination_for_convenience_clause", ""),
        governing_law_clause=clauses.get("governing_law_clause", ""),
        liability_clause=clauses.get("cap_on_liability_clause", ""),
        license_grant_clause=clauses.get("license_grant_clause", ""),
        ip_clause=clauses.get("ip_ownership_assignment_clause", ""),
        non_solicitation_clause=clauses.get("non-solicitation_of_employees_clause", "")
    )
    
    # Get the specific clause for this sample
    clause_key = clause_type.lower().replace(" ", "_") + "_clause"
    relevant_passage = clauses.get(clause_key, "")
    
    return ACORDSample(
        contract_id=f"synthetic_{contract_type.lower()}_{random.randint(1000, 9999)}",
        contract_text=contract_text,
        clause_type=clause_type,
        query=f"Extract the {clause_type} clause from this contract.",
        relevant_passages=[relevant_passage] if relevant_passage else [],
        has_clause=bool(relevant_passage),
        metadata={'synthetic': True, 'contract_type': contract_type}
    )


def create_synthetic_dataset(num_samples: int = 100, seed: int = 42) -> ACORDDataset:
    """Create a synthetic ACORD dataset."""
    random.seed(seed)
    
    samples = []
    clause_types = list(CLAUSE_VARIATIONS.keys())
    contract_types = list(CONTRACT_TEMPLATES.keys())
    
    for i in range(num_samples):
        contract_type = contract_types[i % len(contract_types)]
        clause_type = clause_types[i % len(clause_types)]
        samples.append(create_synthetic_sample(contract_type, clause_type))
    
    return ACORDDataset(
        samples=samples,
        metadata={'synthetic': True, 'num_samples': num_samples, 'seed': seed}
    )
