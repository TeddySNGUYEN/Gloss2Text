"""
Finite State Transducer for Gloss Morphological Analysis
Implements Table 1 rules and Algorithm 1 from the paper
"""
import re
from typing import List, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class Role(Enum):
    """Functional roles for gloss components"""
    REFERENCE = "REFERENCE"
    AGREEMENT = "AGREEMENT"
    NEGATION = "NEGATION"
    LOCATIVE = "LOCATIVE"
    ASPECT = "ASPECT"
    LEXICALROOT = "LEXICALROOT"


class Relation(Enum):
    """Edge relations in gloss-internal graph"""
    REFERENCE_REL = "reference"
    AGREEMENT_REL = "agreement"
    MODIFICATION = "modification"
    REL_AGREEMENT = "rel_agreement"  # refinement edge


@dataclass
class Component:
    """A morphological component within a compound gloss"""
    text: str
    role: Role
    relation: Relation
    index: int


class GlossFST:
    """
    Deterministic Finite State Transducer for gloss parsing
    Implements the FST described in Section 3.2
    """
    
    def __init__(self):
        # Pattern rules from Table 1
        self.reference_patterns = [r'^IX-.*', r'^IX$', r'^INDEX.*']
        self.agreement_patterns = [r'^1P$', r'^2P$', r'^3P$']
        self.negation_patterns = [r'^NOT$', r'^NO$']
        self.locative_patterns = [r'^LOC$', r'^LOCATION$']
        
        # Common aspect markers (can be extended)
        self.aspect_markers = {
            'UP', 'DOWN', 'CONT', 'CONTINUOUS', 'FINISH', 'DONE',
            'AGAIN', 'REPEAT', 'START', 'BEGIN'
        }
    
    def parse(self, gloss: str) -> List[Component]:
        """
        Parse a compound gloss into components with roles and relations
        
        Args:
            gloss: Raw gloss string (e.g., "IX-2P-WAKE-UP")
            
        Returns:
            List of Component objects with assigned roles and relations
        """
        # Segmentation: split at hyphen boundaries
        raw_components = self._segment_gloss(gloss)
        n = len(raw_components)
        
        # Handle single-component gloss
        if n == 1:
            return [Component(
                text=raw_components[0],
                role=Role.LEXICALROOT,
                relation=Relation.REFERENCE_REL,  # placeholder
                index=0
            )]
        
        # Role assignment
        components = []
        for i, text in enumerate(raw_components):
            role = self._assign_role(text)
            components.append(Component(
                text=text,
                role=role,
                relation=None,  # will be assigned later
                index=i
            ))
        
        # Root selection
        root_idx = self._select_root(components)
        
        # Assign relations for root-attached edges
        for i, comp in enumerate(components):
            if i != root_idx:
                comp.relation = self._get_relation(comp.role)
        
        return components, root_idx
    
    def _segment_gloss(self, gloss: str) -> List[str]:
        """Segment gloss at hyphen and plus boundaries"""
        # Split by hyphen or plus
        components = re.split(r'[-+]', gloss)
        # Remove empty strings
        components = [c.strip() for c in components if c.strip()]
        return components
    
    def _assign_role(self, component: str) -> Role:
        """
        Assign functional role using pattern matching (Table 1)
        
        Args:
            component: Individual morpheme string
            
        Returns:
            Assigned Role
        """
        # Check reference patterns
        for pattern in self.reference_patterns:
            if re.match(pattern, component, re.IGNORECASE):
                return Role.REFERENCE
        
        # Check agreement patterns
        for pattern in self.agreement_patterns:
            if re.match(pattern, component, re.IGNORECASE):
                return Role.AGREEMENT
        
        # Check negation patterns
        for pattern in self.negation_patterns:
            if re.match(pattern, component, re.IGNORECASE):
                return Role.NEGATION
        
        # Check locative patterns
        for pattern in self.locative_patterns:
            if re.match(pattern, component, re.IGNORECASE):
                return Role.LOCATIVE
        
        # Check aspect markers
        if component.upper() in self.aspect_markers:
            return Role.ASPECT
        
        # Default: lexical root
        return Role.LEXICALROOT
    
    def _select_root(self, components: List[Component]) -> int:
        """
        Select root node (first LEXICALROOT or last component)
        
        Args:
            components: List of parsed components
            
        Returns:
            Index of root component
        """
        for i, comp in enumerate(components):
            if comp.role == Role.LEXICALROOT:
                return i
        
        # Fallback: last component
        return len(components) - 1
    
    def _get_relation(self, role: Role) -> Relation:
        """
        Map role to relation type for root attachment
        
        Args:
            role: Component role
            
        Returns:
            Relation type
        """
        if role == Role.REFERENCE:
            return Relation.REFERENCE_REL
        elif role == Role.AGREEMENT:
            return Relation.AGREEMENT_REL
        else:  # NEGATION, LOCATIVE, ASPECT
            return Relation.MODIFICATION


def parse_gloss_sequence(gloss_sequence: List[str]) -> List[Tuple[List[Component], int]]:
    """
    Parse a sequence of glosses
    
    Args:
        gloss_sequence: List of gloss strings
        
    Returns:
        List of (components, root_idx) tuples
    """
    parser = GlossFST()
    parsed = []
    
    for gloss in gloss_sequence:
        components, root_idx = parser.parse(gloss)
        parsed.append((components, root_idx))
    
    return parsed


# Example usage
if __name__ == "__main__":
    parser = GlossFST()
    
    # Test cases from the paper
    test_glosses = [
        "IX-2P-WAKE-UP",
        "BATH",
        "IX-1P-LOVE-3P",
        "NOT-FINISH",
        "WAKE-UP"
    ]
    
    for gloss in test_glosses:
        print(f"\nParsing: {gloss}")
        components, root_idx = parser.parse(gloss)
        print(f"Root: {components[root_idx].text} (index {root_idx})")
        for comp in components:
            print(f"  {comp.index}: {comp.text} -> {comp.role.value} ({comp.relation.value if comp.relation else 'N/A'})")
