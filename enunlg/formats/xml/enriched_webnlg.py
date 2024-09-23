from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Union

import logging

from xsdata.models.datatype import XmlDate

logger = logging.getLogger(__name__)




@dataclass
class EntityMap:
    """EnrichedWebNLG-style mapping from entity placeholders to the canonical forms for the entities they refer to."""
    class Meta:
        name = "entitymap"

    entity: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )


@dataclass
class ModifiedTripleSet:
    """WebNLG-style set of RDF triples which have been modified, typically by a normalisation process"""
    class Meta:
        name = "modifiedtripleset"

    # TODO change the name of mtriple to reflect that it's a list of mtriples
    mtriple: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )


@dataclass
class OriginalTripleSet:
    """WebNLG-style set of RDF triples, as taken from DBPedia"""
    class Meta:
        name = "originaltripleset"

    otriple: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )


@dataclass
class Reference:
    class Meta:
        name = "reference"

    entity: Optional[Union[str, float, int]] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    number: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    tag: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    type_value: Optional[str] = field(
        default=None,
        metadata={
            "name": "type",
            "type": "Attribute",
            "required": True,
        }
    )
    value: Union[str, float, int, XmlDate, Decimal] = field(
        default="",
        metadata={
            "required": True,
        }
    )


@dataclass
class SentenceGrouping:
    """EnrichedWebNLG-style set of triples which should be expressed in a single sentence."""
    class Meta:
        name = "sentence"

    id: Optional[int] = field(
        default=None,
        metadata={
            "name": "ID",
            "type": "Attribute",
            "required": True,
        }
    )
    striple: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class References:
    class Meta:
        name = "references"

    reference: List[Reference] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )


@dataclass
class SortedTripleSet:
    """EnrichedWebNLG-style set of RDF triples where the order has been fixed (i.e. they have been sorted somehow)"""
    class Meta:
        name = "sortedtripleset"

    sentence: List[SentenceGrouping] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )


@dataclass
class Lex:
    """WebNLG-style files use <lex> elements for all the information relating to a particular surface realisation."""
    class Meta:
        name = "lex"

    comment: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    lid: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    sortedtripleset: SortedTripleSet = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    references: Optional[References] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    text: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    template: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    lexicalization: Optional[str] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class EnrichedWebNLGEntry:
    """
    (Enriched) WebNLG-style entry

    Attributes
    --------
    category:
        The DBPedia category for this entry
    eid:

    size:

    originaltripleset:

    modifiedtripleset:

    lex:

    entitymap:
        Only present in EnrichedWebNLG datasets
    """
    class Meta:
        name = "entry"

    category: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    eid: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    size: Optional[int] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    originaltripleset: Optional[OriginalTripleSet] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    modifiedtripleset: Optional[ModifiedTripleSet] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    lex: List[Lex] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )
    entitymap: Optional[EntityMap] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )


@dataclass
class EnrichedWebNLGEntries:
    """
    WebNLG-style files contain a set of <entries> comprising the <benchmark>

    Note the odd name, that the attribute storing the list of entries is called "entry" rather than "entries";
    this is an artefact of the library we're using to parse XML to dataclasses

    Attributes
    ----
    entry:
        "list of entries for the parent benchmark
    """
    class Meta:
        name = "entries"

    entry: List[EnrichedWebNLGEntry] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )


@dataclass
class EnrichedWebNLGBenchmark:
    """
    WebNLG-style files use <benchmark> as the root element.

    Attributes
    --------
    entries:
        the (possibly empty) (XML) element for containing the entries for this WebNLGBenchmark.
    """
    class Meta:
        name = "benchmark"

    entries: Optional[EnrichedWebNLGEntries] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
