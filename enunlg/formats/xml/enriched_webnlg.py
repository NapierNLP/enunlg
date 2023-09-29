from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Dbpedialink:
    class Meta:
        name = "dbpedialink"

    direction: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )



@dataclass
class Link:
    class Meta:
        name = "link"

    direction: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    value: str = field(
        default="",
        metadata={
            "required": True,
        }
    )



@dataclass
class Dbpedialinks:
    class Meta:
        name = "dbpedialinks"

    dbpedialink: List[Dbpedialink] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )


@dataclass
class Links:
    class Meta:
        name = "links"

    link: List[Link] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
        }
    )


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
    lang: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": False,
        }
    )
    lid: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    tree: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
            "sequence": 1,
        }
    )
    sortedtripleset: List[SortedTripleSet] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
            "sequence": 1,
        }
    )
    references: List[object] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
            "sequence": 1,
        }
    )
    text: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
            "sequence": 1,
        }
    )
    template: List[str] = field(
        default_factory=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
            "sequence": 1,
        }
    )
    value: str = field(
        default="",
        metadata={
            "required": True,
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
    shape: Optional[str] = field(
        default=None,
        metadata={
            "type": "Attribute",
            "required": True,
        }
    )
    shape_type: Optional[str] = field(
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
    originaltripleset: List[OriginalTripleSet] = field(
        default=list,
        metadata={
            "type": "Element",
            "min_occurs": 1,
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

    dbpedialinks: Optional[Dbpedialinks] = field(
        default=None,
        metadata={
            "type": "Element",
        }
    )
    links: Optional[Links] = field(
        default=None,
        metadata={
            "type": "Element",
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
