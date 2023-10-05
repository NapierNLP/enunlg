"""
IO data classes for the EnrichedE2E dataset.

Use dataclasses.asdict() to get a JSON-able dict from this.
Use xsdata.formats.dataclass.serializers.XmlSerializer() to generate XML outputs from this.
"""


from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class EnrichedE2ESlotValuePair:
    class Meta:
        name = "inputType"

    attribute: Optional[str] = field(default=None, metadata={"type": "Attribute"})
    tag: Optional[str] = field(default=None, metadata={"type": "Attribute"})
    value: Optional[str] = field(default=None,metadata={"type": "Attribute"})


@dataclass
class EnrichedE2ESentenceGrouping:
    class Meta:
        name = "sentenceType"

    content: List[object] = field(
        default_factory=list,
        metadata={
            "type": "Wildcard",
            "namespace": "##any",
            "mixed": True,
            "choices": (
                {
                    "name": "input",
                    "type": EnrichedE2ESlotValuePair,
                },
            ),
        }
    )


@dataclass
class EnrichedE2ESource:
    class Meta:
        name = "sourceType"

    input: List[EnrichedE2ESlotValuePair] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class EnrichedE2EContentPlan:
    class Meta:
        name = "structuringType"

    sentence: List[EnrichedE2ESentenceGrouping] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class EnrichedE2ETarget:
    class Meta:
        name = "targetType"

    annotator: Optional[str] = field(default=None, metadata={"type": "Attribute"})
    comment: Optional[str] = field(default=None, metadata={"type": "Attribute"})
    lid: Optional[str] = field(default=None, metadata={"type": "Attribute"})
    ref: Optional[str] = field(default=None, metadata={"type": "Attribute"})
    structuring: Optional[EnrichedE2EContentPlan] = field(
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
            "required": True,
        }
    )


@dataclass
class EnrichedE2EEntry:
    class Meta:
        name = "entryType"

    eid: Optional[str] = field(default=None, metadata={"type": "Attribute"})
    size: Optional[str] = field(default=None, metadata={"type": "Attribute"})
    source: Optional[EnrichedE2ESource] = field(
        default=None,
        metadata={
            "type": "Element",
            "required": True,
        }
    )
    target: List[EnrichedE2ETarget] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class EnrichedE2EEntries:
    class Meta:
        name = "entriesType"

    entry: List[EnrichedE2EEntry] = field(
        default_factory=list,
        metadata={
            "type": "Element",
        }
    )


@dataclass
class Entries(EnrichedE2EEntries):
    class Meta:
        name = "entries"
