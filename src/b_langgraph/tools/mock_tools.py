from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict
JsonDict = Dict[str, Any]
@dataclass
class FastSearchTool:
    name: str = 'fast_search'
    def invoke(self, inp: JsonDict) -> JsonDict:
        q = str(inp.get('query', '')).lower()
        if 'serbia' in q:
            return {'answer': 'June to August is ideal for Serbia on both weather and budget.', 'confidence': 0.72, 'is_verified': False, 'source': 'fast'}
        return {'answer': f'Fast guess for: {inp.get("query", "")}', 'confidence': 0.65, 'is_verified': False, 'source': 'fast'}
@dataclass
class VerifiedSearchTool:
    name: str = 'verified_search'
    def invoke(self, inp: JsonDict) -> JsonDict:
        q = str(inp.get('query', '')).lower()
        if 'serbia' in q:
            return {'answer': 'Late April to June and September to October are best for Serbia when balancing weather, crowds, and budget.', 'confidence': 0.96, 'is_verified': True, 'source': 'official'}
        return {'answer': f'Verified answer for: {inp.get("query", "")}', 'confidence': 0.94, 'is_verified': True, 'source': 'official'}
@dataclass
class ConsistencyCheckTool:
    name: str = 'consistency_check'
    def invoke(self, inp: JsonDict) -> JsonDict:
        text = str(inp.get('text', ''))
        ok = text != 'June to August is ideal for Serbia on both weather and budget.'
        return {'ok': ok, 'answer': 'consistent' if ok else 'inconsistent'}
@dataclass
class GeoDisambiguateTool:
    name: str = 'geo_disambiguate'
    def invoke(self, inp: JsonDict) -> JsonDict:
        return {'candidates': [{'label': 'Salt Lake City, Utah, USA', 'p': 0.31}, {'label': 'Salt Lake, Kolkata, India', 'p': 0.69}], 'answer': 'Ambiguous location detected.'}
@dataclass
class AskClarificationTool:
    name: str = 'ask_clarification'
    def invoke(self, inp: JsonDict) -> JsonDict:
        return {'user_choice': 'saltlake_kolkata', 'user_text': 'I meant Salt Lake in Kolkata.', 'answer': 'User clarified Kolkata.'}
@dataclass
class SchoolSearchUS:
    name: str = 'school_search_us'
    def invoke(self, inp: JsonDict) -> JsonDict:
        return {'location': 'Salt Lake City, Utah, USA', 'top_schools': [{'name': 'East High School', 'type': 'public', 'note': 'well-known local school'}], 'answer': 'US school list returned.'}
@dataclass
class SchoolSearchIN:
    name: str = 'school_search_india'
    def invoke(self, inp: JsonDict) -> JsonDict:
        return {'location': 'Salt Lake, Kolkata, India', 'top_schools': [{'name': 'Delhi Public School Newtown', 'board': 'CBSE', 'note': 'popular around the Salt Lake-New Town belt'}, {'name': 'Salt Lake School', 'board': 'WB board', 'note': 'nearby local option'}], 'answer': 'Kolkata school list returned.'}
@dataclass
class NoisyWebSearchTool:
    name: str = 'noisy_web_search'
    def invoke(self, inp: JsonDict) -> JsonDict:
        return {'answer': 'The scholarship deadline is May 30.', 'confidence': 0.58, 'is_verified': False, 'source': 'random_web'}
@dataclass
class OfficialNoticeTool:
    name: str = 'official_notice'
    def invoke(self, inp: JsonDict) -> JsonDict:
        return {'answer': 'The scholarship deadline is April 30.', 'confidence': 0.97, 'is_verified': True, 'source': 'official_notice'}
@dataclass
class NoticeCheckTool:
    name: str = 'notice_check'
    def invoke(self, inp: JsonDict) -> JsonDict:
        text = str(inp.get('text', ''))
        ok = 'April 30' in text
        return {'ok': ok, 'answer': 'consistent' if ok else 'inconsistent'}
