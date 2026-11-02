"""Multi-Agent AI Scientist Agent Definitions."""

from agents.critic import CriticAgent
from agents.investigator import InvestigatorAgent
from agents.librarian import LibrarianAgent
from agents.miner import MinerAgent
from agents.writer import WriterAgent

__all__ = [
    "LibrarianAgent",
    "MinerAgent",
    "InvestigatorAgent",
    "CriticAgent",
    "WriterAgent",
]
