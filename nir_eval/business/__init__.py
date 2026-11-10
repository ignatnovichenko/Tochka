"""Generic business-text evaluation, independent of FinQA and XLSX adapters."""
from .evaluation import evaluate, evaluate_matrix, build_solver_messages, build_judge_messages, parse_judgement
from .providers import JudgeProvider, JudgeReply, GenerationJudgeProvider, DisabledJudge, make_judge
from .schema import import_inputs, ValidationError
from .reporting import summarize, write_report
from .cli import main
__all__ = ['evaluate', 'evaluate_matrix', 'build_solver_messages', 'build_judge_messages',
           'parse_judgement', 'JudgeProvider', 'JudgeReply', 'GenerationJudgeProvider',
           'DisabledJudge', 'make_judge', 'import_inputs', 'ValidationError', 'summarize', 'write_report', 'main']
