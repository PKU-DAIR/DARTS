# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Reward scoring for DeepScaler/DeepMath dataset
Handles numeric answers, yes/no questions, and complex mathematical expressions
"""

import re
from typing import Optional
import sympy
from sympy.parsing.latex import parse_latex


def extract_answer_from_boxed(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} format, handling nested braces"""
    if "\\boxed{" not in text:
        return None
    
    try:
        # Find the last occurrence of \boxed{
        start = text.rfind("\\boxed{")
        if start == -1:
            return None
        
        start += 7  # len("\\boxed{")
        depth = 1
        end = start
        
        while depth > 0 and end < len(text):
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
            end += 1
        
        if depth == 0:
            return text[start:end-1].strip()
    except Exception:
        pass
    
    return None


def extract_answer_from_text(text: str) -> Optional[str]:
    """Extract answer from various text patterns"""
    # Try common answer patterns
    patterns = [
        r"(?i)(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\\boxed\{([^}]+)\}",
        r"(?i)(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*\$([^$]+)\$",
        r"(?i)(?:the\s+)?(?:final\s+)?answer\s+is\s*:?\s*([^\n.]+)",
        r"(?i)therefore\s*,?\s*([^\n.]+)",
        r"(?i)thus\s*,?\s*([^\n.]+)",
        r"(?i)so\s+(?:the\s+answer\s+is\s+)?([^\n.]+)",
        r"####\s*(.+)",  # GSM8K style
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1).strip()
            # Clean up LaTeX math mode markers
            answer = answer.strip('$').strip()
            return answer
    
    return None


def normalize_latex(text: str) -> str:
    """Normalize LaTeX expressions"""
    if not text:
        return ""
    
    # Remove outer dollar signs
    text = text.strip().strip('$')
    
    # Standardize common LaTeX commands
    replacements = {
        '\\text{': '',
        '\\mathrm{': '',
        '\\mathbf{': '',
        '\\boldsymbol{': '',
        '\\,': '',
        '\\;': '',
        '\\:': '',
        '\\!': '',
        '\\ ': '',
        '~': '',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Remove unmatched braces
    text = text.replace('}', '')
    
    return text.strip()


def try_parse_as_sympy(text: str) -> Optional[sympy.Basic]:
    """Try to parse text as a SymPy expression"""
    try:
        # First try direct parsing
        expr = sympy.sympify(text, evaluate=True)
        return expr
    except Exception:
        pass
    
    try:
        # Try parsing as LaTeX
        expr = parse_latex(text)
        return expr
    except Exception:
        pass
    
    return None


def normalize_mathematical_expression(text: str) -> str:
    """Normalize mathematical expressions for comparison"""
    if not text:
        return ""
    
    text = normalize_latex(text)
    
    # Try to parse and simplify with SymPy
    expr = try_parse_as_sympy(text)
    if expr is not None:
        try:
            # Simplify and convert to string
            simplified = sympy.simplify(expr)
            return str(simplified)
        except Exception:
            pass
    
    # Fall back to string normalization
    # Remove spaces
    text = ''.join(text.split())
    
    # Standardize common patterns
    text = text.lower()
    text = text.replace('\\frac', 'frac')
    text = text.replace('\\dfrac', 'frac')
    text = text.replace('\\tfrac', 'frac')
    text = text.replace('\\log', 'log')
    text = text.replace('\\ln', 'ln')
    text = text.replace('\\pi', 'pi')
    text = text.replace('\\infty', 'inf')
    
    return text


def is_yes_no_answer(text: str) -> bool:
    """Check if answer is yes/no type"""
    normalized = text.strip().lower()
    yes_no_words = ['yes', 'no', 'true', 'false']
    return normalized in yes_no_words


def compare_yes_no(pred: str, gt: str) -> bool:
    """Compare yes/no answers"""
    pred_norm = pred.strip().lower()
    gt_norm = gt.strip().lower()
    
    # Direct match
    if pred_norm == gt_norm:
        return True
    
    # Handle variations
    yes_variants = ['yes', 'true', 'correct', 'right', 't']
    no_variants = ['no', 'false', 'incorrect', 'wrong', 'f']
    
    if gt_norm in yes_variants:
        return any(variant in pred_norm for variant in yes_variants)
    elif gt_norm in no_variants:
        return any(variant in pred_norm for variant in no_variants)
    
    return False


def compare_numeric(pred: str, gt: str, tolerance: float = 1e-6) -> bool:
    """Compare numeric values using SymPy"""
    try:
        pred_expr = try_parse_as_sympy(pred)
        gt_expr = try_parse_as_sympy(gt)
        
        if pred_expr is not None and gt_expr is not None:
            # Try to evaluate numerically
            pred_val = complex(pred_expr.evalf())
            gt_val = complex(gt_expr.evalf())
            
            # Compare with tolerance
            return abs(pred_val - gt_val) < tolerance
    except Exception:
        pass
    
    # Fall back to simple float comparison
    try:
        pred_float = float(pred.strip())
        gt_float = float(gt.strip())
        return abs(pred_float - gt_float) < tolerance
    except (ValueError, TypeError):
        pass
    
    return False


def compare_expressions(pred: str, gt: str) -> bool:
    """Compare mathematical expressions using SymPy"""
    try:
        pred_expr = try_parse_as_sympy(pred)
        gt_expr = try_parse_as_sympy(gt)
        
        if pred_expr is not None and gt_expr is not None:
            # Check if expressions are equal
            diff = sympy.simplify(pred_expr - gt_expr)
            return diff == 0 or diff.equals(0)
    except Exception:
        pass
    
    return False


def compute_score(solution_str: str, ground_truth: str) -> dict:
    """
    Compute reward score for DeepScaler/DeepMath dataset
    
    Args:
        solution_str: The model's solution string
        ground_truth: The correct answer
    
    Returns:
        dict: Contains 'score' (0.0 or 1.0) and 'extracted_answer'
    """
    if not solution_str or not ground_truth:
        return {"score": 0.0, "extracted_answer": None}
    
    # Step 1: Extract answer from solution
    extracted = extract_answer_from_boxed(solution_str)
    
    if not extracted:
        extracted = extract_answer_from_text(solution_str)
    
    if not extracted:
        # Use last non-empty line as fallback
        lines = [line.strip() for line in solution_str.split('\n') if line.strip()]
        if lines:
            extracted = lines[-1]
    
    if not extracted:
        return {"score": 0.0, "extracted_answer": None}
    
    # Step 2: Compare based on answer type
    is_correct = False
    
    # Try yes/no comparison
    if is_yes_no_answer(ground_truth):
        is_correct = compare_yes_no(extracted, ground_truth)
    else:
        # Try numeric comparison
        is_correct = compare_numeric(extracted, ground_truth)
        
        if not is_correct:
            # Try symbolic expression comparison
            is_correct = compare_expressions(extracted, ground_truth)
        
        if not is_correct:
            # Try string matching after normalization
            pred_norm = normalize_mathematical_expression(extracted)
            gt_norm = normalize_mathematical_expression(ground_truth)
            
            is_correct = pred_norm == gt_norm
            
            # Last resort: substring matching
            if not is_correct and len(gt_norm) > 2:
                is_correct = gt_norm in pred_norm or pred_norm in gt_norm
    
    return {
        "score": 1.0 if is_correct else 0.0,
        "extracted_answer": extracted
    }


def verify(solution_str: str, ground_truth: str) -> bool:
    """Simple verification function that returns boolean"""
    result = compute_score(solution_str, ground_truth)
    return result["score"] > 0.5