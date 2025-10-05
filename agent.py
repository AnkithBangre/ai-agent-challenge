#!/usr/bin/env python3
"""
Agent-as-Coder: Autonomous parser generator for bank statement PDFs.
Implements a loop: plan → generate code → test → self-fix (max 3 attempts).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional
import google.generativeai as genai


class ParserAgent:
    """Autonomous agent that writes and debugs bank statement parsers."""
    
    def __init__(self, bank_name: str, max_attempts: int = 3):
        self.bank_name = bank_name.lower()
        self.max_attempts = max_attempts
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data" / self.bank_name
        self.parser_path = self.project_root / "custom_parsers" / f"{self.bank_name}_parser.py"
        
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
        # Agent memory
        self.conversation_history = []
        self.attempt_count = 0
        
    def run(self) -> bool:
        """Execute the agent loop: plan → code → test → fix."""
        print(f"🤖 Agent starting for {self.bank_name.upper()} bank parser...")
        
        # Phase 1: Plan
        if not self._plan():
            return False
        
        # Phase 2: Generate initial parser
        if not self._generate_parser():
            return False
        
        # Phase 3: Test and self-correct loop
        while self.attempt_count < self.max_attempts:
            self.attempt_count += 1
            print(f"\n🔍 Attempt {self.attempt_count}/{self.max_attempts}: Testing parser...")
            
            test_result = self._run_tests()
            
            if test_result["success"]:
                print(f"✅ Parser working! Tests passed on attempt {self.attempt_count}")
                return True
            
            print(f"❌ Tests failed. Self-correcting...")
            if not self._self_correct(test_result):
                print("⚠️  Self-correction failed")
                
        print(f"❌ Failed after {self.max_attempts} attempts")
        return False
    
    def _plan(self) -> bool:
        """Analyze the bank statement structure and plan the parser."""
        print("\n📋 Phase 1: Planning...")
        
        # Find sample files
        pdf_files = list(self.data_dir.glob("*.pdf"))
        csv_files = list(self.data_dir.glob("*.csv"))
        
        if not pdf_files or not csv_files:
            print(f"❌ Missing sample files in {self.data_dir}")
            return False
        
        pdf_path = pdf_files[0]
        csv_path = csv_files[0]
        
        # Read expected CSV structure
        with open(csv_path, 'r') as f:
            csv_content = f.read()
        
        # Extract text from PDF for analysis
        pdf_text = self._extract_pdf_text(pdf_path)
        
        planning_prompt = f"""You are a code planning expert. Analyze this bank statement and plan a parser.

Bank: {self.bank_name.upper()}
PDF Sample Text (first 2000 chars):
{pdf_text[:2000]}

Expected CSV Output:
{csv_content}

Create a detailed plan for parsing this PDF:
1. Identify the transaction table structure
2. List key patterns to extract (dates, descriptions, amounts)
3. Describe the parsing strategy (regex, table extraction, etc.)
4. Note any special formatting or edge cases

Be specific and concise."""

        try:
            response = self.model.generate_content(planning_prompt)
            plan = response.text
            self.conversation_history.append(("plan", plan))
            print(f"📝 Plan created:\n{plan[:300]}...")
            return True
        except Exception as e:
            print(f"❌ Planning failed: {e}")
            return False
    
    def _generate_parser(self) -> bool:
        """Generate the parser code based on the plan."""
        print("\n⚙️  Phase 2: Generating parser code...")
        
        # Get plan from history
        plan = next((msg[1] for msg in self.conversation_history if msg[0] == "plan"), "")
        
        # Read sample files
        pdf_path = list(self.data_dir.glob("*.pdf"))[0]
        csv_path = list(self.data_dir.glob("*.csv"))[0]
        
        with open(csv_path, 'r') as f:
            csv_content = f.read()
        
        pdf_text = self._extract_pdf_text(pdf_path)
        
        code_prompt = f"""Generate a Python parser for {self.bank_name.upper()} bank statements.

PLAN:
{plan}

REQUIREMENTS:
1. Function signature: def parse(pdf_path: str) -> pd.DataFrame
2. Return DataFrame matching this CSV schema:
{csv_content[:500]}

3. Use libraries: pandas, pdfplumber (or PyPDF2), re
4. Handle errors gracefully
5. Add type hints and docstrings

PDF SAMPLE TEXT:
{pdf_text[:1500]}

Generate ONLY the Python code, no explanations. Start with imports."""

        try:
            response = self.model.generate_content(code_prompt)
            code = self._extract_code(response.text)
            
            # Save parser
            self.parser_path.parent.mkdir(exist_ok=True)
            with open(self.parser_path, 'w') as f:
                f.write(code)
            
            self.conversation_history.append(("code", code))
            print(f"💾 Parser saved to {self.parser_path}")
            return True
            
        except Exception as e:
            print(f"❌ Code generation failed: {e}")
            return False
    
    def _run_tests(self) -> dict:
        """Run pytest on the generated parser."""
        test_file = self.project_root / "tests" / f"test_{self.bank_name}_parser.py"
        
        # Generate test if it doesn't exist
        if not test_file.exists():
            self._generate_test_file(test_file)
        
        try:
            result = subprocess.run(
                ["pytest", str(test_file), "-v"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e)
            }
    
    def _self_correct(self, test_result: dict) -> bool:
        """Fix the parser based on test failures."""
        print("\n🔧 Self-correcting based on test results...")
        
        current_code = self.parser_path.read_text()
        
        fix_prompt = f"""The parser failed tests. Fix the code.

CURRENT CODE:
{current_code}

TEST OUTPUT:
{test_result['stdout'][-1000:]}

ERROR:
{test_result['stderr'][-500:]}

Analyze the error and provide ONLY the corrected Python code. No explanations."""

        try:
            response = self.model.generate_content(fix_prompt)
            fixed_code = self._extract_code(response.text)
            
            with open(self.parser_path, 'w') as f:
                f.write(fixed_code)
            
            self.conversation_history.append(("fix", fixed_code))
            print("🔄 Parser updated")
            return True
            
        except Exception as e:
            print(f"❌ Self-correction failed: {e}")
            return False
    
    def _generate_test_file(self, test_path: Path):
        """Generate pytest test file for the parser."""
        csv_path = list(self.data_dir.glob("*.csv"))[0]
        pdf_path = list(self.data_dir.glob("*.pdf"))[0]
        
        test_code = f'''import pandas as pd
from pathlib import Path
from custom_parsers.{self.bank_name}_parser import parse

def test_{self.bank_name}_parser():
    """Test that parser output matches expected CSV."""
    pdf_path = Path("data/{self.bank_name}/{pdf_path.name}")
    expected_df = pd.read_csv("data/{self.bank_name}/{csv_path.name}")
    
    result_df = parse(str(pdf_path))
    
    assert result_df is not None, "Parser returned None"
    assert isinstance(result_df, pd.DataFrame), "Parser must return DataFrame"
    assert list(result_df.columns) == list(expected_df.columns), "Column mismatch"
    
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)
'''
        
        test_path.parent.mkdir(exist_ok=True)
        test_path.write_text(test_code)
    
    def _extract_pdf_text(self, pdf_path: Path) -> str:
        """Extract text from PDF for analysis."""
        try:
            import pdfplumber
            with pdfplumber.open(pdf_path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages[:3])
        except:
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as f:
                    reader = PyPDF2.PdfReader(f)
                    return "\n".join(page.extract_text() for page in reader.pages[:3])
            except:
                return ""
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response."""
        # Remove markdown code blocks
        if "```python" in text:
            text = text.split("```python")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Agent-as-Coder: Generate bank statement parsers")
    parser.add_argument("--target", required=True, help="Bank name (e.g., icici, sbi)")
    args = parser.parse_args()
    
    agent = ParserAgent(args.target)
    success = agent.run()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
