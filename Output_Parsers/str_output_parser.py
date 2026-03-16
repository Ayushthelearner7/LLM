import os
import traceback
from huggingface_hub import InferenceClient
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

load_dotenv()

# ── 1. Validate Token ─────────────────────────────────────────────────────────
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not token:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file!")

# ── 2. InferenceClient ────────────────────────────────────────────────────────
# Using meta-llama/Llama-3.1-8B-Instruct — already active on your HF account!
client = InferenceClient(token=token)

def call_llm(prompt_value) -> str:
    prompt_str = str(prompt_value)
    response = client.chat_completion(
        model="meta-llama/Llama-3.1-8B-Instruct",  # Already working on your account
        messages=[{"role": "user", "content": prompt_str}],
        max_tokens=512,
        temperature=0.7,
    )
    return response.choices[0].message.content

# ── 3. Prompt Templates ───────────────────────────────────────────────────────
template1 = PromptTemplate.from_template(
    "Write a detailed report on the following topic.\n\nTopic: {topic}\n\nReport:"
)
template2 = PromptTemplate.from_template(
    "Summarize the following text in exactly 5 lines.\n\nText: {text}\n\nSummary:"
)

# ── 4. Output Parser ──────────────────────────────────────────────────────────
parser = StrOutputParser()

# ── 5. LCEL Chains ────────────────────────────────────────────────────────────
chain1 = template1 | RunnableLambda(call_llm) | parser
chain2 = template2 | RunnableLambda(call_llm) | parser

# ── 6. Pipeline Function ──────────────────────────────────────────────────────
def run_chain(topic: str):
    print(f"\n⏳ Step 1: Generating report on '{topic}'...")
    report = chain1.invoke({"topic": topic})
    print("✅ Report generated!")

    print("\n⏳ Step 2: Summarizing the report...")
    summary = chain2.invoke({"text": report})
    print("✅ Summary generated!")

    return report, summary

# ── 7. Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    try:
        report_out, summary_out = run_chain("the atom")

        print("\n" + "=" * 40)
        print("        DETAILED REPORT PREVIEW")
        print("=" * 40)
        print(report_out[:500] + ("..." if len(report_out) > 500 else ""))

        print("\n" + "=" * 40)
        print("             FINAL SUMMARY")
        print("=" * 40)
        print(summary_out)

    except Exception as e:
        print(f"\n❌ Error Type    : {type(e).__name__}")
        print(f"❌ Error Message : {e}")
        print("\n--- Full Traceback ---")
        traceback.print_exc() 
