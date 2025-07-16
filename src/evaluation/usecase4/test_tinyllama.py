import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer(model_path):
    """Load the fine-tuned TinyLlama model and tokenizer"""
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded on device: {device}")
    return model, tokenizer, device

def format_prompt(text):
    """Format the input text as a prompt for summarization"""
    instruction = "Summarize this CBP ruling in 2-3 clear sentences."
    return f"### Instruction:\n{instruction}\n\n### Input:\n{text}\n\n### Response:\n"

def generate_summary(model, tokenizer, device, text):
    """Generate a summary for the given text"""
    prompt = format_prompt(text)
    
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        max_length=1024, 
        truncation=True,
        padding=True
    ).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            min_length=inputs.input_ids.shape[1] + 15,
            temperature=0.3,
            do_sample=True,
            top_p=0.8,
            repetition_penalty=1.5,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Extract only the new generated tokens
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    summary = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    # Clean up the response
    summary = summary.replace("### Instruction:", "").replace("### Input:", "").strip()
    
    return summary

def test_cbp_rule():
    """Test with the specific CBP rule text"""
    MODEL_PATH = "/disk/diamond-scratch/cvaro009/data/usecase4/tinyllama_tos_finetuned"
    
    # Your CBP rule text
    cbp_rule_text = """
CBP Rule 34.7.112 governs the importation of personal electronic devices (PEDs) for non-commercial use. This includes items such as smartphones, laptops, tablets, smartwatches, and e-readers that are carried by travelers for personal or professional use, excluding any devices intended for sale or distribution. Commercial shipments fall under separate regulations (19 CFR §§ 141–144). PEDs are defined as portable, electricity-powered devices capable of data storage, computation, or wireless communication. Non-commercial use refers to any use that does not involve resale, leasing, or profit-generating distribution. All travelers entering the U.S. must declare PEDs valued over $2,500 or those purchased within 60 days of entry. Devices under the $800 threshold per 19 U.S.C. § 1321 are duty-exempt if the exemption hasn’t been used in the prior 30 days. Every PED may be inspected by Customs and Border Protection (CBP) via physical, x-ray, or forensic examination (per CBP Directive No. 3340-049A). CBP officers are legally empowered (under 8 U.S.C. § 1357 and 19 U.S.C. § 1582) to search devices at the border without a warrant. Refusing inspection can result in detention (19 U.S.C. § 1499) and denial of entry. Forensic exams must be completed in 15 days unless extended under 19 CFR § 162.7. Any data retrieved is handled under DHS/CBP-006 (Automated Targeting System), with a retention window of up to 75 years. Data may be shared with DHS, FBI, and foreign partners. There are no categorical exemptions based on citizenship; U.S. citizens, lawful permanent residents, and visa holders are all subject to these rules. Diplomats must present credentials to claim immunity (8 CFR § 235.1). Journalists’ materials are protected under the Privacy Protection Act of 1980, though inspection is allowed if national interest is proven and supervisor approval is granted. Failure to declare PEDs for commercial intent or exceeding value limits may be punished under 19 U.S.C. § 1592, with possible seizure and fines. False declarations (CBP Form 6059B or verbal) are punishable under 18 U.S.C. § 1001, including possible criminal prosecution. Repeat violations may impact eligibility for expedited entry programs like Global Entry or SENTRI.
    """.strip()
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(MODEL_PATH)
    
    print("="*80)
    print("TESTING TINYLLAMA WITH CBP RULE 34.7.112")
    print("="*80)
    
    print(f"\nInput Text Length: {len(cbp_rule_text.split())} words")
    print(f"\nInput Text Preview:")
    print(f"{cbp_rule_text[:200]}...\n")
    
    print("Generating summary...")
    summary = generate_summary(model, tokenizer, device, cbp_rule_text)
    
    print(f"\nGenerated Summary:")
    print(f"Length: {len(summary.split())} words")
    print(f"Compression Ratio: {len(summary.split()) / len(cbp_rule_text.split()):.3f}")
    print(f"\nSummary Text:")
    print(f'"{summary}"')
    
    print("\n" + "="*80)
    print("EVALUATION")
    print("="*80)
    
    # Basic quality checks
    quality_checks = {
        "Contains key terms": any(term in summary.lower() for term in ['ped', 'electronic', 'device', 'cbp', 'import']),
        "Reasonable length": 20 <= len(summary.split()) <= 100,
        "Good compression": 0.05 <= len(summary.split()) / len(cbp_rule_text.split()) <= 0.25,
        "Coherent text": len(summary.strip()) > 10 and not summary.startswith("###")
    }
    
    for check, passed in quality_checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{check}: {status}")
    
    overall_score = sum(quality_checks.values()) / len(quality_checks)
    print(f"\nOverall Quality Score: {overall_score:.1%}")
    
    if overall_score >= 0.75:
        print("🎉 GOOD SUMMARIZATION!")
    elif overall_score >= 0.5:
        print("⚠️  MODERATE SUMMARIZATION")
    else:
        print("❌ POOR SUMMARIZATION")

if __name__ == "__main__":
    test_cbp_rule()