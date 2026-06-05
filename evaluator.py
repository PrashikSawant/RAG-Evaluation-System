from groq import Groq
from dotenv import load_dotenv
import os
import json

load_dotenv()
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def evaluate_metric(metric_name, prompt):
    """Run a single evaluation metric using LLaMA"""
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
        temperature=0.0
    )
    raw = response.choices[0].message.content.strip()

    # extract JSON from response
    try:
        start = raw.find("{")
        end = raw.rfind("}") + 1
        json_str = raw[start:end]
        result = json.loads(json_str)
        score = float(result.get("score", 0))
        reason = result.get("reason", "No reason provided")
        return round(min(max(score, 0.0), 1.0), 2), reason
    except:
        return 0.5, "Could not parse evaluation response"

def evaluate_faithfulness(question, context, answer):
    """Is the answer faithful to the context?"""
    prompt = f"""You are an expert RAG evaluator.
Evaluate if the answer is faithful to the context.
Faithful means the answer only contains information from the context.
Score 1.0 if fully faithful, 0.0 if answer contains hallucinations.

Question: {question}
Context: {context}
Answer: {answer}

Respond ONLY with valid JSON:
{{"score": 0.0 to 1.0, "reason": "one sentence explanation"}}"""
    return evaluate_metric("faithfulness", prompt)

def evaluate_answer_relevance(question, answer):
    """Does the answer actually address the question?"""
    prompt = f"""You are an expert RAG evaluator.
Evaluate if the answer is relevant to the question.
Score 1.0 if fully answers the question, 0.0 if completely off-topic.

Question: {question}
Answer: {answer}

Respond ONLY with valid JSON:
{{"score": 0.0 to 1.0, "reason": "one sentence explanation"}}"""
    return evaluate_metric("answer_relevance", prompt)

def evaluate_context_precision(question, context):
    """Were the retrieved chunks actually useful?"""
    prompt = f"""You are an expert RAG evaluator.
Evaluate if the retrieved context is precise and relevant to the question.
Score 1.0 if context is highly relevant, 0.0 if context is irrelevant.

Question: {question}
Context: {context}

Respond ONLY with valid JSON:
{{"score": 0.0 to 1.0, "reason": "one sentence explanation"}}"""
    return evaluate_metric("context_precision", prompt)

def evaluate_context_recall(question, context, answer):
    """Did the context contain enough info to answer fully?"""
    prompt = f"""You are an expert RAG evaluator.
Evaluate if the context contained sufficient information to answer the question completely.
Score 1.0 if context had all needed info, 0.0 if critical info was missing.

Question: {question}
Context: {context}
Answer: {answer}

Respond ONLY with valid JSON:
{{"score": 0.0 to 1.0, "reason": "one sentence explanation"}}"""
    return evaluate_metric("context_recall", prompt)

def run_full_evaluation(question, context, answer):
    """Run all 4 metrics and return results"""
    faithfulness_score, faithfulness_reason = evaluate_faithfulness(
        question, context, answer
    )
    relevance_score, relevance_reason = evaluate_answer_relevance(
        question, answer
    )
    precision_score, precision_reason = evaluate_context_precision(
        question, context
    )
    recall_score, recall_reason = evaluate_context_recall(
        question, context, answer
    )

    overall = round(
        (faithfulness_score + relevance_score +
         precision_score + recall_score) / 4, 2
    )

    return {
        "question": question,
        "answer": answer,
        "overall_score": overall,
        "metrics": {
            "faithfulness": {
                "score": faithfulness_score,
                "reason": faithfulness_reason
            },
            "answer_relevance": {
                "score": relevance_score,
                "reason": relevance_reason
            },
            "context_precision": {
                "score": precision_score,
                "reason": precision_reason
            },
            "context_recall": {
                "score": recall_score,
                "reason": recall_reason
            }
        }
    }