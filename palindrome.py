"""
AI Solutions Engineer Demo: Conversation Risk Analysis Engine
Author: Ebebe Clarence
Description:
    Analyses a conversation transcript and produces:
        - HIV acquisition risk score (0-100)
        - Mental-health disorder risk score (0-100)
        - Sentiment trend analysis
        - Risk trajectory over time
        - Urgent flag detection (PEP window, suicidality etc.)
        - A SA NDoH-aligned recommendation & treatment plan

Usage:
    python risk_engine.py plaindrome_data.txt
"""

import sys
import re
import json
from collections import Counter, defaultdict
from statistics import mean

# ---------------------------------------------------------
#                CONFIGURABLE KEYWORD MODELS
# ---------------------------------------------------------

HIV_KEYWORDS = {
    # High-risk exposures
    "unprotected sex": 35,
    "condomless": 35,
    "no condom": 30,
    "sexual assault": 60,
    "rape": 60,
    "bleeding after sex": 35,
    "partner hiv": 40,
    "partner positive": 40,
    "multiple partners": 25,
    "new partner": 15,
    "sti": 20,
    "ulcer": 20,
    "sore": 15,
    "discharge": 15,
    "recent exposure": 30,
    "72 hours": 30,
}

MENTAL_KEYWORDS = {
    # Emotional distress
    "stressed": 20,
    "anxious": 15,
    "anxiety": 20,
    "panic": 30,
    "panic attack": 40,

    # Depression
    "feeling down": 30,
    "hopeless": 40,
    "worthless": 40,
    "can't cope": 40,
    "overwhelmed": 30,
    "not sleeping": 20,
    "insomnia": 15,

    # Self-harm & suicidality (critical)
    "suicide": 120,
    "kill myself": 120,
    "end my life": 120,
    "self harm": 110,
    "hurt myself": 110
}

URGENT_PHRASES = [
    "suicide", "kill myself", "end my life",
    "self harm", "hurt myself", "rape", "sexual assault"
]

SENTIMENT_POS = ["good", "okay", "fine", "better", "improving", "relieved"]
SENTIMENT_NEG = ["sad", "bad", "angry", "upset", "scared", "worried", "hopeless"]


# ---------------------------------------------------------
#                        HELPERS
# ---------------------------------------------------------

def clean(text):
    return text.lower().strip()


def parse_conversation(text):
    """
    Expected format:
    [DD/MM/YYYY, HH:MM] Name: Message
    """
    pattern = re.compile(r'^\[(.*?)\]\s*(.*?):\s*(.*)$')
    messages = []

    for line in text.splitlines():
        m = pattern.match(line)
        if m:
            messages.append({
                "timestamp": m.group(1),
                "speaker": m.group(2),
                "message": m.group(3).strip()
            })
    return messages


def score_keywords(text, model):
    score = 0
    matches = Counter()
    t = clean(text)

    for phrase, pts in sorted(model.items(), key=lambda x: -len(x[0])):
        if phrase in t:
            score += pts
            matches[phrase] += 1

    return score, matches


def sentiment_score(text):
    t = clean(text)
    pos = sum(1 for w in SENTIMENT_POS if w in t)
    neg = sum(1 for w in SENTIMENT_NEG if w in t)
    return pos - neg


# ---------------------------------------------------------
#                    CORE ANALYSIS ENGINE
# ---------------------------------------------------------

def analyse(messages):
    hiv_total = 0
    mh_total = 0
    hiv_matches = Counter()
    mh_matches = Counter()
    urgent = []

    sentiment_history = []

    for m in messages:
        msg = m["message"]

        # sentiment
        sentiment_history.append(sentiment_score(msg))

        # keyword scoring
        h_s, h_m = score_keywords(msg, HIV_KEYWORDS)
        m_s, m_m = score_keywords(msg, MENTAL_KEYWORDS)

        hiv_total += h_s
        mh_total += m_s
        hiv_matches.update(h_m)
        mh_matches.update(m_m)

        # urgent flags
        for u in URGENT_PHRASES:
            if u in clean(msg):
                urgent.append({
                    "phrase": u,
                    "timestamp": m["timestamp"],
                    "speaker": m["speaker"],
                    "message": msg
                })

    # normalize to 0–100 scale
    hiv_score = min(100, int((hiv_total / 120) * 100))
    mh_score = min(100, int((mh_total / 120) * 100))

    trend = (
        "Improving" if mean(sentiment_history[-5:]) > mean(sentiment_history[:5])
        else "Worsening" if mean(sentiment_history[-5:]) < mean(sentiment_history[:5])
        else "Stable"
    )

    return {
        "scores": {"hiv": hiv_score, "mental": mh_score},
        "matches": {"hiv": hiv_matches, "mental": mh_matches},
        "urgent": urgent,
        "sentiment_trend": trend,
        "raw_sentiments": sentiment_history
    }


# ---------------------------------------------------------
#              NDoH-ALIGNED RECOMMENDATIONS
# ---------------------------------------------------------

def ndoh_recommendations(result):

    hiv = result["scores"]["hiv"]
    mental = result["scores"]["mental"]
    urgent = result["urgent"]

    recs = []

    # HIV — South African Guidelines
    if hiv >= 70:
        recs.append(
            "HIV Risk HIGH — Follow NDoH HTS & PEP guidelines: "
            "• Immediate HIV test\n"
            "• If exposure <72h: urgent PEP evaluation\n"
            "• Screen for STIs\n"
            "• Offer PrEP for ongoing risk"
        )
    elif hiv >= 35:
        recs.append(
            "HIV Risk MODERATE — Recommend HTS testing soon, partner testing, "
            "and discuss PrEP per NDoH prevention program."
        )
    else:
        recs.append(
            "HIV Risk LOW — Routine HTS testing recommended per NDoH policy."
        )

    # Mental — South Africa Mental Health Policy
    if any(u["phrase"] in ["suicide", "kill myself", "self harm", "hurt myself"] for u in urgent):
        recs.append(
            "Mental Health CRITICAL — Follow NDoH 72-hour Emergency Mental-Health policy: "
            "• Immediate safety assessment\n"
            "• Do not leave patient alone\n"
            "• Urgent psychiatric evaluation"
        )
    elif mental >= 70:
        recs.append(
            "Mental Health HIGH — Urgent referral to mental-health professional. "
            "Screen with PHQ-9/GAD-7 and initiate brief counselling."
        )
    elif mental >= 35:
        recs.append(
            "Mental Health MODERATE — Begin supportive counselling, lifestyle interventions, "
            "and schedule follow-up in 1–2 weeks."
        )
    else:
        recs.append(
            "Mental Health LOW — Mild symptoms: recommend self-care, sleep hygiene, "
            "and monitoring for escalation."
        )

    return recs


# ---------------------------------------------------------
#                      REPORT GENERATOR
# ---------------------------------------------------------

def make_report(path, analysis):
    r = []

    r.append("========== AI Conversation Risk Analysis ==========\n")
    r.append(f"Source file: {path}\n")

    # Scores
    r.append("---- Risk Scores (0–100) ----")
    r.append(f"HIV acquisition risk: {analysis['scores']['hiv']}")
    r.append(f"Mental-health risk: {analysis['scores']['mental']}")
    r.append("")

    # Sentiment Trend
    r.append(f"Sentiment trend: {analysis['sentiment_trend']}\n")

    # Matches
    r.append("---- Strongest HIV indicators detected ----")
    if analysis["matches"]["hiv"]:
        for k, v in analysis["matches"]["hiv"].items():
            r.append(f"  - {k} (x{v})")
    else:
        r.append("  none")

    r.append("\n---- Strongest Mental-Health indicators detected ----")
    if analysis["matches"]["mental"]:
        for k, v in analysis["matches"]["mental"].items():
            r.append(f"  - {k} (x{v})")
    else:
        r.append("  none")

    # Urgent
    r.append("\n---- Urgent Red Flags ----")
    if analysis["urgent"]:
        for f in analysis["urgent"]:
            r.append(f"  !!! {f['phrase']} @ {f['timestamp']} — {f['message']}")
    else:
        r.append("  none")

    # Recommendations
    r.append("\n---- NDoH-Aligned Treatment Plan ----")
    for rec in ndoh_recommendations(analysis):
        r.append(f"• {rec}")

    return "\n".join(r)


# ---------------------------------------------------------
#                           MAIN
# ---------------------------------------------------------

def main(path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    messages = parse_conversation(content)
    analysis = analyse(messages)
    report = make_report(path, analysis)

    print(report)

    json_path = path + ".analysis.json"
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(analysis, jf, indent=2)

    print(f"\nJSON results saved to: {json_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python risk_engine.py conversation.txt")
        sys.exit(1)
    main(sys.argv[1])
