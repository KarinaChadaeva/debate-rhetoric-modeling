import os
import re
import json
import time
import random
from getpass import getpass
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
import cohere


INPUT_PATH = "unlabeled_df.csv"
CHECKPOINT_PATH = "cohere_fs_annotation.csv"
MODEL = "command-a-03-2025"

MAX_WORKERS = 8
SAVE_EVERY = 100
MAX_RETRIES = 5
BASE_BACKOFF = 2.0



ANNOTATION_PROMPT = """You are an expert annotator of rhetorical strategies in American presidential and vice-presidential debate transcripts.
## Your task
Given a speech fragment, assign one primary rhetorical strategy (strategy1). Assign strategy2 only in rare cases when two strategies are equally and clearly present.
## Taxonomy

1. 'presentation' (positive framing, self-presentation) - speaker explicitly represents themselves, their party, their candidate, situation, event, policy outcome, or trend in a positive light. Positive evaluation of a situation counts as presentation only when the speaker associates the positive outcome with their own side (their party, administration, or policy).
2. 'accusation' (negative framing of others) - speaker attributes blame or wrongdoing to an opponent, or describes a situation, event, or trend negatively while attributing the cause to a specific actor (opponent, opposing party, current administration). Includes alleging specific improper actions, lies, or intentions. A neutral statement of a problem within the speaker's own program is not accusation.
3. 'self-justification' - speaker can either explain and provide context for an action, decision, or record, or deny and reject an accusation. Apply only when the speaker's main rhetorical move is to deny, explain, or contextualize a specific accusation against themselves or their record. If the speaker pivots to attacking the opponent or laying out their own positive program, use accusation or presentation instead.
4. 'appeal' - an audience-oriented rhetorical move: a direct call to the audience to act, vote, support, or adopt a position, OR a broader appeal that invokes shared values, identity, or collective responsibility ("we as a nation must...", "we need to..."), OR emotional storytelling addressed to the audience to mobilize support for a stance. The defining feature is audience orientation, not just programmatic content.
Special labels:
- '-' - moderator or question-asker utterance.
- 'no_strategy' - fragment is too short, interrupted, purely emotional, procedural, or lacks sufficient context to identify a strategy (broken-off thoughts, one-word interjections, simple acknowledgments like "thank you", "yes", "well", procedural requests like "May I respond?", "Can I have a rebuttal?").
## Decision procedure
Apply these checks in order:
1. Is this a moderator/question-asker? -> strategy1 = "-", strategy2 = null. Stop.
2. Is the fragment too short, interrupted, purely emotional, or procedural? -> strategy1 = "no_strategy", strategy2 = null. Stop.
3. Is the speaker's main rhetorical move to deny, explain, or contextualize a specific accusation against themselves or their record? -> "self-justification". (If the speaker pivots to attacking the opponent or laying out their own positive program, skip this step.)
4. Is the speaker explicitly praising a named person/party (themselves, running mate, their party), or positively evaluating a situation/outcome that they associate with their own side? -> "presentation"
5. Is the speaker explicitly blaming a named opponent, or negatively describing a situation/trend while attributing the cause to a specific actor (opponent, opposing party, current administration)? -> "accusation"
6. Is the speaker addressing the audience - calling them to act/vote/decide, invoking shared values or collective responsibility ("we as a nation must..."), or using emotional storytelling to mobilize support? -> "appeal"
## Rules for strategy2
**Default: strategy2 should be null.** Most fragments have one dominant strategy. Only assign strategy2 when both strategies are EQUALLY prominent in the fragment.
The most common legitimate dual case is 'accusation' + 'presentation': the speaker explicitly contrasts the opponent's failures with their own/their party's achievements in the same fragment, with both halves substantively developed.
Do NOT assign strategy2 when:
- One strategy is clearly dominant and the other is only briefly hinted at
- You are merely uncertain between two strategies - pick the better one for strategy1, leave strategy2 null
- The fragment expresses a single coherent rhetorical move
If only one strategy applies, set strategy2 to null.
## Output format
Return strictly valid JSON, no extra text:
{
  "strategy1": "...",
  "strategy2": "..." or null,
  "confidence": 0.0-1.0,
  "explanation": "one sentence explaining the choice"
}

## Examples
Text: "We need real results and we need them now. I've done that in my whole career and I'll do it as president."
{"strategy1": "presentation", "strategy2": null, "confidence": 0.9, "explanation": "The speaker explicitly presents themselves as a concrete actor with a track record of delivering results throughout their career."}

Text: "No, check out the deal that they signed with Judicial Watch. They agreed that that many people either voted illegally, shouldn't have been voting, a lot of things. … But they play a very dirty game."
{"strategy1": "accusation", "strategy2": null, "confidence": 0.9, "explanation": "The speaker explicitly blames a specific opponent of illegal voting practices and concludes with a direct negative characterization ('they play a very dirty game')."}

Text: "They had the slowest economic recovery since 1929. It was the slowest recovery. Also, they took over something that was down here. All you had to do is turn on the lights and you pick up a lot. But they had the slowest economic recovery since 1929."
{"strategy1": "accusation", "strategy2": null, "confidence": 0.85, "explanation": "The speaker negatively describes the economic recovery and attributes it to a specific actor ('they' = previous administration), with repeated emphasis on the negative characterization."}

Text: "Tonight, I am also asking you to join me in another fight that all Americans can get behind: the fight against childhood cancer."
{"strategy1": "appeal", "strategy2": null, "confidence": 0.9, "explanation": "The speaker directly calls on the audience ('I am asking you to join me') to take a particular action - supporting the fight against childhood cancer."}

Text: "We must be united at home to defeat our adversaries abroad."
{"strategy1": "appeal", "strategy2": null, "confidence": 0.9, "explanation": "The speaker appeals to shared values through inclusive 'we' rhetoric and frames national unity as a collective necessity."}

Text: "Those pre-existing conditions, insurance companies are going to love this. And so it's just not appropriate to do this before this election. … Number one, he knows what I proposed. What I proposed is that we expand Obamacare and we increase it. We do not wipe any. … The platform of the Democratic Party is what I, in fact, approved of, what I approved of. Now, here's the deal. The deal is that it's going to wipe out pre-existing conditions."
{"strategy1": "self-justification", "strategy2": null, "confidence": 0.85, "explanation": "The speaker defends their own healthcare policy stance by clarifying what they actually proposed and explaining their relationship to the party platform, rather than disputing that any action took place."}

Text: "He's absolutely wrong, number one. Number two, if in fact, during our administration in the recovery act, I was in charge able to bring down the cost of renewable energy to cheaper than are as cheap as coal and gas and oil. Nobody's going to build another coal fired plant in America. No one's going to build another oil fire plant in America."
{"strategy1": "self-justification", "strategy2": null, "confidence": 0.9, "explanation": "The speaker directly rejects the opponent's accusation and supports the denial by pointing to a concrete achievement during their administration."}

Text: "… we inherited the worst recession, short of a depression in American history. I was asked to bring it back. We were able to have an economic recovery that created the jobs you're talking about. We handed him a booming economy, he blew it."
{"strategy1": "accusation", "strategy2": "presentation", "confidence": 0.9, "explanation": "The speaker positively presents their own administration's economic record ('we were able to', 'we handed him a booming economy') and contrasts it with a direct accusation that the opponent ruined it ('he blew it')."}

Text: "Well, I must admit, it surprised me tonight. We're seeing all over this nation, all cities and all parts of the country, indeed across the world, an outpouring of joy, of hope, renewed faith in tomorrow to bring a better day."
{"strategy1": "presentation", "strategy2": null, "confidence": 0.65, "explanation": "The speaker positively evaluates the current situation by describing widespread joy, hope, and renewed faith without attributing this positive state to any specific actor."}

Text: "I appreciate that."
{"strategy1": "no_strategy", "strategy2": null, "confidence": 0.9, "explanation": "Short acknowledgment of an opponent's remark with no rhetorical content."}

"""

ANNOTATION_SCHEMA = {
    "type": "object",
    "required": ["strategy1", "confidence", "explanation"],
    "properties": {
        "strategy1": {
            "type": "string",
            "enum": ["presentation", "accusation", "self-justification", "appeal", "-", "no_strategy"]
        },
        "strategy2": {
            "anyOf": [
                {
                    "type": "string",
                    "enum": ["presentation", "accusation", "self-justification", "appeal", "no_strategy"]
                },
                {"type": "null"}
            ]
        },
        "confidence": {"type": "number"},
        "explanation": {"type": "string"}
    }
}



def extract_json(raw_text):
    if raw_text is None:
        return {
            "strategy1": None,
            "strategy2": None,
            "confidence": None,
            "explanation": "Empty response"
        }

    raw_text = str(raw_text).strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {
        "strategy1": None,
        "strategy2": None,
        "confidence": None,
        "explanation": f"Could not parse JSON: {raw_text[:300]}"
    }


def annotate_cohere(client, text, model=MODEL):
    """One API call. Raises on failure - retries handled by caller."""
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": ANNOTATION_PROMPT},
            {"role": "user", "content": f"Annotate this fragment and return a JSON object:\n\n{text}"}
        ],
        temperature=0,
        max_tokens=300,
        response_format={
            "type": "json_object",
            "schema": ANNOTATION_SCHEMA
        }
    )
    raw_text = response.message.content[0].text
    return extract_json(raw_text)


def annotate_with_retry(client, text):
    """Wraps annotate_cohere with exponential backoff on transient errors."""
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return annotate_cohere(client, text), None
        except Exception as e:
            last_err = e
            msg = str(e).lower()
            is_transient = any(s in msg for s in ["429", "rate", "timeout", "503", "502", "504", "connection"])
            if not is_transient and attempt > 0:
                break
            sleep_for = BASE_BACKOFF * (2 ** attempt) + random.uniform(0, 1)
            time.sleep(sleep_for)
    return None, str(last_err)



def main():
    api_key = os.environ.get("COHERE_API_KEY") or getpass("Cohere API key: ").strip()
    client = cohere.ClientV2(api_key=api_key, log_warning_experimental_features=False)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Found checkpoint at {CHECKPOINT_PATH}, resuming.")
        annotated_df = pd.read_csv(CHECKPOINT_PATH, sep=";")
        done = annotated_df["cohere_strategy1"].notna().sum()
        print(f"Already annotated: {done} rows")
    else:
        print(f"No checkpoint found. Loading fresh from {INPUT_PATH}.")
        unlabeled_df = pd.read_csv(INPUT_PATH, sep=";")
        annotated_df = unlabeled_df.copy()
        for col in ["cohere_strategy1", "cohere_strategy2", "cohere_confidence",
                    "cohere_explanation", "cohere_error"]:
            annotated_df[col] = None

        mod_mask = annotated_df["strategy1"] == "-"
        annotated_df.loc[mod_mask, "cohere_strategy1"] = "-"
        print(f"Total rows: {len(annotated_df)}")
        print(f"Marked {mod_mask.sum()} moderator rows as '-' instantly.")

    # Rows that still need work: no strategy1 yet, OR previous run errored out
    needs_work = annotated_df["cohere_strategy1"].isna()
    rows_to_annotate = annotated_df[needs_work].index.tolist()
    print(f"Rows to annotate via Cohere API: {len(rows_to_annotate)}")

    if not rows_to_annotate:
        print("Nothing to do. All rows already annotated.")
        return

    # Worker function for the pool
    def process_one(idx):
        text = annotated_df.at[idx, "text"]
        result, err = annotate_with_retry(client, text)
        return idx, result, err

    completed_since_save = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_one, idx): idx for idx in rows_to_annotate}

        pbar = tqdm(total=len(futures), desc="Cohere zs")
        try:
            for future in as_completed(futures):
                idx, result, err = future.result()
                if result is not None:
                    annotated_df.at[idx, "cohere_strategy1"] = result.get("strategy1")
                    annotated_df.at[idx, "cohere_strategy2"] = result.get("strategy2")
                    annotated_df.at[idx, "cohere_confidence"] = result.get("confidence")
                    annotated_df.at[idx, "cohere_explanation"] = result.get("explanation")
                    annotated_df.at[idx, "cohere_error"] = None
                else:
                    annotated_df.at[idx, "cohere_error"] = err

                completed_since_save += 1
                pbar.update(1)

                if completed_since_save >= SAVE_EVERY:
                    annotated_df.to_csv(CHECKPOINT_PATH, sep=";", index=False)
                    completed_since_save = 0
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress...")
        finally:
            pbar.close()
            annotated_df.to_csv(CHECKPOINT_PATH, sep=";", index=False)

    # Final summary
    print(f"\nDone. Final size: {len(annotated_df)} rows")
    print(f"Errors: {annotated_df['cohere_error'].notna().sum()}")
    print(f"Rows still without cohere_strategy1: {annotated_df['cohere_strategy1'].isna().sum()}")
    print(f"Saved to {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
