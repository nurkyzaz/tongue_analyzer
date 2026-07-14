"""Retrieval-quality check for the RAG corpus: for each query, does a relevant chunk appear in the top-k?
Keeps corpus growth honest (add cards -> rerun -> retrieval shouldn't regress)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "stage2_interpretation"))
from rag import Retriever

# query -> substrings, ANY of which appearing in a retrieved chunk id counts as a hit
CASES = [
    ("red tongue with greasy yellow coating, bitter mouth", ["damp_heat", "damp-heat"]),
    ("pale swollen wet tongue, cold hands", ["yang_deficiency", "yang-deficiency"]),
    ("pale tongue with tooth marks, tired and breathless", ["qi_deficiency", "qi_vs_blood", "spleen_qi"]),
    ("red tongue, peeled coating, cracks, dry mouth", ["yin_deficiency", "yin_vs", "yin_deficiency_vs"]),
    ("purple dusky tongue, dark lips, bruising", ["blood_stasis"]),
    ("tongue tip much redder than the rest, poor sleep", ["red_tip", "red tip"]),
    ("thin pale tongue, dizzy, palpitations", ["blood_deficiency", "qi_vs_blood"]),
    ("oily skin, acne, sticky stools, humid weather", ["damp_heat"]),
    ("bloating, worry, overthinking, mood swings", ["qi_stagnation"]),
    ("thick greasy white coating, heavy limbs, overweight", ["phlegm_dampness", "phlegm-damp"]),
    ("cold hands and feet, worse after cold food", ["yang", "cold_limbs"]),
    ("night sweats, warm palms, thirsty for cold drinks", ["yin", "night_sweats"]),
    ("bloating after meals, loose stools, poor appetite", ["spleen_qi", "bloating_loose", "qi_deficiency"]),
    ("foggy head, phlegm, sticky mouth, puffiness", ["phlegm", "phlegm_heaviness"]),
    ("fixed stabbing pain, purple tongue, dark veins", ["blood_stasis"]),
    ("coating turning yellow, feeling hot and thirsty", ["yellow_coating", "damp_heat", "white_vs_yellow"]),
    ("tongue coating wiped off easily, glossy patch", ["rooted", "peeled", "rootless"]),
    ("red spots on the tip of the tongue", ["red_dots", "red tip", "red_tip"]),
    ("stress makes my bloating and digestion worse", ["liver_qi", "qi_stagnation", "spleen"]),
    ("cold lower back, early morning loose stool, puffy", ["spleen_kidney_yang", "yang"]),
    ("what foods help a cold constitution", ["warming", "yang"]),
    ("what to eat for damp heat and oily skin", ["cooling", "damp_heat", "damp"]),
    ("red tongue, thick yellow coat, thirst, constipation", ["excess_vs_deficient", "damp_heat", "yellow"]),
    ("pale swollen tongue with greasy coating, tired", ["spleen_qi_with_dampness", "phlegm", "spleen"]),
]


def main():
    r = Retriever()
    if not r.ok:
        print("no index — run: python stage2_interpretation/rag.py --build"); return
    hits = 0
    for q, keys in CASES:
        got = r.retrieve(q, k=4)
        ids = [h["id"] for h in got]
        ok = any(any(k in i for i in ids) for k in keys)
        hits += ok
        print(f"[{'HIT ' if ok else 'MISS'}] {q[:44]:44} -> {ids[:3]}")
    print(f"\nretrieval hit@4: {hits}/{len(CASES)} = {hits/len(CASES):.0%}")


if __name__ == "__main__":
    main()
