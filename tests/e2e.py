#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""End-to-end test: the 6502 program answers exactly as the Python integer engine.

    python tests/e2e.py sim65           # build/sim65/chatcbm.prg in sim65
    python tests/e2e.py c64             # build/c64/chatcbm-test.prg in VICE (x64sc in PATH)

Writes the questions to build/e2e/<target>/questions and the expected answers to
build/e2e/<target>/expected: for every question the greedy answer and the answer sampled
with seed 2026 (temperature 0.6, top-p 0.9). The program writes its answers to answers:

- sim65 runs the program with the two file names as arguments;
- VICE maps disk drive 8 to the directory (host file system, no true drive emulation);
  the test build of the C64 program opens QUESTIONS and ANSWERS (lower case on the host)
  with the kernal file routines, then executes a JAM instruction: with -jamaction 5 VICE
  quits.

Questions do not go through the VICE keyboard buffer (-keybuf): VICE writes it into
memory while the program is computing with the kernal zero page swapped out.
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from chatcbm import intengine  # noqa: E402
from chatcbm.generate import chat_prompt  # noqa: E402
from chatcbm.tokenizer import BOT, EOS, PAD, UNK, USER, Tokenizer  # noqa: E402

SEED, MAX_NEW = 2026, 80
QUESTIONS = [
    "what color is the frog?",
    "hi",
    "which is bigger, the fox or the owl?",
    "does the bee eat flowers?",
    "which animals live in the forest?",
    "tell me a story about the rabbit and the owl",
    "  How many LEGS has the duck?!",
    "who are you?",
]


def expected_answers(models, questions):
    tok = Tokenizer.load(models / "tokenizer.json")
    m = intengine.IntModel(intengine.load(models / "model_int.npz"))
    stop, banned = (tok.id(EOS), tok.id(USER)), (tok.id(PAD), tok.id(UNK), tok.id(BOT))
    lines = []
    for q in questions:
        for sampler in (None, intengine.IntSampler(0.6, 0.9, SEED)):
            out = m.generate(chat_prompt(tok, q), MAX_NEW, stop, banned, sampler)
            # as chat.py and the 6502: the leading spaces of the first token are not printed
            lines.append("".join(tok.decode([t]).lstrip() if i == 0 else tok.decode([t])
                                 for i, t in enumerate(out)))
    return "\n".join(lines) + "\n"


def run_sim65(work):
    prg = ROOT / "build" / "sim65" / "chatcbm.prg"
    subprocess.run(["sim65", str(prg), str(work / "questions"), str(work / "answers")], check=True)


def run_vice(work, vice, screenshot):
    prg = ROOT / "build" / "c64" / "chatcbm-test.prg"
    cmd = [vice, "-default", "+confirmonexit", "-warp", "+sound", "-jamaction", "5",
           "-limitcycles", "2000000000", "+drive8truedrive", "-busdevice8", "-fs8", str(work),
           "-autostartprgmode", "1"]
    if screenshot:
        cmd += ["-exitscreenshot", str(work / "screen.png")]
    subprocess.run(cmd + ["-autostart", str(prg)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", choices=("sim65", "c64"))
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--vice", default="x64sc", help="VICE C64 emulator")
    ap.add_argument("--screenshot", action="store_true", help="save the final C64 screen (screen.png)")
    ap.add_argument("-q", "--question", action="append", help="ask this instead of the test questions")
    args = ap.parse_args()

    work = ROOT / "build" / "e2e" / args.target
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    questions = args.question or QUESTIONS
    (work / "questions").write_text("\n".join(questions) + "\n", encoding="ascii", newline="\n")
    expected = expected_answers(Path(args.models), questions)
    (work / "expected").write_text(expected, encoding="ascii", newline="\n")

    if args.target == "sim65":
        run_sim65(work)
    else:
        run_vice(work, args.vice, args.screenshot)
    answers = work / "answers"
    if not answers.exists():
        sys.exit(f"{args.target}: no answers ({answers})")
    got = answers.read_text(encoding="ascii")
    if got != expected:
        print(f"{args.target}: answers differ from the Python engine")
        for i, (a, b) in enumerate(zip(got.splitlines(), expected.splitlines())):
            if a != b:
                print(f"  line {i + 1}:\n    got      {a}\n    expected {b}")
        sys.exit(1)
    print(f"{args.target}: {len(expected.splitlines())} answers identical to the Python engine")


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------
