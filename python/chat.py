#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Chat with the trained model.

    python python/chat.py                           # conversation
    python python/chat.py -p "who are you?"         # a single question
    python python/chat.py --float                   # float32 reference model

By default it runs the all-integer engine (models/model_int.npz), the same arithmetic as
the 6502 version. Each question is a turn of its own: there is no conversation history.
"""
import argparse
import sys

import numpy as np

from chatcbm import intengine, model
from chatcbm.generate import chat_prompt, reply
from chatcbm.tokenizer import BOT, EOS, PAD, UNK, USER, Tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--prompt", help="ask a single question and exit")
    ap.add_argument("--models", default="models")
    ap.add_argument("--float", action="store_true", help="use the float32 model instead of the integer engine")
    ap.add_argument("-t", "--temperature", type=float, default=0.6, help="0 = always the most likely token")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    tok = Tokenizer.load(f"{args.models}/tokenizer.json")
    rng = np.random.default_rng(args.seed)
    opts = dict(temperature=args.temperature, top_p=args.top_p, max_new_tokens=args.max_tokens)
    if args.float:
        p, cfg = model.load(f"{args.models}/model.npz")
        engine = f"float32, {model.count_params(p)} parameters"
    else:
        m = intengine.IntModel(intengine.load(f"{args.models}/model_int.npz"))
        engine = f"integer engine, {sum(intengine.image_bytes(m.img).values())} bytes"

    def pieces(text):
        if args.float:
            yield from reply(p, cfg, tok, text, rng, **opts)
            return
        t = opts["temperature"]
        sampler = intengine.IntSampler(t, opts["top_p"], seed=int(rng.integers(1, 2 ** 31))) if t > 0 else None
        for nxt in m.generate(chat_prompt(tok, text), opts["max_new_tokens"], (tok.id(EOS), tok.id(USER)),
                              (tok.id(PAD), tok.id(UNK), tok.id(BOT)), sampler):
            yield tok.decode([nxt])

    def answer(text):
        print("bot> ", end="", flush=True)
        for i, piece in enumerate(pieces(text)):
            piece = piece.lstrip() if i == 0 else piece
            print(piece, end="", flush=True)  # streaming, one token at a time
        print()

    if args.prompt:
        answer(args.prompt)
        return

    print(f"ChatCBM ({engine}). Type /quit to leave, /temp 0.8 to change the temperature.")
    while True:
        try:
            text = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text.startswith("/temp"):
            opts["temperature"] = float(text.split()[1])
            print(f"temperature = {opts['temperature']}")
            continue
        answer(text)


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------
