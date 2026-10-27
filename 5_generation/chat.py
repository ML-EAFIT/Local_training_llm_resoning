"""Interactive chat / completion with any checkpoint.

python 5_generation/chat.py --ckpt work/7_post_training/final.pt          (chat template)
python 5_generation/chat.py --ckpt work/4_pre_train/final.pt --raw        (base model completion)
Commands: /reset  /temp 0.7  /quit
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common.data import Tok  # noqa: E402
from common.model import load_model  # noqa: E402
from engine import Engine  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--raw", action="store_true")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max_new", type=int, default=256)
    a = ap.parse_args()
    tok = Tok()
    model, _ = load_model(a.ckpt)
    model.half().eval()
    eng = Engine(model, tok)
    history, temp = [], a.temp
    print("type /quit to exit, /reset to clear history, /temp X to change temperature")
    while True:
        try:
            msg = input("\nyou> ").strip()
        except EOFError:
            break
        if msg == "/quit":
            break
        if msg == "/reset":
            history = []
            continue
        if msg.startswith("/temp"):
            temp = float(msg.split()[1])
            continue
        if a.raw:
            ids = [tok.bos] + tok.encode(msg)
            print(msg + tok.decode(eng.generate(ids, a.max_new, temperature=temp, top_p=0.9)[0]))
            continue
        reply = eng.chat(msg, history=history, max_new=a.max_new, temperature=temp, top_p=0.9)
        print("model>", reply)
        history += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()
