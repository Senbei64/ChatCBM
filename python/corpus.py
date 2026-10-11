#- ChatCBM - Copyright 2026 Fabio Carignano ----------------------------------
"""Generate the training corpus: a small, controlled world of animals.

    python python/corpus.py            # writes data/corpus.txt

Few animals, few facts, few words: questions and answers, small talk and short
stories. Sentences are compact so that a whole turn fits the 64-token context.
Each line is one chat turn:

    <user> question<bot> answer<eos>

facts() lists every question with a single correct answer, by category: the
compiler (compile.py) uses it to count exact answers. Correct stories are built
from meetings(), endings() and story_text().
"""
import random
from itertools import product
from pathlib import Path

# name, color, home, food, sound, action, size (1 = smallest), legs
ANIMALS = [
    ("cat", "black", "in the house", "fish", "meows", "sleeps on the bed", 9, 4),
    ("dog", "brown", "in the garden", "the bone", "barks", "runs in the meadow", 11, 4),
    ("mouse", "grey", "in the burrow", "cheese", "squeaks", "runs from the cat", 3, 4),
    ("wolf", "grey", "in the forest", "meat", "howls", "looks at the moon", 13, 4),
    ("fish", "red", "in the sea", "seaweed", "makes no sound", "swims in the sea", 4, 0),
    ("cow", "white", "in the barn", "grass", "moos", "makes milk", 15, 4),
    ("frog", "green", "in the pond", "flies", "croaks", "jumps in the pond", 2, 4),
    ("sheep", "white", "in the meadow", "grass", "bleats", "sleeps in the meadow", 12, 4),
    ("hen", "brown", "in the henhouse", "grain", "clucks", "lays eggs", 6, 2),
    ("fox", "red", "in the forest", "hens", "yelps", "hunts at night", 10, 4),
    ("horse", "brown", "in the stable", "hay", "neighs", "pulls the cart", 16, 4),
    ("pig", "pink", "in the barn", "apples", "oinks", "rolls in the mud", 14, 4),
    ("duck", "yellow", "in the pond", "bread", "quacks", "swims in the pond", 7, 2),
    ("owl", "brown", "in the forest", "mice", "hoots", "flies at night", 8, 2),
    ("bee", "yellow", "in the hive", "flowers", "buzzes", "makes honey", 1, 6),
    ("rabbit", "white", "in the burrow", "carrots", "makes no sound", "jumps in the meadow", 5, 4),
]
BIG = 10                        # size from which an animal is "big"
LEGS = {0: "no legs", 2: "two legs", 4: "four legs", 6: "six legs"}

SMALL_TALK = [
    (["hi", "hi!", "good morning", "hello", "hey"],
     ["hi! how are you?", "hi! how can i help you?"]),
    (["how are you?", "how is it going?", "all good?"],
     ["i am fine, thanks! and you?", "fine, thanks!"]),
    (["i am fine", "fine", "fine, thanks", "all good"],
     ["how nice! i am happy."]),
    (["who are you?", "what is your name?", "who are you then?"],
     ["i am chatcbm, a small language model."]),
    (["what can you do?", "what do you know?", "help"],
     ["i can talk about animals and tell stories."]),
    (["which animals do you know?", "what animals do you know?"],
     ["i know the cat, the dog, the cow, the frog and many others."]),
    (["thanks", "thank you very much"],
     ["you are welcome!", "no problem!"]),
    (["bye bye", "goodbye", "see you soon"],
     ["bye! see you soon!"]),
]


class Animal:
    """An animal and its facts."""

    def __init__(self, row):
        (self.name, self.color, self.home, self.food, self.sound, self.action,
         self.size, self.legs) = row
        self.the = "the " + self.name
        self.big = "big" if self.size >= BIG else "small"


def listing(names):
    """["the cat", "the dog", "the cow"] -> "the cat, the dog and the cow"."""
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def questions(a):
    """Questions about one animal: (category, question, answer)."""
    v = dict(the=a.the, color=a.color, home=a.home, food=a.food, sound=a.sound, action=a.action,
             big=a.big, legs=LEGS[a.legs])
    templates = [
        ("facts", ["what color is {the}?", "which color is {the}?", "{the} is what color?"], "{the} is {color}."),
        ("facts", ["where does {the} live?", "where is the home of {the}?", "{the} lives where?"],
         "{the} lives {home}."),
        ("facts", ["what does {the} eat?", "what food does {the} eat?", "{the} eats what?"], "{the} eats {food}."),
        ("facts", ["what does {the} say?", "what sound does {the} make?"], "{the} {sound}."),
        ("facts", ["what does {the} do?", "what is {the} doing?"], "{the} {action}."),
        ("facts", ["tell me about {the}", "what do you know about {the}?", "say something about {the}"],
         "{the} is {color}. it lives {home} and eats {food}."),
        ("size", ["is {the} big or small?", "how big is {the}?"], "{the} is {big}."),
        ("legs", ["how many legs does {the} have?", "how many legs has {the}?"], "{the} has {legs}."),
    ]
    for category, qs, answer in templates:
        for q in qs:
            yield category, q.format(**v), answer.format(**v)
    for c in sorted({r[1] for r in ANIMALS}):
        yield "yes/no", f"is {a.the} {c}?", (f"yes, {a.the} is {c}." if c == a.color
                                              else f"no, {a.the} is {a.color}.")
    for f in sorted({r[3] for r in ANIMALS}):
        yield "yes/no", f"does {a.the} eat {f}?", (f"yes, {a.the} eats {f}." if f == a.food
                                                    else f"no, {a.the} eats {a.food}.")


def facts():
    """Every question with a single correct answer: (category, question, answer)."""
    animals = [Animal(r) for r in ANIMALS]
    out = [f for a in animals for f in questions(a)]
    for a, b in product(animals, animals):           # size comparisons
        if a.name != b.name:
            big, small = (a, b) if a.size > b.size else (b, a)
            out.append(("comparisons", f"which is bigger, {a.the} or {b.the}?",
                        f"{big.the} is bigger than {small.the}."))
            out.append(("comparisons", f"which is smaller, {a.the} or {b.the}?",
                        f"{small.the} is smaller than {big.the}."))
    for key, question, verb in ((lambda a: a.home, "which animals live {}?", ("lives", "live")),
                                (lambda a: a.color, "which animals are {}?", ("is", "are")),
                                (lambda a: LEGS[a.legs], "which animals have {}?", ("has", "have"))):
        for k in sorted({key(a) for a in animals}):
            names = [a.the for a in animals if key(a) == k]
            out.append(("lists", question.format(k), f"{listing(names)} {verb[len(names) > 1]} {k}."))
    return out


def meetings(a, b):
    return [f"{b.the} said: hi, want to play?",
            f"{b.the} was sad and {a.the} helped it.",
            f"{b.the} was hungry and {a.the} gave it {a.food}.",
            f"{b.the} was lost and {a.the} showed it the way.",
            f"{b.the} was cold and {a.the} gave it a warm bed."]


def endings(a, b):
    return [f"{a.the} and {b.the} became friends.",
            f"they played {b.home}.",
            f"they ate {a.food}.",
            "they sang together."]


def story_text(a, b, meeting, ending):
    return (f"once a {a.color} {a.name} lived {a.home}. one day it met {b.the}. "
            f"{meeting} {ending} then {a.the} went home, happy.")


def story(a, b, rng):
    return story_text(a, b, rng.choice(meetings(a, b)), rng.choice(endings(a, b)))


def story_request(a, b=None):
    return f"tell me a story about {a.the}" + (f" and {b.the}" if b else "")


def turn(question, answer):
    return f"<user> {question}<bot> {answer}<eos>"


def generate(seed=0, fact_repeats=4, n_stories=4000):
    rng = random.Random(seed)
    animals = [Animal(r) for r in ANIMALS]
    lines = []
    every = facts()
    # yes/no questions answered "yes" are few (1 color in 8, 1 food in 14): repeat them as
    # often as the "no" ones, otherwise the model learns to always answer "no"
    n_no = sum(r.startswith("no,") for _, _, r in every)
    n_yes = sum(r.startswith("yes,") for _, _, r in every)
    weighted = [(q, r) for _, q, r in every for _ in range(n_no // n_yes if r.startswith("yes,") else 1)]
    for _ in range(fact_repeats):
        lines += [turn(q, r) for q, r in weighted]
        for qs, rs in SMALL_TALK:
            lines += [turn(q, rng.choice(rs)) for q in qs]
    for _ in range(n_stories):
        a, b = rng.sample(animals, 2)
        request = rng.choice(["tell me a story", story_request(a), story_request(a, b),
                              f"a story about {a.the}", f"once a {a.name}"])
        lines.append(turn(request, story(a, b, rng)))
    rng.shuffle(lines)
    return lines


def main():
    lines = generate()
    out = Path(__file__).resolve().parents[1] / "data" / "corpus.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    words = {w for line in lines for w in line.replace("<", " <").split()}
    print(f"{len(lines)} lines, {sum(len(line) for line in lines)} characters, "
          f"~{len(words)} distinct words -> {out}")


if __name__ == "__main__":
    main()

#-----------------------------------------------------------------------------
