"""Download the Abilene traffic matrices used by the demo.

Source: Yin Zhang's Abilene traffic matrices, derived from Internet2 NetFlow data
(https://www.cs.utexas.edu/~yzhang/research/AbileneTM/). The demo needs six consecutive
weeks (X10 to X15) plus the demand list. About 45 MB in total.
"""
from pathlib import Path
from urllib.request import urlretrieve

BASE = "https://www.cs.utexas.edu/~yzhang/research/AbileneTM/"
FILES = [f"X{i:02d}.gz" for i in range(10, 16)] + ["demands", "links", "readme.txt"]


def main(target="data/raw"):
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = target / name
        if path.exists():
            print(f"{name}: already there")
            continue
        print(f"{name}: downloading")
        urlretrieve(BASE + name, path)
    print("done")


if __name__ == "__main__":
    main()
