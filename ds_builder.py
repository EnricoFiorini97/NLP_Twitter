import sys
import pandas as pd

def main() -> None:
        try:
                with open(sys.argv[1], "r") as _:
                        pass
                if not sys.argv[1].endswith(".txt"):
                        raise Exception
        except Exception:
                print("You must provide a valid path to a .txt file containing your raw data! [syntax -> python this_script.py <file_path.txt>")
                return




if __name__ == "__main__":
        main() 