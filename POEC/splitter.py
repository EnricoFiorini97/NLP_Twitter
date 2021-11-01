#Tool for dataset creation and splitting

import sys
import os

def main() -> None:
    target_path, destination_folder = "", ""

    try:
        target_path = sys.argv[1]
        destination_folder = sys.argv[2]
        if not os.path.exists(target_path) or not os.path.isfile(target_path) or not os.path.exists(destination_folder) or not os.path.isdir(destination_folder):
            raise ValueError
    except (IndexError, FileNotFoundError, ValueError):
        print("You must provide a valid target file/destination folder!")
        return 

    try:
        with open(target_path, "r") as f:
            for idx, line in enumerate(f):
                with open(f"{destination_folder}{os.path.sep}{str(idx + 1)}.txt", "w") as tmp:
                    tmp.write(line.strip())
    except IOError:
        print("Your target path seems to be corrupted or you haven't permissions!")
        return 

if __name__ == "__main__":
    main()

    
