from pathlib import Path
import sys

def find_dirs_with_keyword(root: str, keyword: str = "MRA", case_sensitive: bool = False):
    root_path = Path(root)

    if not root_path.exists():
        raise FileNotFoundError(f"Root path not found: {root_path}")

    key = keyword if case_sensitive else keyword.lower()

    for p in root_path.rglob("*"):
        if p.is_dir():
            name = p.name
            hay = name if case_sensitive else name.lower()
            if key in hay:
                print(str(p.resolve()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python find_mra_dirs.py <root_folder> [keyword]")
        sys.exit(1)

    root = sys.argv[1]
    keyword = sys.argv[2] if len(sys.argv) >= 3 else "MRA"

    # 預設不分大小寫；要分大小寫就把 case_sensitive 改 True
    find_dirs_with_keyword(root, keyword, case_sensitive=False)
