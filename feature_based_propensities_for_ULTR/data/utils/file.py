import hashlib
import shutil
from pathlib import Path



def verify_file(path: Path, checksum: str) -> bool:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    return True


def unarchive(in_path: Path, out_path: Path) -> Path:
    if not out_path.exists():
        print(f"Unpack archived dataset to: {out_path}")
        shutil.unpack_archive(in_path, out_path)

    assert out_path.exists()
    return out_path
