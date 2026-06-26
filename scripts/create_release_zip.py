import subprocess
import zipfile
from pathlib import Path

# Repository root (current working directory)
REPO_ROOT = Path.cwd()

# Create zip one directory above the repository
OUTPUT = REPO_ROOT.parent / "MultiPoint_AD.zip"

files = subprocess.check_output(
    ["git", "ls-files", "-co", "--exclude-standard"],
    text=True
).splitlines()

with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in files:
        p = REPO_ROOT / f
        if p.exists():
            # Preserve the repository folder structure inside the zip
            z.write(p, arcname=f)

print(f"✓ Created: {OUTPUT}")
print(f"✓ Added {len(files)} files")