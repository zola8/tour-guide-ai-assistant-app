import subprocess
import sys

if __name__ == '__main__':
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "a03_streamlit.py"],
        check=True,
    )
