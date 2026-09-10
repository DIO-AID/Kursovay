import sys
from pathlib import Path
import streamlit as st

APP_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = APP_DIR.parent

sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from ui import render_ui


def main():
    st.set_page_config(
        page_title="ML Auto Pipeline",
        layout="wide"
    )

    render_ui()


if __name__ == "__main__":
    main()