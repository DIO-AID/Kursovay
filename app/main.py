import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.append(str(PROJECT_ROOT.parent))

from ui import render_ui


def main():
    st.set_page_config(
        page_title="ML Auto Pipeline",
        layout="wide"
    )

    render_ui()


if __name__ == "__main__":
    main()