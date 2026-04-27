"""
Legacy entry point — now delegates to the XAI + recommendations app.

Previously this file only rendered templates/result.html (prediction only).
If you run `python app.py` you now get the same behavior as `python app_xai.py`:
SHAP explanations and personalized recommendations on result_xai.html.

Run: python app.py
"""

from app_xai import app

if __name__ == "__main__":
    app.run(debug=True)
