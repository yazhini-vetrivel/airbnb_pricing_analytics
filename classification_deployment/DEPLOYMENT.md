# Deploying the app (bonus: +1 GUI, +1 public deployment)

The app is `app/app.py` (Streamlit). It loads `models/final_model.joblib`, which the last cells of
`notebooks/classification.ipynb` create. Streamlit Community Cloud is free and deploys straight from GitHub.

## 1. Run it locally first (this is the GUI bonus)

```bash
# from the repo root
pip install -r app/requirements.txt
streamlit run app/app.py
```

A browser tab opens at `http://localhost:8501`. Pick a typical reading from the dropdown, or type your own values.
If this works on your laptop, the deployment will work.

## 2. Make the model file match your environment

A pickled scikit-learn model only loads reliably with the **same scikit-learn version** that trained it.

1. Re-run the notebook end to end on your own machine. The last cell rewrites `models/final_model.joblib`
   and prints the line to pin, e.g. `scikit-learn==1.8.0`.
2. Put the versions of your environment into `app/requirements.txt`:
   ```bash
   pip freeze | findstr /i "scikit-learn numpy pandas joblib"      # Windows
   pip freeze | grep -iE "scikit-learn|numpy|pandas|joblib"        # Mac / Linux
   ```
3. Check the model file is under GitHub's 100 MB limit: `dir models` (Windows) / `ls -lh models` (Mac / Linux).

## 3. Push to GitHub

The repo must be reachable by Streamlit (a public repo is simplest).

```bash
git add app/ models/final_model.joblib requirements.txt .streamlit/ DEPLOYMENT.md
git commit -m "Add Streamlit app and final model for deployment"
git push origin main
```

## 4. Deploy on Streamlit Community Cloud

1. Go to **share.streamlit.io** and sign in with GitHub (authorise access to the repo).
2. Click **Create app** and choose to deploy from an existing GitHub repo.
3. Fill in **Repository**, **Branch** (`main`) and **Main file path**: `app/app.py`.
4. Open **Advanced settings** and choose a **Python version** (3.12 or 3.13 both work with scikit-learn 1.8).
   The Python version is set here, not in a `runtime.txt` file (Community Cloud ignores that file).
5. Click **Deploy**. The first build takes a few minutes. You get a public URL like
   `https://<your-subdomain>.streamlit.app`.

Streamlit looks for `requirements.txt` in the folder of the main file first (`app/requirements.txt`) and only then in the
repo root, so the slim app file is the one used for the build. The big root `requirements.txt` stays for the notebooks.

## 5. Evidence for the marks

- Paste the public URL at the top of `README.md`, and add one screenshot of the app.
- Open the URL once shortly before the review: free apps go to sleep after a period of inactivity and need a moment to wake.
- Be ready to demo: choose the **Toaster** preset (100% confident), then the **Iron** preset, and point at the "looks
  identical" message. That message is the project's main finding, shown live. The app does not ask for the time of day on
  purpose: the notebook (PB10) shows the recording hour is an artefact of the lab's schedule, so the deployed model uses
  electrical features only.
- Enter the current with all three decimals. TV and Desktop Computer differ only in the third decimal of the current:
  rounding it to one decimal drops the model's accuracy on TV rows from 100% to about 32%.

## If something goes wrong

| Symptom | Fix |
|---|---|
| `InconsistentVersionWarning` or an unpickling error on startup | scikit-learn / numpy in `app/requirements.txt` differ from the training environment. Re-pin (step 2) and redeploy. |
| Build fails while installing packages | Try Python 3.12 in Advanced settings; remove everything not needed from `app/requirements.txt`. |
| `FileNotFoundError: final_model.joblib` | The model file was not pushed (check `.gitignore`) or the folder layout changed. The app expects `models/` next to `app/`. |
| Push rejected, file too large | The model is over 100 MB. Re-save with `compress=3` (already used) or choose a smaller final model. |
