# Publishing the project on GitHub
```bash
cd PhishGuard-XAI
git init
printf ".venv/\n__pycache__/\ndata/*.csv\n" > .gitignore
git add . && git commit -m "Initial commit: PhishGuard-XAI"
# create an empty repo named PhishGuard-XAI on github.com, then:
git branch -M main
git remote add origin https://github.com/<your-username>/PhishGuard-XAI.git
git push -u origin main
```
Add a LICENSE (MIT/Apache-2.0), then archive a release on Zenodo to get a DOI and cite it in the paper's "Code availability" statement (Q1 journals expect this).
