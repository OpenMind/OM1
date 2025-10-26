## Contributing to OM1

We welcome contributions from the community!  OM1 is an open-source project, and we appreciate your help in making it better.  Whether you're fixing bugs, adding features, improving documentation, or suggesting new ideas, your contributions are valuable.

Before contributing, please take a moment to read through the following guidelines. This helps streamline the process and ensures everyone is on the same page.

**Ways to Contribute:**
# Contributing to OM1

Thanks for helping! This guide shows how to set up a development environment, run tests, and submit PRs.

## 1. Code of conduct
Please read `CODE_OF_CONDUCT.md` and be respectful.

## 2. Quick dev setup (Linux/macOS recommended)
1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/<your-username>/OM1.git
   cd OM1
Create a Python virtualenv and activate it:
python -m venv .venv
source .venv/bin/activate
pip install -U pip

Install the project dependencies in editable mode:
pip install -e .

Run linters & formatters (pre-commit)
pip install pre-commit
pre-commit install
pre-commit run --all-files

3. Branching & commit style
   Create a feature branch: git checkout -b feat/<short-descr> or fix/<short-descr>.

Keep commits small and focused.

Commit message format:
<scope>(<subsystem>): <short summary>

Longer description (optional). Reference issues using Fixes #<issue>.

Tests & CI
Run unit tests locally:
uv run pytest

Opening a PR

Rebase or merge main branch first:
git fetch upstream
git rebase upstream/main
Push and open a PR against main.

Add GitHub Actions CI (.github/workflows/ci.yml) — minimal example
name: CI

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python: [3.10, 3.11]
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python }}
      - name: Install deps
        run: |
          python -m venv .venv
          . .venv/bin/activate
          pip install -U pip
          pip install -e .
          pip install pytest pre-commit
      - name: Run linters (pre-commit)
        run: |
          pre-commit run --all-files || true
      - name: Run tests
        run: |
          uv run pytest -q


Good first contributions

Look for good first issue or documentation labels. Small wins:

Fix README typos

Improve docs for an example config

Add a unit test for a small utility function

Where to ask for help

GitHub Issues / Discussions

Discord: <link>

Mention maintainers in PR if you need a review

Use the PR template (.github/PULL_REQUEST_TEMPLATE.md) and add tests/screenshots where relevant.

### Commit message suggestion
docs(contributing): add step-by-step dev setup, commit style, tests instructions

---

# 3) Issues / labels / templates — improve triage & new contributor visibility

## Problems
- Issues may be unlabelled or inconsistent; newcomers struggle to find easy work.
- No PR or issue templates to guide contributors.

## Suggested changes

### Add Issue templates (.github/ISSUE_TEMPLATE/*.md)
`bug_report.md` and `feature_request.md` (basic content) — example:

```yaml
# .github/ISSUE_TEMPLATE/bug_report.md
name: Bug report
about: Create a report to help us fix bugs
---
**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. ...
2. ...

**Expected behavior**
A clear and concise description.

**Environment (please complete the following information):**
 - OS: [e.g. Ubuntu 22.04]
 - Python version: [e.g. 3.10]
 - OM1 commit: [git sha]
Add a "smoke" test for examples (tests/test_smoke_examples.py)
def test_spot_sim_config_loads():
    from om1 import load_config
    cfg = load_config('config/spot.json5')
    assert 'robot' in cfg
Commit message suggestion
ci: add GitHub Actions for lint & tests and add simple smoke test for example configs

**Additional context**
Add any other context about the problem here.

Add labels (recommended set)

good first issue

help wanted

documentation

bug

enhancement

priority/high

needs triage
You can add label descriptions to explain what type of tasks go into each.
Add a PR template .github/PULL_REQUEST_TEMPLATE.md
## What this PR does
(Concise summary)

## Related issues
Fixes #<issue-number> / Related to #<issue>

## How to test
Steps to reproduce or commands to run

## Checklist
- [ ] I have run `pre-commit`
- [ ] I have added/updated tests
- [ ] I have updated documentation if needed
Commit message suggestion
chore(.github): add issue & PR templates and recommended labels


*   **Report Bugs:** If you find a bug, please [open an issue](https://github.com/OpenmindAGI/OM1/issues) on GitHub. Be sure to include:
    *   A clear and concise description of the bug.
    *   Steps to reproduce the bug.
    *   Your operating system and Python version.
    *   Relevant error messages or stack traces.
    *   Screenshots (if applicable).

*   **Suggest Features:**  Have an idea for a new feature or improvement?  [Open an issue](https://github.com/OpenmindAGI/OM1/issues) on GitHub and describe your suggestion. Explain the motivation behind the feature and how it would benefit OM1 users.  We encourage discussion on feature requests before implementation.

*   **Improve Documentation:**  Good documentation is crucial.  If you find anything unclear, incomplete, or outdated in the documentation, please submit a pull request with your changes. This includes the README, docstrings, and any other documentation files. Visit [OM1 docs](https://docs.openmind.org/), and [source code](https://github.com/OpenmindAGI/OM1/tree/main/docs).

*   **Fix Bugs:** Browse the [open issues](https://github.com/OpenmindAGI/OM1/issues) and look for bugs labeled "bug" or "help wanted." If you want to tackle a bug, comment on the issue to let us know you're working on it.

*   **Implement Features:**  Check the [open issues](https://github.com/OpenmindAGI/OM1/issues) for features labeled "enhancement" or "bounty" or "help wanted".  It's best to discuss your approach in the issue comments *before* starting significant development.

*   **Write Tests:**  OM1 aims for high test coverage.  If you're adding new code, please include corresponding tests. If you find areas with insufficient test coverage, adding tests is a great contribution.

*   **Code Review:** Reviewing pull requests is a valuable way to contribute.  It helps ensure code quality and maintainability.

**Contribution Workflow (Pull Requests):**

1.  **Fork the Repository:**  Click the "Fork" button on the top-right of the OM1 repository page to create your own copy.

2.  **Clone Your Fork with CLI:**
    ```bash
    git clone [https://github.com/](https://github.com/)<your-username>/OM1.git
    cd OM1
    ```
    (Replace `<your-username>` with your GitHub username.)

3.  **Create a Branch:**  Create a new branch for your work.  Use a descriptive name that reflects the purpose of your changes (e.g., `fix-bug-xyz`, `add-feature-abc`, `docs-improve-readme`).
    ```bash
    git checkout -b your-branch-name
    ```

4.  **Make Changes:**  Make your code changes, add tests, and update documentation as needed.

5.  **Commit Changes:**  Commit your changes with clear and concise commit messages.  Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification if possible (e.g., `feat: Add new feature`, `fix: Correct bug in module X`, `docs: Update README`).
    ```bash
    git commit -m "feat: Add support for XYZ"
    ```

6.  **Push Changes:** Push your branch to your forked repository.
    ```bash
    git push origin your-branch-name
    ```

7.  **Create a Pull Request (PR):**  Go to the [original OM1 repository](https://github.com/OpenmindAGI/OM1/) on GitHub. You should see a prompt to create a pull request from your newly pushed branch.  Click "Compare & pull request."

8.  **Write a Clear PR Description:**
    *   Describe the purpose of your pull request.
    *   Link to any relevant issues it addresses (e.g., "Closes #123").
    *   Explain your changes and your design choices.
    *   Include any relevant screenshots or GIFs (if applicable).

9.  **Request Review:**  Your pull request will be reviewed by the maintainers.  Be prepared to address any feedback or make further changes.

10. **Merge:** Once your pull request is reviewed and approved, it will be merged into the main branch.

**Coding Style and Conventions:**

*   **Code Style:**  Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide.  We may use a code formatter like `black` or `ruff` (check the `pyproject.toml` or `setup.cfg` for project-specific configuration).  Run `pre-commit run --all-files` before committing.
*   **Docstrings:**  Write clear and comprehensive docstrings for all functions, classes, and modules.  We may use a specific docstring format (e.g., Google style, NumPy style).
*   **Tests:** Write unit tests to ensure your code works as expected.  Use a testing framework like `pytest`.
*   **Type Hints:** Use type hints (PEP 484) to improve code readability and maintainability.

**Code of Conduct:**

Please review and adhere to our [Code of Conduct](./). We expect all contributors to be respectful and inclusive.

**Getting Help:**

If you have any questions or need help with the contribution process, feel free to:

*   Open an issue on GitHub.
*   Ask questions in the comments of relevant issues or pull requests.
*   Join our [developer telegram group](https://t.me/openminddev).

Thank you for contributing to OM1!
