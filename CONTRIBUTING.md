## Contributing to OM1

We welcome contributions from the community! **OM1** is an open-source project, and we appreciate your help in making it better. Whether you’re fixing bugs, adding features, improving documentation, or suggesting ideas — your input is valuable.

Before you contribute, please read these guidelines. They help ensure a smooth process and consistent collaboration.

---

### 💡 Ways to Contribute

#### 🐛 Report Bugs
If you find a bug, please [open an issue](https://github.com/OpenmindAGI/OM1/issues) with:
- A clear, concise description.
- Steps to reproduce.
- Your OS and Python version.
- Relevant error messages or stack traces.
- Screenshots, if applicable.

#### 💡 Suggest Features
Got an idea? [Open an issue](https://github.com/OpenmindAGI/OM1/issues) and describe:
- The problem or motivation.
- How it benefits users.
- Optional: early design thoughts.

We encourage discussion before implementation.

#### 📝 Improve Documentation
If anything is unclear, outdated, or missing, please submit a pull request!  
Docs live in the [OM1 Docs](https://docs.openmind.org/) and [source code](https://github.com/OpenmindAGI/OM1/tree/main/docs).

#### 🔧 Fix Bugs
Browse [open issues](https://github.com/OpenmindAGI/OM1/issues) labeled `bug` or `help wanted`.  
Comment on the issue to claim it.

#### 🚀 Implement Features
Look for issues labeled `enhancement`, `bounty`, or `help wanted`.  
Discuss your plan in comments before coding.

#### 🧪 Write Tests
We aim for **high test coverage**. Add or improve tests in areas that need them.

#### 👀 Code Review
Reviewing pull requests helps maintain quality and consistency — and is highly appreciated!

---

### 🧭 Contribution Workflow

1. **Fork the Repository**  
   Click **Fork** on the [OM1 repo](https://github.com/OpenmindAGI/OM1/).

2. **Clone Your Fork**
   ```bash
   git clone https://github.com/<your-username>/OM1.git
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
