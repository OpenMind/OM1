## Contributing to OM1

We welcome contributions from the community!  OM1 is an open-source project, and we appreciate your help in making it better.  Whether you're fixing bugs, adding features, improving documentation, or suggesting new ideas, your contributions are valuable.

Before contributing, please take a moment to read through the following guidelines. This helps streamline the process and ensures everyone is on the same page.

**PRs must clearly state the problem being solved. Changes without a clear problem statement may be closed without review.**

**Ways to Contribute:**

*   **Report Bugs:** If you find a bug, please [open an issue](https://github.com/OpenMind/OM1/issues) on GitHub. Be sure to include:
    *   A clear and concise description of the bug.
    *   Steps to reproduce the bug.
    *   Your operating system and Python version.
    *   Relevant error messages or stack traces.
    *   Screenshots (if applicable).

*   **Suggest Features:**  Have an idea for a new feature or improvement?  [Open an issue](https://github.com/OpenMind/OM1/issues) on GitHub and describe your suggestion. Explain the motivation behind the feature and how it would benefit OM1 users.  We encourage discussion on feature requests before implementation.

*   **Improve Documentation:**  Good documentation is crucial.  If you find anything unclear, incomplete, or outdated in the documentation, please submit a pull request with your changes. This includes the README, docstrings, and any other documentation files. Visit [OM1 docs](https://docs.openmind.org/), and [source code](https://github.com/OpenMind/OM1/tree/main/docs).

*   **Fix Bugs:** Browse the [open issues](https://github.com/OpenMind/OM1/issues) and look for bugs labeled "bug" or "help wanted." If you want to tackle a bug, comment on the issue to let us know you're working on it.

*   **Implement Features:**  Check the [open issues](https://github.com/OpenMind/OM1/issues) for features labeled "enhancement" or "bounty" or "help wanted".  It's best to discuss your approach in the issue comments *before* starting significant development.

*   **Write Tests:**  OM1 aims for high test coverage.  If you're adding new code, please include corresponding tests. If you find areas with insufficient test coverage, adding tests is a great contribution.

*   **Code Review:** Reviewing pull requests is a valuable way to contribute.  It helps ensure code quality and maintainability.

**Out of Scope**

- Documentation Translations: Multilingual versions of documentation are not supported. PRs translating docs will be closed.

- Stylistic or Minor Changes: Changes that only affect formatting, variable names, or style without functional improvement are out of scope.

- Trivial or Cosmetic Fixes: Small changes that do not fix bugs or meaningfully improve usability are out of scope.

- Opinion-Driven Refactors: Refactors made solely for personal style or preference without measurable benefit are out of scope.

**Contribution Workflow (Pull Requests):**

**Please open an issue for discussion before submitting PRs that introduce new features, refactors, stylistic changes, or dependency upgrades. PRs of this type without prior discussion may be closed. Once approval is granted, you can start working and open a PR.**

1.  **Fork the Repository:**  Click the "Fork" button on the top-right of the OM1 repository page to create your own copy.

2.  **Clone Your Fork with CLI:**
    ```bash
    git clone https://github.com/<your-username>/OM1.git
    cd OM1
    ```
    (Replace `<your-username>` with your GitHub username.)

3. **Setup Development Environment**
    Refer [documentation](https://docs.openmind.org/developing/1_get-started) to setup your development environment.

4.  **Create a Branch:**  Create a new branch for your work.  Use a descriptive name that reflects the purpose of your changes (e.g., `fix-bug-xyz`, `add-feature-abc`, `docs-improve-readme`).
    ```bash
    git checkout -b your-branch-name
    ```

5.  **Make Changes:**  Make your code changes, add tests, and update documentation as needed.

6.  **Commit Changes:**  Commit your changes with clear and concise commit messages.  Follow the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification if possible (e.g., `feat: Add new feature`, `fix: Correct bug in module X`, `docs: Update README`).
    ```bash
    git commit -m "feat: Add support for XYZ"
    ```

7. **Local Testing**

    Install pre-commit and execute `pre-commit install`. This ensures that pre-commit checks run before each commit. Alternatively, you can manually trigger all checks by running

    ```bash
    pre-commit run --all-files
    ```

    To unit test the system, run
    ```bash
    uv run pytest --log-cli-level=DEBUG -s
    ```

8.  **Push Changes:** Once all the tests pass locally, push your branch to your forked repository.
    ```bash
    git push origin your-branch-name
    ```

9.  **Create a Pull Request (PR):**  Go to the [original OM1 repository](https://github.com/OpenMind/OM1/) on GitHub. You should see a prompt to create a pull request from your newly pushed branch.  Click "Compare & pull request."

10.  **Write a Clear PR Description:**
    *   Describe the purpose of your pull request.
    *   Link to any relevant issues it addresses (e.g., "Closes #123").
    *   Explain your changes and your design choices.
    *   Include any relevant screenshots or GIFs (if applicable).

11.  **Request Review:**  Your pull request will be reviewed by the maintainers.  Be prepared to address any feedback or make further changes.

12. **Merge:** Once your pull request is reviewed and approved, it will be merged into the main branch.

**Coding Style and Conventions:**

*   **Code Style:**  Follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide.  We may use a code formatter like `black` or `ruff` (check the `pyproject.toml` or `setup.cfg` for project-specific configuration).  Run `pre-commit run --all-files` before committing.
*   **Docstrings:**  Write clear and comprehensive docstrings for all functions, classes, and modules.  We may use a specific docstring format (e.g., Google style, NumPy style).
*   **Tests:** Write unit tests to ensure your code works as expected.  Use a testing framework like `pytest`.
*   **Type Hints:** Use type hints (PEP 484) to improve code readability and maintainability.

**Code of Conduct:**

We expect all contributors to be respectful and inclusive. Please follow GitHub's community guidelines and maintain a positive, collaborative environment.

**Review Policy**

- Maintainers may close PRs that do not align with project goals
- Closed PRs may not receive detailed feedback

**Getting Help:**

If you have any questions or need help with the contribution process, feel free to:

*   Open an issue on GitHub.
*   Ask questions in the comments of relevant issues or pull requests.
*   Join our [developer telegram group](https://t.me/openminddev).

Thank you for contributing to OM1!
