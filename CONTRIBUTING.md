# Contributing to Stable Outcome Reward Modeling

Thank you for your interest in contributing! This project aims to advance research in pairwise preference learning for agentic reasoning systems. We welcome contributions of all kinds.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guidelines](#style-guidelines)
- [Reporting Issues](#reporting-issues)

---

## Code of Conduct

This project follows a simple code of conduct: **be respectful, be constructive, and be collaborative**. We are committed to providing a welcoming environment for everyone.

---

## How Can I Contribute?

### 🐛 Bug Reports

Found a bug? Please open an issue with:
- A clear, descriptive title
- Steps to reproduce the behavior
- Expected vs. actual behavior
- Environment details (Python version, PyTorch version, GPU, OS)
- Relevant logs or error messages

### 💡 Feature Requests

Have an idea? We'd love to hear it! Open an issue with:
- A clear description of the feature
- The motivation and use case
- Any implementation ideas you have

### 🔬 Research Contributions

We especially welcome contributions that:
- Extend evaluation to new domains (code, science, multi-turn)
- Improve training stability or efficiency
- Add new robustness tests or metrics
- Integrate with other agentic frameworks
- Provide multilingual support

### 📖 Documentation

Help improve our docs:
- Fix typos or clarify explanations
- Add usage examples
- Improve code comments
- Write tutorials or guides

### 🧪 Code Contributions

- Bug fixes
- Performance improvements
- New features
- Test coverage improvements

---

## Development Setup

### 1. Fork and Clone

```bash
# Fork the repo on GitHub, then:
git clone https://github.com/YOUR_USERNAME/stable-outcome-reward-modeling.git
cd stable-outcome-reward-modeling
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install in development mode
pip install -r requirements.txt

# Install development dependencies (if available)
pip install pytest black isort flake8
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

---

## Pull Request Process

### Before Submitting

1. **Test your changes**: Ensure existing tests pass and add new tests if applicable
   ```bash
   pytest tests/  # if tests exist
   ```

2. **Format your code**: Follow our style guidelines
   ```bash
   black src/ tools/
   isort src/ tools/
   ```

3. **Update documentation**: If your changes affect usage, update the README or relevant docs

4. **Write clear commit messages**: Use descriptive commit messages
   ```
   feat: add length-stratified evaluation metrics
   fix: handle edge case in pairwise loss computation
   docs: clarify training configuration options
   ```

### Submitting

1. Push your branch to your fork
   ```bash
   git push origin feature/your-feature-name
   ```

2. Open a Pull Request against the `main` branch

3. Fill out the PR template with:
   - Description of changes
   - Motivation and context
   - How it was tested
   - Any breaking changes

### Review Process

- A maintainer will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

---

## Style Guidelines

### Python Code

- **Formatter**: Use [Black](https://github.com/psf/black) with default settings
- **Imports**: Use [isort](https://github.com/PyCQA/isort) for import sorting
- **Linting**: Code should pass [flake8](https://flake8.pycqa.org/) checks
- **Type hints**: Encouraged for function signatures
- **Docstrings**: Use Google-style docstrings for public functions

```python
def compute_pairwise_accuracy(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor
) -> float:
    """Compute pairwise ranking accuracy.
    
    Args:
        chosen_scores: Tensor of scores for preferred examples.
        rejected_scores: Tensor of scores for rejected examples.
    
    Returns:
        Accuracy as a float between 0 and 1.
    """
    return (chosen_scores > rejected_scores).float().mean().item()
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code restructuring |
| `test` | Adding or updating tests |
| `chore` | Maintenance tasks |

---

## Reporting Issues

### Bug Reports

Use the bug report template:

```markdown
**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
1. Run '...'
2. With config '...'
3. See error

**Expected behavior**
What you expected to happen.

**Environment**
- OS: [e.g., Ubuntu 22.04]
- Python: [e.g., 3.10]
- PyTorch: [e.g., 2.1.0]
- GPU: [e.g., A100 40GB]

**Logs**
```
Paste relevant error logs here
```
```

### Feature Requests

```markdown
**Is your feature request related to a problem?**
A clear description of the problem.

**Describe the solution you'd like**
What you want to happen.

**Describe alternatives you've considered**
Other solutions you've thought about.

**Additional context**
Any other information.
```

---

## Questions?

- Open a [Discussion](https://github.com/Coder-12/stable-outcome-reward-modeling/discussions) for general questions
- Reach out on Twitter: [@iminevitable10](https://x.com/iminevitable10)
- Email: akleshmishra7@gmail.com

---

## Recognition

Contributors will be acknowledged in:
- The README contributors section
- Release notes for significant contributions
- Paper acknowledgments for research contributions

---

Thank you for contributing to advancing research in preference learning and agentic reasoning! 🚀
