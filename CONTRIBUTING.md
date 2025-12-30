# Contributing to Depression Detection NLP System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Code of Conduct

Please be respectful and constructive in all interactions related to this project.

## How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- A clear title and description
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)
- Any relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please open an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Any implementation ideas you have

### Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the coding standards below
3. **Add tests** if applicable
4. **Update documentation** if needed
5. **Ensure tests pass** by running `pytest tests/`
6. **Submit a pull request** with a clear description of the changes

## Coding Standards

### Python Style Guide

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and concise

### Docstring Format

Use Google-style docstrings:

```python
def function_name(arg1: str, arg2: int) -> bool:
    """
    Short description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When something goes wrong
    """
    pass
```

### Code Organization

- Keep related functionality together
- Use meaningful module and file names
- Maintain separation of concerns
- Follow existing project structure

## Testing

- Write tests for new features
- Ensure existing tests pass
- Aim for meaningful test coverage
- Use pytest for testing

Run tests with:
```bash
pytest tests/
```

## Documentation

- Update README.md if adding new features
- Update API.md for API changes
- Add examples in notebooks if applicable
- Keep documentation clear and concise

## Commit Messages

Write clear commit messages:
- Use present tense ("Add feature" not "Added feature")
- Keep first line under 72 characters
- Add detailed description if needed
- Reference issues and PRs when applicable

Example:
```
Add SHAP explanation caching feature

- Implement caching mechanism for SHAP values
- Add configuration option for cache size
- Update documentation

Fixes #123
```

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/Simeonakwu/Predicting-Mental-Health-Risk-from-Patient-Storytelling-Patterns.git
cd Predicting-Mental-Health-Risk-from-Patient-Storytelling-Patterns
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

4. Install development tools:
```bash
pip install pytest black flake8
```

## Code Review Process

All contributions will be reviewed by maintainers. We look for:
- Code quality and readability
- Adherence to project standards
- Adequate testing
- Clear documentation
- No breaking changes (unless discussed)

## Questions?

Feel free to open an issue for any questions or clarifications needed.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing! 🎉
