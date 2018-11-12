# Contributing to SafeAI
We appreciate your interest in this project! We want to make contributing to this project 
as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing the new model or additional features in model
- Becoming a maintainer

## We follow [Git Flow](https://guides.github.com/introduction/flow/index.html), So All Code Changes Happen Through Pull Requests
Pull requests are the best way to propose changes to the codebase 
(we use [Github Flow](https://guides.github.com/introduction/flow/index.html)). 
We actively welcome your pull requests:

1. Fork the repo and create your own branch from `master`.
2. If you've added code that should be tested, add tests so that it prevents the bugs from future development.
To see the example test code for uncertainty model: [safeai/tests/test_joint_confident.py](https://github.com/EpiSci/SafeAI/blob/master/safeai/tests/test_joint_confident.py)
3. Ensure the test suite passes with command `python -m unittest discover`, executed at the root of the project directory
4. Make sure your code lints.(Check [README.md - Pylint](https://github.com/EpiSci/SafeAI/blob/master/README.md))
5. Issue that pull request!

## Any contributions you make will be under the MIT Software License
In short, when you submit code changes, your submissions are understood to be under the same [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.

## Report bugs using Github's [issues](https://github.com/EpiSci/SafeAI/issues)
We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/EpiSci/SafeAI/issues/new); it's that easy!

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Please be specific!
  - Give sample code if you can. [This example issue in tensorflow/tensorflow](https://github.com/tensorflow/tensorflow/issues/22793) 
  includes short code that anyone with a base setup can run to reproduce what the issuer was seeing
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

People *love* thorough bug reports. I'm not even kidding.

## Use a Consistent Coding Style

* 4 spaces for indentation rather than tabs
* You can try running `pylint` for style unification

## License
By contributing, you agree that your contributions will be licensed under its MIT License.

## References
This document was adapted from the open-source contribution guidelines for [Facebook's Draft](https://github.com/facebook/draft-js/blob/a9316a723f9e918afde44dea68b5f9f39b7d9b00/CONTRIBUTING.md), 
and this [Gist by briandk](https://gist.github.com/briandk/3d2e8b3ec8daf5a27a62)
