# Contributing to biomodal_codebase

We welcome contributions from our team members and from other teams within the company who are interested in collaborating on this unified codebase. This document provides guidelines for contributing to the `biomodal_codebase` repository. Please follow these guidelines to maintain the coherence and quality of the code.

## Getting Started

Before contributing, make sure you are familiar with the project's goals and the existing codebase. All experimental designs and planned code integrations are documented on our Notion page. If you wish to propose a new experiment or feature that is not listed on the Notion page, please add your proposal there first for discussion.

### Prerequisites

- Access to the project's [Notion page](https://www.notion.so/Code-specifications-509accc4ffc24abb95c77fe134e51c7a).
- Familiarity with Git and GitHub workflows.

## Contribution Workflow

### Step 1: Propose Your Experiment

1. Visit our [Notion page](https://www.notion.so/Code-specifications-509accc4ffc24abb95c77fe134e51c7a) and ensure your experiment or feature is not already listed or in discussion.
2. Add your proposal to the [Notion page](https://www.notion.so/Code-specifications-509accc4ffc24abb95c77fe134e51c7a) if it is a new idea. Wait for the team's feedback or for someone to undertake its implementation.

### Step 2: Code Development

Once your proposal is accepted or if you are implementing an already approved experiment:

1. **Clone the repository**: Start by cloning the repository to your local machine.

```
git clone git@github.com:valence-labs/biomodal_codebase.git
```

2. **Select the appropriate branch**: 
- The `main` branch holds stable, running code used for trainings and experiments.
- The `dev` branch is for developing new features.
- Based on your focus, select from one of the specific branches (`transcriptomics`, `phenomics`, `perturbants`, `multimodality`) forked from `dev`.

3. **Create your feature branch**:

```
git checkout -b your-branch-name [parent-branch-name]
```

Example: If you are working on a transcriptomics feature:

```
git checkout -b feature-awesome-transcriptomics transcriptomics
```

A good naming practice is to keep the name of the parent branch in the complete name of the child branch, to avoid potential confusions.

### Step 3: Development and Testing

- Implement your feature or bug fix.
- Ensure your code adheres to the existing code structure and coding standards.
- Add docstrings defining usage, input and output to all your added/modified function.
- Test your code thoroughly to ensure stability and compatibility.

### Step 4: Submitting Changes

1. Push your changes to your forked branch on GitHub.

```
git push origin your-branch-name
````

2. Create a Pull Request (PR) to the relevant branch (e.g., `transcriptomics`, `phenomics`). You can find templates relevant to any type of PR in this [folder](.github/PULL_REQUEST_TEMPLATE/)
3. Provide a clear description of the changes and reference any related Notion discussions or documents.

## Code Review Process

- Pull Requests require at least one review from the team members.
- Reviewers will provide feedback and may request changes to the PR.
- Once approved, a designated repository maintainer will merge the PR into the target branch.

## Guidelines

- **Code Quality**: Aim for clarity, efficiency, and adherence to in-house coding standards.
- **Documentation**: Update the README.md and other documentation as necessary.

Thank you for contributing to the Biomodality codebase.
