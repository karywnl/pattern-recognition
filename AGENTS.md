# Repository Guide

## Purpose

This repository contains coursework for **26AI5707: Pattern Recognition and
Machine Learning**. The repository owner is a master's student in Artificial
Intelligence and Data Science at **Shiv Nadar University, Chennai
(Kalavakkam)**.

The course instructor is **Hema Arunachalam Murthy**, whom the student
identifies as a retired IIT Madras professor.

The goal of this repository is practical and academic: understand each
assignment, implement the required algorithms, generate results and figures,
and prepare submission-ready code and reports.

## Repository Layout

- `docs/assignments/` contains the official assignment PDFs and supplied data.
  Treat these files as the source of truth for each lab.
- `lab1.py`, `lab2.py`, and future `lab<N>.py` files are the runnable lab
  drivers.
- `src/` contains reusable implementations for linear algebra, probability,
  image processing, and plotting.
- `docs/submissions/lab<N>/` contains report sources, generated figures, and
  submission-ready PDFs.
- `test.py` is a scratch diagnostic file. Leave it alone unless the student
  explicitly asks for changes.

## Working Principles

- Read the relevant assignment in `docs/assignments/` before changing a lab.
- Keep solutions small and appropriate for coursework; avoid unnecessary
  architecture or abstraction.
- Follow the instructor's hand-coding requirements. Do not replace an assigned
  algorithm with a library shortcut unless the student has explicitly allowed
  that exception.
- Use snake_case for functions and multi-word variables. Capitalized
  single-letter names such as `A`, `Q`, and `U` are acceptable for standard
  mathematical notation.
- Keep code comments short. Put conceptual explanations in the report or the
  conversation, not in long code comments.
- Run the relevant lab after changing code and verify the numerical results.
- Keep generated figures reproducible and use the filenames referenced by the
  corresponding LaTeX report.

## Reports and Student Voice

The student writes report explanations in their own words. Preserve that voice.

- Do not rewrite, polish, or correct unrelated report sections unless directly
  requested.
- When asked to edit one topic or page, change only that topic or page.
- Prefer checking factual correctness over making the prose sound formal.
- Do not add long AI-written analysis to the report. Explain concepts to the
  student first so they can express the idea themselves.
- Tables, equations, figures, formatting, compilation, and small grammar fixes
  may be handled when requested.
- Never reuse prose or analysis from another student's report.

## Teaching Preference

For conceptual questions, explain the idea directly from first principles in
plain language. Define jargon before using it and connect the explanation to
the actual code and measured values. For an exercise whose purpose is for the
student to implement the algorithm, guide them without immediately replacing
their work with a complete library-based solution.
