---
language:
- en
license: other
task_categories:
- text-generation
pretty_name: Lincoln Style Chat Data
---

# Lincoln Style Chat Data

This private dataset contains supervised chat examples from Abraham Lincoln's public-domain speeches and letters.

The source text comes from Project Gutenberg eBook 14721, *Speeches & Letters of Abraham Lincoln, 1832-1865*.

The dataset includes topic-response, passage-continuation, and reviewed modern-domain tasks.

The train split contains 381 historical rows and 16 synthetic rows across 16 modern domains. The validation split contains 23 historical rows.

The split reserves 10% of source document titles for validation. All examples with the same title stay in one split. Synthetic examples appear only in the train split.

Synthetic rows include `synthetic`, `modern_domain`, and `historical_basis` fields. They distinguish documented principles from speculative analysis.

The system prompt requires a transparent historical writing assistant. It forbids false identity claims and invented quotations.

Source: https://www.gutenberg.org/ebooks/14721

Project Gutenberg marks the source text as public domain in the United States. Check the applicable law outside the United States.
