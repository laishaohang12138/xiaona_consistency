# Custom GPT Knowledge Policy

## Confirmed Policy
- Custom GPT knowledge ingests frozen rule documents only
- Dynamic batch data does not enter knowledge

## Allowed Into Knowledge
- Project charter
- Anchor registry
- Body constitution spec
- Training layer doctrine
- Review rubric
- Prompt architecture rules
- Minimal patch playbook
- Stable QA reason dictionary

## Not Allowed Into Knowledge
- `outputs/qa_report.json`
- Raw batch result dumps
- Candidate image sets
- Temporary threshold experiments
- Unfrozen retrospective notes

## Publication Rule
- `docs/` is the canonical authoring layer
- `kb_export/` is the publication layer for Custom GPT
- Every knowledge export should be generated from frozen docs, not from memory

