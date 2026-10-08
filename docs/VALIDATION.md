# Release validation

The usability update was checked on 2026-10-08 on Windows with Python 3.12.10 in a new virtual environment installed from `requirements.txt`. The installed packages are recorded in `requirements-lock.txt`.

| Check | Observed result |
| --- | --- |
| Paired CSV example | Eight synthetic task pairs produced a report, summary, normalized stream, decision and trace records, and run metadata. |
| CSV workflow tests | All 14 test methods passed, including malformed inputs, output preservation, score direction, arbitrary score scales, group coverage, and paths with spaces. |
| CSV engine parity and prefix isolation | Decision records matched the existing generic engine exactly. Changing later scores did not change the earlier selected-prefix record, apart from its input-derived run identifier. |
| Original JSONL example | Two terminal decision records and six trace records were produced. Both files matched the pre-update files byte for byte. |
| Stream contract test | Passed. All 13 malformed-stream cases were rejected, including run, system, and supplied source mismatches. |
| Terminal-Bench adapter verification | Schema, batch equivalence, prefix isolation, budget accounting, trace integrity, audit-field separation, and invalid-stream checks passed. |

These checks exercise the packaged interface. They do not rerun the full paper analyses, establish an anytime-valid stopping guarantee, or constitute independent reproduction of the paper's numerical results. OSWorld input rows are included with their source audit; its full replay was not rerun during packaging.

The release manifest records file hashes. Run `python reproduce.py --verify` after copying or checking out the repository. Git attributes preserve line endings so that a checkout retains those bytes. The GitHub Actions workflow runs the contract checks, paired CSV example, and manifest verification on Windows and Linux. Hosted run status is available in the repository's Actions tab.

In artifact-review terms, availability is partial for the full paper because some inputs must be obtained separately. Local functionality checks passed for the paths above. Full result reproduction has not been tested in this release preparation.
