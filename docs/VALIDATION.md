# Release validation

The following checks ran locally on Windows with Python 3.12.10 in a new virtual environment. The installed core packages are recorded in `requirements-lock.txt`.

| Check | Observed result |
| --- | --- |
| Synthetic example | Two terminal decision records and six trace records were produced. |
| Synthetic contract test | Passed. All six malformed-stream cases were rejected. |
| Terminal-Bench compact-data pipeline | Passed, including stream and record validation and comparison with the batch driver. |
| Terminal-Bench adapter verification | Schema, batch equivalence, prefix isolation, budget accounting, trace integrity, audit-field separation, and invalid-stream checks passed. |

These checks exercise the packaged interface. They do not rerun the full paper analyses, establish an anytime-valid stopping guarantee, or constitute independent reproduction of the paper's numerical results. OSWorld input rows are included with their source audit; its full replay was not rerun during packaging.

The release manifest records file hashes. Run `python reproduce.py --verify` after copying or checking out the repository. Git attributes preserve line endings so that a checkout retains those bytes. The GitHub Actions workflow is supplied for subsequent runs; no hosted Actions run was observed during local preparation.

In artifact-review terms, availability is partial for the full paper because some inputs must be obtained separately. Local functionality checks passed for the paths above. Full result reproduction has not been tested in this release preparation.
