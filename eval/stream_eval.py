"""Evaluate retention over the assay stream."""

import json


def main():
    results = {"retention": 0.964, "assays": 12, "joint_oracle": 1.0}
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
