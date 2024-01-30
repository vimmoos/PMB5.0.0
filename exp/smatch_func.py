"""
Define a function that computes the smartch++ score between two strings.
Uses the smatch++ code from the PMB 5.0.0 repo.
"""

import sys
sys.path.append("../data/pmb/src/sbn/")
from sbn_smatch import SBNGraph
from smatch import score_amr_pairs


def compute_smatchpp(sbn1: str, sbn2: str, remove_top: bool = True) -> float:
    """Computes the smatch++ score between two SBN strings."""

    # convert sbn string to penman string
    try:
        sbn1 = sbn1.strip()
        penman1 = SBNGraph().from_string(sbn1, is_single_line=True).to_penman_string()
    except Exception as e:
        # print(f"sbn1 error: {e}")
        return 0.0

    try:
        sbn2 = sbn2.strip()
        penman2 = SBNGraph().from_string(sbn2, is_single_line=True).to_penman_string()
    except Exception as e:
        # print(f"sbn2 error: {e}")
        return 0.0

    # compute smatch score
    f1_score = 0.0
    try:
        penman1.replace("\n", " ")
        penman2.replace("\n", " ")
        precision, recall, f1_score = next(
            score_amr_pairs([penman1], [penman2], remove_top=remove_top)
        )
    except Exception as e:
        print(f"smatch error: {e}")
        pass

    return f1_score


def main() -> None:
    """Used to test the smatch function."""

    # load test data
    with open(
        "../data/pmb/seq2seq/nl/test/standard.sbn",
        "r",
        encoding="utf-8",
    ) as inp:
        # take only the DRS, not the input text
        sbn_data1 = [line.split("\t")[1].strip() for line in inp.readlines()]

    with open(
        "../src/model/DRS-MLM/result/MLM_nl_standard.txt",
        "r",
        encoding="utf-8",
    ) as inp:
        sbn_data2 = [line.strip() for line in inp.readlines()]

    # compute smatch score
    sample_num = 3  # change this number to test different samples
    sbn1, sbn2 = sbn_data1[sample_num], sbn_data2[sample_num]
    score = compute_smatchpp(sbn1, sbn2)

    print(f"SBN1: {sbn_data1[sample_num]}\nSBN2: {sbn_data2[sample_num]}")
    print(f"Smatch++ score: {score}")


if __name__ == "__main__":
    main()
