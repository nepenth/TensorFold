"""The NVFP4 serial path's allocation contract, checked without a GPU."""

from tensorfold.cuda.geometry import nvfp4_inventory, split_weights
from tensorfold.families.glm5_next.cuda.split import rule


def test_inventory_has_one_owner_and_nine_fp32_slots():
    inventory = nvfp4_inventory()
    assert inventory["storage"] == "pointer-table"
    assert inventory["duplicate_expert_storage"] == 0
    assert inventory["draft_head"] == 0
    assert inventory["bf16_ey"] == 0
    assert inventory["shared_slot"] == 8
    assert inventory["fp32_slots"] == 2048 * 9 * 4096 * 4


def test_nvfp4_admission_drops_the_draft_head_term():
    info = {"dtype": "BF16", "shape": [77440, 4096]}
    keep = split_weights(rule)("lm_head.weight", info)[0]
    drop = split_weights(rule, draft_head=False)("lm_head.weight", info)[0]
    half = 77440 // 2
    assert keep - drop == half * 4096 * 9 // 16
    assert drop == half * 4096 * 2
