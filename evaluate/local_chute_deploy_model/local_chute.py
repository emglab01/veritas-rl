import os
from chutes.chute import NodeSelector
from chutes.chute.template.sglang import build_sglang_chute
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = build_sglang_chute(
    username="<chutes_username>",
    readme="kiwikiw/Affine-0001-5GxTqXLzESa6FThGdcfHANa1b8XmafCshj4yw7PVKwDZuUE2",
    model_name="kiwikiw/Affine-0001-5GxTqXLzESa6FThGdcfHANa1b8XmafCshj4yw7PVKwDZuUE2",
    image="chutes/sglang:0.5.1.post3",
    concurrency=16,
    revision="09b10414a8c84292006431fd7ede2970d85379ce",
    node_selector=NodeSelector(
        gpu_count=1,
        include=["h200"],
    ),
)