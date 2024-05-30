import torch
from diffusers import StableDiffusionPipeline
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default=None, type=str, help="label that you want to generate")
    parser.add_argument("--checkpoint", default=None,type=str)
    args = parser.parse_args()
    return args

args = parse_args()
pipe = StableDiffusionPipeline.from_pretrained(args.checkpoint, torch_dtype=torch.float16)
pipe = pipe.to("cuda")
prompt = "A photo of {}, photorealistic".format(args.label)
image = pipe(prompt).images[0]

image.save("./mmdetection/infer_imgs/{}_2.jpg".format(args.label))